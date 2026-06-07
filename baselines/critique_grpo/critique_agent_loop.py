"""Critique-GRPO agent loop for VERL.

Implements the Critique-GRPO baseline (Zhang et al., arXiv:2506.03106).

For each prompt, the loop:
    1. Generates N thought chains (first attempts)
    2. For incorrect responses, generates a critique ("incorrect") and
       a refinement using the structured refinement prompt
    3. Selects the best refinement (highest reward, preferring correct ones)
    4. Returns N trajectories: 1 refinement + (N-1) original responses

The refinement is marked as off-policy via extra_fields["is_refinement"]
so the scoring hook can apply the p/(p+gamma) shaping function.

Uses only binary feedback ("simple" critique mode): the model is told
"The generated solution is incorrect" without seeing the ground truth.

Reference:
    Zhang et al. "Critique-GRPO: Advancing LLM Reasoning with Natural
    Language and Numerical Feedback." arXiv:2506.03106.
"""

import asyncio
import logging
import os
from typing import Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
)
from verl.utils.reward_score.math_reward import compute_score as math_compute_score

from training.thought_agent_loop import ThoughtAgentLoop, _ThoughtChainResult

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# --------------------------------------------------------------------------- #
# Critique and refinement prompt templates
# --------------------------------------------------------------------------- #

_SIMPLE_CRITIQUE_CORRECT = "The generated solution is correct."
_SIMPLE_CRITIQUE_INCORRECT = "The generated solution is incorrect."

_REFINEMENT_PROMPT = """Given the following inputs:
**Question**: {question}
**Previous Solution**: {failed_solution}
**Critique**: {critique}

Please re-answer by:
- Correcting potential errors identified in the critique, if they exist.
- Providing clear, step-by-step reasoning.
- Formatting each reasoning step ending in </thought>.
- Formatting the final answer as \\boxed{{answer}}.

Ensure the revised solution addresses all issues raised in the critique."""


# --------------------------------------------------------------------------- #
# Shared buffer for coordinator pattern
# --------------------------------------------------------------------------- #

class _CritiqueBuffer:
    """Per-prompt coordination buffer for Critique-GRPO rollouts."""

    __slots__ = ("trajectories", "stats", "done", "claim_lock", "next_slot")

    def __init__(self):
        self.trajectories: list[dict] = []
        self.stats: dict = {}
        self.done = asyncio.Event()
        self.claim_lock = asyncio.Lock()
        self.next_slot = 0


class CritiqueGRPOAgentLoop(ThoughtAgentLoop):
    """Critique-GRPO: generate responses, critique wrong ones, refine.

    The coordinator (slot 0) generates all N responses, critiques incorrect
    ones, generates refinements, and fills the buffer with (N-1) original
    responses + 1 best refinement. Other slots wait and claim trajectories.

    Extra fields attached:
        - is_refinement (bool): True for the refinement slot
        - critique_triggered (bool): Whether any critique was generated
    """

    _buffers: dict = {}

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        max_thoughts: int = 10,
        max_tokens_per_thought: int = 512,
        thought_delimiter: str = "</thought>",
        use_examples: bool = True,
        **kwargs,
    ):
        super().__init__(
            trainer_config=trainer_config,
            server_manager=server_manager,
            max_thoughts=max_thoughts,
            max_tokens_per_thought=max_tokens_per_thought,
            thought_delimiter=thought_delimiter,
            use_examples=use_examples,
            **kwargs,
        )
        self.rollout_n = self.config.actor_rollout_ref.rollout.n
        logger.info(f"[CritiqueGRPO] Init: rollout_n={self.rollout_n}")

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        """Entry point -- coordinate rollout slots."""
        if sampling_params.get("temperature", 1.0) == 0:
            return await super().run(sampling_params, **kwargs)

        messages = kwargs["raw_prompt"]
        question = self._extract_question(messages)
        ground_truth = kwargs["reward_model"]["ground_truth"]

        buffer_key = question[:200]
        if buffer_key not in self._buffers or self._buffers[buffer_key].done.is_set():
            self._buffers[buffer_key] = _CritiqueBuffer()
        buffer = self._buffers[buffer_key]

        async with buffer.claim_lock:
            my_slot = buffer.next_slot
            buffer.next_slot += 1

        if my_slot == 0:
            logger.info(f"[CritiqueGRPO] Coordinator starting for: {question[:80]}...")
            try:
                trajectories, stats = await self._fill_rollout_buffer(
                    sampling_params, question, ground_truth
                )
                buffer.trajectories = trajectories
                buffer.stats = stats
            except Exception as e:
                logger.error(f"[CritiqueGRPO] Coordinator failed: {e}")
            finally:
                buffer.done.set()

            if buffer.trajectories:
                traj = buffer.trajectories[0]
            else:
                chain = await self._generate_thought_chain(sampling_params, question)
                traj = {"chain": chain, "is_refinement": False}

            if stats.get("critiques_generated", 0) > 0:
                logger.warning(
                    f"[CritiqueGRPO] {question[:60]}... | "
                    f"critiques={stats['critiques_generated']} "
                    f"refinements_correct={stats['refinements_correct']} "
                    f"originals_correct={stats['originals_correct']} "
                    f"total={len(buffer.trajectories)}"
                )
        else:
            await buffer.done.wait()

            if my_slot < len(buffer.trajectories):
                traj = buffer.trajectories[my_slot]
            else:
                chain = await self._generate_thought_chain(sampling_params, question)
                traj = {"chain": chain, "is_refinement": False}

        output = self._chain_result_to_output(traj["chain"])
        output.extra_fields["is_refinement"] = traj["is_refinement"]
        output.extra_fields["critique_triggered"] = buffer.stats.get(
            "critiques_generated", 0
        ) > 0

        return output

    # ------------------------------------------------------------------ #
    # Buffer filling
    # ------------------------------------------------------------------ #

    async def _fill_rollout_buffer(
        self,
        sampling_params: dict,
        question: str,
        ground_truth: str,
    ) -> tuple[list[dict], dict]:
        """Generate N responses, critique+refine wrong ones, mix 7:1.

        Strategy:
            1. Generate N fresh chains
            2. Score each for correctness
            3. For each incorrect chain, generate critique + refinement
            4. Select best refinement (prefer correct, then highest reward)
            5. Return: [best_refinement] + [N-1 original chains]
               If no refinement available (all correct), return N originals.
        """
        n = self.rollout_n
        stats = {
            "originals_correct": 0,
            "critiques_generated": 0,
            "refinements_correct": 0,
        }

        # Step 1: Generate N fresh chains
        chains: list[_ThoughtChainResult] = []
        correctness: list[bool] = []

        for _ in range(n):
            chain = await self._generate_thought_chain(sampling_params, question)
            chains.append(chain)

            if chain.num_thoughts > 0:
                correct = await self._check_correctness(chain, ground_truth)
            else:
                correct = False
            correctness.append(correct)
            if correct:
                stats["originals_correct"] += 1

        # Step 2: Generate critiques + refinements for ALL chains
        best_refinement: Optional[dict] = None
        best_refinement_correct = False
        best_refinement_score = -float("inf")

        # Generate refinements in parallel for all N chains
        refinement_tasks = []
        for idx in range(n):
            refinement_tasks.append(
                self._generate_critique_and_refinement(
                    sampling_params, question, chains[idx],
                    is_correct=correctness[idx],
                )
            )

        refinement_results = await asyncio.gather(
            *refinement_tasks, return_exceptions=True
        )

        for idx, result in enumerate(refinement_results):
            if isinstance(result, Exception):
                logger.warning(
                    f"[CritiqueGRPO] Refinement failed for chain {idx}: {result}"
                )
                continue

            stats["critiques_generated"] += 1
            refinement_chain = result

            if refinement_chain.num_thoughts > 0:
                ref_correct = await self._check_correctness(
                    refinement_chain, ground_truth
                )
                ref_score = 1.0 if ref_correct else 0.0
            else:
                ref_correct = False
                ref_score = 0.0

            if ref_correct:
                stats["refinements_correct"] += 1

            # Select best refinement: prefer correct, then highest score
            if (best_refinement is None
                    or (ref_correct and not best_refinement_correct)
                    or (ref_correct == best_refinement_correct
                        and ref_score > best_refinement_score)):
                best_refinement = {
                    "chain": refinement_chain,
                    "is_refinement": True,
                }
                best_refinement_correct = ref_correct
                best_refinement_score = ref_score

        # Step 3: Assemble buffer — 1 refinement + (N-1) originals
        buffer: list[dict] = []

        if best_refinement is not None:
            # Slot 0: refinement
            buffer.append(best_refinement)
            # Slots 1..N-1: original chains (skip one to maintain N total)
            for i in range(min(n - 1, len(chains))):
                buffer.append({
                    "chain": chains[i],
                    "is_refinement": False,
                })
        else:
            # No refinement (all correct or all refinements failed)
            for chain in chains[:n]:
                buffer.append({
                    "chain": chain,
                    "is_refinement": False,
                })

        return buffer[:n], stats

    # ------------------------------------------------------------------ #
    # Critique and refinement generation
    # ------------------------------------------------------------------ #

    async def _generate_critique_and_refinement(
        self,
        sampling_params: dict,
        question: str,
        failed_chain: _ThoughtChainResult,
        is_correct: bool = False,
    ) -> _ThoughtChainResult:
        """Generate a binary critique and then a refinement.

        Uses "simple" critique mode with binary feedback.
        No ground truth is revealed to the model.

        If the combined prompt would be too long, truncates the failed
        response to fit within the available token budget.
        """
        # Decode the response
        failed_text = await self.loop.run_in_executor(
            None,
            lambda ids=failed_chain.response_ids: self.tokenizer.decode(
                ids, skip_special_tokens=False
            ),
        )

        # Build refinement prompt with binary critique
        critique = _SIMPLE_CRITIQUE_CORRECT if is_correct else _SIMPLE_CRITIQUE_INCORRECT
        refinement_text = _REFINEMENT_PROMPT.format(
            question=question,
            failed_solution=failed_text,
            critique=critique,
        )

        refinement_prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(refinement_text, add_special_tokens=True),
        )

        # Budget check: prompt must leave room for at least
        # max_tokens_per_thought tokens of response. Hard ceiling is
        # max_model_len (vLLM KV cache) minus the response budget.
        max_model_len = self.config.actor_rollout_ref.rollout.get("max_model_len", 8192)
        max_prompt = min(
            max_model_len - self.max_tokens_per_thought,
            self.prompt_length + self.response_length - self.max_tokens_per_thought,
        )
        if len(refinement_prompt_ids) > max_prompt:
            overhead = len(refinement_prompt_ids) - len(failed_chain.response_ids)
            max_resp_tokens = max(0, max_prompt - overhead)
            truncated_ids = failed_chain.response_ids[:max_resp_tokens]
            failed_text = await self.loop.run_in_executor(
                None,
                lambda ids=truncated_ids: self.tokenizer.decode(
                    ids, skip_special_tokens=False
                ),
            )
            refinement_text = _REFINEMENT_PROMPT.format(
                question=question,
                failed_solution=failed_text,
                critique=critique,
            )
            refinement_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(refinement_text, add_special_tokens=True),
            )
            logger.info(
                f"[CritiqueGRPO] Truncated response for refinement prompt: "
                f"{len(refinement_prompt_ids)} tokens (max {max_prompt})"
            )

        # Generate the refinement as a fresh thought chain
        result = await self._generate_thoughts_core(
            list(refinement_prompt_ids), sampling_params
        )

        # Replace prompt_ids with the original question prompt so VERL's
        # batch collation sees the standard prompt_length. The refinement
        # context was only needed for generation, not for training.
        original_prompt_text = self.template.format(question=question)
        original_prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(original_prompt_text, add_special_tokens=True),
        )
        result.prompt_ids = list(original_prompt_ids)
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    async def _check_correctness(
        self, chain: _ThoughtChainResult, ground_truth: str
    ) -> bool:
        decoded = await self.loop.run_in_executor(
            None,
            lambda ids=chain.response_ids: self.tokenizer.decode(
                ids, skip_special_tokens=False
            ),
        )
        return math_compute_score(decoded, ground_truth) > 0.5
