"""SCoRe (Self-Correction via Reinforcement Learning) agent loop for VERL.

Implements the SCoRe baseline (Kumar et al., arXiv:2409.12917, ICLR 2025).

The loop generates two attempts per prompt:
    1. y1: A full thought chain (first attempt)
    2. y2: A correction attempt generated unconditionally for every y1.
       Uses the paper's hedged prompt ("there might be an error ...").

The buffer alternates y1, y2, y1, y2, ... up to n slots. Each y2 carries
metadata ``is_correction=True`` plus y1's correctness flag, so the scoring
hook can apply shaped reward appropriately.

Key differences from SRPO (ThoughtICSAgentLoop):
    - No error localization: SCoRe does NOT identify which step went wrong.
    - Full regeneration: y2 is generated from scratch (with y1 as context),
      not from a backtracked prefix.
    - Hedged correction: the correction prompt does NOT reveal whether y1 is
      correct, preserving the paper's evaluation-time applicability.

Coordinator pattern (slot 0 fills the buffer) is reused from ICS to avoid
redundant generation across rollout slots.

Limitation -- Stage I joint loss:
    The paper's Stage I optimises a *joint* loss that couples y1 and y2
    gradients (Eq. 2, Kumar et al. 2025). VERL's rollout buffer treats each
    trajectory independently, so there is no mechanism to back-propagate a
    joint loss through both y1 and y2 in a single update step. Our Stage I
    approximation therefore zeros y1's reward (via the scoring hook) while
    training y2 independently.  This is a known limitation.

References:
    Kumar et al. "Training Language Models to Self-Correct via Reinforcement
    Learning." ICLR 2025. arXiv:2409.12917.
"""

import asyncio
import logging
import os
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
# Correction prompt template
# --------------------------------------------------------------------------- #

_CORRECTION_PROMPT = (
    "\n\nThere might be an error in the solution above because of lack of "
    "understanding of the question. Please correct the error, if any, and "
    "rewrite the solution. Show your reasoning step by step, with each step "
    "ending in </thought>, and provide your final answer in \\boxed{answer} "
    "format.\n"
)


# --------------------------------------------------------------------------- #
# Shared buffer for coordinator pattern
# --------------------------------------------------------------------------- #

class _SCoReBuffer:
    """Per-prompt coordination buffer for SCoRe rollouts.

    Slot 0 (coordinator) generates all trajectory pairs, then signals done.
    Other slots wait and claim their pre-generated trajectory.
    """

    __slots__ = ("trajectories", "stats", "done", "claim_lock", "next_slot")

    def __init__(self):
        self.trajectories: list[dict] = []
        self.stats: dict = {}
        self.done = asyncio.Event()
        self.claim_lock = asyncio.Lock()
        self.next_slot = 0


class SCoReAgentLoop(ThoughtAgentLoop):
    """SCoRe: generate y1, then unconditionally generate y2 with hedged prompt.

    The buffer alternates y1, y2 pairs. Each rollout slot receives a
    single trajectory. y2 is always generated (regardless of y1
    correctness), matching the paper's evaluation-time protocol.

    SCoRe-specific metadata (score_y1_correct, score_is_correction) is
    attached to extra_fields for the scoring hook / advantage estimator.
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
        logger.info(f"[SCoRe] Init: rollout_n={self.rollout_n}")

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        """Entry point -- coordinate rollout slots for SCoRe generation.

        During validation (temperature=0), skip correction and fall back to
        the parent's standard thought generation.
        """
        if sampling_params.get("temperature", 1.0) == 0:
            return await super().run(sampling_params, **kwargs)

        messages = kwargs["raw_prompt"]
        question = self._extract_question(messages)
        ground_truth = kwargs["reward_model"]["ground_truth"]

        # Get or create buffer
        buffer_key = question[:200]
        if buffer_key not in self._buffers or self._buffers[buffer_key].done.is_set():
            self._buffers[buffer_key] = _SCoReBuffer()
        buffer = self._buffers[buffer_key]

        # Claim a slot atomically
        async with buffer.claim_lock:
            my_slot = buffer.next_slot
            buffer.next_slot += 1

        if my_slot == 0:
            # Coordinator: fill entire rollout buffer
            logger.info(f"[SCoRe] Coordinator starting for: {question[:80]}...")
            stats = {"y1_correct_count": 0, "corrections_attempted": 0, "corrections_successful": 0}
            try:
                trajectories, stats = await self._fill_rollout_buffer(
                    sampling_params, question, ground_truth
                )
                buffer.trajectories = trajectories
                buffer.stats = stats
            except Exception as e:
                logger.error(f"[SCoRe] Coordinator failed: {e}")
            finally:
                buffer.done.set()

            if buffer.trajectories:
                traj = buffer.trajectories[0]
            else:
                logger.warning("[SCoRe] No trajectories, generating emergency fallback")
                chain = await self._generate_thought_chain(sampling_params, question)
                traj = {
                    "chain": chain,
                    "y1_correct": False,
                    "y2_correct": False,
                    "is_correction": False,
                }

            if stats.get("corrections_attempted", 0) > 0:
                logger.warning(
                    f"[SCoRe] {question[:60]}... | "
                    f"corrections={stats['corrections_attempted']} "
                    f"corrected={stats['corrections_successful']} "
                    f"y1_correct_count={stats['y1_correct_count']} "
                    f"total={len(buffer.trajectories)}"
                )
        else:
            # Non-coordinator: wait and claim
            await buffer.done.wait()

            if my_slot < len(buffer.trajectories):
                traj = buffer.trajectories[my_slot]
            else:
                logger.warning(
                    f"[SCoRe] Slot {my_slot}: buffer has {len(buffer.trajectories)} "
                    f"trajectories, generating emergency fallback"
                )
                chain = await self._generate_thought_chain(sampling_params, question)
                traj = {
                    "chain": chain,
                    "y1_correct": False,
                    "y2_correct": False,
                    "is_correction": False,
                }

        output = self._chain_result_to_output(traj["chain"])

        # Attach SCoRe metadata for shaped reward computation
        output.extra_fields["score_y1_correct"] = traj["y1_correct"]
        output.extra_fields["score_y2_correct"] = traj.get("y2_correct", traj["y1_correct"])
        output.extra_fields["score_is_correction"] = traj.get("is_correction", False)

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
        """Fill rollout buffer with n trajectories using SCoRe strategy.

        For each y1 chain we *unconditionally* generate a y2 correction
        attempt (matching the paper's hedged prompt that does not reveal
        correctness).  The buffer alternates: y1, y2, y1, y2, ... until n
        slots are filled.  Each y2 carries ``is_correction=True`` with the
        corresponding y1's correctness so the scoring hook can compute the
        shaped reward.
        """
        n = self.rollout_n
        buffer: list[dict] = []
        stats = {
            "y1_correct_count": 0,
            "corrections_attempted": 0,
            "corrections_successful": 0,
        }

        while len(buffer) < n:
            # Generate y1
            y1 = await self._generate_thought_chain(sampling_params, question)

            if y1.num_thoughts == 0:
                buffer.append({
                    "chain": y1,
                    "y1_correct": False,
                    "y2_correct": False,
                    "is_correction": False,
                })
                continue

            y1_correct = await self._check_correctness(y1, ground_truth)
            if y1_correct:
                stats["y1_correct_count"] += 1

            # Always add y1 to buffer
            buffer.append({
                "chain": y1,
                "y1_correct": y1_correct,
                "y2_correct": False,   # placeholder, updated after y2
                "is_correction": False,
            })

            if len(buffer) >= n:
                break

            # Unconditionally generate y2 correction attempt
            stats["corrections_attempted"] += 1
            y2 = await self._generate_correction(sampling_params, question, y1)

            y2_correct = await self._check_correctness(y2, ground_truth)
            if y2_correct:
                stats["corrections_successful"] += 1

            buffer.append({
                "chain": y2,
                "y1_correct": y1_correct,
                "y2_correct": y2_correct,
                "is_correction": True,
            })

        return buffer[:n], stats

    # ------------------------------------------------------------------ #
    # Correction generation
    # ------------------------------------------------------------------ #

    async def _generate_correction(
        self,
        sampling_params: dict,
        question: str,
        y1: _ThoughtChainResult,
    ) -> _ThoughtChainResult:
        """Generate y2: a fresh thought chain after 'try again' feedback.

        Builds a new prompt:  original_prompt + y1_response + correction_prompt
        Then generates a completely fresh chain (no prefix reuse from y1).

        If the combined prompt would be too long, truncates the y1 response
        to fit within the available token budget (keeping the correction
        prompt and question intact).
        """
        # Decode y1 response to include as context
        y1_text = await self.loop.run_in_executor(
            None,
            lambda ids=y1.response_ids: self.tokenizer.decode(
                ids, skip_special_tokens=False
            ),
        )

        # Build the correction prompt: original question + y1 + "try again"
        prompt_text = self.template.format(question=question)
        correction_text = prompt_text + y1_text + _CORRECTION_PROMPT

        correction_prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(correction_text, add_special_tokens=True),
        )

        # Budget check: correction prompt must leave room for at least
        # max_tokens_per_thought tokens of response. The hard ceiling is
        # max_model_len (vLLM KV cache) minus the response budget.
        max_model_len = self.config.actor_rollout_ref.rollout.get("max_model_len", 8192)
        max_prompt = min(
            max_model_len - self.max_tokens_per_thought,
            self.prompt_length + self.response_length - self.max_tokens_per_thought,
        )
        if len(correction_prompt_ids) > max_prompt:
            # Truncate y1 response IDs to fit
            overhead = len(correction_prompt_ids) - len(y1.response_ids)
            max_y1_tokens = max(0, max_prompt - overhead)
            truncated_y1_ids = y1.response_ids[:max_y1_tokens]
            y1_text = await self.loop.run_in_executor(
                None,
                lambda ids=truncated_y1_ids: self.tokenizer.decode(
                    ids, skip_special_tokens=False
                ),
            )
            correction_text = prompt_text + y1_text + _CORRECTION_PROMPT
            correction_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(correction_text, add_special_tokens=True),
            )
            logger.info(
                f"[SCoRe] Truncated y1 for correction prompt: "
                f"{len(correction_prompt_ids)} tokens (max {max_prompt})"
            )

        # Generate fresh chain from the correction prompt
        result = await self._generate_thoughts_core(
            list(correction_prompt_ids), sampling_params
        )

        # Replace prompt_ids with the original question prompt so VERL's
        # batch collation sees the standard prompt_length. The correction
        # context was only needed for generation, not for training.
        original_prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(prompt_text, add_special_tokens=True),
        )
        result.prompt_ids = list(original_prompt_ids)
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    async def _check_correctness(
        self, chain: _ThoughtChainResult, ground_truth: str
    ) -> bool:
        """Check if chain's answer matches ground truth."""
        decoded = await self.loop.run_in_executor(
            None,
            lambda ids=chain.response_ids: self.tokenizer.decode(
                ids, skip_special_tokens=False
            ),
        )
        return math_compute_score(decoded, ground_truth) > 0.5
