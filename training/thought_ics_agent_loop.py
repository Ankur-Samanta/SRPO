"""Thought ICS agent loop for VERL GRPO.

Integrates Iterative Self-Correction (ICS) into the thought-by-thought generation
pipeline. The coordinator (slot 0) fills the ENTIRE rollout buffer of N slots by
alternating fresh chain generation and ICS correction attempts.

When a fresh chain is wrong, ICS is triggered: localize error, backtrack, and
regenerate. This continues until the chain is corrected or the budget is exhausted,
then the next fresh chain is generated. All slots (fresh and ICS) count as rollouts.

Non-coordinator slots simply wait for the coordinator to finish and claim their
pre-generated trajectory. This ensures no rollout budget is wasted on redundant
independent generation.
"""

import asyncio
import logging
import os
import re
from typing import Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
)
from verl.utils.reward_score.math_reward import compute_score as math_compute_score

from training.branch_logger import dump_branch_group
from training.thought_agent_loop import ThoughtAgentLoop, _ThoughtChainResult

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Datasets that opt out of math-only correctness in the ICS agent loop.
# For each, the value is the score above which a rollout counts as
# correct=True for SRPO's correction-credit semantics. Math-style datasets
# (math500, aime, numina_oly, gpqa, etc.) are intentionally absent —
# they continue to use math_compute_score > 0.5 with no behavior change.
#
# To override at runtime without editing this table, set ICS_RETRO_THRESHOLD
# (or per-dataset env var, see _resolve_correctness_threshold below).
_CORRECTNESS_THRESHOLDS: dict[str, float] = {
    "retrosynthesis_uspto50k": 0.8,  # template-match tier or better
    "livecodebench": 1.0,        # all test cases must pass
    "livecodebench_medium": 1.0,
    "livecodebench_hard": 1.0,
}


def _resolve_correctness_threshold(data_source: str) -> Optional[float]:
    """Return the score threshold for binary correctness on this dataset, or
    None if the dataset uses default math correctness."""
    if data_source not in _CORRECTNESS_THRESHOLDS:
        return None
    # Optional env override for retrosynthesis specifically; future datasets
    # can add their own ICS_<NAME>_THRESHOLD here.
    if data_source == "retrosynthesis_uspto50k":
        env = os.environ.get("ICS_RETRO_THRESHOLD", "").strip()
        if env:
            try:
                return float(env)
            except ValueError:
                logger.warning(f"[ICS] Invalid ICS_RETRO_THRESHOLD={env!r}; using default")
    return _CORRECTNESS_THRESHOLDS[data_source]


class _ICSBuffer:
    """Per-prompt coordination for ICS rollouts.

    Slot 0 (coordinator) fills the entire buffer, then signals done.
    Other slots wait on the event, then claim their pre-generated trajectory.
    """

    __slots__ = ("trajectories", "ics_stats", "loc_tensors", "lppo_tensors", "verifier_tensors", "done", "claim_lock", "next_slot", "buffer_meta")

    def __init__(self):
        self.trajectories: list = []
        self.ics_stats: dict = {}
        self.loc_tensors: Optional[dict] = None
        self.lppo_tensors: Optional[dict] = None
        self.verifier_tensors: Optional[dict] = None
        self.done = asyncio.Event()
        self.claim_lock = asyncio.Lock()
        self.next_slot = 0
        self.buffer_meta: list = []  # parallel to trajectories: {suffix_start_idx, reward}


class ThoughtICSAgentLoop(ThoughtAgentLoop):
    """Thought-by-thought generation with Iterative Self-Correction.

    Coordinates N rollout slots per prompt. Slot 0 (coordinator) fills the
    entire rollout buffer by alternating fresh generation and ICS correction.
    Other slots wait and take their pre-generated trajectory.
    """

    # Class-level buffer shared across all instances, keyed by question text.
    # Safe for asyncio (single-threaded event loop; dict mutations are atomic
    # between await points).
    _buffers: dict = {}

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        max_thoughts: int = 10,
        max_tokens_per_thought: int = 512,
        thought_delimiter: str = "</thought>",
        use_examples: bool = True,
        max_ics_iterations: Optional[int] = None,
        autonomy_level: int = 2,
        use_context: bool = False,
        localization_temp: float = 0.3,
        localization_max_tokens: int = 4096,
        # Localization training params (no-op when train_localization=False)
        train_localization: bool = False,
        localization_rollout_n: int = 4,
        localization_rollout_temp: float = 1.0,
        max_loc_groups_per_prompt: int = 3,
        loc_response_length: int = 512,
        loc_seq_length: int = 4096,
        # Localization PPO params (no-op when train_loc_ppo=False)
        train_loc_ppo: bool = False,
        max_corrections_per_prompt: int = 4,
        # Localization SFT params (no-op when train_loc_sft=False).
        # Reuses the lppo data path: filters to successful corrections and
        # runs NLL on those localization responses (handled in loc_patch.py).
        train_loc_sft: bool = False,
        # Localization KTO. Also reuses the lppo data path. Uses both
        # successful (desirable) and failed (undesirable) corrections with
        # the KTO sigmoid-shaped loss. Anchor = old_logprobs (Option B).
        train_loc_kto: bool = False,
        # Self-verifier shadow-mode training. Verifier runs on every fresh
        # chain but does NOT gate ICS — oracle still drives control flow.
        # Reward = (verifier_predicted == oracle_truth). Three flavors:
        train_verifier_sft: bool = False,
        train_verifier_ppo: bool = False,
        train_verifier_grpo: bool = False,
        train_verifier_kto: bool = False,
        verifier_temp: float = 0.3,
        verifier_grpo_rollout_n: int = 4,
        verifier_grpo_rollout_temp: float = 1.0,
        max_verifier_groups_per_prompt: int = 4,
        # Eval-time verifier flag — wired in a follow-up change. Currently
        # only stored on self for visibility; run() does not consult it yet.
        use_verifier_at_eval: bool = False,
        force_ics_at_eval: bool = False,
        # Random localization baseline: skip LLM localization and pick a
        # uniformly random step in [1, n_steps] instead.
        random_localization: bool = False,
        # Random reset baseline: apply random localization + reset on ALL
        # trajectories (including correct ones), not just failed ones.
        # Forces random_localization=True. Tests whether verification
        # gating matters vs pure resampling diversity.
        random_reset_all: bool = False,
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
        if max_ics_iterations is not None:
            self.max_ics_iterations = max_ics_iterations
        else:
            self.max_ics_iterations = self.rollout_n - 1
        logger.info(
            f"[ICS] Init: max_ics_iterations={self.max_ics_iterations} "
            f"(from_config={max_ics_iterations is not None}, rollout_n={self.rollout_n}, "
            f"autonomy_level={autonomy_level}, use_context={use_context})"
        )
        self.autonomy_level = autonomy_level
        self.use_context = use_context
        self.localization_temp = localization_temp
        self.localization_max_tokens = localization_max_tokens

        # Localization training
        self.train_localization = train_localization
        self.localization_rollout_n = localization_rollout_n
        self.localization_rollout_temp = localization_rollout_temp
        self.max_loc_groups_per_prompt = max_loc_groups_per_prompt
        self.loc_response_length = loc_response_length
        self.loc_seq_length = loc_seq_length
        if train_localization:
            logger.info(
                f"[ICS] Localization training ENABLED: "
                f"rollout_n={localization_rollout_n}, "
                f"rollout_temp={localization_rollout_temp}, "
                f"max_groups={max_loc_groups_per_prompt}"
            )

        # Localization PPO training (reward = correction outcome)
        self.train_loc_ppo = train_loc_ppo
        self.train_loc_sft = train_loc_sft
        self.train_loc_kto = train_loc_kto
        self.max_corrections_per_prompt = max_corrections_per_prompt
        modes_on = sum([
            bool(train_localization), bool(train_loc_ppo),
            bool(train_loc_sft), bool(train_loc_kto),
        ])
        if modes_on > 1:
            raise ValueError(
                "train_localization, train_loc_ppo, train_loc_sft, train_loc_kto "
                "are mutually exclusive"
            )
        if train_loc_ppo:
            logger.info(
                f"[ICS] Localization PPO ENABLED: "
                f"max_corrections={max_corrections_per_prompt}"
            )
        if train_loc_sft:
            logger.info(
                f"[ICS] Localization SFT ENABLED (NLL on successful localizations): "
                f"max_corrections={max_corrections_per_prompt}"
            )
        if train_loc_kto:
            logger.info(
                f"[ICS] Localization KTO ENABLED (sigmoid-shaped loss on success+failure): "
                f"max_corrections={max_corrections_per_prompt}"
            )

        # Self-verifier training (shadow-mode; orthogonal to loc modes)
        self.train_verifier_sft = train_verifier_sft
        self.train_verifier_ppo = train_verifier_ppo
        self.train_verifier_grpo = train_verifier_grpo
        self.train_verifier_kto = train_verifier_kto
        self.verifier_temp = verifier_temp
        self.verifier_grpo_rollout_n = verifier_grpo_rollout_n
        self.verifier_grpo_rollout_temp = verifier_grpo_rollout_temp
        self.max_verifier_groups_per_prompt = max_verifier_groups_per_prompt
        self.use_verifier_at_eval = use_verifier_at_eval
        self.force_ics_at_eval = force_ics_at_eval
        self.random_reset_all = random_reset_all
        if random_reset_all:
            random_localization = True  # force random loc when resetting all
        self.random_localization = random_localization
        if random_localization:
            logger.info("[ICS] Random localization ENABLED: skipping LLM, picking uniform random step")
        if random_reset_all:
            logger.info("[ICS] Random reset ALL ENABLED: resetting all trajectories (including correct)")

        # Optional override for the localization sampling temperature (e.g.
        # ICS_LOC_TEMP=0.0 for greedy). Falls back to the constructor default.
        loc_temp_override = os.environ.get("ICS_LOC_TEMP", "").strip()
        if loc_temp_override:
            try:
                self.localization_temp = float(loc_temp_override)
                logger.info(f"[ICS] Localization temperature override: {self.localization_temp}")
            except ValueError:
                logger.warning(
                    f"[ICS] Invalid ICS_LOC_TEMP={loc_temp_override!r}; keeping {self.localization_temp}"
                )

        if use_verifier_at_eval:
            logger.info(
                f"[ICS] Self-verification gating ENABLED: verifier gates ICS "
                f"instead of oracle (verifier_temp={verifier_temp})"
            )
        v_modes_on = sum([
            bool(train_verifier_sft), bool(train_verifier_ppo),
            bool(train_verifier_grpo), bool(train_verifier_kto),
        ])
        if v_modes_on > 1:
            raise ValueError(
                "train_verifier_sft, train_verifier_ppo, train_verifier_grpo, "
                "train_verifier_kto are mutually exclusive"
            )
        if v_modes_on:
            if train_verifier_sft:
                mode = "sft"
            elif train_verifier_ppo:
                mode = "ppo"
            elif train_verifier_grpo:
                mode = "grpo"
            else:
                mode = "kto"
            logger.info(
                f"[ICS] Self-verifier training ENABLED (mode={mode}, "
                f"temp={verifier_temp}, K={verifier_grpo_rollout_n if train_verifier_grpo else 1})"
            )

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        """Entry point -- coordinate with other rollout slots for ICS.

        During validation (greedy decoding, temperature=0), ICS is skipped
        and we fall back to the parent's standard thought generation so that
        validation accuracy reflects the model's standalone performance.
        """
        # Detect validation: VERL overrides temperature to 0 for val rollouts.
        # ICS requires ground_truth to decide when to trigger correction, so it
        # cannot run at eval time. Fall back to the standard thought_agent loop
        # (super().run), which generates a single chain without correction.
        # This means SRPO eval is identical to TGRPO eval — both use thought_agent.
        # Exception: force_ics_at_eval=True bypasses this for dedicated ICS eval runs.
        if sampling_params.get("temperature", 1.0) == 0 and not self.force_ics_at_eval:
            return await super().run(sampling_params, **kwargs)

        messages = kwargs["raw_prompt"]
        question = self._extract_question(messages)
        ground_truth = kwargs["reward_model"]["ground_truth"]
        # Captured for _check_correctness dispatch on non-math datasets.
        # Unused on math-style datasets (whose data_source isn't in
        # _CORRECTNESS_THRESHOLDS), so behavior there is unchanged.
        self._current_data_source = kwargs.get("data_source", "") or ""
        self._current_extra_info = kwargs.get("extra_info", {}) or {}

        # Get or create buffer for this prompt.  If a buffer exists from a
        # previous batch (done is already set), replace it with a fresh one.
        buffer_key = question[:200]
        if buffer_key not in self._buffers or self._buffers[buffer_key].done.is_set():
            self._buffers[buffer_key] = _ICSBuffer()
        buffer = self._buffers[buffer_key]

        # Claim a slot atomically
        async with buffer.claim_lock:
            my_slot = buffer.next_slot
            buffer.next_slot += 1

        ics_stats = None

        if my_slot == 0:
            # Coordinator: fill entire rollout buffer
            logger.info(
                f"[ICS] Coordinator starting for: {question[:80]}..."
            )
            loc_training_data = []
            loc_ppo_data = []
            verifier_data = []
            try:
                trajectories, ics_stats, loc_training_data, loc_ppo_data, verifier_data, buffer_meta = await self._fill_rollout_buffer(
                    sampling_params, question, ground_truth
                )
                buffer.trajectories = trajectories
                buffer.ics_stats = ics_stats
                buffer.buffer_meta = buffer_meta

                # Opt-in branch-structure dump for offline credit-assignment viz.
                # No-op unless SRPO_BRANCH_DUMP_DIR is set. Silent on errors.
                dump_branch_group(
                    question=question,
                    trajectories=trajectories,
                    ics_stats=ics_stats,
                    ground_truth=ground_truth,
                )

                # Generate localization rollouts if training is enabled
                if self.train_localization and loc_training_data:
                    loc_rollouts = await self._generate_localization_rollouts(
                        loc_training_data
                    )
                    buffer.loc_tensors = self._pack_localization_tensors(loc_rollouts)
                elif self.train_localization:
                    buffer.loc_tensors = self._pack_localization_tensors([])

                # Pack loc PPO/SFT tensors if either is enabled (same data layout)
                if self.train_loc_ppo or self.train_loc_sft or self.train_loc_kto:
                    buffer.lppo_tensors = self._pack_loc_ppo_tensors(loc_ppo_data)

                # Pack verifier tensors based on which mode is enabled
                if self.train_verifier_grpo:
                    if verifier_data:
                        v_rollouts = await self._generate_verifier_grpo_rollouts(
                            verifier_data
                        )
                        buffer.verifier_tensors = self._pack_verifier_grpo_tensors(v_rollouts)
                    else:
                        buffer.verifier_tensors = self._pack_verifier_grpo_tensors([])
                elif self.train_verifier_ppo or self.train_verifier_kto:
                    # vkto reuses the vppo data layout (needs old_logprobs)
                    buffer.verifier_tensors = self._pack_verifier_ppo_tensors(verifier_data)
                elif self.train_verifier_sft:
                    buffer.verifier_tensors = self._pack_verifier_sft_tensors(verifier_data)
            except Exception as e:
                logger.error(f"[ICS] Coordinator failed: {e}")
            finally:
                buffer.done.set()

            # Take first trajectory (or generate fresh on failure)
            if buffer.trajectories:
                result = buffer.trajectories[0]
            else:
                logger.warning("[ICS] Coordinator produced no trajectories, generating emergency fallback")
                result = await self._generate_thought_chain(
                    sampling_params, question
                )

            # Log ICS summary at WARN level so it's always visible
            if ics_stats and ics_stats["ics_triggered"]:
                logger.warning(
                    f"[ICS] {question[:60]}... | "
                    f"triggered=True iters={ics_stats['ics_iterations']} "
                    f"corrected={ics_stats['ics_corrected']} "
                    f"triggers={ics_stats['ics_triggers']} "
                    f"fresh={ics_stats['fresh_chains']} "
                    f"error_steps={ics_stats['ics_error_steps']} "
                    f"trajectories={len(buffer.trajectories)}"
                )
        else:
            # Non-coordinator: wait for coordinator to fill the buffer
            await buffer.done.wait()
            ics_stats = getattr(buffer, "ics_stats", None)

            if my_slot < len(buffer.trajectories):
                result = buffer.trajectories[my_slot]
                logger.info(
                    f"[ICS] Slot {my_slot}: took trajectory "
                    f"({result.num_thoughts} thoughts)"
                )
            else:
                logger.warning(
                    f"[ICS] Slot {my_slot}: buffer has {len(buffer.trajectories)} "
                    f"trajectories but slot needs index {my_slot}, generating emergency fallback"
                )
                result = await self._generate_thought_chain(
                    sampling_params, question
                )

        output = self._chain_result_to_output(result)

        # Attach per-trajectory SRPO metadata (reward + suffix start for gradient masking)
        _srpo_meta = None
        if my_slot < len(buffer.buffer_meta):
            _srpo_meta = buffer.buffer_meta[my_slot]
        output.extra_fields["srpo_reward"] = _srpo_meta["reward"] if _srpo_meta else 0.0
        output.extra_fields["suffix_start_idx"] = _srpo_meta["suffix_start_idx"] if _srpo_meta else 0

        # Attach ICS stats to extra_fields for downstream logging
        if ics_stats is not None:
            output.extra_fields["ics_triggered"] = ics_stats["ics_triggered"]
            output.extra_fields["ics_iterations"] = ics_stats["ics_iterations"]
            output.extra_fields["ics_corrected"] = ics_stats["ics_corrected"]
            # Per-iteration oracle correctness (slot 0 only — one list per prompt).
            # iter_oracle_correct[0] = fresh chain, [k] = k-th correction.
            # Downstream: aggregate across prompts to get accuracy-over-iterations curve.
            if my_slot == 0:
                output.extra_fields["ics_iter_oracle_correct"] = ics_stats["iter_oracle_correct"]
        else:
            output.extra_fields["ics_triggered"] = False
            output.extra_fields["ics_iterations"] = 0
            output.extra_fields["ics_corrected"] = False

        # Attach localization training tensors. ONLY slot 0 gets the real
        # data — slots 1..N-1 get an empty (all-padding) version with the
        # same shape so VERL collation works. This avoids training on
        # duplicated loc data 8x per prompt.
        if self.train_localization and buffer.loc_tensors is not None:
            if my_slot == 0:
                output.extra_fields.update(buffer.loc_tensors)
            else:
                output.extra_fields.update(self._pack_localization_tensors([]))

        # Same for loc PPO tensors
        if (self.train_loc_ppo or self.train_loc_sft or self.train_loc_kto) and buffer.lppo_tensors is not None:
            if my_slot == 0:
                output.extra_fields.update(buffer.lppo_tensors)
            else:
                output.extra_fields.update(self._pack_loc_ppo_tensors([]))

        # Verifier tensors (slot-0 only — same dedup pattern)
        v_enabled = (
            self.train_verifier_sft or self.train_verifier_ppo
            or self.train_verifier_grpo or self.train_verifier_kto
        )
        if v_enabled and buffer.verifier_tensors is not None:
            if my_slot == 0:
                output.extra_fields.update(buffer.verifier_tensors)
            else:
                output.extra_fields.update(self._pack_verifier_empty())

        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _check_correctness(self, chain: _ThoughtChainResult, ground_truth: str) -> bool:
        """Decode a chain's response and check if the answer is correct.

        Default behavior (unchanged for math-style datasets): use
        math_compute_score with a 0.5 threshold.

        Non-math datasets opt in via _CORRECTNESS_THRESHOLDS — those route
        through the dispatched compute_score and threshold at the table value
        (e.g. 0.8 for retrosynthesis_uspto50k = template-match tier).
        """
        is_correct, _ = await self._check_correctness_and_score(chain, ground_truth)
        return is_correct

    async def _check_correctness_and_score(
        self, chain: _ThoughtChainResult, ground_truth: str
    ) -> tuple[bool, float]:
        """Return (is_correct, raw_score) for a chain.

        For math-style datasets (data_source not in _CORRECTNESS_THRESHOLDS):
        raw_score is math_compute_score's float output (0.0 or 1.0), and
        is_correct = raw_score > 0.5. So float(is_correct) == raw_score
        for math — using raw_score as the advantage reward is bit-exact
        equivalent to using float(is_correct).

        For graded datasets (retrosynthesis_uspto50k, etc.): raw_score is
        the continuous compute_score output, and is_correct is the
        threshold check used to gate ICS branching.
        """
        decoded = await self.loop.run_in_executor(
            None,
            lambda ids=chain.response_ids: self.tokenizer.decode(
                ids, skip_special_tokens=False
            ),
        )

        ds = getattr(self, "_current_data_source", "")
        threshold = _resolve_correctness_threshold(ds)
        if threshold is not None:
            # Lazy import: only loaded for opted-in datasets, so math runs
            # don't pay the import cost or risk circular imports at module load.
            from training.reward_fn import compute_score
            try:
                result = compute_score(
                    ds, decoded, ground_truth,
                    extra_info=getattr(self, "_current_extra_info", {}) or {},
                )
                score = float(result["score"]) if isinstance(result, dict) else float(result)
                return score >= threshold, score
            except Exception as e:
                logger.warning(f"[ICS] dispatched correctness failed for {ds}: {e}; falling back to math")

        raw = float(math_compute_score(decoded, ground_truth))
        return raw > 0.5, raw

    # ------------------------------------------------------------------
    # Self-verification (shadow mode — does NOT gate ICS)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_boxed(text: str) -> str:
        """Extract content of the last \\boxed{...} in text. Empty if none."""
        # Find the last \boxed{ and balance braces from there.
        idx = text.rfind("\\boxed{")
        if idx < 0:
            return ""
        i = idx + len("\\boxed{")
        depth = 1
        out = []
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            out.append(c)
            i += 1
        return "".join(out).strip()

    @staticmethod
    def _parse_verifier_yesno(response_text: str) -> Optional[bool]:
        """Parse YES/NO verdict from a verifier response.

        Tries \\boxed{YES|NO} first, then falls back to a lowercase scan.
        Returns None if neither is found (caller decides default).
        """
        boxed = ThoughtICSAgentLoop._extract_boxed(response_text).upper()
        if "YES" in boxed:
            return True
        if "NO" in boxed:
            return False
        lower = response_text.lower()
        has_yes = "yes" in lower
        has_no = "no" in lower
        if has_yes and not has_no:
            return True
        if has_no and not has_yes:
            return False
        return None

    def _build_verification_prompt(
        self, question: str, chain: _ThoughtChainResult
    ) -> tuple[str, str]:
        """Build the verifier prompt text. Returns (prompt_text, final_answer)."""
        # Build chain representation: numbered steps stripped of delimiters,
        # mirroring _localize_error.
        steps = []
        for thought in chain.decoded_thoughts:
            clean = thought.replace(self.thought_delimiter, "").strip()
            if clean:
                steps.append(clean)
        chain_text = "\n".join(f"Step {i}: {s}" for i, s in enumerate(steps, 1))

        # Final answer = \boxed{} extracted from the last step.
        final_answer = self._extract_boxed(steps[-1] if steps else "")

        # Verbatim from TREE iterative_self_correction.verify_solution_correctness
        prompt_text = (
            f"You are reviewing a solution to a problem. Analyze it carefully "
            f"to see if they arrived at the right answer.\n\n"
            f"Problem: {question}\n\n"
            f"Solution to review:\n{chain_text}\n\n"
            f"Final answer: {final_answer}\n\n"
            f"Verify the reasoning step by step and determine whether the "
            f"final answer is correct or not.\n\n"
            f"Conclude with \\boxed{{YES}} if the solution is correct, or "
            f"\\boxed{{NO}} if it contains errors."
        )
        return prompt_text, final_answer

    async def _self_verify(
        self,
        sampling_params: dict,
        question: str,
        chain: _ThoughtChainResult,
    ) -> Optional[dict]:
        """Run one self-verification call on a chain.

        Returns dict with prompt_ids, response_ids, response_logprobs,
        predicted (bool), raw_text. Returns None if generation produced
        no tokens.
        """
        if chain.num_thoughts == 0:
            return None

        prompt_text, _ = self._build_verification_prompt(question, chain)
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(prompt_text, add_special_tokens=True),
        )

        v_params = dict(sampling_params)
        v_params["max_new_tokens"] = self.localization_max_tokens
        v_params.pop("max_tokens", None)
        v_params["temperature"] = self.verifier_temp
        v_params.pop("stop", None)
        v_params.pop("include_stop_str_in_output", None)

        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=list(prompt_ids),
            sampling_params=v_params,
        )
        if not output.token_ids:
            return None

        response_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(
                output.token_ids, skip_special_tokens=False
            ),
        )
        parsed = self._parse_verifier_yesno(response_text)
        # Default to False (model couldn't decide → assume needs correction)
        predicted = bool(parsed) if parsed is not None else False

        return {
            "prompt_ids": list(prompt_ids),
            "response_ids": list(output.token_ids),
            "response_logprobs": list(output.log_probs) if output.log_probs else [],
            "predicted": predicted,
            "parsed_ok": parsed is not None,
            "raw_text": response_text,
        }

    async def _generate_verifier_grpo_rollouts(
        self,
        verifier_data: list[dict],
    ) -> list[dict]:
        """For each verifier_data entry, generate K independent rollouts of
        the same verification prompt and score each by oracle agreement.

        Mirrors _generate_localization_rollouts. Returns a list of groups,
        each with prompt_ids + K {responses, rewards}.
        """
        results = []
        for entry in verifier_data[: self.max_verifier_groups_per_prompt]:
            prompt_ids = entry["prompt_ids"]
            oracle = entry["oracle"]

            v_params = {
                "max_new_tokens": self.localization_max_tokens,
                "temperature": self.verifier_grpo_rollout_temp,
            }

            async def _gen_one():
                out = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=list(prompt_ids),
                    sampling_params=v_params,
                )
                tids = list(out.token_ids) if out.token_ids else []
                lps = list(out.log_probs) if out.log_probs else []
                if tids:
                    dec = await self.loop.run_in_executor(
                        None,
                        lambda ids=tids: self.tokenizer.decode(
                            ids, skip_special_tokens=False
                        ),
                    )
                    parsed = self._parse_verifier_yesno(dec)
                    predicted = bool(parsed) if parsed is not None else False
                    rwd = 1.0 if predicted == oracle else 0.0
                else:
                    rwd = 0.0
                return {"token_ids": tids, "log_probs": lps}, rwd

            rollout_results = await asyncio.gather(
                *[_gen_one() for _ in range(self.verifier_grpo_rollout_n)]
            )
            group = {"prompt_ids": prompt_ids, "responses": [], "rewards": []}
            for resp_data, reward in rollout_results:
                group["responses"].append(resp_data)
                group["rewards"].append(reward)
            results.append(group)

        logger.info(
            f"[ICS-V] Generated {len(results)} verifier groups, "
            f"K={self.verifier_grpo_rollout_n} rollouts each"
        )
        return results

    # ------------------------------------------------------------------
    # ICS core loop
    # ------------------------------------------------------------------

    async def _fill_rollout_buffer(
        self,
        sampling_params: dict,
        question: str,
        ground_truth: str,
    ) -> tuple[list, dict, list]:
        """Fill the entire rollout buffer with n trajectories.

        Alternates between fresh chain generation and ICS correction.
        When a fresh chain is wrong, triggers ICS corrections until the
        chain is corrected or the buffer is full.

        Returns:
            (buffer, ics_stats, loc_training_data) where buffer has up to
            n trajectories and loc_training_data contains metadata for
            successful localizations (empty when train_localization=False).
        """
        n = self.rollout_n
        buffer: list = []
        buffer_meta: list = []  # parallel to buffer: {"suffix_start_idx": int, "reward": float}
        loc_training_data: list = []
        loc_ppo_data: list = []
        verifier_data: list = []
        verifier_enabled = (
            self.train_verifier_sft or self.train_verifier_ppo
            or self.train_verifier_grpo or self.train_verifier_kto
        )
        ics_stats = {
            "ics_triggered": False,
            "ics_iterations": 0,
            "ics_corrected": False,
            "ics_error_steps": [],
            "ics_triggers": 0,
            "fresh_chains": 0,
            # Per-iteration oracle correctness: index 0 = fresh chain,
            # index k = k-th correction. Used to compute accuracy curves.
            "iter_oracle_correct": [],
            # Offline analysis: every localization call recorded in parallel lists.
            # Includes 0-step "no error found" outputs that are excluded from
            # ics_error_steps. Only populated when SRPO_BRANCH_DUMP_DIR is set.
            "ics_loc_error_steps": [],
            "ics_loc_n_steps": [],
            "ics_loc_reasonings": [],
            # Full localization prompt text (one per loc call). Lets offline
            # analysis pair (prompt, response) without reconstructing from chain.
            "ics_loc_prompts": [],
        }
        _capture_loc = bool(os.environ.get("SRPO_BRANCH_DUMP_DIR", "").strip())

        while len(buffer) < n:
            # Generate a fresh chain
            chain = await self._generate_thought_chain(sampling_params, question)
            buffer.append(chain)
            _fresh_idx = len(buffer) - 1
            buffer_meta.append({"suffix_start_idx": 0, "reward": 0.0})
            ics_stats["fresh_chains"] += 1

            if len(buffer) >= n:
                # Still need correctness for SRPO reward on the last fresh chain
                _ok, _raw = await self._check_correctness_and_score(chain, ground_truth)
                buffer_meta[_fresh_idx]["reward"] = _raw
                ics_stats["iter_oracle_correct"].append(_ok)
                break

            # Skip ICS if chain is empty (nothing to correct)
            if chain.num_thoughts == 0:
                continue

            if self.use_verifier_at_eval:
                v_result = await self._self_verify(sampling_params, question, chain)
                is_correct = v_result["predicted"] if v_result is not None else False
                oracle_correct, raw_score = await self._check_correctness_and_score(chain, ground_truth)
                logger.warning(
                    f"[ICS] Fresh #{ics_stats['fresh_chains']}: "
                    f"{chain.num_thoughts} thoughts, "
                    f"self_verify={is_correct}, oracle={oracle_correct}"
                )
            else:
                oracle_correct, raw_score = await self._check_correctness_and_score(chain, ground_truth)
                is_correct = oracle_correct
                logger.warning(
                    f"[ICS] Fresh #{ics_stats['fresh_chains']}: "
                    f"{chain.num_thoughts} thoughts, correct={is_correct}"
                )
            ics_stats["iter_oracle_correct"].append(oracle_correct)
            buffer_meta[_fresh_idx]["reward"] = raw_score

            # Shadow-mode self-verification: runs on every fresh chain (correct
            # AND incorrect), captures (prompt, response, predicted, oracle, reward).
            # Does NOT influence ICS control flow — oracle still gates everything.
            #
            # For vgrpo we cap collection at max_verifier_groups_per_prompt
            # because _generate_verifier_grpo_rollouts only iterates over that
            # many groups — collecting more would waste shadow verification calls.
            # vsft/vppo cap at rollout_n (= max possible fresh chains).
            if verifier_enabled:
                v_cap = (
                    self.max_verifier_groups_per_prompt
                    if self.train_verifier_grpo
                    else self.rollout_n
                )
                if len(verifier_data) < v_cap:
                    v_result = await self._self_verify(sampling_params, question, chain)
                    if v_result is not None:
                        v_result["oracle"] = is_correct
                        v_result["reward"] = (
                            1.0 if (v_result["predicted"] == is_correct) else 0.0
                        )
                        verifier_data.append(v_result)

            if is_correct and not self.random_reset_all:
                continue

            # Wrong chain (or random_reset_all) → trigger ICS
            ics_stats["ics_triggered"] = True
            ics_stats["ics_triggers"] += 1

            per_trigger_budget = min(self.max_ics_iterations, n - len(buffer))
            current_chain = chain

            for ics_iter in range(per_trigger_budget):
                try:
                    ics_stats["ics_iterations"] += 1

                    # Localize error
                    error_step, loc_prompt_ids, error_reasoning, loc_resp_ids, loc_resp_lps, loc_prompt_text = await self._localize_error(
                        sampling_params, question, ground_truth,
                        current_chain.decoded_thoughts,
                    )

                    # Offline analysis capture: every call (including step 0).
                    if _capture_loc:
                        ics_stats["ics_loc_error_steps"].append(error_step)
                        ics_stats["ics_loc_n_steps"].append(current_chain.num_thoughts)
                        ics_stats["ics_loc_reasonings"].append(error_reasoning)
                        ics_stats["ics_loc_prompts"].append(loc_prompt_text)

                    if error_step == 0:
                        logger.info(
                            f"[ICS] Trigger {ics_stats['ics_triggers']} "
                            f"iter {ics_iter + 1}: no error found, stopping"
                        )
                        break

                    ics_stats["ics_error_steps"].append(error_step)

                    logger.info(
                        f"[ICS] Trigger {ics_stats['ics_triggers']} "
                        f"iter {ics_iter + 1}: error at step {error_step}/"
                        f"{current_chain.num_thoughts}"
                    )

                    # Backtrack: keep steps before the error
                    if error_step <= 1:
                        prefix_response_ids = []
                        prefix_logprobs = []
                        prefix_boundaries = []
                        prefix_thoughts = []
                    else:
                        cut_idx = current_chain.thought_boundaries[error_step - 2][1]
                        prefix_response_ids = current_chain.response_ids[:cut_idx]
                        prefix_logprobs = (
                            current_chain.response_logprobs[:cut_idx]
                            if current_chain.response_logprobs
                            else []
                        )
                        prefix_boundaries = (
                            current_chain.thought_boundaries[: error_step - 1]
                        )
                        prefix_thoughts = (
                            current_chain.decoded_thoughts[: error_step - 1]
                        )
                    _suffix_start = len(prefix_response_ids)

                    # Regenerate from prefix
                    if self.use_context:
                        correction = await self._generate_thought_chain_from_prefix(
                            sampling_params, question,
                            prefix_response_ids, prefix_logprobs,
                            prefix_boundaries, prefix_thoughts,
                            previous_chain_thoughts=current_chain.decoded_thoughts,
                            error_reasoning=error_reasoning,
                            error_step=error_step,
                        )
                    else:
                        correction = await self._generate_thought_chain_from_prefix(
                            sampling_params, question,
                            prefix_response_ids, prefix_logprobs,
                            prefix_boundaries, prefix_thoughts,
                        )
                    buffer.append(correction)
                    _corr_idx = len(buffer) - 1
                    buffer_meta.append({"suffix_start_idx": _suffix_start, "reward": 0.0})

                    if self.use_verifier_at_eval:
                        v_result = await self._self_verify(sampling_params, question, correction)
                        is_correct = v_result["predicted"] if v_result is not None else False
                        oracle_correct, raw_score = await self._check_correctness_and_score(correction, ground_truth)
                        logger.warning(
                            f"[ICS] Trigger {ics_stats['ics_triggers']} "
                            f"iter {ics_iter + 1}: {correction.num_thoughts} "
                            f"thoughts, self_verify={is_correct}, oracle={oracle_correct}"
                        )
                    else:
                        oracle_correct, raw_score = await self._check_correctness_and_score(correction, ground_truth)
                        is_correct = oracle_correct
                        logger.warning(
                            f"[ICS] Trigger {ics_stats['ics_triggers']} "
                            f"iter {ics_iter + 1}: {correction.num_thoughts} "
                            f"thoughts, correct={is_correct}"
                        )
                    ics_stats["iter_oracle_correct"].append(oracle_correct)
                    buffer_meta[_corr_idx]["reward"] = raw_score

                    # Capture for loc PPO training (both success AND failure)
                    if ((self.train_loc_ppo or self.train_loc_sft or self.train_loc_kto)
                            and loc_prompt_ids is not None
                            and loc_resp_ids is not None
                            and len(loc_ppo_data) < self.max_corrections_per_prompt):
                        # Precision-weighted reward: reward proportional to
                        # fraction of steps kept (higher = more precise localization)
                        if is_correct and os.environ.get("LPPO_PRECISION_REWARD", "0") == "1":
                            n_total = current_chain.num_thoughts
                            steps_kept = max(error_step - 1, 0)
                            reward = steps_kept / max(n_total, 1)
                        else:
                            reward = 1.0 if is_correct else 0.0
                        loc_ppo_data.append({
                            "prompt_ids": loc_prompt_ids,
                            "response_ids": loc_resp_ids,
                            "response_logprobs": loc_resp_lps or [],
                            "reward": reward,
                        })

                    if is_correct:
                        ics_stats["ics_corrected"] = True
                        # Capture for localization training
                        if (self.train_localization
                                and loc_prompt_ids is not None
                                and len(loc_training_data) < self.max_loc_groups_per_prompt):
                            loc_training_data.append({
                                "prompt_ids": loc_prompt_ids,
                                "ground_truth_step": error_step,
                                "n_steps": len(current_chain.decoded_thoughts),
                            })
                        if not self.random_reset_all:
                            break

                    if len(buffer) >= n:
                        break

                    current_chain = correction

                except Exception as e:
                    logger.warning(
                        f"[ICS] Trigger {ics_stats['ics_triggers']} "
                        f"iter {ics_iter + 1} failed: {e}"
                    )
                    break

        return buffer, ics_stats, loc_training_data, loc_ppo_data, verifier_data, buffer_meta

    # ------------------------------------------------------------------
    # ICS-specific generation (prefix-based continuation)
    # ------------------------------------------------------------------

    async def _generate_thought_chain_from_prefix(
        self,
        sampling_params: dict,
        question: str,
        prefix_response_ids: list,
        prefix_logprobs: list,
        prefix_boundaries: list,
        prefix_thoughts: list,
        previous_chain_thoughts: Optional[list] = None,
        error_reasoning: Optional[str] = None,
        error_step: Optional[int] = None,
    ) -> _ThoughtChainResult:
        """Generate a thought chain continuing from a given prefix.

        When previous_chain_thoughts and error_reasoning are provided,
        historical context is injected into the generation prompt so the
        model knows what went wrong.  The returned result's prompt_ids
        are always the original (context-free) prompt so that all rollouts
        in the GRPO group share the same training prompt.
        """
        # Original prompt — always needed for training
        original_prompt_text = self.template.format(question=question)
        original_prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(original_prompt_text, add_special_tokens=True),
        )

        # Build generation prompt (may include historical context)
        has_context = previous_chain_thoughts is not None and error_reasoning is not None
        if has_context:
            # Budget: context prompt + prefix + new thoughts must fit in
            # prompt_length + response_length (the vLLM operational limit).
            max_seq = self.prompt_length + self.response_length
            budget = max_seq - len(prefix_response_ids) - self.max_tokens_per_thought
            # Try full chain first; trim from the top if too long
            chain_to_include = list(previous_chain_thoughts)
            while True:
                context_text = original_prompt_text
                context_text += "\n\n### Previous Failed Attempt\n"
                context_text += "The following reasoning chain led to an incorrect answer:\n"
                for i, thought in enumerate(chain_to_include, 1):
                    clean = thought.replace(self.thought_delimiter, "").strip()
                    context_text += f"\nStep {i}: {clean}"
                context_text += f"\n\n### Error Analysis\n{error_reasoning}\n"
                context_text += "\nNow let's try again with the correct approach:\n"

                generation_prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda ct=context_text: self.tokenizer.encode(ct, add_special_tokens=True),
                )
                if len(generation_prompt_ids) <= budget or len(chain_to_include) <= 1:
                    break
                # Drop the earliest step (least relevant to the error)
                chain_to_include = chain_to_include[1:]

            n_trimmed = len(previous_chain_thoughts) - len(chain_to_include)
            logger.info(
                f"[ICS] Context-enriched prompt: {len(generation_prompt_ids)} tokens "
                f"(original: {len(original_prompt_ids)}, "
                f"trimmed {n_trimmed}/{len(previous_chain_thoughts)} steps)"
            )
        else:
            generation_prompt_ids = original_prompt_ids  # same object — no context

        # Generate using (possibly context-enriched) prompt
        result = await self._generate_thoughts_core(
            list(generation_prompt_ids), sampling_params,
            prefix_response_ids=prefix_response_ids,
            prefix_logprobs=prefix_logprobs,
            prefix_boundaries=prefix_boundaries,
            prefix_thoughts=prefix_thoughts,
        )

        # Swap prompt_ids back to original for training (GRPO group consistency)
        # and recompute on-policy log-probs via a scoring pass
        if generation_prompt_ids is not original_prompt_ids:
            result.prompt_ids = list(original_prompt_ids)

            # Scoring pass: recompute log-probs under original prompt.
            # The response tokens were generated conditioned on the context-
            # enriched prompt, so their log-probs are off-policy w.r.t. the
            # original prompt used for training.  Feed [original_prompt +
            # response_ids] as the "prompt" to vLLM with prompt_logprobs=1
            # and max_tokens=1 to get per-token log-probs under the original
            # prompt.
            # Wrapped in try/except so a scoring failure degrades gracefully
            # (keeps off-policy log-probs) rather than losing the trajectory.
            if result.response_ids:
                try:
                    scoring_prompt_ids = list(original_prompt_ids) + list(result.response_ids)
                    scoring_params = {
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "prompt_logprobs": 1,
                    }
                    scoring_output = await self.server_manager.generate(
                        request_id=uuid4().hex,
                        prompt_ids=scoring_prompt_ids,
                        sampling_params=scoring_params,
                    )
                    if scoring_output.prompt_log_probs is not None:
                        # prompt_log_probs covers the full scoring prompt;
                        # skip the original prompt portion to get response log-probs.
                        # First token of prompt has no log-prob (None), so offset by
                        # len(original_prompt_ids) to align with response tokens.
                        n_prompt = len(original_prompt_ids)
                        response_logprobs = scoring_output.prompt_log_probs[n_prompt:]
                        # Replace off-policy log-probs with on-policy ones
                        on_policy = [lp if lp is not None else 0.0 for lp in response_logprobs]
                        # Truncate/pad to match response_ids length
                        n_resp = len(result.response_ids)
                        if len(on_policy) >= n_resp:
                            result.response_logprobs = on_policy[:n_resp]
                        else:
                            result.response_logprobs = on_policy + [0.0] * (n_resp - len(on_policy))
                        logger.info(
                            f"[ICS] Scoring pass: recomputed {n_resp} on-policy log-probs"
                        )
                    else:
                        logger.warning("[ICS] Scoring pass returned no prompt_log_probs")
                except Exception as e:
                    logger.warning(f"[ICS] Scoring pass failed, keeping off-policy log-probs: {e}")

        return result

    # ------------------------------------------------------------------
    # Error localization
    # ------------------------------------------------------------------

    async def _localize_error(
        self,
        sampling_params: dict,
        question: str,
        ground_truth: str,
        decoded_thoughts: list,
    ) -> tuple[int, Optional[list], str]:
        """Localize error step via a standard CoT call (no stop tokens).

        Localization tokens are discarded -- they are NOT training data
        (unless train_localization is enabled, in which case prompt IDs
        are returned for downstream rollout generation).

        Returns:
            (error_step, loc_prompt_ids, error_reasoning) where error_step
            is 1-indexed (0 if no error found), loc_prompt_ids is the
            tokenized localization prompt (None when train_localization is
            off), and error_reasoning is the full localization response text.
        """
        if not decoded_thoughts:
            return 0, None, "", None, None, ""

        n_steps = len(decoded_thoughts)

        # Random localization baseline: skip the LLM call entirely
        if self.random_localization:
            import random
            error_step = random.randint(1, n_steps)
            logger.info(f"[ICS] Random localization: picked step {error_step}/{n_steps}")
            return error_step, None, f"[random localization: step {error_step}]", None, None, ""

        # Build chain text, stripping thought delimiters for readability
        chain_text = ""
        for i, thought in enumerate(decoded_thoughts, 1):
            clean = thought.replace(self.thought_delimiter, "").strip()
            chain_text += f"\nStep {i}: {clean}"

        # Build localization prompt based on autonomy level
        if self.autonomy_level == 1:
            # L1 (Oracle): model sees the correct ground truth answer
            prompt_text = (
                f"Problem: {question}\n\n"
                f"Current reasoning chain (WRONG - got incorrect answer):\n"
                f"{chain_text}\n\n"
                f"The correct answer should be {ground_truth}.\n\n"
                f"Analyze the reasoning chain step by step to identify where "
                f"the error occurred. Which step number (1 to {n_steps}) "
                f"contains the first critical error that led to the wrong "
                f"answer?\n\n"
                f"Do NOT solve the problem again. Your ONLY task is to "
                f"identify the first erroneous step. Provide your reasoning, "
                f"then put ONLY the step number (an integer from 1 to "
                f"{n_steps}) in the format: \\boxed{{step_number}}\n"
            )
        else:
            # L2 (Binary Feedback, early-biasing localization):
            # Frames errors as compounding so subtle early mistakes that look
            # fine at the time are still the right localization target. This
            # avoids pushing picks into low-success late bins (last-quartile
            # mean#ok/4 ~0.3 vs first-quartile ~1.0+ across qwen14b / olmo7b).
            prompt_text = (
                f"You are tasked with localizing the first erroneous "
                f"thought in your previous solution to this problem.\n\n"
                f"Problem: {question}\n\n"
                f"Your incorrect reasoning chain:\n{chain_text}\n\n"
                f"The final answer this chain produces is incorrect — "
                f"therefore at least one step contains an error. The error "
                f"you are looking for is the originating step where a key "
                f"decision or action derailed the reasoning, not just the "
                f"step where the failure ultimately becomes visible. A "
                f"misread of the problem, an unjustified assumption, or a "
                f"logical flaw can look fine for several follow-on steps "
                f"before it surfaces in the wrong answer. A step is "
                f"erroneous if you cannot justify its claims from the "
                f"problem statement and earlier verified steps alone. "
                f"Find the originating step, not just the symptom.\n\n"
                f"Do NOT re-solve the problem. Your ONLY task is to "
                f"identify the step number of that originating error.\n\n"
                f"Requirements:\n"
                f"- Commit to exactly ONE step number (1 to {n_steps}).\n"
                f"- Stop at the first step you cannot justify.\n"
                f"- MANDATORY final line: your response MUST end with "
                f"\\boxed{{N}} on its own line, where N is the step index "
                f"(1-indexed) of the first erroneous step in the chain "
                f"above — NOT the answer to the problem. Do NOT add any "
                f"text after the \\boxed{{N}}.\n"
            )

        # Wrap as a chat turn for chat-tuned models (e.g. OLMo-3-Instruct emits
        # EOS immediately on raw completion prompts because it was fine-tuned
        # to expect <|im_start|>user ... <|im_end|>\n<|im_start|>assistant\n
        # scaffolding). Falls back to raw encode when no chat_template is set.
        if getattr(self.tokenizer, "chat_template", None):
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=True,
                    add_generation_prompt=True,
                ),
            )
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(prompt_text, add_special_tokens=True),
            )

        # Capture prompt IDs for localization training (before generation)
        loc_prompt_ids = list(prompt_ids) if (self.train_localization or self.train_loc_ppo or self.train_loc_sft or self.train_loc_kto) else None

        # Standard CoT call -- NO stop tokens
        loc_params = dict(sampling_params)
        loc_params["max_new_tokens"] = self.localization_max_tokens
        loc_params.pop("max_tokens", None)
        loc_params["temperature"] = self.localization_temp
        loc_params.pop("stop", None)
        loc_params.pop("include_stop_str_in_output", None)

        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=list(prompt_ids),
            sampling_params=loc_params,
        )

        if not output.token_ids:
            logger.warning("[ICS] Localization returned empty output")
            return max(1, n_steps // 2), loc_prompt_ids, "", None, None

        response_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(
                output.token_ids, skip_special_tokens=False
            ),
        )

        # Capture response tokens/logprobs for loc PPO training
        loc_resp_ids = list(output.token_ids) if (self.train_loc_ppo or self.train_loc_sft or self.train_loc_kto) else None
        loc_resp_lps = list(output.log_probs) if ((self.train_loc_ppo or self.train_loc_sft or self.train_loc_kto) and output.log_probs) else None

        return self._parse_error_step(response_text, n_steps), loc_prompt_ids, response_text, loc_resp_ids, loc_resp_lps, prompt_text

    # ------------------------------------------------------------------
    # Localization training: rollout generation and tensor packing
    # ------------------------------------------------------------------

    async def _generate_localization_rollouts(
        self,
        loc_training_data: list[dict],
    ) -> list[dict]:
        """Generate K rollouts per successful localization and score them.

        For each entry in loc_training_data, generates localization_rollout_n
        independent responses from the localization prompt, then scores each
        with a binary reward (predicted step == ground truth step).

        Returns:
            List of dicts, each with keys: prompt_ids, responses (list of
            {token_ids, log_probs}), rewards (list of floats).
        """
        results = []
        for entry in loc_training_data:
            prompt_ids = entry["prompt_ids"]
            gt_step = entry["ground_truth_step"]
            n_steps = entry["n_steps"]

            loc_params = {
                "max_new_tokens": self.loc_response_length,
                "temperature": self.localization_rollout_temp,
            }

            group: dict = {
                "prompt_ids": prompt_ids,
                "responses": [],
                "rewards": [],
            }

            # Generate K rollouts in parallel via asyncio.gather
            async def _gen_one():
                out = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=list(prompt_ids),
                    sampling_params=loc_params,
                )
                tids = list(out.token_ids) if out.token_ids else []
                lps = list(out.log_probs) if out.log_probs else []
                if tids:
                    dec = await self.loop.run_in_executor(
                        None,
                        lambda ids=tids: self.tokenizer.decode(
                            ids, skip_special_tokens=False
                        ),
                    )
                    pred = self._parse_error_step(dec, n_steps)
                    rwd = 1.0 if pred == gt_step else 0.0
                else:
                    rwd = 0.0
                return {"token_ids": tids, "log_probs": lps}, rwd

            rollout_results = await asyncio.gather(
                *[_gen_one() for _ in range(self.localization_rollout_n)]
            )
            for resp_data, reward in rollout_results:
                group["responses"].append(resp_data)
                group["rewards"].append(reward)

            results.append(group)

        logger.info(
            f"[ICS-LOC] Generated {len(results)} localization groups, "
            f"K={self.localization_rollout_n} rollouts each"
        )
        return results

    def _pack_localization_tensors(self, loc_rollouts: list[dict]) -> dict:
        """Pack localization rollout data into fixed-size lists for extra_fields.

        Layout: flatten all rollouts into (max_total_rollouts, ...) arrays.
        max_total_rollouts = max_loc_groups_per_prompt * localization_rollout_n.

        Sequence length S is computed from the ACTUAL max localization prompt
        length (not self.prompt_length, which is for math prompts and may be
        too small for localization prompts that include full reasoning chains).

        Returns dict of lists (VERL converts to tensors during batch collation).
        """
        K = self.localization_rollout_n
        max_total = self.max_loc_groups_per_prompt * K
        R = self.loc_response_length

        # S is fixed via config (loc_seq_length) to control GPU memory usage.
        # Localization prompt + response are truncated to fit within S.
        # Default 4096 covers most cases (typical loc prompt ~1500 tokens,
        # response ~300 tokens). Increase if truncation warnings appear.
        S = self.loc_seq_length

        # Initialize zero-filled arrays
        loc_input_ids = [[0] * S for _ in range(max_total)]
        loc_attention_mask = [[0] * S for _ in range(max_total)]
        loc_old_logprobs = [[0.0] * R for _ in range(max_total)]
        loc_response_mask = [[0.0] * R for _ in range(max_total)]
        loc_rewards = [0.0] * max_total
        loc_group_ids = [0] * max_total  # 0 = padding
        loc_resp_start = [0] * max_total

        idx = 0
        for group_i, group in enumerate(loc_rollouts):
            prompt_ids = group["prompt_ids"]
            for resp_data, reward in zip(group["responses"], group["rewards"]):
                if idx >= max_total:
                    break

                token_ids = resp_data["token_ids"]
                log_probs = resp_data["log_probs"]

                # Build full input sequence: prompt + response (truncated to S)
                full_ids = prompt_ids + token_ids[:R]
                if len(full_ids) > S:
                    logger.warning(
                        f"[ICS-LOC] Localization sequence truncated: "
                        f"{len(full_ids)} > {S} (prompt={len(prompt_ids)}, "
                        f"resp={len(token_ids[:R])})"
                    )
                seq_len = min(len(full_ids), S)
                # Adjust resp/lp lengths for truncation: only count response
                # tokens that actually fit within the truncated sequence
                actual_resp_in_seq = max(0, seq_len - len(prompt_ids))
                resp_len = min(len(token_ids), R, actual_resp_in_seq)
                lp_len = min(len(log_probs), R, actual_resp_in_seq)

                for j in range(seq_len):
                    loc_input_ids[idx][j] = full_ids[j]
                    loc_attention_mask[idx][j] = 1
                for j in range(lp_len):
                    loc_old_logprobs[idx][j] = log_probs[j]
                for j in range(resp_len):
                    loc_response_mask[idx][j] = 1.0

                loc_rewards[idx] = reward
                loc_group_ids[idx] = group_i + 1  # 1-indexed
                # Clamp resp_start to S-1 so downstream indexing stays in-bounds
                # even when the localization prompt is very long
                loc_resp_start[idx] = min(len(prompt_ids), S - 1)
                idx += 1

        return {
            "loc_input_ids": loc_input_ids,
            "loc_attention_mask": loc_attention_mask,
            "loc_old_logprobs": loc_old_logprobs,
            "loc_response_mask": loc_response_mask,
            "loc_rewards": loc_rewards,
            "loc_group_ids": loc_group_ids,
            "loc_resp_start": loc_resp_start,
            "loc_count": [idx],  # list so VERL collates it as a tensor, not scalar
        }

    def _pack_loc_ppo_tensors(self, loc_ppo_data: list[dict]) -> dict:
        """Pack localization PPO data into fixed-size lists for extra_fields.

        Each entry is a single localization response from an ICS iteration,
        rewarded by whether the subsequent correction succeeded.

        Layout: (max_corrections_per_prompt, ...) arrays.
        """
        max_corr = self.max_corrections_per_prompt
        S = self.loc_seq_length
        R = self.localization_max_tokens  # response length for localization

        # Initialize zero-filled arrays
        lppo_input_ids = [[0] * S for _ in range(max_corr)]
        lppo_attention_mask = [[0] * S for _ in range(max_corr)]
        lppo_old_logprobs = [[0.0] * R for _ in range(max_corr)]
        lppo_response_mask = [[0.0] * R for _ in range(max_corr)]
        lppo_rewards = [0.0] * max_corr
        lppo_valid = [0] * max_corr
        lppo_resp_start = [0] * max_corr

        idx = 0
        for entry in loc_ppo_data:
            if idx >= max_corr:
                break

            prompt_ids = entry["prompt_ids"]
            response_ids = entry["response_ids"]
            response_logprobs = entry["response_logprobs"]
            reward = entry["reward"]

            # Build full input: prompt + response, truncated to S
            full_ids = prompt_ids + response_ids[:R]
            seq_len = min(len(full_ids), S)
            actual_resp_in_seq = max(0, seq_len - len(prompt_ids))
            resp_len = min(len(response_ids), R, actual_resp_in_seq)
            lp_len = min(len(response_logprobs), R, actual_resp_in_seq)

            if seq_len > S:
                logger.warning(
                    f"[ICS-LPPO] Sequence truncated: "
                    f"{len(full_ids)} > {S} (prompt={len(prompt_ids)}, "
                    f"resp={len(response_ids[:R])})"
                )

            for j in range(seq_len):
                lppo_input_ids[idx][j] = full_ids[j]
                lppo_attention_mask[idx][j] = 1
            for j in range(lp_len):
                lppo_old_logprobs[idx][j] = response_logprobs[j]
            for j in range(resp_len):
                lppo_response_mask[idx][j] = 1.0

            lppo_rewards[idx] = reward
            lppo_valid[idx] = 1
            lppo_resp_start[idx] = min(len(prompt_ids), S - 1)
            idx += 1

        return {
            "lppo_input_ids": lppo_input_ids,
            "lppo_attention_mask": lppo_attention_mask,
            "lppo_old_logprobs": lppo_old_logprobs,
            "lppo_response_mask": lppo_response_mask,
            "lppo_rewards": lppo_rewards,
            "lppo_valid": lppo_valid,
            "lppo_resp_start": lppo_resp_start,
            "lppo_count": [idx],
        }

    # ------------------------------------------------------------------
    # Verifier tensor packers (vsft / vppo / vgrpo)
    # ------------------------------------------------------------------

    def _pack_verifier_sft_tensors(self, verifier_data: list[dict]) -> dict:
        """Pack verifier responses for SFT.

        One row per fresh chain (max = rollout_n). NLL is run on rows
        where reward > 0.5 (verifier prediction matched oracle).
        """
        max_v = self.rollout_n
        S = self.loc_seq_length
        R = self.localization_max_tokens

        vsft_input_ids = [[0] * S for _ in range(max_v)]
        vsft_attention_mask = [[0] * S for _ in range(max_v)]
        vsft_response_mask = [[0.0] * R for _ in range(max_v)]
        vsft_resp_start = [0] * max_v
        vsft_rewards = [0.0] * max_v
        vsft_valid = [0] * max_v

        idx = 0
        for entry in verifier_data:
            if idx >= max_v:
                break
            prompt_ids = entry["prompt_ids"]
            response_ids = entry["response_ids"]

            full_ids = prompt_ids + response_ids[:R]
            seq_len = min(len(full_ids), S)
            actual_resp_in_seq = max(0, seq_len - len(prompt_ids))
            resp_len = min(len(response_ids), R, actual_resp_in_seq)

            if len(full_ids) > S:
                logger.warning(
                    f"[ICS-VSFT] Sequence truncated: {len(full_ids)} > {S}"
                )

            for j in range(seq_len):
                vsft_input_ids[idx][j] = full_ids[j]
                vsft_attention_mask[idx][j] = 1
            for j in range(resp_len):
                vsft_response_mask[idx][j] = 1.0

            vsft_rewards[idx] = entry["reward"]
            vsft_valid[idx] = 1
            vsft_resp_start[idx] = min(len(prompt_ids), S - 1)
            idx += 1

        return {
            "vsft_input_ids": vsft_input_ids,
            "vsft_attention_mask": vsft_attention_mask,
            "vsft_response_mask": vsft_response_mask,
            "vsft_resp_start": vsft_resp_start,
            "vsft_rewards": vsft_rewards,
            "vsft_valid": vsft_valid,
            "vsft_count": [idx],
        }

    def _pack_verifier_ppo_tensors(self, verifier_data: list[dict]) -> dict:
        """Pack verifier responses for PPO. Adds old_logprobs vs SFT."""
        max_v = self.rollout_n
        S = self.loc_seq_length
        R = self.localization_max_tokens

        vppo_input_ids = [[0] * S for _ in range(max_v)]
        vppo_attention_mask = [[0] * S for _ in range(max_v)]
        vppo_old_logprobs = [[0.0] * R for _ in range(max_v)]
        vppo_response_mask = [[0.0] * R for _ in range(max_v)]
        vppo_resp_start = [0] * max_v
        vppo_rewards = [0.0] * max_v
        vppo_oracle_labels = [0.0] * max_v
        vppo_valid = [0] * max_v

        idx = 0
        for entry in verifier_data:
            if idx >= max_v:
                break
            prompt_ids = entry["prompt_ids"]
            response_ids = entry["response_ids"]
            response_logprobs = entry["response_logprobs"]

            full_ids = prompt_ids + response_ids[:R]
            seq_len = min(len(full_ids), S)
            actual_resp_in_seq = max(0, seq_len - len(prompt_ids))
            resp_len = min(len(response_ids), R, actual_resp_in_seq)
            lp_len = min(len(response_logprobs), R, actual_resp_in_seq)

            if len(full_ids) > S:
                logger.warning(
                    f"[ICS-VPPO] Sequence truncated: {len(full_ids)} > {S}"
                )

            for j in range(seq_len):
                vppo_input_ids[idx][j] = full_ids[j]
                vppo_attention_mask[idx][j] = 1
            for j in range(lp_len):
                vppo_old_logprobs[idx][j] = response_logprobs[j]
            for j in range(resp_len):
                vppo_response_mask[idx][j] = 1.0

            vppo_rewards[idx] = entry["reward"]
            vppo_oracle_labels[idx] = 1.0 if entry.get("oracle") else 0.0
            vppo_valid[idx] = 1
            vppo_resp_start[idx] = min(len(prompt_ids), S - 1)
            idx += 1

        return {
            "vppo_input_ids": vppo_input_ids,
            "vppo_attention_mask": vppo_attention_mask,
            "vppo_old_logprobs": vppo_old_logprobs,
            "vppo_response_mask": vppo_response_mask,
            "vppo_resp_start": vppo_resp_start,
            "vppo_rewards": vppo_rewards,
            "vppo_oracle_labels": vppo_oracle_labels,
            "vppo_valid": vppo_valid,
            "vppo_count": [idx],
        }

    def _pack_verifier_grpo_tensors(self, v_rollouts: list[dict]) -> dict:
        """Pack verifier GRPO rollouts.

        Layout mirrors _pack_localization_tensors:
        max_total = max_verifier_groups_per_prompt * verifier_grpo_rollout_n.
        """
        K = self.verifier_grpo_rollout_n
        max_total = self.max_verifier_groups_per_prompt * K
        S = self.loc_seq_length
        R = self.localization_max_tokens

        vgrpo_input_ids = [[0] * S for _ in range(max_total)]
        vgrpo_attention_mask = [[0] * S for _ in range(max_total)]
        vgrpo_old_logprobs = [[0.0] * R for _ in range(max_total)]
        vgrpo_response_mask = [[0.0] * R for _ in range(max_total)]
        vgrpo_resp_start = [0] * max_total
        vgrpo_rewards = [0.0] * max_total
        vgrpo_group_ids = [0] * max_total  # 0 = padding

        idx = 0
        for group_i, group in enumerate(v_rollouts):
            prompt_ids = group["prompt_ids"]
            for resp_data, reward in zip(group["responses"], group["rewards"]):
                if idx >= max_total:
                    break
                token_ids = resp_data["token_ids"]
                log_probs = resp_data["log_probs"]
                if not token_ids:
                    continue

                full_ids = prompt_ids + token_ids[:R]
                seq_len = min(len(full_ids), S)
                actual_resp_in_seq = max(0, seq_len - len(prompt_ids))
                resp_len = min(len(token_ids), R, actual_resp_in_seq)
                lp_len = min(len(log_probs), R, actual_resp_in_seq)

                if len(full_ids) > S:
                    logger.warning(
                        f"[ICS-VGRPO] Sequence truncated: {len(full_ids)} > {S}"
                    )

                for j in range(seq_len):
                    vgrpo_input_ids[idx][j] = full_ids[j]
                    vgrpo_attention_mask[idx][j] = 1
                for j in range(lp_len):
                    vgrpo_old_logprobs[idx][j] = log_probs[j]
                for j in range(resp_len):
                    vgrpo_response_mask[idx][j] = 1.0

                vgrpo_rewards[idx] = reward
                vgrpo_group_ids[idx] = group_i + 1
                vgrpo_resp_start[idx] = min(len(prompt_ids), S - 1)
                idx += 1

        return {
            "vgrpo_input_ids": vgrpo_input_ids,
            "vgrpo_attention_mask": vgrpo_attention_mask,
            "vgrpo_old_logprobs": vgrpo_old_logprobs,
            "vgrpo_response_mask": vgrpo_response_mask,
            "vgrpo_resp_start": vgrpo_resp_start,
            "vgrpo_rewards": vgrpo_rewards,
            "vgrpo_group_ids": vgrpo_group_ids,
            "vgrpo_count": [idx],
        }

    def _pack_verifier_empty(self) -> dict:
        """Empty pack for non-coordinator slots — branches on enabled mode."""
        if self.train_verifier_grpo:
            return self._pack_verifier_grpo_tensors([])
        if self.train_verifier_ppo or self.train_verifier_kto:
            return self._pack_verifier_ppo_tensors([])
        if self.train_verifier_sft:
            return self._pack_verifier_sft_tensors([])
        return {}

    # ------------------------------------------------------------------
    # Parsing and conversion utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_error_step(response_text: str, chain_length: int) -> int:
        """Parse error step number from localization response.

        Tries \\boxed{step_number} first, then falls back to first valid
        integer in the response, then defaults to middle of chain.

        Returns:
            Step number (1-indexed), or 0 if model found no errors.
        """
        # Primary: last \boxed{N} with brace-balanced parsing.
        matches = list(re.finditer(r"\\boxed\{", response_text))
        if matches:
            start_pos = matches[-1].end()
            brace_count = 1
            i = start_pos
            while i < len(response_text) and brace_count > 0:
                if response_text[i] == "{":
                    brace_count += 1
                elif response_text[i] == "}":
                    brace_count -= 1
                i += 1
            if brace_count == 0:
                boxed = response_text[start_pos : i - 1].strip()
                try:
                    step_num = int(boxed)
                    if step_num == 0:
                        return 0
                    if 1 <= step_num <= chain_length:
                        return step_num
                except (ValueError, TypeError):
                    pass

        # Fallback 1: last "Step N" phrase in prose (case-insensitive).
        # Catches cases where the model writes its conclusion as "...the
        # first error is in Step 3" without wrapping in \boxed{}.
        step_matches = list(re.finditer(r"[Ss]tep\s+(\d+)", response_text))
        if step_matches:
            try:
                step_num = int(step_matches[-1].group(1))
                if 1 <= step_num <= chain_length:
                    return step_num
            except (ValueError, TypeError):
                pass

        # Fallback 2: default to step 1 (was: middle of chain).
        return 1

