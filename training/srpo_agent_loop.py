"""SRPO agent loop: 4 i.i.d. fresh (Group 1) + 4 counterfactuals (Group 2, no parent).

Same as SRPO except Group 2 drops the failed parent slot in favor of a 4th
correction. The failed parent is still sampled and used for error localization,
but not added to the training buffer. This makes G2 a clean i.i.d. group of 4
samples from π(·|q, prefix), giving an unbiased baseline estimate for GRPO
normalization.

Buffer layout:
  Group 1 (slots 0-3): 4 i.i.d. fresh rollouts, suffix_start=0 (full-response gradient)
  Group 2 (slots 4-7): 4 counterfactuals from the localized prefix, suffix_start>0 (suffix-only)

If no failure is found within MAX_FRESH_ATTEMPTS chains, G2 is skipped and all
8 slots are filled with G1 chains (vanilla GRPO fallback, same as SRPO).
"""

import logging
import os
from typing import Optional

import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

from training.branch_logger import dump_branch_group
from training.thought_ics_agent_loop import ThoughtICSAgentLoop, _ICSBuffer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Maximum fresh chains to generate before giving up on finding a failure for
# Group 2. On hard problems (p_fail ≈ 0.9) this is almost never hit; on easy
# problems Group 2 is simply skipped and Group 1 provides the training signal.
_MAX_FRESH_ATTEMPTS = 12


class SRPOAgentLoop(ThoughtICSAgentLoop):
    """SRPO: 4 i.i.d. fresh chains (Group 1) + 4 counterfactuals (Group 2, no parent).

    The 8-slot buffer is split into two independently-normalized GRPO groups:
      Group 1 (slots 0-3): 4 i.i.d. fresh rollouts, suffix_start=0 (full-response gradient)
      Group 2 (slots 4-7): 4 counterfactuals from the localized prefix, suffix_start>0

    The first failed fresh chain is used for error localization only — it is NOT
    added to the training buffer — so Group 2 is a clean i.i.d. set of 4 samples
    from π(·|q, prefix), giving an unbiased baseline for GRPO normalization.

    This class is the base for the SRPO sampling-ablation variants
    (SRPO_1x8/SRPOx4/SRPO2x4/SRPO_1x8) and the no-mask variant SRPONM, which
    inherit run() and _compute_advantages and override _fill_rollout_buffer.
    """

    # Separate buffer pool per subclass.
    _buffers: dict = {}

    # Shared-prefix masking flags (overridden by the srpo_nomask variant).
    # Default: corrections get suffix-only gradient. mask_parent_prefix is
    # retained for ablation variants that keep a parent slot.
    mask_parent_prefix: bool = False
    mask_correction_prefix: bool = True

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        """Coordinate with other rollout slots for SRPO buffer construction."""
        if sampling_params.get("temperature", 1.0) == 0 and not self.force_ics_at_eval:
            return await super(ThoughtICSAgentLoop, self).run(sampling_params, **kwargs)

        messages = kwargs["raw_prompt"]
        question = self._extract_question(messages)
        ground_truth = kwargs["reward_model"]["ground_truth"]
        # Capture for _check_correctness_and_score dispatch on non-math datasets.
        # For datasets absent from _CORRECTNESS_THRESHOLDS (all math-style), the
        # threshold lookup returns None and behavior is unchanged.
        self._current_data_source = kwargs.get("data_source", "") or ""
        self._current_extra_info = kwargs.get("extra_info", {}) or {}

        buffer_key = question[:200]
        if buffer_key not in self._buffers or self._buffers[buffer_key].done.is_set():
            self._buffers[buffer_key] = _ICSBuffer()
        buffer = self._buffers[buffer_key]

        async with buffer.claim_lock:
            my_slot = buffer.next_slot
            buffer.next_slot += 1

        ics_stats = None

        if my_slot == 0:
            logger.info(f"[SRPO] Coordinator starting for: {question[:80]}...")
            try:
                trajectories, ics_stats, _, _, _, buffer_meta = await self._fill_rollout_buffer(
                    sampling_params, question, ground_truth
                )
                buffer.trajectories = trajectories
                buffer.ics_stats = ics_stats
                buffer.buffer_meta = buffer_meta

                # Opt-in branch-structure dump (no-op unless SRPO_BRANCH_DUMP_DIR set).
                # Base ThoughtICSAgentLoop.run calls this; SRPO overrides run and
                # must call it explicitly.
                dump_branch_group(
                    question=question,
                    trajectories=trajectories,
                    ics_stats=ics_stats,
                    ground_truth=ground_truth,
                )

                if ics_stats and ics_stats["ics_triggered"]:
                    logger.warning(
                        f"[SRPO] {question[:60]}... | "
                        f"triggered=True corrected={ics_stats['ics_corrected']} "
                        f"error_steps={ics_stats['ics_error_steps']} "
                        f"trajectories={len(buffer.trajectories)}"
                    )
            except Exception as e:
                logger.error(f"[SRPO] Coordinator failed: {e}")
            finally:
                buffer.done.set()

            if buffer.trajectories:
                result = buffer.trajectories[0]
            else:
                logger.warning("[SRPO] Coordinator produced no trajectories, generating emergency fallback")
                result = await self._generate_thought_chain(sampling_params, question)
        else:
            await buffer.done.wait()
            ics_stats = getattr(buffer, "ics_stats", None)

            if my_slot < len(buffer.trajectories):
                result = buffer.trajectories[my_slot]
            else:
                logger.warning(
                    f"[SRPO] Slot {my_slot}: buffer has {len(buffer.trajectories)} "
                    f"trajectories, generating emergency fallback"
                )
                result = await self._generate_thought_chain(sampling_params, question)

        output = self._chain_result_to_output(result)

        # Attach SRPO metadata: pre-computed advantage + suffix start for gradient masking
        _meta = buffer.buffer_meta[my_slot] if my_slot < len(buffer.buffer_meta) else None
        output.extra_fields["reset_advantage"] = _meta["reset_advantage"] if _meta else 0.0
        output.extra_fields["suffix_start_idx"] = _meta["suffix_start_idx"] if _meta else 0
        # Also emit raw reward under srpo_reward so ics_metrics.py fresh/correction
        # accuracy tracking works without modification.
        output.extra_fields["srpo_reward"] = _meta["reward"] if _meta else 0.0

        # ICS stats for wandb logging (same keys as ThoughtICSAgentLoop)
        if ics_stats is not None:
            output.extra_fields["ics_triggered"] = ics_stats["ics_triggered"]
            output.extra_fields["ics_iterations"] = ics_stats["ics_iterations"]
            output.extra_fields["ics_corrected"] = ics_stats["ics_corrected"]
            if my_slot == 0:
                output.extra_fields["ics_iter_oracle_correct"] = ics_stats["iter_oracle_correct"]
        else:
            output.extra_fields["ics_triggered"] = False
            output.extra_fields["ics_iterations"] = 0
            output.extra_fields["ics_corrected"] = False

        return output

    async def _fill_rollout_buffer(
        self,
        sampling_params: dict,
        question: str,
        ground_truth: str,
    ) -> tuple[list, dict, list, list, list, list]:
        """Fill 8 rollout slots with 4 G1 fresh chains + 4 G2 counterfactuals.

        Returns the same 6-tuple as SRPOAgentLoop so that run() works unchanged.
        """
        assert self.rollout_n == 8, (
            f"SRPO requires actor_rollout_ref.rollout.n == 8 "
            f"(got {self.rollout_n}). The 4+4 Group-1/Group-2 split is hardcoded."
        )

        ics_stats = {
            "ics_triggered": False,
            "ics_iterations": 0,
            "ics_corrected": False,
            "ics_error_steps": [],
            "ics_triggers": 0,
            "fresh_chains": 0,
            "iter_oracle_correct": [],
            # Offline analysis: populated when SRPO_BRANCH_DUMP_DIR is set.
            "ics_loc_error_steps": [],
            "ics_loc_n_steps": [],
            "ics_loc_reasonings": [],
            "ics_loc_prompts": [],
        }
        _capture_loc = bool(os.environ.get("SRPO_BRANCH_DUMP_DIR", "").strip())

        # Phase 1: collect 4 Group-1 chains and find the first failure probe.
        # Reward stored is the raw graded score (in [0,1]) so advantage normalization
        # sees a continuous signal on graded datasets (retrosynthesis_uspto50k).
        # On math (binary 0/1 score) raw_score == float(is_correct) — bit-exact unchanged.
        group1: list = []
        group2_parent: Optional[tuple] = None
        n_fresh = 0

        while (len(group1) < 4 or group2_parent is None) and n_fresh < _MAX_FRESH_ATTEMPTS:
            chain = await self._generate_thought_chain(sampling_params, question)
            n_fresh += 1
            ics_stats["fresh_chains"] += 1

            if chain.num_thoughts == 0:
                if len(group1) < 4:
                    group1.append((chain, 0.0))
                    ics_stats["iter_oracle_correct"].append(False)
                continue

            is_correct, raw_score = await self._check_correctness_and_score(chain, ground_truth)
            logger.info(f"[SRPO] Fresh #{n_fresh}: {chain.num_thoughts} thoughts, correct={is_correct}, score={raw_score:.3f}")

            if group2_parent is None and not is_correct:
                # Parent is dropped from the buffer in SRPO (only used for localization),
                # so its reward never reaches advantage computation.
                group2_parent = (chain, raw_score)
            elif len(group1) < 4:
                group1.append((chain, raw_score))
                ics_stats["iter_oracle_correct"].append(is_correct)

        has_group2 = group2_parent is not None

        # Phase 2: no failure found → fill all 8 slots with G1 (vanilla GRPO fallback)
        if not has_group2:
            logger.info("[SRPO] No failure found — filling all 8 slots with Group 1 (vanilla GRPO fallback)")
            while len(group1) < 8:
                chain = await self._generate_thought_chain(sampling_params, question)
                is_correct, raw_score = await self._check_correctness_and_score(chain, ground_truth)
                group1.append((chain, raw_score))
                ics_stats["iter_oracle_correct"].append(is_correct)
                ics_stats["fresh_chains"] += 1

            buffer = [c for c, _ in group1[:8]]
            buffer_meta = [
                {"suffix_start_idx": 0, "reward": r, "reset_advantage": 0.0}
                for _, r in group1[:8]
            ]
            self._compute_advantages(buffer_meta, has_group2=False)
            return buffer, ics_stats, [], [], [], buffer_meta

        # Phase 3: localize error in the parent and generate 4 counterfactuals
        parent_chain, _ = group2_parent
        ics_stats["ics_triggered"] = True
        ics_stats["ics_triggers"] += 1

        try:
            error_step, _, error_reasoning, _, _, loc_prompt_text = await self._localize_error(
                sampling_params, question, ground_truth,
                parent_chain.decoded_thoughts,
            )
        except Exception as e:
            logger.warning(f"[SRPO] Localization failed: {e}. Using error_step=1.")
            error_step = 1
            error_reasoning = f"[localization exception: {e}]"
            loc_prompt_text = ""

        if _capture_loc:
            # Capture the raw localizer output before any offset adjustment so
            # downstream localization-quality analyses see what the model picked.
            ics_stats["ics_loc_error_steps"].append(error_step)
            ics_stats["ics_loc_n_steps"].append(parent_chain.num_thoughts)
            ics_stats["ics_loc_reasonings"].append(error_reasoning)
            ics_stats["ics_loc_prompts"].append(loc_prompt_text)

        # SRPO-l2n-o variant: pull the reset back by N step boundaries when
        # SRPO_LOC_OFFSET=N is set. Default 0 preserves standard SRPO behavior.
        # `error_step` is 1-indexed; floor at 1 (= empty prefix).
        loc_offset = int(os.environ.get("SRPO_LOC_OFFSET", "0"))
        if loc_offset > 0:
            offset_step = max(1, error_step - loc_offset)
            if offset_step != error_step:
                logger.info(
                    f"[SRPO] Offset applied: localizer picked step {error_step}, "
                    f"using step {offset_step} (offset={loc_offset})"
                )
            error_step = offset_step

        error_step = max(1, min(error_step, parent_chain.num_thoughts))
        ics_stats["ics_error_steps"].append(error_step)
        logger.info(
            f"[SRPO] Parent: error at step {error_step}/{parent_chain.num_thoughts}"
        )

        if error_step <= 1:
            prefix_response_ids = []
            prefix_logprobs = []
            prefix_boundaries = []
            prefix_thoughts = []
        else:
            cut_idx = parent_chain.thought_boundaries[error_step - 2][1]
            prefix_response_ids = parent_chain.response_ids[:cut_idx]
            prefix_logprobs = (
                parent_chain.response_logprobs[:cut_idx]
                if parent_chain.response_logprobs else []
            )
            prefix_boundaries = parent_chain.thought_boundaries[:error_step - 1]
            prefix_thoughts = parent_chain.decoded_thoughts[:error_step - 1]

        suffix_start = len(prefix_response_ids)

        # Generate 4 counterfactuals from the localized state
        corrections: list = []
        for i in range(4):
            try:
                corr = await self._generate_thought_chain_from_prefix(
                    sampling_params, question,
                    prefix_response_ids, prefix_logprobs,
                    prefix_boundaries, prefix_thoughts,
                )
                corr_correct, corr_score = await self._check_correctness_and_score(corr, ground_truth)
                logger.info(f"[SRPO] Correction {i+1}/4: {corr.num_thoughts} thoughts, correct={corr_correct}, score={corr_score:.3f}")
                if corr_correct:
                    ics_stats["ics_corrected"] = True
                ics_stats["iter_oracle_correct"].append(corr_correct)
                corrections.append((corr, corr_score, suffix_start))
            except Exception as e:
                logger.warning(f"[SRPO] Correction {i+1}/4 failed: {e}")
                fallback = await self._generate_thought_chain(sampling_params, question)
                fb_correct, fb_score = await self._check_correctness_and_score(fallback, ground_truth)
                ics_stats["iter_oracle_correct"].append(fb_correct)
                corrections.append((fallback, fb_score, 0))

        ics_stats["ics_iterations"] = 4  # always exactly 4 correction attempts

        # Phase 4: assemble buffer (4 G1 + 4 G2 counterfactuals, no parent)
        while len(group1) < 4:
            chain = await self._generate_thought_chain(sampling_params, question)
            is_correct, raw_score = await self._check_correctness_and_score(chain, ground_truth)
            group1.append((chain, raw_score))
            ics_stats["fresh_chains"] += 1
            ics_stats["iter_oracle_correct"].append(is_correct)

        buffer = (
            [c for c, _ in group1[:4]]
            + [c for c, _, _ in corrections]
        )
        buffer_meta = (
            [{"suffix_start_idx": 0, "reward": r, "reset_advantage": 0.0} for _, r in group1[:4]]
            + [
                {"suffix_start_idx": (s if self.mask_correction_prefix else 0), "reward": r, "reset_advantage": 0.0}
                for _, r, s in corrections
            ]
        )

        self._compute_advantages(buffer_meta, has_group2=True)

        return buffer, ics_stats, [], [], [], buffer_meta

    @staticmethod
    def _compute_advantages(buffer_meta: list, has_group2: bool) -> None:
        """Normalize rewards within each GRPO group and store as reset_advantage.

        Modifies buffer_meta in-place.
        """
        def _normalize(indices: list) -> None:
            rewards = np.array([buffer_meta[i]["reward"] for i in indices], dtype=float)
            std = float(rewards.std())
            mean = float(rewards.mean())
            if std < 1e-8:
                # Zero variance — no learning signal; advantages all zero
                for i in indices:
                    buffer_meta[i]["reset_advantage"] = 0.0
            else:
                for j, i in enumerate(indices):
                    buffer_meta[i]["reset_advantage"] = float((rewards[j] - mean) / std)

        if not has_group2:
            _normalize(list(range(len(buffer_meta))))
        else:
            # Group 1: slots 0–3
            _normalize(list(range(4)))
            # Group 2: slots 4–7
            _normalize(list(range(4, 8)))
