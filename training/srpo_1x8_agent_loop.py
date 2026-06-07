"""SRPO_1x8 agent loop: 8 counterfactuals from a single localized prefix, no parent.

Same first-failure → localize → fresh-from-prefix scaffold as SRPO_1x8, but
the failure parent is DISCARDED rather than kept in the buffer. The result
is 8 counterfactuals sharing one prefix, forming a single GRPO group of 8.

Relation to existing variants:
    SRPO_1x8: 1 failure parent + 7 corrections (parent kept) -> 8 in single group
    SRPO_1x8 (this): 0 parents + 8 corrections (parent dumped) -> 8 in single group
    (Mirrors how SRPO dropped the parent from SRPO's correction group.)

Buffer layout:
  Slots 0-7: 8 counterfactuals from the localized prefix (suffix_start=cut_idx)

If no failure is found within MAX_FRESH_ATTEMPTS chains, falls back to 8
fresh i.i.d. chains (vanilla GRPO), reusing already-sampled chains.
"""

import logging
import os
from typing import Optional

from training.srpo_agent_loop import SRPOAgentLoop, _MAX_FRESH_ATTEMPTS

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SRPO1x8AgentLoop(SRPOAgentLoop):
    """SRPO_1x8: 8 counterfactuals from one localized prefix (no parent in buffer).

    Inherits run() and _compute_advantages from SRPOAgentLoop. Overrides
    _fill_rollout_buffer to produce a single shared-prefix group of 8 corrections.
    """

    # Separate buffer pool from the parent class.
    _buffers: dict = {}

    async def _fill_rollout_buffer(
        self,
        sampling_params: dict,
        question: str,
        ground_truth: str,
    ) -> tuple[list, dict, list, list, list, list]:
        """Fill 8 rollout slots with 8 counterfactuals from one localized prefix.

        Returns the same 6-tuple as SRPOAgentLoop so that run() works unchanged.
        """
        assert self.rollout_n == 8, (
            f"SRPO_1x8 requires actor_rollout_ref.rollout.n == 8 "
            f"(got {self.rollout_n}). The 0+8 single-group layout is hardcoded."
        )

        ics_stats = {
            "ics_triggered": False,
            "ics_iterations": 0,
            "ics_corrected": False,
            "ics_error_steps": [],
            "ics_triggers": 0,
            "fresh_chains": 0,
            "iter_oracle_correct": [],
            "ics_loc_error_steps": [],
            "ics_loc_n_steps": [],
            "ics_loc_reasonings": [],
            "ics_loc_prompts": [],
        }
        _capture_loc = bool(os.environ.get("SRPO_BRANCH_DUMP_DIR", "").strip())

        # Phase 1: sample fresh chains until the first incorrect one.
        sampled: list = []  # (chain, reward) — used only in the no-failure fallback
        parent_chain = None
        n_fresh = 0

        while parent_chain is None and n_fresh < _MAX_FRESH_ATTEMPTS:
            chain = await self._generate_thought_chain(sampling_params, question)
            n_fresh += 1
            ics_stats["fresh_chains"] += 1

            if chain.num_thoughts == 0:
                sampled.append((chain, 0.0))
                ics_stats["iter_oracle_correct"].append(False)
                continue

            is_correct = await self._check_correctness(chain, ground_truth)
            logger.info(f"[SRPO_1x8] Fresh #{n_fresh}: {chain.num_thoughts} thoughts, correct={is_correct}")

            if not is_correct:
                parent_chain = chain
                break
            sampled.append((chain, 1.0))
            ics_stats["iter_oracle_correct"].append(is_correct)

        # Phase 2: no failure found → vanilla GRPO fallback (reuse sampled, top up to 8).
        if parent_chain is None:
            logger.info("[SRPO_1x8] No failure found — filling all 8 slots with fresh chains (vanilla GRPO fallback)")
            while len(sampled) < 8:
                chain = await self._generate_thought_chain(sampling_params, question)
                is_correct = await self._check_correctness(chain, ground_truth)
                sampled.append((chain, float(is_correct)))
                ics_stats["iter_oracle_correct"].append(is_correct)
                ics_stats["fresh_chains"] += 1

            buffer = [c for c, _ in sampled[:8]]
            buffer_meta = [
                {"suffix_start_idx": 0, "reward": r, "reset_advantage": 0.0}
                for _, r in sampled[:8]
            ]
            self._compute_advantages(buffer_meta, has_group2=False)
            return buffer, ics_stats, [], [], [], buffer_meta

        # Phase 3: localize the error on the parent.
        ics_stats["ics_triggered"] = True
        ics_stats["ics_triggers"] += 1

        try:
            error_step, _, error_reasoning, _, _, loc_prompt_text = await self._localize_error(
                sampling_params, question, ground_truth,
                parent_chain.decoded_thoughts,
            )
        except Exception as e:
            logger.warning(f"[SRPO_1x8] Localization failed: {e}. Using error_step=1.")
            error_step = 1
            error_reasoning = f"[localization exception: {e}]"
            loc_prompt_text = ""

        if _capture_loc:
            ics_stats["ics_loc_error_steps"].append(error_step)
            ics_stats["ics_loc_n_steps"].append(parent_chain.num_thoughts)
            ics_stats["ics_loc_reasonings"].append(error_reasoning)
            ics_stats["ics_loc_prompts"].append(loc_prompt_text)

        error_step = max(1, min(error_step, parent_chain.num_thoughts))
        ics_stats["ics_error_steps"].append(error_step)
        logger.info(
            f"[SRPO_1x8] Parent: error at step {error_step}/{parent_chain.num_thoughts}"
        )

        # Compute the shared prefix (up to but not including the error step).
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

        # Phase 4: generate 8 counterfactuals from the shared prefix (parent dropped).
        corrections: list = []  # (chain, reward, suffix_start) — suffix_start differs only on fallback
        for i in range(8):
            try:
                corr = await self._generate_thought_chain_from_prefix(
                    sampling_params, question,
                    prefix_response_ids, prefix_logprobs,
                    prefix_boundaries, prefix_thoughts,
                )
                corr_correct = await self._check_correctness(corr, ground_truth)
                logger.info(f"[SRPO_1x8] Correction {i+1}/8: {corr.num_thoughts} thoughts, correct={corr_correct}")
                if corr_correct:
                    ics_stats["ics_corrected"] = True
                ics_stats["iter_oracle_correct"].append(corr_correct)
                corrections.append((corr, float(corr_correct), suffix_start))
            except Exception as e:
                logger.warning(f"[SRPO_1x8] Correction {i+1}/8 failed: {e}")
                # Emergency fallback: fresh chain with no shared prefix.
                fallback = await self._generate_thought_chain(sampling_params, question)
                fb_correct = await self._check_correctness(fallback, ground_truth)
                ics_stats["iter_oracle_correct"].append(fb_correct)
                corrections.append((fallback, float(fb_correct), 0))

        ics_stats["ics_iterations"] = 8

        # Phase 5: assemble buffer (8 corrections, no parent) as one group of 8.
        buffer = [c for c, _, _ in corrections]
        buffer_meta = [
            {"suffix_start_idx": s, "reward": r, "reset_advantage": 0.0}
            for _, r, s in corrections
        ]

        # Single group of 8 — standard GRPO normalization (has_group2=False).
        self._compute_advantages(buffer_meta, has_group2=False)

        return buffer, ics_stats, [], [], [], buffer_meta
