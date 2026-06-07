"""SRPO2x4 agent loop: 2 independent groups of 4, no parent reuse.

Like SRPOx4 but the parents are discarded after localization — each group's
4 slots are filled with counterfactuals sampled from the shared prefix. The
two groups use different failure parents (and thus different prefixes) and
are normalized separately for GRPO advantages.

Think of this as "SRPO doubled": each group matches SRPO's G2 layout
(4 counterfactuals, no parent anchor), but we run two independent groups
driven by two distinct failure prefixes.

If fewer than 2 failures are found within MAX_FRESH_ATTEMPTS chains, falls
back to 8 fresh i.i.d. chains (vanilla GRPO), same as SRPOx4.

Buffer layout:
  Group 1 (slots 0-3): 4 counterfactuals from prefix1 (suffix_start=cut_idx1)
  Group 2 (slots 4-7): 4 counterfactuals from prefix2 (suffix_start=cut_idx2)
"""

import logging
import os

from training.srpo_agent_loop import SRPOAgentLoop, _MAX_FRESH_ATTEMPTS

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SRPO2x4AgentLoop(SRPOAgentLoop):
    """SRPO2x4: 2 independent failure prefixes × 4 counterfactuals each, no parent in buffer.

    Inherits run() and _compute_advantages from SRPOAgentLoop. Overrides
    _fill_rollout_buffer to sample 2 parents (for localization only) and build
    2 prefix-conditioned groups of 4 counterfactuals each. Normalized separately
    per group.
    """

    # Separate buffer pool from the parent class.
    _buffers: dict = {}

    async def _fill_rollout_buffer(
        self,
        sampling_params: dict,
        question: str,
        ground_truth: str,
    ) -> tuple[list, dict, list, list, list, list]:
        """Fill 8 rollout slots with 2 groups of 4 counterfactuals, no parents.

        Returns the same 6-tuple as SRPOAgentLoop so that run() works unchanged.
        """
        assert self.rollout_n == 8, (
            f"SRPO2x4 requires actor_rollout_ref.rollout.n == 8 "
            f"(got {self.rollout_n}). The 2-group 4+4 layout is hardcoded."
        )

        ics_stats = {
            "ics_triggered": False,
            "ics_iterations": 0,
            "ics_corrected": False,
            "ics_error_steps": [],
            "ics_triggers": 0,
            "fresh_chains": 0,
            "iter_oracle_correct": [],
        }

        # Phase 1: sample fresh chains until 2 incorrect ones are found.
        # Non-failure chains are kept only as fallback material — discarded on success.
        sampled: list = []  # (chain, reward) — used only in the <2-failures fallback
        parents: list = []
        n_fresh = 0

        while len(parents) < 2 and n_fresh < _MAX_FRESH_ATTEMPTS:
            chain = await self._generate_thought_chain(sampling_params, question)
            n_fresh += 1
            ics_stats["fresh_chains"] += 1

            if chain.num_thoughts == 0:
                sampled.append((chain, 0.0))
                ics_stats["iter_oracle_correct"].append(False)
                continue

            is_correct = await self._check_correctness(chain, ground_truth)
            logger.info(f"[SRPO2x4] Fresh #{n_fresh}: {chain.num_thoughts} thoughts, correct={is_correct}")

            if not is_correct:
                parents.append(chain)
            else:
                sampled.append((chain, 1.0))
                ics_stats["iter_oracle_correct"].append(is_correct)

        # Phase 2: fewer than 2 failures → vanilla GRPO fallback.
        if len(parents) < 2:
            logger.info(
                f"[SRPO2x4] Only {len(parents)} failure(s) within budget — "
                "filling all 8 slots with fresh chains (vanilla GRPO fallback)"
            )
            # Reuse the failure(s) we did find as 0-reward chains.
            for p in parents:
                sampled.append((p, 0.0))
                ics_stats["iter_oracle_correct"].append(False)
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

        ics_stats["ics_triggered"] = True
        ics_stats["ics_triggers"] += 1

        # Phase 3: for each parent, localize and generate 4 counterfactuals.
        # Parents are used only as the source of the localization prefix — they
        # are NOT added to the training buffer (unlike SRPOx4).
        groups: list = []  # list of (suffix_start, corrections) — one per parent
        for p_idx, parent_chain in enumerate(parents):
            try:
                error_step, _, _, _, _, _ = await self._localize_error(
                    sampling_params, question, ground_truth,
                    parent_chain.decoded_thoughts,
                )
            except Exception as e:
                logger.warning(f"[SRPO2x4] Parent-{p_idx} localization failed: {e}. Using error_step=1.")
                error_step = 1

            error_step = max(1, min(error_step, parent_chain.num_thoughts))
            ics_stats["ics_error_steps"].append(error_step)
            logger.info(
                f"[SRPO2x4] Parent-{p_idx}: error at step {error_step}/{parent_chain.num_thoughts}"
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

            corrections: list = []  # (chain, reward, suffix_start)
            for i in range(4):
                try:
                    corr = await self._generate_thought_chain_from_prefix(
                        sampling_params, question,
                        prefix_response_ids, prefix_logprobs,
                        prefix_boundaries, prefix_thoughts,
                    )
                    corr_correct = await self._check_correctness(corr, ground_truth)
                    logger.info(
                        f"[SRPO2x4] Parent-{p_idx} correction {i+1}/4: "
                        f"{corr.num_thoughts} thoughts, correct={corr_correct}"
                    )
                    if corr_correct:
                        ics_stats["ics_corrected"] = True
                    ics_stats["iter_oracle_correct"].append(corr_correct)
                    corrections.append((corr, float(corr_correct), suffix_start))
                except Exception as e:
                    logger.warning(f"[SRPO2x4] Parent-{p_idx} correction {i+1}/4 failed: {e}")
                    fallback = await self._generate_thought_chain(sampling_params, question)
                    fb_correct = await self._check_correctness(fallback, ground_truth)
                    ics_stats["iter_oracle_correct"].append(fb_correct)
                    corrections.append((fallback, float(fb_correct), 0))

            groups.append((suffix_start, corrections))

        ics_stats["ics_iterations"] = 8  # 4 corrections × 2 parents

        # Phase 4: assemble buffer — group 1 (slots 0-3) + group 2 (slots 4-7).
        # No parents in the buffer; each group is 4 counterfactuals sharing a prefix.
        buffer = []
        buffer_meta = []
        for suffix_start, corrections in groups:
            for corr, reward, s in corrections:
                buffer.append(corr)
                buffer_meta.append({
                    "suffix_start_idx": s,
                    "reward": reward,
                    "reset_advantage": 0.0,
                })

        # 4+4 layout matches SRPO's group convention; normalize each group separately.
        self._compute_advantages(buffer_meta, has_group2=True)

        return buffer, ics_stats, [], [], [], buffer_meta
