"""SCoRe scoring hook: applies shaped reward AND per-sequence KL penalty.

The SCoRe agent loop attaches metadata to each trajectory:
    - score_is_correction: whether this is y2 (correction attempt)
    - score_y1_correct: whether y1 was correct (for the corresponding y1)

This hook monkey-patches verl's compute_advantage() to (a) apply SCoRe's
shaped outcome reward and (b) fold a length-normalized (mean-KL) penalty
into token_level_rewards with a per-sequence coefficient keyed on
score_is_correction. Mean-KL is used instead of sum-KL to make β values
length-invariant — this matches the aggregation convention used in the
public SCoRe reimplementations (BY571/SCoRe, daje0601/Google_SCoRe) and
keeps the paper's β=0.1/0.01 hyperparameters in a sensible range for our
longer thought-chain generations.

Stage I (Eq. 3 in Kumar et al.):
    y1 tokens: KL coeff = beta2 (heavy); outcome reward = 0
    y2 tokens: KL coeff = beta1 (small); outcome reward = R(y2)

Stage II (Eq. 4 + bonus):
    y1 tokens: KL coeff = beta1; outcome reward = R(y1)
    y2 tokens: KL coeff = beta1; outcome reward = R(y2) + alpha*(R(y2)-R(y1))

Mid-run stage switch:
    If SCORE_STAGE1_STEPS > 0, the first SCORE_STAGE1_STEPS gradient steps
    run Stage I and subsequent steps run Stage II, using the Stage I
    weights already in GPU memory as the Stage II initialization. This
    reproduces the paper's two-stage pipeline in a single training run.

The "score" advantage estimator broadcasts the sum of token_level_rewards
uniformly across tokens, which gives exactly the sequence-level REINFORCE
gradient: E[grad log pi * (R - beta*sum_t kl_t)].

Relies on ref_log_prob being populated in data.batch, which requires
algorithm.use_kl_in_reward=True in config. VERL's own KL-in-reward
subtraction is neutralised by setting algorithm.kl_ctrl.kl_coef=0.

Reference:
    Kumar et al. "Training Language Models to Self-Correct via Reinforcement
    Learning." ICLR 2025. Equations 2-4.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Configurable via environment variables (also overridable in config)
SCORE_STAGE = int(os.getenv("SCORE_STAGE", "2"))
SCORE_ALPHA = float(os.getenv("SCORE_ALPHA", "10.0"))
# Paper's beta_1: default (small) KL coefficient, applied to y2 in Stage I
# and to both y1 and y2 in Stage II.
SCORE_BETA1 = float(os.getenv("SCORE_BETA1", "0.01"))
# Paper's beta_2: heavy KL coefficient applied ONLY to y1 tokens in Stage I
# to keep the first-attempt distribution close to pi_ref (decoupling).
# Ignored in Stage II.
SCORE_BETA2 = float(os.getenv("SCORE_BETA2", "0.1"))
# Mid-run Stage I -> Stage II switch. If > 0, the scoring hook runs Stage I
# for the first SCORE_STAGE1_STEPS training steps (compute_advantage calls)
# and Stage II thereafter, using the Stage I weights already in GPU memory
# as the Stage II initialization. If 0/unset, SCORE_STAGE is used as a
# static setting (legacy behaviour).
SCORE_STAGE1_STEPS = int(os.getenv("SCORE_STAGE1_STEPS", "0"))

# Module-level step counter, incremented on each compute_advantage call
# that routes through the SCoRe path. Used only when SCORE_STAGE1_STEPS > 0.
_SCORE_STEP_COUNT = 0


def _resolve_stage() -> int:
    """Return current SCoRe stage (1 or 2).

    If SCORE_STAGE1_STEPS > 0, stage is determined by the step counter:
    Stage I for counter < threshold, Stage II otherwise. With threshold=K,
    calls 1..K run Stage I (counter 0..K-1) and calls K+1.. run Stage II.

    Otherwise falls back to the static SCORE_STAGE env var.
    """
    if SCORE_STAGE1_STEPS > 0:
        return 1 if _SCORE_STEP_COUNT < SCORE_STAGE1_STEPS else 2
    return SCORE_STAGE


def apply_shaped_reward(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    score_is_correction: np.ndarray,
    score_y1_correct: np.ndarray,
    stage: int = 2,
    alpha: float = 10.0,
    old_log_probs: torch.Tensor | None = None,
    ref_log_probs: torch.Tensor | None = None,
    beta1: float = 0.01,
    beta2: float = 0.1,
) -> torch.Tensor:
    """Apply SCoRe's shaped outcome reward plus per-sequence KL penalty.

    The returned tensor is used directly by the "score" advantage estimator,
    which broadcasts the per-sequence sum uniformly — giving REINFORCE with
    a KL-regularized objective.

    Args:
        token_level_rewards: Shape (bs, T). Base outcome reward, placed at
            the last valid token (VERL convention).
        response_mask: Shape (bs, T).
        score_is_correction: Shape (bs,). True for y2 attempts.
        score_y1_correct: Shape (bs,). True if the corresponding y1 was
            correct.
        stage: 1 or 2.
        alpha: Stage II self-correction bonus weight.
        old_log_probs, ref_log_probs: Shape (bs, T). Used to compute the k1
            KL estimator log pi_theta - log pi_ref. If either is None, the
            KL term is skipped (caller should warn).
        beta1: Default KL coefficient (applied to y2 in Stage I, both in II).
        beta2: Heavy y1-only KL coefficient for Stage I; ignored in Stage II.

    Returns:
        Shape (bs, T) token-level rewards with outcome scalar at the last
        valid token and per-token KL penalty subtracted on every valid token.
    """
    bs, _ = token_level_rewards.shape
    shaped = torch.zeros_like(token_level_rewards)

    # --- Outcome reward: scalar at the last valid token ---
    for i in range(bs):
        base_reward = (token_level_rewards[i] * response_mask[i]).sum().item()
        is_correction = bool(score_is_correction[i])
        y1_was_correct = bool(score_y1_correct[i])

        if stage == 1:
            new_reward = base_reward if is_correction else 0.0
        else:
            if is_correction:
                r_y1 = 1.0 if y1_was_correct else 0.0
                r_y2 = base_reward
                new_reward = r_y2 + alpha * (r_y2 - r_y1)
            else:
                new_reward = base_reward

        valid_positions = (response_mask[i] > 0).nonzero(as_tuple=True)[0]
        if len(valid_positions) > 0:
            shaped[i, valid_positions[-1]] = new_reward

    # --- Per-token KL penalty, length-normalized (mean-KL) ---
    #
    # We use mean-KL (sum of per-token KL / sequence length) rather than
    # sum-KL. The paper's equations are written as sequence-level D_KL but
    # every public SCoRe reimplementation (BY571, daje0601) uses mean-KL
    # aggregation; with the paper's β values (0.1, 0.01), sum-KL over our
    # 2000-token thought chains produces penalties that swamp the sparse
    # binary reward signal. Mean-KL makes β length-invariant and matches
    # the aggregation convention implicit in the paper's hyperparameters.
    #
    # The broadcast-sum advantage estimator sums token_level_rewards across
    # the sequence. Dividing per-token KL by seq_len here means that sum
    # equals (β · mean-KL), giving the REINFORCE gradient:
    #   ∇log π · (R − β · mean-KL)
    if old_log_probs is None or ref_log_probs is None:
        return shaped

    # k1 estimator: log π_θ - log π_ref.
    kl_per_token = (old_log_probs - ref_log_probs) * response_mask
    # Normalize by sequence length so the broadcast-sum = mean-KL.
    seq_len = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    kl_per_token = kl_per_token / seq_len

    is_corr_t = torch.as_tensor(
        score_is_correction.astype(bool), device=shaped.device
    )
    if stage == 1:
        # y1 (is_corr=False) -> beta2 (heavy); y2 (is_corr=True) -> beta1.
        beta_per_seq = torch.where(
            is_corr_t,
            torch.tensor(beta1, device=shaped.device),
            torch.tensor(beta2, device=shaped.device),
        ).unsqueeze(1)
    else:
        beta_per_seq = torch.full(
            (bs, 1), beta1, device=shaped.device, dtype=shaped.dtype
        )

    shaped = shaped - beta_per_seq * kl_per_token
    return shaped


def patch_compute_advantage():
    """Monkey-patch verl's compute_advantage to support SCoRe.

    Wraps the original to intercept calls where adv_estimator == "score".
    For SCoRe:
        1. Extract SCoRe metadata from data.non_tensor_batch
        2. Apply shaped reward to token_level_rewards
        3. Call the SCoRe advantage estimator

    For all other estimators, the original function is called unchanged.

    This patch is idempotent.
    """
    import verl.trainer.ppo.ray_trainer as ray_trainer_module

    original_compute_advantage = ray_trainer_module.compute_advantage

    # Guard against double-patching
    if getattr(original_compute_advantage, "_score_patched", False):
        return

    def patched_compute_advantage(data, adv_estimator, **kwargs):
        """Wrapped compute_advantage with SCoRe shaped reward support."""
        adv_name = (
            adv_estimator.value
            if hasattr(adv_estimator, "value")
            else str(adv_estimator)
        )

        if adv_name != "score":
            return original_compute_advantage(data, adv_estimator, **kwargs)

        # --- SCoRe path: apply shaped reward ---
        from verl.trainer.ppo import core_algos

        # Check if SCoRe metadata is available
        has_score_data = (
            "score_is_correction" in data.non_tensor_batch
            and "score_y1_correct" in data.non_tensor_batch
        )

        if has_score_data:
            # Resolve stage BEFORE incrementing, so call #K with threshold=K
            # still sees Stage I (counter is K-1 at entry).
            global _SCORE_STEP_COUNT
            current_stage = _resolve_stage()
            transitioning = (
                SCORE_STAGE1_STEPS > 0
                and _SCORE_STEP_COUNT == SCORE_STAGE1_STEPS
            )
            if transitioning:
                logger.warning(
                    f"SCoRe: Stage I -> Stage II transition at step "
                    f"{_SCORE_STEP_COUNT + 1} (threshold={SCORE_STAGE1_STEPS})"
                )

            # Pull log probs for per-token KL. Both are expected when
            # algorithm.use_kl_in_reward=True (triggers the ref-policy path
            # in ray_trainer.py:_compute_ref_log_prob). If missing, fall
            # back to outcome-only shaping with a warning — Stage I will
            # degrade to the old (broken) behaviour, Stage II is fine.
            old_log_probs = data.batch.get("old_log_probs", None)
            ref_log_probs = data.batch.get("ref_log_prob", None)
            if old_log_probs is None or ref_log_probs is None:
                logger.warning(
                    "SCoRe scoring: ref_log_prob or old_log_probs missing "
                    "from batch; KL penalty will NOT be applied. Set "
                    "algorithm.use_kl_in_reward=True in the config."
                )

            data.batch["token_level_rewards"] = apply_shaped_reward(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                score_is_correction=data.non_tensor_batch["score_is_correction"],
                score_y1_correct=data.non_tensor_batch["score_y1_correct"],
                stage=current_stage,
                alpha=SCORE_ALPHA,
                old_log_probs=old_log_probs,
                ref_log_probs=ref_log_probs,
                beta1=SCORE_BETA1,
                beta2=SCORE_BETA2,
            )

            n_corrections = sum(
                1 for x in data.non_tensor_batch["score_is_correction"] if x
            )
            kl_enabled = old_log_probs is not None and ref_log_probs is not None
            logger.info(
                f"SCoRe scoring: step={_SCORE_STEP_COUNT + 1} "
                f"stage={current_stage} (static={SCORE_STAGE}, "
                f"threshold={SCORE_STAGE1_STEPS}) alpha={SCORE_ALPHA} "
                f"beta1={SCORE_BETA1} beta2={SCORE_BETA2} kl={kl_enabled} "
                f"seqs={len(data.non_tensor_batch['score_is_correction'])} "
                f"(corrections={n_corrections})"
            )
            _SCORE_STEP_COUNT += 1
        else:
            logger.info("SCoRe scoring: no SCoRe metadata found, using base rewards")

        # Call SCoRe advantage estimator
        adv_estimator_fn = core_algos.get_adv_estimator_fn("score")
        config = kwargs.get("config", None)

        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]

        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    patched_compute_advantage._score_patched = True
    ray_trainer_module.compute_advantage = patched_compute_advantage
    logger.info("SCoRe: patched verl.trainer.ppo.ray_trainer.compute_advantage")
