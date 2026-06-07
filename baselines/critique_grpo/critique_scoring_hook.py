"""Critique-GRPO scoring hook: off-policy shaping for refinement tokens.

The Critique-GRPO agent loop marks refinement trajectories with
is_refinement=True in extra_fields. The loss for these off-policy tokens
uses a shaping function instead of the standard importance ratio:

    On-policy (original responses):  ratio = pi_new / pi_old  (standard)
    Off-policy (refinements):        rho = pi_new / (pi_new + gamma)

where gamma=0.1. This amplifies gradients for tokens the model currently
assigns low probability to (correct but unfamiliar refinement tokens).

This hook monkey-patches verl's compute_advantage to:
    1. Extract is_refinement flags from non_tensor_batch
    2. Pass them as extra kwargs to the advantage estimator
    3. Store a prefix_mask in the batch for the loss function

The actual shaping is applied in the loss computation via prefix_mask.
Since verl's vanilla loss doesn't support prefix_mask natively, we store
it and rely on a custom loss function (registered separately).

Reference:
    Zhang et al. "Critique-GRPO: Advancing LLM Reasoning with Natural
    Language and Numerical Feedback." arXiv:2506.03106. Section 3.3.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

CRITIQUE_GAMMA = float(os.getenv("CRITIQUE_GAMMA", "0.1"))


def build_prefix_mask(
    is_refinement_batch: np.ndarray,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Build a binary mask indicating off-policy (refinement) tokens.

    Args:
        is_refinement_batch: Array of bools, shape (bs,).
        response_mask: Shape (bs, response_length).

    Returns:
        prefix_mask of shape (bs, response_length). 1.0 for refinement
        tokens, 0.0 for original response tokens.
    """
    bs, response_length = response_mask.shape
    prefix_mask = torch.zeros_like(response_mask)
    for i in range(bs):
        if bool(is_refinement_batch[i]):
            prefix_mask[i] = response_mask[i]
    return prefix_mask


def patch_compute_advantage():
    """Monkey-patch verl's compute_advantage to support Critique-GRPO.

    For adv_estimator == "critique_grpo":
        1. Extract is_refinement flags from non_tensor_batch
        2. Build prefix_mask and store in batch
        3. Call the critique_grpo advantage estimator

    For all other estimators, the original function is called unchanged.
    This patch is idempotent.
    """
    import verl.trainer.ppo.ray_trainer as ray_trainer_module

    original_compute_advantage = ray_trainer_module.compute_advantage

    if getattr(original_compute_advantage, "_critique_grpo_patched", False):
        return

    def patched_compute_advantage(data, adv_estimator, **kwargs):
        adv_name = (
            adv_estimator.value
            if hasattr(adv_estimator, "value")
            else str(adv_estimator)
        )

        if adv_name != "critique_grpo":
            return original_compute_advantage(data, adv_estimator, **kwargs)

        from verl.trainer.ppo import core_algos

        # Build prefix_mask from is_refinement flags
        has_critique_data = "is_refinement" in data.non_tensor_batch

        if has_critique_data:
            prefix_mask = build_prefix_mask(
                is_refinement_batch=data.non_tensor_batch["is_refinement"],
                response_mask=data.batch["response_mask"],
            )
            data.batch["prefix_mask"] = prefix_mask

            n_refinements = sum(
                1 for x in data.non_tensor_batch["is_refinement"] if x
            )
            logger.info(
                f"Critique-GRPO scoring: {n_refinements} refinements out of "
                f"{len(data.non_tensor_batch['is_refinement'])} sequences, "
                f"gamma={CRITIQUE_GAMMA}"
            )
        else:
            data.batch["prefix_mask"] = torch.zeros_like(
                data.batch["response_mask"]
            )
            logger.info(
                "Critique-GRPO scoring: no is_refinement data, "
                "all tokens treated as on-policy"
            )

        # Call critique_grpo advantage estimator
        adv_estimator_fn = core_algos.get_adv_estimator_fn("critique_grpo")
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

    patched_compute_advantage._critique_grpo_patched = True
    ray_trainer_module.compute_advantage = patched_compute_advantage
    logger.info(
        "Critique-GRPO: patched verl.trainer.ppo.ray_trainer.compute_advantage"
    )
