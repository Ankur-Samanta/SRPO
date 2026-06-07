"""Critique-GRPO advantage estimation for VERL.

Implements GRPO advantages without std normalization (Dr.GRPO style)
as used in Critique-GRPO (Zhang et al., arXiv:2506.03106).

Advantages:
    A(i) = R(i) - mean(R(group))

where the group includes both original responses and the refinement.
No std normalization is applied (grpo_use_std=False in the paper).

The scoring hook applies the off-policy shaping function p/(p+gamma) for
refinement tokens AFTER advantages are computed, so this estimator only
handles the advantage computation.

Registers "critique_grpo" advantage estimator with VERL's registry.
"""

from __future__ import annotations

import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


def _normalize_advantages_no_std(
    advantages: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Center advantages (subtract mean) without std normalization.

    This matches the Dr.GRPO formulation used by Critique-GRPO:
    advantages are group-mean-subtracted but NOT divided by std.
    """
    valid = mask.bool()
    if valid.sum() == 0:
        return advantages
    valid_advs = advantages[valid]
    mean = valid_advs.mean()
    advantages = advantages.clone()
    advantages[valid] = valid_advs - mean
    return advantages


try:
    from verl.trainer.ppo.core_algos import register_adv_est

    @register_adv_est("critique_grpo")
    def compute_critique_grpo_advantage(
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        config=None,
        index=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Critique-GRPO advantage estimator for VERL.

        Group-relative advantages (GRPO) without std normalization.
        Groups are determined by the index (uid) array — all rollouts
        sharing the same prompt form one group.

        Args:
            token_level_rewards: Shape (bs, response_length).
            response_mask: Shape (bs, response_length).
            config: Algorithm config (unused).
            index: UID array for grouping, shape (bs,).

        Returns:
            Tuple (advantages, returns) both of shape (bs, response_length).
        """
        bs, response_length = token_level_rewards.shape

        # Extract per-sequence outcome rewards
        outcome_rewards = (token_level_rewards * response_mask).sum(dim=-1)  # (bs,)

        # Compute group-relative advantages
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)

        if index is not None:
            # Group by prompt uid
            unique_ids = np.unique(index)
            for uid in unique_ids:
                group_mask = torch.tensor(
                    [1 if idx == uid else 0 for idx in index],
                    dtype=torch.bool,
                    device=token_level_rewards.device,
                )
                group_rewards = outcome_rewards[group_mask]
                group_mean = group_rewards.mean()

                # A(i) = R(i) - mean(R(group)), NO std normalization
                for i in range(bs):
                    if index[i] == uid:
                        adv = outcome_rewards[i] - group_mean
                        advantages[i] = adv * response_mask[i]
                        returns[i] = outcome_rewards[i] * response_mask[i]
        else:
            # No grouping info: treat entire batch as one group
            batch_mean = outcome_rewards.mean()
            for i in range(bs):
                adv = outcome_rewards[i] - batch_mean
                advantages[i] = adv * response_mask[i]
                returns[i] = outcome_rewards[i] * response_mask[i]

        return advantages, returns

except ImportError:
    logger.debug(
        "verl not available; skipping registration of 'critique_grpo' "
        "advantage estimator."
    )
