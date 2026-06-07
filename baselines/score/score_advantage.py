"""SCoRe REINFORCE advantage estimation for VERL.

Implements the REINFORCE advantage from SCoRe (Kumar et al., ICLR 2025).

SCoRe uses plain REINFORCE with NO advantage normalization.  Each token
in a trajectory receives the (shaped) outcome reward as its advantage.
No mean subtraction, no whitening -- the shaped reward IS the advantage.

The shaped reward R_shaped is computed by the scoring hook:
    - y1 (first attempt):  R(y1)  [Stage 2] or 0 [Stage 1]
    - y2 (correction):     R(y2) + alpha * (R(y2) - R(y1))  [Stage 2]
                           R(y2)  [Stage 1]

The scoring hook writes the shaped reward into token_level_rewards before
this estimator is called.

Registers "score" advantage estimator with VERL's registry.

Reference:
    Kumar et al. "Training Language Models to Self-Correct via Reinforcement
    Learning." ICLR 2025. Section 4.2.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


try:
    from verl.trainer.ppo.core_algos import register_adv_est

    @register_adv_est("score")
    def compute_score_outcome_advantage(
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        config=None,
        index=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SCoRe REINFORCE advantage estimator for VERL.

        Each token receives the sequence-level outcome reward as its
        advantage. This is standard REINFORCE — no baseline, no
        normalization, no value function.  The shaped reward IS the
        advantage.

        The shaped reward (R(y2) + alpha*(R(y2)-R(y1))) has already been
        applied to token_level_rewards by the scoring hook.

        Args:
            token_level_rewards: Shape (bs, response_length). Outcome reward
                is placed at the last valid token by VERL.
            response_mask: Shape (bs, response_length).
            config: Algorithm config (unused).
            index: UID array for grouping (unused — SCoRe does not group).

        Returns:
            Tuple (advantages, returns) both of shape (bs, response_length).
        """
        # Extract outcome reward per sequence
        outcome_rewards = (token_level_rewards * response_mask).sum(
            dim=-1, keepdim=True
        )

        # Broadcast uniformly across valid tokens — no normalization
        advantages = outcome_rewards.expand_as(response_mask) * response_mask
        returns = advantages.clone()

        return advantages, returns

except ImportError:
    logger.debug(
        "verl not available; skipping registration of 'score' advantage "
        "estimator. Core functions are still usable standalone."
    )
