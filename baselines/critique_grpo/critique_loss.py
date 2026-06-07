"""Critique-GRPO loss function with on/off-policy token handling.

Implements the dual loss from Critique-GRPO (Zhang et al., arXiv:2506.03106):

On-policy tokens (original responses, prefix_mask=0):
    L_on = -advantage * f(pi_new) / f(pi_old)
    where f(p) = p / (p + gamma)  [shaping function]

Off-policy tokens (refinement responses, prefix_mask=1):
    L_off = -advantage * f(pi_new)
    where f(p) = p / (p + gamma)  [shaping function, no pi_old denominator]

The shaping function f(p) = p/(p+gamma) with gamma=0.1:
    - Amplifies gradients for low-probability tokens (unfamiliar but correct)
    - Naturally down-weights high-probability tokens (already known)
    - Bounded in [0, 1), preventing extreme ratio explosions

No PPO clipping is used (loss_remove_clip=True in the paper).

Registers as "critique_grpo" via the verl policy loss registry.
"""

import os
from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig

CRITIQUE_GAMMA = float(os.getenv("CRITIQUE_GAMMA", "0.1"))


def _shaping_fn(log_prob: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Compute f(p) = p / (p + gamma) from log probabilities.

    Args:
        log_prob: Log-probabilities, shape (bs, response_length).
        gamma: Shaping parameter. Default 0.1.

    Returns:
        Shaped values in [0, 1), same shape.
    """
    prob = torch.exp(log_prob.clamp(min=-20.0, max=0.0))
    return prob / (prob + gamma)


@register_policy_loss("critique_grpo")  # type: ignore[arg-type]
def compute_policy_loss_critique_grpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    prefix_mask: torch.Tensor | None = None,
    thought_segment_ids: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute Critique-GRPO loss with on/off-policy token handling.

    Args:
        old_log_prob: Log-probs under old policy, shape (bs, response_length).
        log_prob: Log-probs under current policy, shape (bs, response_length).
        advantages: Advantage estimates, shape (bs, response_length).
        response_mask: Token mask, shape (bs, response_length).
        loss_agg_mode: Aggregation mode (unused, always seq-mean-token-sum).
        config: Actor config.
        rollout_is_weights: Optional importance sampling weights.
        prefix_mask: Binary mask, shape (bs, response_length). 1.0 for
            off-policy (refinement) tokens, 0.0 for on-policy tokens.
        thought_segment_ids: Unused (kept for interface compatibility).
    """
    assert config is not None
    gamma = CRITIQUE_GAMMA

    # Compute shaped values for off-policy tokens
    f_new = _shaping_fn(log_prob, gamma)

    # On-policy ratio: standard importance ratio exp(log_new - log_old)
    on_ratio = torch.exp((log_prob - old_log_prob).clamp(min=-20.0, max=20.0))

    # Off-policy ratio: just f(pi_new) (no pi_old denominator)
    off_ratio = f_new

    # KL tracking (for metrics, not used in loss)
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Combine on-policy and off-policy losses via prefix_mask
    if prefix_mask is None:
        # No refinements: all on-policy
        pg_losses = -advantages * on_ratio
    else:
        on_policy_mask = 1.0 - prefix_mask
        off_pg_losses = -advantages * off_ratio
        on_pg_losses = -advantages * on_ratio
        pg_losses = off_pg_losses * prefix_mask + on_pg_losses * on_policy_mask

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # Aggregation: divide by response_length (constant denominator,
    # matching the paper's loss_remove_token_mean=True)
    masked_losses = pg_losses * response_mask
    pg_loss = masked_losses.sum() / response_mask.shape[-1]

    # Metrics
    n_refinement_tokens = prefix_mask.sum().item() if prefix_mask is not None else 0
    n_total_tokens = response_mask.sum().item()

    pg_metrics = {
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/on_ratio_mean": verl_F.masked_mean(
            on_ratio, response_mask
        ).detach().item(),
        "actor/off_ratio_mean": (
            verl_F.masked_mean(off_ratio, prefix_mask).detach().item()
            if prefix_mask is not None and prefix_mask.sum() > 0
            else 0.0
        ),
        "actor/refinement_token_frac": (
            n_refinement_tokens / max(n_total_tokens, 1)
        ),
    }
    return pg_loss, pg_metrics
