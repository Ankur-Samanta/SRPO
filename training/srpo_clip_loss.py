"""SRPO_clip policy loss: standard GRPO/PPO clipped ratio loss with srpo's
two-group advantage scheme and suffix-only gradient masking.

Uses the full PPO machinery (ratio, clip, dual-clip, agg_loss) but advantages
come pre-computed and group-normalized from
srpo_agent_loop._compute_advantages — Group 1 (slots 0-3) and Group 2
(slots 4-7) are normalized independently. Suffix-only masking is always on:
tokens before suffix_start_idx are excluded from the loss so the kept ICS
prefix gets no gradient.

Loss form:
    L = E_t [ min(r_t * A_t, clip(r_t, 1±ε) * A_t) ]   masked to suffix tokens
where r_t = exp(log_prob_t - old_log_prob_t) and A_t is the per-trajectory
reset_advantage broadcast to every (suffix) token.

This is what srpo was supposed to be all along: GRPO advantages + GRPO loss.
The original srpo_loss dropped the ratio/clip/dual-clip machinery, leaving
unclipped policy gradient — see s24 collapse for the consequence.
"""

from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig


@register_policy_loss("srpo_clip")
def compute_policy_loss_srpo_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
    thought_segment_ids: Optional[torch.Tensor] = None,
    reset_advantage: Optional[torch.Tensor] = None,
    suffix_start_idx: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Standard PPO clipped surrogate with srpo advantages and suffix mask.

    Args:
        old_log_prob: (bs, response_length) log-probs under π_old
        log_prob:     (bs, response_length) log-probs under π_θ
        advantages:   (bs, response_length) — IGNORED; we use reset_advantage.
        response_mask: (bs, response_length) — 1 for valid response tokens
        reset_advantage: (bs,) — pre-computed per-trajectory GRPO advantages,
            normalized within Group 1 and Group 2 independently.
        suffix_start_idx: (bs,) — per-sequence index where the suffix begins;
            tokens before this are masked out (the kept ICS prefix).
    """
    assert config is not None, "srpo_clip loss requires ActorConfig"
    bs = log_prob.shape[0]

    # Suffix mask: zero out tokens before suffix_start_idx (kept ICS prefix)
    loss_mask = response_mask.clone()
    if suffix_start_idx is not None:
        for i in range(bs):
            start = int(suffix_start_idx[i].item())
            if start > 0:
                loss_mask[i, :start] = 0

    # Per-trajectory advantage → broadcast to per-token (constant within each seq)
    if reset_advantage is None:
        adv = log_prob.new_zeros(bs)
    else:
        adv = reset_advantage.to(log_prob.device).float()
    token_adv = adv.unsqueeze(-1).expand_as(log_prob)

    # Clip-ratio config (same as vanilla PPO/GRPO)
    clip_ratio = config.clip_ratio
    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    assert clip_ratio_c > 1.0, f"clip_ratio_c must be > 1.0, got {clip_ratio_c}"

    # Importance-sampling ratio (clamped for numerical stability)
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, loss_mask)

    # Standard PPO clipped surrogate
    pg_losses1 = -token_adv * ratio
    pg_losses2 = -token_adv * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), loss_mask
    )

    # Dual-clip for negative advantages (vanilla PPO mechanism)
    pg_losses3 = -token_adv * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(token_adv < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=loss_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    correction_frac = 0.0
    if suffix_start_idx is not None:
        correction_frac = (suffix_start_idx > 0).float().mean().item()

    loss_tok_count = loss_mask.sum(dim=-1).clamp(min=1).float()
    metrics = {
        "actor/reset_advantage_mean": adv.mean().item(),
        "actor/reset_advantage_std": adv.std().item() if bs > 1 else 0.0,
        "actor/srpo_loss_tokens_mean": loss_tok_count.mean().item(),
        "actor/srpo_correction_frac": correction_frac,
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    }
    return pg_loss, metrics
