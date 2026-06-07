"""SRPO policy loss: two-group GRPO with pre-computed advantages + suffix-only gradients.

Advantages are pre-computed in SRPOAgentLoop._fill_rollout_buffer (within-group
normalization, Group 1 and Group 2 separately) and passed as reset_advantage.
This avoids the biased baseline problem of SRPO while retaining GRPO's
credit-assignment benefit within each group.

Group 1 (slots 0-3): full-response gradient (suffix_start_idx=0)
Group 2 corrections (slots 4-7): suffix-only gradient (suffix_start_idx > 0)
"""

import os
from pathlib import Path
from typing import Any

import torch

from verl.trainer.ppo.core_algos import register_policy_loss

# Module-level counter for env-gated per-token tensor dump. ZERO effect on
# training math; only used inside the dump block which itself is gated on
# SRPO_PER_TOKEN_DUMP_DIR being set. Per-process state (each FSDP rank gets its own).
_DUMP_CALL_IDX = 0


@register_policy_loss("srpo")
def compute_policy_loss_srpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config=None,
    rollout_is_weights=None,
    thought_segment_ids=None,
    reset_advantage: torch.Tensor | None = None,
    suffix_start_idx: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """SRPO loss: pre-computed GRPO advantages applied over suffix-masked tokens.

    Args:
        log_prob: (bs, response_length) — current policy log-probs
        response_mask: (bs, response_length) — 1 for valid response tokens
        reset_advantage: (bs,) — per-trajectory GRPO advantages, normalized within
            Group 1 and Group 2 independently (pre-computed in agent loop)
        suffix_start_idx: (bs,) — per-sequence index into response where the suffix
            begins; tokens before this are masked out (the kept ICS prefix)

    Returns:
        Scalar loss and metrics dict.
    """
    bs = log_prob.shape[0]

    # Build suffix mask: response tokens at or after suffix_start_idx
    suffix_mask = response_mask.clone()
    if suffix_start_idx is not None:
        for i in range(bs):
            start = int(suffix_start_idx[i].item())
            if start > 0:
                suffix_mask[i, :start] = 0

    # Per-sequence token-mean log_prob over suffix
    suffix_lp_sum = (log_prob * suffix_mask).sum(dim=-1)          # (bs,)
    suffix_tok_count = suffix_mask.sum(dim=-1).clamp(min=1)       # (bs,)
    suffix_lp_mean = suffix_lp_sum / suffix_tok_count             # (bs,)

    # Use pre-computed per-trajectory advantage (default 1.0 if absent — pure NLL)
    if reset_advantage is None:
        adv = log_prob.new_ones(bs)
    else:
        adv = reset_advantage.to(log_prob.device).float()

    loss = -(adv * suffix_lp_mean).mean()

    # Fraction of trajectories that are Group-2 corrections (suffix_start_idx > 0)
    if suffix_start_idx is not None:
        is_correction = (suffix_start_idx > 0).float()
        correction_frac = is_correction.mean().item()
    else:
        correction_frac = 0.0

    metrics = {
        "actor/reset_advantage_mean": adv.mean().item(),
        "actor/reset_advantage_std": adv.std().item() if bs > 1 else 0.0,
        "actor/srpo_suffix_tokens_mean": suffix_tok_count.float().mean().item(),
        "actor/srpo_correction_frac": correction_frac,
    }

    # === Env-gated per-token tensor dump (strict no-op unless dir is set). ===
    # Production runs do not set SRPO_PER_TOKEN_DUMP_DIR, so this short-circuits
    # at the first os.environ.get call. Wrapped in try/except so a dump failure
    # cannot affect the returned loss/metrics. All tensors are detached.
    _dump_dir = os.environ.get("SRPO_PER_TOKEN_DUMP_DIR", "").strip()
    if _dump_dir:
        try:
            global _DUMP_CALL_IDX
            _max = int(os.environ.get("SRPO_PER_TOKEN_DUMP_MAX", "200") or "200")
            if _DUMP_CALL_IDX < _max:
                _p = Path(_dump_dir)
                _p.mkdir(parents=True, exist_ok=True)
                _fname = _p / f"loss_pid{os.getpid()}_call{_DUMP_CALL_IDX:04d}.pt"
                # Extra inputs needed to re-run forward+backward for actual
                # per-token gradient computation. All optional; if the caller
                # didn't pass them in kwargs the corresponding fields are None.
                _input_ids       = kwargs.get("input_ids")
                _responses       = kwargs.get("responses")
                _attention_mask  = kwargs.get("attention_mask")
                _position_ids    = kwargs.get("position_ids")
                _payload = {
                    "call_idx":            _DUMP_CALL_IDX,
                    "old_log_prob":        old_log_prob.detach().float().cpu(),
                    "log_prob":            log_prob.detach().float().cpu(),
                    "response_mask":       response_mask.detach().cpu(),
                    "suffix_mask":         suffix_mask.detach().cpu(),
                    "suffix_start_idx":    None if suffix_start_idx is None else suffix_start_idx.detach().cpu(),
                    "reset_advantage":      adv.detach().cpu(),
                    "thought_segment_ids": None if thought_segment_ids is None else thought_segment_ids.detach().cpu(),
                    "input_ids":           None if _input_ids is None      else _input_ids.detach().cpu(),
                    "responses":           None if _responses is None      else _responses.detach().cpu(),
                    "attention_mask":      None if _attention_mask is None else _attention_mask.detach().cpu(),
                    "position_ids":        None if _position_ids is None   else _position_ids.detach().cpu(),
                }
                torch.save(_payload, _fname)
                _DUMP_CALL_IDX += 1
        except Exception:
            pass

    return loss, metrics
