"""SPO segment-level advantage estimator (paper arXiv:2505.23564).

Implements Eq. 2 and the probability-mask + Z-normalization of Eq. 3.

Per-sequence notation:
    t_0 = 0, t_K = T (response length), and t_1 < ... < t_{K-1} are interior
    segment boundaries from the cutpoint partition in the agent loop.

    V̂(s_{t_0})  := group-mean outcome reward (the group of n siblings rolled
                   out from the same prompt is already N i.i.d. trajectories
                   from s_0; using their mean avoids redundant MC rollouts).
    V̂(s_{t_k})  := mean MC-rollout reward at boundary k   (1 ≤ k ≤ K-1)
    V̂(s_{t_K}) := R(τ), the observed terminal reward of this trajectory.

    A_k := V̂(s_{t_k}) - V̂(s_{t_{k-1}}),  1 ≤ k ≤ K.  (Eq. 2)

Token-level advantages assign A_k to every token position in segment k.

Probability mask (Eq. 3):
    M_t := 𝕀[π_θ_old(y_t | s_t) < ρ]
Only masked-in (low-probability, "decision") tokens contribute gradient.

Paper Eq. 3 normalizes the per-trajectory PPO-clipped objective by
Z_s := Σ_t M_t (count of decision tokens in trajectory s). Verl aggregates
with token-mean (1/T_valid) Σ loss·mask. To reproduce Eq. 3 exactly we
pre-scale per-token advantages by (T_valid / (B · Z_s)):

    L_verl = (1/T_valid) Σ_{s,t} mask · ratio · A'_{s,t}
           = (1/B) Σ_s (1/Z_s) Σ_t M_t · ratio · A_{s,t}      ✓ matches Eq. 3

Scaling advantages is loss-equivalent to scaling the PPO-clipped objective
because the clip is min(r·A, clip(r)·A) and min commutes with multiplication
by a non-negative scalar.

Reference:
    SPO: Segment Policy Optimization (arXiv:2505.23564), §3–§4
    https://github.com/AIFrameResearch/SPO
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core algorithmic functions (no VERL dependency)
# ---------------------------------------------------------------------------


def compute_segment_advantages(
    mc_values: list[float],
    segment_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    response_length: int,
) -> torch.Tensor:
    """Assign A_k = V(t_k) - V(t_{k-1}) to every token in segment k.

    Args:
        mc_values: [V(t_0), V(t_1), ..., V(t_K)] — length K+1.
            V(t_0) is the group-mean baseline; V(t_K) is the trajectory
            terminal reward; the interior values are MC estimates.
        segment_token_ids: (response_length,) 1-indexed segment ids, 0 = pad.
        response_mask: (response_length,) binary.
        response_length: int.

    Returns:
        (response_length,) per-token advantages tensor.
    """
    advantages = torch.zeros(response_length, dtype=torch.float32)
    K = len(mc_values) - 1
    if K <= 0:
        return advantages

    for t in range(response_length):
        seg = int(segment_token_ids[t].item())
        if seg == 0 or response_mask[t] <= 0:
            continue
        k = min(seg, K)  # clamp against off-by-one
        advantages[t] = mc_values[k] - mc_values[k - 1]
    return advantages


def apply_probability_mask(
    advantages: torch.Tensor,
    token_probs: torch.Tensor,
    threshold: float = 0.9,
) -> torch.Tensor:
    """Zero advantages for high-probability (non-decision) tokens.

    M_t = 𝕀[π_θ_old(y_t|s_t) < threshold]. Per paper §4: ``These tokens
    primarily contribute to the segment's advantage'' — only low-probability
    ("decision") tokens should receive gradient.

    Args:
        advantages: (bs, response_length).
        token_probs: (bs, response_length), π_θ_old.
        threshold: ρ, default 0.9 (paper).

    Returns:
        Masked advantages, same shape.
    """
    mask = (token_probs < threshold).float()
    return advantages * mask


def probability_mask(
    token_probs: torch.Tensor,
    threshold: float = 0.9,
) -> torch.Tensor:
    """Returns the probability mask M_t directly (for Z computation)."""
    return (token_probs < threshold).float()


def z_scale_advantages(
    advantages: torch.Tensor,
    prob_mask: torch.Tensor,
    response_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pre-scale advantages so verl's token-mean agg yields paper Eq. 3.

    For each sequence s with Z_s = Σ_t M_t and the batch's T_valid = Σ mask,
    scale A_{s,t} → A_{s,t} · T_valid / (B · Z_s). Fully-masked sequences
    (Z_s = 0) get scale 0 so they contribute nothing.

    Args:
        advantages: (bs, response_length). Assumed already mask-zeroed at
            M_t = 0 positions.
        prob_mask: (bs, response_length) binary M_t.
        response_mask: (bs, response_length) binary.
        eps: numerical safety for the 1/Z_s divide.

    Returns:
        Scaled advantages, same shape.
    """
    bs = advantages.shape[0]
    Z_s = prob_mask.sum(dim=-1)                          # (bs,)
    T_valid = response_mask.sum()                        # scalar
    alive = (Z_s > 0).float()                            # (bs,)
    scale = alive * T_valid / (bs * (Z_s + eps))         # (bs,)
    return advantages * scale.unsqueeze(-1)


def v0_from_group_outcomes(
    outcome_rewards: torch.Tensor,
    uids: np.ndarray,
) -> torch.Tensor:
    """V̂(s_0) = mean outcome reward within each prompt group.

    Paper §3: V(s_{t_0}) is estimated from N independent trajectories from
    s_0. The n sibling rollouts per prompt ARE those trajectories, so the
    group-mean outcome is exactly the paper's MC estimate with N = n.

    Args:
        outcome_rewards: (bs,) scalar per trajectory.
        uids: (bs,) object ndarray — same uid == same prompt group.

    Returns:
        (bs,) tensor with the group-mean assigned to every member.
    """
    if len(uids) == 0:
        return outcome_rewards.clone()
    device = outcome_rewards.device
    bs = outcome_rewards.shape[0]
    v0 = torch.zeros(bs, device=device, dtype=outcome_rewards.dtype)

    # Group indices by uid.
    uid_to_idx: dict[object, list[int]] = {}
    for i, u in enumerate(uids):
        uid_to_idx.setdefault(u, []).append(i)

    for members in uid_to_idx.values():
        mean = outcome_rewards[members].mean()
        for i in members:
            v0[i] = mean
    return v0


# ---------------------------------------------------------------------------
# REINFORCE++ fallback (kept for when MC data is missing)
# ---------------------------------------------------------------------------


def _reinforce_fallback(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spread the sequence outcome reward uniformly over valid tokens.

    Used when MC data is unavailable (e.g., validation, degenerate generation
    with no cutpoints).
    """
    outcome = (token_level_rewards * response_mask).sum(dim=-1, keepdim=True)
    advantages = outcome.expand_as(response_mask) * response_mask
    returns = advantages.clone()
    return advantages, returns


# ---------------------------------------------------------------------------
# VERL registration
# ---------------------------------------------------------------------------

try:
    from verl.trainer.ppo.core_algos import register_adv_est

    @register_adv_est("spo")
    def compute_spo_outcome_advantage(
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        config=None,
        index=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SPO segment-level advantage estimator registered with VERL.

        The scoring hook (``spo_scoring_hook.patch_compute_advantage``)
        computes `mc_values` (with V(s_0) already filled from group means and
        V(s_T) from trajectory outcome) and passes them here alongside
        `segment_token_ids` and `token_probs`.

        Kwargs consumed:
            mc_values: list[list[float]] of length bs; each inner list is
                [V(t_0), V(t_1), ..., V(t_K_s)] for sequence s.
            segment_token_ids: (bs, response_length) 1-indexed.
            token_probs: (bs, response_length) π_θ_old probabilities.
            prob_mask_threshold: float, default 0.9.
        """
        mc_values_batch = kwargs.get("mc_values", None)
        segment_token_ids = kwargs.get("segment_token_ids", None)
        if segment_token_ids is None:
            # Backward-compat with the old key name.
            segment_token_ids = kwargs.get("thought_segment_ids", None)

        if mc_values_batch is None or segment_token_ids is None:
            logger.info("SPO advantage: MC data missing, falling back to REINFORCE++.")
            return _reinforce_fallback(token_level_rewards, response_mask)

        bs, response_length = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)

        # A_k = V(t_k) - V(t_{k-1})
        for i in range(bs):
            seq_mc = mc_values_batch[i]
            seq_seg = segment_token_ids[i]
            seq_mask = response_mask[i]
            if seq_mc is None or len(seq_mc) < 2:
                # Degenerate (no interior boundaries AND no V0/V_T pair); skip.
                continue
            advantages[i] = compute_segment_advantages(
                mc_values=seq_mc,
                segment_token_ids=seq_seg,
                response_mask=seq_mask,
                response_length=response_length,
            )
            # Returns: keep the outcome reward for logging consistency.
            returns[i] = (token_level_rewards[i] * seq_mask).sum() * seq_mask

        # Probability mask (Eq. 3)
        token_probs = kwargs.get("token_probs", None)
        threshold = kwargs.get("prob_mask_threshold", 0.9)
        if token_probs is not None:
            p_mask = probability_mask(token_probs, threshold) * response_mask
            advantages = advantages * p_mask
            # Z-normalization (Eq. 3): L_verl_token-mean = (1/B) Σ (1/Z_s) …
            advantages = z_scale_advantages(advantages, p_mask, response_mask)
        else:
            logger.warning(
                "SPO advantage: token_probs not provided — probability mask "
                "disabled and Z-normalization skipped. Loss scale will differ "
                "from paper Eq. 3."
            )

        return advantages, returns

except ImportError:
    logger.debug(
        "verl not importable; 'spo' advantage estimator not registered. "
        "Core functions remain usable standalone."
    )


# ---------------------------------------------------------------------------
# SPO-tree (paper §5)
# ---------------------------------------------------------------------------


def build_tree_values(
    paths: list[tuple[int, ...]],
    outcome_rewards: list[float],
) -> dict[tuple[int, ...], float]:
    """Bottom-up V̂ for every node in the tree defined by `paths`.

    Leaves get V̂(leaf) = R(trajectory). Internal nodes get the mean of their
    children's V̂ (paper §5.2). The root is the empty tuple ().

    Args:
        paths: list of root-to-leaf paths, one per trajectory. All paths
            must have the same length L.
        outcome_rewards: per-trajectory terminal reward, aligned with paths.

    Returns:
        V: dict mapping node tuple → V̂.
    """
    if not paths:
        return {(): 0.0}
    L = len(paths[0])
    assert all(len(p) == L for p in paths), "all paths must share depth"

    # Collect node membership: node_key -> list of trajectory indices in its subtree.
    subtree_indices: dict[tuple[int, ...], list[int]] = {}
    for i, p in enumerate(paths):
        for depth in range(L + 1):
            key = tuple(p[:depth])
            subtree_indices.setdefault(key, []).append(i)

    # V̂(leaf) = R(leaf); V̂(internal) = mean of descendant leaf rewards
    # (equivalent to bottom-up child averaging when the tree is balanced,
    # which it is by construction).
    V: dict[tuple[int, ...], float] = {}
    for key, idxs in subtree_indices.items():
        V[key] = sum(outcome_rewards[i] for i in idxs) / len(idxs)
    return V


def sibling_relative_advantages(
    paths: list[tuple[int, ...]],
    V: dict[tuple[int, ...], float],
    normalize_by_std: bool = False,
    eps: float = 1e-8,
) -> list[list[float]]:
    """Per-trajectory, per-depth sibling-relative advantage Â(n) (paper §5.2).

    For each trajectory i and each depth d ∈ {1, ..., L}:
        node      = paths[i][:d]
        siblings  = {(paths[i][:d-1] + (j,)) for j in range(B_{d-1})}  -- from paths
        Â_i[d]   = V(node) - mean_{sib} V(sib)
                  [optionally divided by std_{sib} V(sib)]

    Args:
        paths: list of L-tuples, one per trajectory.
        V: tree values dict from build_tree_values.
        normalize_by_std: if True, also divide by sibling std.

    Returns:
        advs_per_traj: list of length N; each inner list has length L.
    """
    if not paths:
        return []
    L = len(paths[0])

    # Group siblings by their parent prefix.
    sibling_groups: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
    for p in paths:
        for d in range(1, L + 1):
            parent = tuple(p[: d - 1])
            node = tuple(p[:d])
            sibling_groups.setdefault(parent, set()).add(node)

    # Compute sibling stats per parent.
    sibling_stats: dict[tuple[int, ...], tuple[float, float]] = {}
    for parent, kids in sibling_groups.items():
        vals = [V[k] for k in kids]
        mean = sum(vals) / len(vals)
        if normalize_by_std and len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(var)
        else:
            std = 1.0
        sibling_stats[parent] = (mean, std)

    advs_per_traj: list[list[float]] = []
    for p in paths:
        row: list[float] = []
        for d in range(1, L + 1):
            parent = tuple(p[: d - 1])
            node = tuple(p[:d])
            mean, std = sibling_stats[parent]
            a = V[node] - mean
            if normalize_by_std:
                a = a / (std + eps)
            row.append(a)
        advs_per_traj.append(row)
    return advs_per_traj


def tree_token_advantages(
    segment_advantages: list[float],
    segment_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    response_length: int,
) -> torch.Tensor:
    """Assign Â(n) to every token in seg(n) (paper §5.3).

    Args:
        segment_advantages: [Â(level 1 node), ..., Â(level L node)] — one
            advantage per depth on this trajectory's path.
        segment_token_ids: (response_length,) 1-indexed segment id (L for
            the leaf segment which extends to end-of-trajectory).
        response_mask: (response_length,) binary.
        response_length: int.

    Returns:
        (response_length,) per-token advantages tensor.
    """
    out = torch.zeros(response_length, dtype=torch.float32)
    L = len(segment_advantages)
    if L == 0:
        return out
    for t in range(response_length):
        seg = int(segment_token_ids[t].item())
        if seg <= 0 or response_mask[t] <= 0:
            continue
        k = min(seg, L) - 1
        out[t] = segment_advantages[k]
    return out


try:
    from verl.trainer.ppo.core_algos import register_adv_est as _register_adv_est_tree

    @_register_adv_est_tree("spo_tree")
    def compute_spo_tree_advantage(
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        config=None,
        index=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SPO-tree advantage estimator (paper Eq. 4).

        Consumes (from kwargs):
            tree_paths: list[tuple[int]] of length bs, root-to-leaf paths.
            tree_uids: list/array of length bs, prompt uid per trajectory
                (siblings share a uid).
            segment_token_ids: (bs, response_length) 1-indexed segment ids
                (tokens in segment k get Â of the node at depth k on this
                trajectory's path).
            token_probs: (bs, response_length) π_θ_old.
            prob_mask_threshold: ρ, default 0.9.
            tree_normalize_by_std: bool, default False (paper uses False).

        The outcome reward for each trajectory is derived from
        ``token_level_rewards``; siblings (same uid) form the tree.
        """
        tree_paths = kwargs.get("tree_paths", None)
        tree_uids = kwargs.get("tree_uids", None)
        segment_token_ids = kwargs.get("segment_token_ids", None)
        token_probs = kwargs.get("token_probs", None)
        threshold = kwargs.get("prob_mask_threshold", 0.9)
        normalize_by_std = bool(kwargs.get("tree_normalize_by_std", False))

        if tree_paths is None or tree_uids is None or segment_token_ids is None:
            logger.info("SPO-tree advantage: tree metadata missing → REINFORCE++ fallback.")
            return _reinforce_fallback(token_level_rewards, response_mask)

        bs, response_length = token_level_rewards.shape
        outcomes = (token_level_rewards * response_mask).sum(dim=-1).tolist()

        # Group trajectories by uid.
        groups: dict[object, list[int]] = {}
        for i, u in enumerate(tree_uids):
            groups.setdefault(u, []).append(i)

        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)

        for uid, members in groups.items():
            # Build tree from this uid's siblings.
            paths = [tuple(tree_paths[i]) for i in members]
            rewards = [float(outcomes[i]) for i in members]

            if len({len(p) for p in paths}) != 1:
                # Malformed tree — skip group.
                logger.warning(
                    f"SPO-tree: uid={uid} has mixed path depths; skipping."
                )
                continue

            V = build_tree_values(paths, rewards)
            adv_rows = sibling_relative_advantages(
                paths, V, normalize_by_std=normalize_by_std
            )

            for i_local, global_idx in enumerate(members):
                seq_adv = tree_token_advantages(
                    segment_advantages=adv_rows[i_local],
                    segment_token_ids=segment_token_ids[global_idx],
                    response_mask=response_mask[global_idx],
                    response_length=response_length,
                )
                advantages[global_idx] = seq_adv
                returns[global_idx] = (
                    token_level_rewards[global_idx] * response_mask[global_idx]
                ).sum() * response_mask[global_idx]

        if token_probs is not None:
            p_mask = probability_mask(token_probs, threshold) * response_mask
            # Paper Eq. 4: sum over nodes n with Â(n) ≠ 0. The token already
            # carries 0 when Â(n)=0, so product with mask preserves that.
            advantages = advantages * p_mask
            advantages = z_scale_advantages(advantages, p_mask, response_mask)
        else:
            logger.warning(
                "SPO-tree advantage: token_probs not provided — probability "
                "mask + Z-normalization skipped."
            )

        return advantages, returns

except ImportError:
    logger.debug("verl not importable; 'spo_tree' advantage estimator not registered.")
