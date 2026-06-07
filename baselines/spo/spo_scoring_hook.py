"""SPO scoring hook: score MC completions and assemble paper-Eq.-2 V-vector.

The agent loop emits raw MC completions at each interior segment boundary;
it cannot score them because it lacks ground-truth access. This module:

    1. Scores MC completions with the math reward function.
    2. Computes V(s_0) per group as the mean outcome reward of the n sibling
       rollouts for each prompt (paper §3: N i.i.d. trajectories from s_0).
    3. Computes V(s_T) as the trajectory's own observed outcome reward.
    4. Assembles the full [V(t_0), V(t_1), ..., V(t_K)] vector per sequence.
    5. Monkey-patches verl's compute_advantage to pass the above plus
       probability data to the SPO advantage estimator.

Reference:
    SPO: Segment Policy Optimization (arXiv:2505.23564), §3–§4.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from verl.utils.reward_score.math_reward import compute_score as math_compute_score

logger = logging.getLogger(__name__)


def _segment_count_from_ids(seg_ids) -> int:
    """Infer K (total segments) from a segment-id sequence."""
    if isinstance(seg_ids, torch.Tensor):
        return int(seg_ids.max().item()) if seg_ids.numel() > 0 else 0
    if isinstance(seg_ids, (list, tuple)) and seg_ids:
        return max(seg_ids)
    return 0


def _seg_ids_to_tensor(seg_ids, response_length: int) -> torch.Tensor:
    if isinstance(seg_ids, torch.Tensor):
        t = seg_ids[:response_length].long()
    else:
        t = torch.tensor(list(seg_ids[:response_length]), dtype=torch.long)
    if t.numel() < response_length:
        pad = torch.zeros(response_length - t.numel(), dtype=torch.long)
        t = torch.cat([t, pad])
    return t


def score_mc_and_compute_values(
    mc_completions_batch: np.ndarray,
    mc_prefixes_batch: np.ndarray,
    mc_boundary_tokens_batch: Optional[np.ndarray],
    segment_token_ids_batch: np.ndarray,
    ground_truths: np.ndarray,
    response_mask: torch.Tensor,
    outcome_rewards: torch.Tensor,
    v0_per_seq: torch.Tensor,
) -> tuple[list[list[float]], torch.Tensor]:
    """Assemble full V-vectors [V(t_0), ..., V(t_K)] and seg-id tensor.

    Args:
        mc_completions_batch: object ndarray (bs,), each element is a list
            of list-of-strings: continuations at each interior boundary.
        mc_prefixes_batch: object ndarray (bs,), each element is a list of
            prefix strings (one per interior boundary).
        mc_boundary_tokens_batch: unused for scoring but retained in the API
            for debugging/inspection. May be None.
        segment_token_ids_batch: object ndarray (bs,), each element is a
            list of 1-indexed segment ids (length ≤ response_length).
        ground_truths: object ndarray (bs,) of ground-truth answer strings.
        response_mask: (bs, response_length) binary tensor.
        outcome_rewards: (bs,) tensor, R(τ) per trajectory. Used as V(t_K).
        v0_per_seq: (bs,) tensor, V(s_0) per sequence (group-mean of sibling
            outcome rewards).

    Returns:
        mc_values_batch: list of length bs; element s is a Python list
            [V(t_0), V(t_1), ..., V(t_K_s)] of length K_s + 1.
        segment_token_ids_tensor: (bs, response_length) long tensor.
    """
    bs = len(segment_token_ids_batch)
    response_length = response_mask.shape[1]
    mc_values_batch: list[list[float]] = []
    seg_ids_list: list[torch.Tensor] = []

    for i in range(bs):
        seg_ids_raw = segment_token_ids_batch[i]
        seg_tensor = _seg_ids_to_tensor(seg_ids_raw, response_length)
        seg_ids_list.append(seg_tensor)

        K = _segment_count_from_ids(seg_ids_raw)

        mc_completions = mc_completions_batch[i] if mc_completions_batch is not None else None
        mc_prefixes = mc_prefixes_batch[i] if mc_prefixes_batch is not None else None
        gt = ground_truths[i] if i < len(ground_truths) else ""

        v0 = float(v0_per_seq[i].item())
        vT = float(outcome_rewards[i].item())

        if K == 0:
            # No segments at all (empty response or missing partition) —
            # estimator will skip this sequence, but return a valid vector.
            mc_values_batch.append([v0, vT])
            continue

        # Interior MC values (K - 1 of them).
        interior: list[float] = []
        num_interior_expected = K - 1
        if (
            mc_completions is None
            or mc_prefixes is None
            or len(mc_completions) < num_interior_expected
        ):
            # MC data absent or incomplete. Fill missing with v0 so the
            # corresponding A_k ≈ 0 — the sequence effectively drops to
            # "outcome only" credit.
            if mc_completions is None or mc_prefixes is None:
                interior = [v0] * num_interior_expected
            else:
                # Use what we have, pad with v0.
                for j in range(num_interior_expected):
                    if j < len(mc_completions) and j < len(mc_prefixes):
                        interior.append(
                            _score_completions(mc_prefixes[j], mc_completions[j], gt)
                        )
                    else:
                        interior.append(v0)
        else:
            for j in range(num_interior_expected):
                interior.append(
                    _score_completions(mc_prefixes[j], mc_completions[j], gt)
                )

        mc_values_batch.append([v0, *interior, vT])

    return mc_values_batch, torch.stack(seg_ids_list)


def _score_completions(prefix: str, completions: list, gt: str) -> float:
    """Mean reward over MC completions given ground truth."""
    if not completions:
        return 0.0
    scores = []
    for c in completions:
        full = prefix + c
        scores.append(float(math_compute_score(full, gt)))
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Monkey-patch of verl.compute_advantage
# ---------------------------------------------------------------------------


def patch_compute_advantage():
    """Wrap verl's compute_advantage to route adv_estimator='spo' through SPO.

    Idempotent.
    """
    import verl.trainer.ppo.ray_trainer as ray_trainer_module
    from baselines.spo.spo_advantage import v0_from_group_outcomes

    original_compute_advantage = ray_trainer_module.compute_advantage
    if getattr(original_compute_advantage, "_spo_patched", False):
        return

    def patched_compute_advantage(data, adv_estimator, **kwargs):
        adv_name = (
            adv_estimator.value
            if hasattr(adv_estimator, "value")
            else str(adv_estimator)
        )

        if adv_name == "spo_tree":
            return _spo_tree_compute_advantage(data, **kwargs)

        if adv_name != "spo":
            return original_compute_advantage(data, adv_estimator, **kwargs)

        # ----- SPO path -----
        from verl.trainer.ppo import core_algos

        if "response_mask" not in data.batch:
            from verl.trainer.ppo.ray_trainer import compute_response_mask
            data.batch["response_mask"] = compute_response_mask(data)

        response_mask = data.batch["response_mask"]
        token_level_rewards = data.batch["token_level_rewards"]
        bs = response_mask.shape[0]

        # Outcome reward per trajectory (sum of token-level rewards over the mask).
        outcome_rewards = (token_level_rewards * response_mask).sum(dim=-1)

        # V(s_0) from group-mean outcome.
        uids = data.non_tensor_batch.get("uid", None)
        if uids is None:
            # Without group info, fall back to batch-mean as V(s_0).
            v0 = torch.full_like(outcome_rewards, outcome_rewards.mean().item())
            logger.warning(
                "SPO scoring: non_tensor_batch lacks 'uid'; using batch-mean as V(s_0)."
            )
        else:
            v0 = v0_from_group_outcomes(outcome_rewards, uids)

        has_mc = (
            "mc_completions" in data.non_tensor_batch
            and "mc_prefixes" in data.non_tensor_batch
            and ("segment_token_ids" in data.non_tensor_batch
                 or "thought_segment_ids" in data.non_tensor_batch)
        )

        if has_mc:
            seg_ids_batch = data.non_tensor_batch.get(
                "segment_token_ids",
                data.non_tensor_batch.get("thought_segment_ids"),
            )

            ground_truths = np.array([
                (item.get("ground_truth", "") if isinstance(item, dict) else "")
                for item in data.non_tensor_batch.get(
                    "reward_model", [{}] * bs
                )
            ])

            mc_values_batch, seg_ids_tensor = score_mc_and_compute_values(
                mc_completions_batch=data.non_tensor_batch["mc_completions"],
                mc_prefixes_batch=data.non_tensor_batch["mc_prefixes"],
                mc_boundary_tokens_batch=data.non_tensor_batch.get(
                    "mc_boundary_tokens", None
                ),
                segment_token_ids_batch=seg_ids_batch,
                ground_truths=ground_truths,
                response_mask=response_mask,
                outcome_rewards=outcome_rewards,
                v0_per_seq=v0,
            )
            logger.info(
                f"SPO scoring: bs={len(mc_values_batch)}, "
                f"avg K={sum(len(v)-1 for v in mc_values_batch) / max(len(mc_values_batch),1):.1f}"
            )
        else:
            mc_values_batch = None
            seg_ids_tensor = None
            logger.info("SPO scoring: MC fields absent → REINFORCE++ fallback.")

        # Probability data for the mask (Eq. 3).
        old_log_prob = data.batch.get("old_log_prob", None)
        token_probs = torch.exp(old_log_prob) if old_log_prob is not None else None
        if token_probs is None:
            logger.warning(
                "SPO scoring: old_log_prob missing — probability mask disabled "
                "and Z-normalization will be skipped."
            )

        adv_estimator_fn = core_algos.get_adv_estimator_fn("spo")
        adv_kwargs = {
            "token_level_rewards": token_level_rewards,
            "response_mask": response_mask,
            "config": kwargs.get("config", None),
            "mc_values": mc_values_batch,
            "segment_token_ids": seg_ids_tensor,
            "token_probs": token_probs,
        }
        if uids is not None:
            adv_kwargs["index"] = uids

        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    patched_compute_advantage._spo_patched = True
    ray_trainer_module.compute_advantage = patched_compute_advantage
    logger.info("SPO: patched verl.trainer.ppo.ray_trainer.compute_advantage")


def _spo_tree_compute_advantage(data, **kwargs):
    """Route SPO-tree advantage: gather tree metadata from non_tensor_batch.

    Trajectories that share a ``uid`` form a sibling group in the tree.
    Each trajectory contributes its root-to-leaf path, outcome reward
    (derived inside the estimator from ``token_level_rewards``), and
    per-token segment ids (level 1 → L along the path).
    """
    from verl.trainer.ppo import core_algos
    import torch as _torch

    if "response_mask" not in data.batch:
        from verl.trainer.ppo.ray_trainer import compute_response_mask
        data.batch["response_mask"] = compute_response_mask(data)

    response_mask = data.batch["response_mask"]
    token_level_rewards = data.batch["token_level_rewards"]

    tree_paths = data.non_tensor_batch.get("tree_path", None)
    seg_ids_raw = data.non_tensor_batch.get(
        "segment_token_ids",
        data.non_tensor_batch.get("thought_segment_ids"),
    )
    uids = data.non_tensor_batch.get("uid", None)

    if tree_paths is None or seg_ids_raw is None or uids is None:
        logger.info(
            "SPO-tree scoring: missing tree_path/segment_token_ids/uid → "
            "REINFORCE++ fallback."
        )
        adv_estimator_fn = core_algos.get_adv_estimator_fn("spo_tree")
        advantages, returns = adv_estimator_fn(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            config=kwargs.get("config", None),
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    response_length = response_mask.shape[1]
    bs = response_mask.shape[0]

    # Materialize segment_token_ids as a (bs, response_length) tensor.
    seg_tensor = _torch.zeros((bs, response_length), dtype=_torch.long)
    for i in range(bs):
        raw = seg_ids_raw[i]
        if isinstance(raw, _torch.Tensor):
            t = raw[:response_length].long()
        else:
            t = _torch.tensor(list(raw[:response_length]), dtype=_torch.long)
        if t.numel() < response_length:
            pad = _torch.zeros(response_length - t.numel(), dtype=_torch.long)
            t = _torch.cat([t, pad])
        seg_tensor[i] = t

    # Probabilities for the mask.
    old_log_prob = data.batch.get("old_log_prob", None)
    token_probs = _torch.exp(old_log_prob) if old_log_prob is not None else None

    adv_estimator_fn = core_algos.get_adv_estimator_fn("spo_tree")
    advantages, returns = adv_estimator_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        config=kwargs.get("config", None),
        tree_paths=[tuple(int(x) for x in p) for p in tree_paths],
        tree_uids=np.asarray(uids),
        segment_token_ids=seg_tensor,
        token_probs=token_probs,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data
