"""Tests for SPO segment-level advantage computation.

Covers Eq. 2 token assignment, the probability mask (Eq. 3), Z-scaling for
the per-trajectory (1/Z_s) normalization, V(s_0) from group means, and the
REINFORCE++ fallback.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from baselines.spo.spo_advantage import (
    compute_segment_advantages,
    apply_probability_mask,
    probability_mask,
    z_scale_advantages,
    v0_from_group_outcomes,
    _reinforce_fallback,
)


# ---------------------------------------------------------------------------
# compute_segment_advantages — Eq. 2
# ---------------------------------------------------------------------------


class TestComputeSegmentAdvantages:
    def test_three_segments(self):
        """A_k = V(t_k) - V(t_{k-1}) assigned to every token in segment k.

        mc_values vector is [V(t_0), V(t_1), V(t_2), V(t_3)] (K = 3 segments).
        """
        mc_values = [0.1, 0.2, 0.5, 0.8]   # V0, V1, V2, V3
        seg_ids = torch.tensor([1, 1, 2, 2, 2, 3, 3, 3])  # K=3 segments
        mask = torch.ones(8)

        adv = compute_segment_advantages(mc_values, seg_ids, mask, 8)

        # A_1 = 0.2 - 0.1 = 0.1   → tokens 0,1
        # A_2 = 0.5 - 0.2 = 0.3   → tokens 2,3,4
        # A_3 = 0.8 - 0.5 = 0.3   → tokens 5,6,7
        for t in (0, 1):
            assert adv[t].item() == pytest.approx(0.1, abs=1e-6)
        for t in (2, 3, 4):
            assert adv[t].item() == pytest.approx(0.3, abs=1e-6)
        for t in (5, 6, 7):
            assert adv[t].item() == pytest.approx(0.3, abs=1e-6)

    def test_padding_tokens_zero(self):
        mc_values = [0.0, 0.5, 1.0]         # V0=0, V1=0.5, V2=1.0 → K=2
        seg_ids = torch.tensor([1, 1, 2, 2, 0, 0])
        mask = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.float32)
        adv = compute_segment_advantages(mc_values, seg_ids, mask, 6)
        assert adv[4].item() == 0.0
        assert adv[5].item() == 0.0
        assert adv[0].item() == pytest.approx(0.5)  # A_1 = 0.5 - 0 = 0.5
        assert adv[2].item() == pytest.approx(0.5)  # A_2 = 1.0 - 0.5 = 0.5

    def test_single_segment_uses_v0_vT(self):
        """K=1: mc_values=[V0, VT], A_1 = VT - V0."""
        mc_values = [0.2, 0.9]
        seg_ids = torch.tensor([1, 1, 1, 1])
        mask = torch.ones(4)
        adv = compute_segment_advantages(mc_values, seg_ids, mask, 4)
        for t in range(4):
            assert adv[t].item() == pytest.approx(0.7, abs=1e-6)

    def test_empty_mc_values(self):
        adv = compute_segment_advantages([], torch.zeros(5, dtype=torch.long),
                                         torch.zeros(5), 5)
        assert (adv == 0).all()

    def test_negative_advantage(self):
        """Values declining across segments yields negative advantages."""
        mc_values = [0.9, 0.5, 0.1]  # K=2
        seg_ids = torch.tensor([1, 1, 2, 2])
        mask = torch.ones(4)
        adv = compute_segment_advantages(mc_values, seg_ids, mask, 4)
        assert adv[0].item() == pytest.approx(-0.4, abs=1e-6)   # 0.5-0.9
        assert adv[2].item() == pytest.approx(-0.4, abs=1e-6)   # 0.1-0.5


# ---------------------------------------------------------------------------
# probability_mask / apply_probability_mask  — Eq. 3
# ---------------------------------------------------------------------------


class TestProbabilityMask:
    def test_strict_inequality(self):
        """M_t = 𝕀[π < ρ] (strict, per paper)."""
        probs = torch.tensor([[0.89, 0.90, 0.91]])
        m = probability_mask(probs, threshold=0.9)
        assert m[0, 0].item() == 1.0
        assert m[0, 1].item() == 0.0   # equal → not a cutpoint
        assert m[0, 2].item() == 0.0

    def test_apply_zeros_high_prob_tokens(self):
        advantages = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        probs = torch.tensor([[0.5, 0.95, 0.1, 0.92]])
        masked = apply_probability_mask(advantages, probs, threshold=0.9)
        assert masked[0, 0].item() == 1.0   # π=0.5 < 0.9 → kept
        assert masked[0, 1].item() == 0.0   # π=0.95 > 0.9 → zeroed
        assert masked[0, 2].item() == 1.0
        assert masked[0, 3].item() == 0.0

    def test_all_high_prob_fully_masked(self):
        advantages = torch.ones(1, 3)
        probs = torch.full((1, 3), 0.99)
        masked = apply_probability_mask(advantages, probs, threshold=0.9)
        assert (masked == 0).all()


# ---------------------------------------------------------------------------
# z_scale_advantages — per-trajectory (1/Z_s) pre-scaling
# ---------------------------------------------------------------------------


class TestZScale:
    def test_reproduces_paper_eq3(self):
        """After scaling, verl's token-mean over a batch equals
        (1/B) Σ_s (1/Z_s) Σ_t M_t · A_t (the paper's (1/Z)-normalized loss)."""
        bs, T = 2, 6
        advantages = torch.tensor([
            [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ])
        p_mask = torch.tensor([
            [1, 0, 1, 0, 1, 0],     # Z_0 = 3
            [1, 1, 0, 0, 0, 0],     # Z_1 = 2
        ], dtype=torch.float32)
        response_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
        ], dtype=torch.float32)

        # Reference (paper): per-sequence weighted sums.
        masked_adv = advantages * p_mask
        per_seq_sum = (masked_adv * response_mask).sum(dim=-1)   # Σ_t M_t·A_t
        Z = p_mask.sum(dim=-1)
        paper_loss = (per_seq_sum / Z).mean().item()             # (1/B) Σ (1/Z_s)·sum

        # Simulated verl token-mean after scaling.
        scaled = z_scale_advantages(masked_adv, p_mask, response_mask)
        # verl: (1/T_valid) Σ mask·scaled
        T_valid = response_mask.sum()
        verl_loss = ((scaled * response_mask).sum() / T_valid).item()

        assert verl_loss == pytest.approx(paper_loss, abs=1e-6)

    def test_zero_Z_sequence_contributes_nothing(self):
        bs, T = 2, 4
        advantages = torch.ones(bs, T)
        p_mask = torch.tensor([
            [0, 0, 0, 0],   # Z_0 = 0 → contributes 0
            [1, 1, 1, 1],   # Z_1 = 4
        ], dtype=torch.float32)
        rmask = torch.ones(bs, T)
        advantages = advantages * p_mask  # zero out first sequence
        scaled = z_scale_advantages(advantages, p_mask, rmask)
        assert (scaled[0] == 0).all()
        # Second sequence: scale = T_valid / (B · Z_1) = 8 / (2·4) = 1.
        assert torch.allclose(scaled[1], advantages[1])


# ---------------------------------------------------------------------------
# v0_from_group_outcomes
# ---------------------------------------------------------------------------


class TestV0FromGroupOutcomes:
    def test_group_mean_assigned_to_members(self):
        outcomes = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        uids = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)
        v0 = v0_from_group_outcomes(outcomes, uids)
        # Group 'a' mean = 2/3; group 'b' mean = 2/3.
        for i in range(3):
            assert v0[i].item() == pytest.approx(2 / 3, abs=1e-6)
        for i in range(3, 6):
            assert v0[i].item() == pytest.approx(2 / 3, abs=1e-6)

    def test_distinct_groups(self):
        outcomes = torch.tensor([1.0, 1.0, 0.0, 0.0])
        uids = np.array(["x", "x", "y", "y"], dtype=object)
        v0 = v0_from_group_outcomes(outcomes, uids)
        assert v0[0].item() == pytest.approx(1.0)
        assert v0[1].item() == pytest.approx(1.0)
        assert v0[2].item() == pytest.approx(0.0)
        assert v0[3].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# REINFORCE++ fallback
# ---------------------------------------------------------------------------


class TestReinforceFallback:
    def test_broadcasts_outcome_reward(self):
        bs, T = 2, 5
        token_level_rewards = torch.zeros(bs, T)
        token_level_rewards[0, 3] = 0.7
        token_level_rewards[1, 4] = 0.2
        mask = torch.tensor([
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ], dtype=torch.float32)

        advantages, returns = _reinforce_fallback(token_level_rewards, mask)

        # Outcome broadcast over valid positions; zero elsewhere.
        assert advantages[0, 0].item() == pytest.approx(0.7)
        assert advantages[0, 3].item() == pytest.approx(0.7)
        assert advantages[0, 4].item() == 0.0
        assert advantages[1, 4].item() == pytest.approx(0.2)
        assert torch.allclose(returns, advantages)
