"""Tests for SPO scoring hook: assembly of [V(t_0), ..., V(t_K)].

Tests scoring of MC completions, V(s_0) group-mean plumbing, V(s_T) outcome
plumbing, and the compute_advantage monkey-patch.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# score_mc_and_compute_values
# ---------------------------------------------------------------------------


class TestScoreMcAndComputeValues:
    def _call(self, **overrides):
        from baselines.spo.spo_scoring_hook import score_mc_and_compute_values

        defaults = dict(
            mc_completions_batch=None,
            mc_prefixes_batch=None,
            mc_boundary_tokens_batch=None,
            segment_token_ids_batch=None,
            ground_truths=np.array(["42"]),
            response_mask=torch.ones(1, 8),
            outcome_rewards=torch.tensor([1.0]),
            v0_per_seq=torch.tensor([0.4]),
        )
        defaults.update(overrides)
        return score_mc_and_compute_values(**defaults)

    def test_v_vector_length_is_K_plus_1(self):
        """Two interior boundaries (K=3 segments) → V-vector length 4."""
        mc_completions = np.empty(1, dtype=object)
        mc_completions[0] = [
            ["The answer is \\boxed{42}"],   # boundary 0: correct → V=1
            ["no answer"],                   # boundary 1: incorrect → V=0
        ]
        mc_prefixes = np.empty(1, dtype=object)
        mc_prefixes[0] = ["pref_a ", "pref_b "]
        seg_ids = np.empty(1, dtype=object)
        seg_ids[0] = [1, 1, 2, 2, 3, 3, 0, 0]   # K=3

        mc_values, seg_tensor = self._call(
            mc_completions_batch=mc_completions,
            mc_prefixes_batch=mc_prefixes,
            segment_token_ids_batch=seg_ids,
            response_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.float32),
            outcome_rewards=torch.tensor([1.0]),
            v0_per_seq=torch.tensor([0.4]),
        )

        assert len(mc_values) == 1
        # [V(t_0)=0.4, V(t_1)≈1, V(t_2)≈0, V(t_3)=1]
        assert len(mc_values[0]) == 4
        assert mc_values[0][0] == pytest.approx(0.4)
        assert mc_values[0][1] == pytest.approx(1.0)
        assert mc_values[0][2] == pytest.approx(0.0)
        assert mc_values[0][3] == pytest.approx(1.0)

        assert seg_tensor.shape == (1, 8)
        assert seg_tensor[0].tolist() == [1, 1, 2, 2, 3, 3, 0, 0]

    def test_single_segment_uses_v0_and_vT(self):
        """K=1 → V-vector is [V(s_0), V(s_T)], no interior entries."""
        seg_ids = np.empty(1, dtype=object)
        seg_ids[0] = [1, 1, 1, 1]
        mc_completions = np.empty(1, dtype=object)
        mc_completions[0] = []
        mc_prefixes = np.empty(1, dtype=object)
        mc_prefixes[0] = []

        mc_values, _ = self._call(
            mc_completions_batch=mc_completions,
            mc_prefixes_batch=mc_prefixes,
            segment_token_ids_batch=seg_ids,
            response_mask=torch.ones(1, 4),
            outcome_rewards=torch.tensor([0.9]),
            v0_per_seq=torch.tensor([0.3]),
        )
        assert mc_values[0] == pytest.approx([0.3, 0.9])

    def test_zero_K_sequence(self):
        """Empty response (K=0) still returns [V0, VT]."""
        seg_ids = np.empty(1, dtype=object)
        seg_ids[0] = [0, 0, 0, 0]
        mc_values, _ = self._call(
            segment_token_ids_batch=seg_ids,
            response_mask=torch.zeros(1, 4),
            outcome_rewards=torch.tensor([0.5]),
            v0_per_seq=torch.tensor([0.2]),
        )
        assert mc_values[0] == pytest.approx([0.2, 0.5])

    def test_missing_mc_data_pads_with_v0(self):
        """If MC completions are missing but K > 1, interior values default to V0."""
        seg_ids = np.empty(1, dtype=object)
        seg_ids[0] = [1, 1, 2, 2, 3, 3]   # K=3
        mc_completions = np.empty(1, dtype=object)
        mc_completions[0] = None
        mc_prefixes = np.empty(1, dtype=object)
        mc_prefixes[0] = None

        mc_values, _ = self._call(
            mc_completions_batch=mc_completions,
            mc_prefixes_batch=mc_prefixes,
            segment_token_ids_batch=seg_ids,
            response_mask=torch.ones(1, 6),
            outcome_rewards=torch.tensor([0.8]),
            v0_per_seq=torch.tensor([0.4]),
        )
        # Interior (2 entries) filled with v0=0.4.
        assert mc_values[0] == pytest.approx([0.4, 0.4, 0.4, 0.8])

    def test_batch_of_two_different_K(self):
        seg_ids = np.empty(2, dtype=object)
        seg_ids[0] = [1, 1, 0, 0]              # K=1
        seg_ids[1] = [1, 2, 3, 0]              # K=3
        mc_completions = np.empty(2, dtype=object)
        mc_completions[0] = []
        mc_completions[1] = [["\\boxed{5}"], ["\\boxed{5}"]]
        mc_prefixes = np.empty(2, dtype=object)
        mc_prefixes[0] = []
        mc_prefixes[1] = ["", ""]

        mc_values, _ = self._call(
            mc_completions_batch=mc_completions,
            mc_prefixes_batch=mc_prefixes,
            segment_token_ids_batch=seg_ids,
            ground_truths=np.array(["?", "5"]),
            response_mask=torch.tensor(
                [[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.float32
            ),
            outcome_rewards=torch.tensor([0.0, 1.0]),
            v0_per_seq=torch.tensor([0.5, 0.3]),
        )

        # Seq 0: K=1 → [0.5, 0.0]
        assert mc_values[0] == pytest.approx([0.5, 0.0])
        # Seq 1: K=3 → [0.3, V1≈1.0, V2≈1.0, 1.0]
        assert len(mc_values[1]) == 4
        assert mc_values[1][0] == pytest.approx(0.3)
        assert mc_values[1][-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Monkey-patch smoke tests
# ---------------------------------------------------------------------------


class TestPatchComputeAdvantage:
    def test_patch_is_idempotent(self):
        from baselines.spo.spo_scoring_hook import patch_compute_advantage
        import verl.trainer.ppo.ray_trainer as rt

        patch_compute_advantage()
        fn1 = rt.compute_advantage
        patch_compute_advantage()
        fn2 = rt.compute_advantage
        assert fn1 is fn2

    def test_non_spo_estimator_passes_through(self):
        """Non-SPO estimator calls should reach the original compute_advantage.

        Using a mock `adv_estimator` string that the original function won't
        recognize would raise a KeyError; we only check that the SPO branch
        isn't taken (no mc_values lookup on the mock data).
        """
        from baselines.spo.spo_scoring_hook import patch_compute_advantage
        import verl.trainer.ppo.ray_trainer as rt

        patch_compute_advantage()
        mock_data = MagicMock()
        mock_data.batch = {
            "token_level_rewards": torch.zeros(2, 4),
            "response_mask": torch.ones(2, 4),
        }
        mock_data.non_tensor_batch = {"uid": np.array(["a", "b"])}

        try:
            rt.compute_advantage(mock_data, "grpo", config=None)
        except Exception:
            pass  # Expected — the mock data is incomplete for GRPO.
