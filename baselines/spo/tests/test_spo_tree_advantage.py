"""Tests for SPO-tree advantage logic (paper §5)."""

from __future__ import annotations

import pytest
import torch

from baselines.spo.spo_advantage import (
    build_tree_values,
    sibling_relative_advantages,
    tree_token_advantages,
)


# ---------------------------------------------------------------------------
# build_tree_values
# ---------------------------------------------------------------------------


class TestBuildTreeValues:
    def test_depth1_is_group_mean(self):
        """Depth-1 tree: V̂(root) = mean(rewards); V̂(leaf) = its own reward."""
        paths = [(0,), (1,), (2,), (3,)]
        rewards = [1.0, 0.0, 1.0, 0.0]
        V = build_tree_values(paths, rewards)
        assert V[()] == pytest.approx(0.5)
        assert V[(0,)] == pytest.approx(1.0)
        assert V[(1,)] == pytest.approx(0.0)

    def test_depth2_bottom_up(self):
        """V̂(internal) = mean of its descendant leaf rewards."""
        # B=(2,2): 4 leaves, paths (0,0), (0,1), (1,0), (1,1).
        paths = [(0, 0), (0, 1), (1, 0), (1, 1)]
        rewards = [1.0, 0.0, 0.0, 1.0]  # parent (0,) avg=0.5, parent (1,) avg=0.5

        V = build_tree_values(paths, rewards)
        assert V[()] == pytest.approx(0.5)
        assert V[(0,)] == pytest.approx(0.5)
        assert V[(1,)] == pytest.approx(0.5)
        # Leaves = their own rewards.
        for p, r in zip(paths, rewards):
            assert V[p] == pytest.approx(r)

    def test_depth3_with_imbalance(self):
        """(2, 2, 2) tree where one subtree is 'good' and one is 'bad'."""
        paths = []
        rewards = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    paths.append((i, j, k))
                    rewards.append(1.0 if i == 0 else 0.0)
        V = build_tree_values(paths, rewards)
        # Left half all 1, right half all 0.
        assert V[(0,)] == pytest.approx(1.0)
        assert V[(1,)] == pytest.approx(0.0)
        assert V[()] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# sibling_relative_advantages
# ---------------------------------------------------------------------------


class TestSiblingRelativeAdvantages:
    def test_depth1_equivalent_to_grpo(self):
        """With a depth-1 tree, Â_leaf = R - mean(siblings)."""
        paths = [(0,), (1,), (2,), (3,)]
        rewards = [1.0, 0.0, 1.0, 0.0]
        V = build_tree_values(paths, rewards)

        advs = sibling_relative_advantages(paths, V, normalize_by_std=False)
        mean = sum(rewards) / len(rewards)
        for i, row in enumerate(advs):
            assert len(row) == 1
            assert row[0] == pytest.approx(rewards[i] - mean)

    def test_depth2_uses_direct_siblings(self):
        """Level-2 advantage compares against same-parent siblings only."""
        # (0,0)=1, (0,1)=0  → parent (0,) has mean 0.5, std 0.5
        # (1,0)=0, (1,1)=1  → parent (1,) has mean 0.5, std 0.5
        paths = [(0, 0), (0, 1), (1, 0), (1, 1)]
        rewards = [1.0, 0.0, 0.0, 1.0]
        V = build_tree_values(paths, rewards)
        advs = sibling_relative_advantages(paths, V, normalize_by_std=False)

        # Level 1: V(0,)=V(1,)=0.5 and root avg=0.5 → both get 0.
        # Level 2: leaf minus its own parent's mean (0.5).
        expected = [
            [0.5 - 0.5, 1.0 - 0.5],   # (0,0): level1=0, level2=+0.5
            [0.5 - 0.5, 0.0 - 0.5],   # (0,1): level1=0, level2=-0.5
            [0.5 - 0.5, 0.0 - 0.5],   # (1,0)
            [0.5 - 0.5, 1.0 - 0.5],   # (1,1)
        ]
        for row, exp in zip(advs, expected):
            assert row[0] == pytest.approx(exp[0])
            assert row[1] == pytest.approx(exp[1])

    def test_std_normalization(self):
        """std normalization divides the advantage by sibling std."""
        paths = [(0,), (1,), (2,), (3,)]
        rewards = [2.0, 0.0, 2.0, 0.0]
        V = build_tree_values(paths, rewards)
        advs = sibling_relative_advantages(paths, V, normalize_by_std=True)
        # Raw: [2-1, 0-1, 2-1, 0-1] = [1, -1, 1, -1], std = 1 → unchanged.
        for row, r in zip(advs, rewards):
            assert row[0] == pytest.approx(r - 1.0)

    def test_zero_advantage_when_all_siblings_equal(self):
        """If siblings all share the same V̂, advantage is 0."""
        paths = [(0,), (1,), (2,)]
        rewards = [0.7, 0.7, 0.7]
        V = build_tree_values(paths, rewards)
        advs = sibling_relative_advantages(paths, V, normalize_by_std=False)
        for row in advs:
            assert row[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# tree_token_advantages
# ---------------------------------------------------------------------------


class TestTreeTokenAdvantages:
    def test_assigns_per_depth(self):
        """Tokens in segment k (1-indexed) get Â of the depth-k node."""
        # depth=3 path, seg_ids: segment 1 on [0..1], segment 2 on [2..3],
        # segment 3 on [4..7] (leaf extends).
        seg_ids = torch.tensor([1, 1, 2, 2, 3, 3, 3, 3])
        mask = torch.ones(8)
        seg_advs = [0.2, -0.3, 0.5]  # level 1, 2, 3

        out = tree_token_advantages(seg_advs, seg_ids, mask, 8)
        for t in (0, 1):
            assert out[t].item() == pytest.approx(0.2)
        for t in (2, 3):
            assert out[t].item() == pytest.approx(-0.3)
        for t in (4, 5, 6, 7):
            assert out[t].item() == pytest.approx(0.5)

    def test_padding_positions_zero(self):
        seg_ids = torch.tensor([1, 1, 2, 0, 0])
        mask = torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32)
        seg_advs = [0.3, 0.4]
        out = tree_token_advantages(seg_advs, seg_ids, mask, 5)
        assert out[3].item() == 0.0
        assert out[4].item() == 0.0
        assert out[0].item() == pytest.approx(0.3)
        assert out[2].item() == pytest.approx(0.4)

    def test_empty_seg_advs(self):
        out = tree_token_advantages([], torch.zeros(5, dtype=torch.long),
                                    torch.zeros(5), 5)
        assert (out == 0).all()


# ---------------------------------------------------------------------------
# End-to-end: sibling advantages should sum to zero within a parent group
# ---------------------------------------------------------------------------


class TestTreeInvariants:
    def test_sum_zero_among_siblings(self):
        """For each parent, mean of children advantages should be 0 (centered)."""
        paths = [(i, j) for i in range(3) for j in range(3)]
        rewards = [float((i * 3 + j) % 5) / 4.0 for i, j in paths]
        V = build_tree_values(paths, rewards)
        advs = sibling_relative_advantages(paths, V, normalize_by_std=False)

        # Group by parent = path[:1]. Depth-2 advantages within same parent
        # should sum to ~0.
        groups: dict[tuple, list[float]] = {}
        for p, row in zip(paths, advs):
            groups.setdefault(p[:1], []).append(row[1])
        for _, vals in groups.items():
            assert sum(vals) == pytest.approx(0.0, abs=1e-9)
