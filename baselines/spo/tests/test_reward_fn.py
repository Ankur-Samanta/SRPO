"""Tests for SPO reward function and MC completion scoring.

Reference: SPO (arXiv:2505.23564) https://github.com/AIFrameResearch/SPO
"""
import pytest
from unittest.mock import patch


def test_compute_score_delegates_to_tgrpo():
    """compute_score should produce same results as TGRPO reward_fn."""
    from baselines.spo.reward_fn import compute_score

    with patch(
        "training.reward_fn.compute_score",
        return_value={"score": 1.0, "math_correct": 1.0, "format_reward": 0.0},
    ) as mock_tgrpo:
        result = compute_score(
            data_source="math500",
            solution_str="The answer is \\boxed{42}",
            ground_truth="42",
        )
        mock_tgrpo.assert_called_once()
        assert result["score"] == 1.0
        assert result["math_correct"] == 1.0


def test_score_mc_completions_basic():
    """Score MC completions with mock math answers."""
    from baselines.spo.reward_fn import score_mc_completions

    # math_compute_score returns 1.0 if boxed answer matches ground truth, 0.0 otherwise
    with patch(
        "baselines.spo.reward_fn.math_compute_score",
        side_effect=lambda sol, gt: 1.0 if "\\boxed{42}" in sol else 0.0,
    ):
        mc_completions = [
            # Segment 0: 2 completions, one correct, one wrong
            ["... therefore \\boxed{42}", "... therefore \\boxed{7}"],
            # Segment 1: 2 completions, both correct
            ["\\boxed{42}", "The answer is \\boxed{42}"],
        ]
        mc_prefixes = [
            "Let me think step by step. ",
            "Let me think step by step. First, compute 6*7=42. ",
        ]
        ground_truth = "42"

        scores = score_mc_completions(mc_completions, mc_prefixes, ground_truth)

        assert len(scores) == 2
        assert scores[0] == [1.0, 0.0]
        assert scores[1] == [1.0, 1.0]


def test_score_mc_completions_empty():
    """Empty completions list should return empty scores."""
    from baselines.spo.reward_fn import score_mc_completions

    scores = score_mc_completions([], [], "42")
    assert scores == []


def test_mc_values_from_scores():
    """MC values should be mean of completion scores per segment."""
    from baselines.spo.reward_fn import mc_values_from_scores

    mc_scores = [
        [1.0, 0.0],        # mean = 0.5
        [1.0, 1.0, 1.0],   # mean = 1.0
        [0.0, 0.0],        # mean = 0.0
    ]
    values = mc_values_from_scores(mc_scores)

    assert len(values) == 3
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(1.0)
    assert values[2] == pytest.approx(0.0)


def test_mc_values_from_scores_empty_segment():
    """Empty scores list for a segment should return 0.0."""
    from baselines.spo.reward_fn import mc_values_from_scores

    mc_scores = [
        [1.0, 0.0],
        [],             # no completions -- should default to 0.0
        [0.5],
    ]
    values = mc_values_from_scores(mc_scores)

    assert len(values) == 3
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(0.0)
    assert values[2] == pytest.approx(0.5)
