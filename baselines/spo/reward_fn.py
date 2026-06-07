"""Reward function for SPO baseline.

Reuses the TGRPO math correctness reward and adds MC completion scoring
for segment-level value estimation.

Reference:
    SPO: Segment Policy Optimization (arXiv:2505.23564)
    https://github.com/AIFrameResearch/SPO
"""

from verl.utils.reward_score.math_reward import compute_score as math_compute_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    format_reward_cap: float = 0.0,
    format_reward_steps: int = 10,
    **kwargs,
) -> dict:
    """Compute math correctness reward + optional format reward.

    Identical to training/reward_fn.py -- reused for consistency.
    """
    # Import and delegate to the existing reward function
    from training.reward_fn import compute_score as tgrpo_compute_score
    return tgrpo_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        format_reward_cap=format_reward_cap,
        format_reward_steps=format_reward_steps,
        **kwargs,
    )


def score_mc_completions(
    mc_completions: list[list[str]],
    mc_prefixes: list[str],
    ground_truth: str,
) -> list[list[float]]:
    """Score MC completions for segment-level value estimation.

    For each segment boundary i, for each MC completion j:
        full_text = mc_prefixes[i] + mc_completions[i][j]
        score = math_compute_score(full_text, ground_truth)

    Args:
        mc_completions: mc_completions[i][j] = decoded text of j-th MC
            completion from segment boundary i.
        mc_prefixes: mc_prefixes[i] = decoded prefix up to segment boundary i.
        ground_truth: Ground truth answer string.

    Returns:
        List of lists of float scores. mc_scores[i][j] = reward for
        completion j from boundary i.
    """
    mc_scores = []
    for i, (prefix, completions) in enumerate(zip(mc_prefixes, mc_completions)):
        segment_scores = []
        for completion in completions:
            full_text = prefix + completion
            score = math_compute_score(full_text, ground_truth)
            segment_scores.append(float(score))
        mc_scores.append(segment_scores)
    return mc_scores


def mc_values_from_scores(mc_scores: list[list[float]]) -> list[float]:
    """Compute MC value estimates from completion scores.

    V(c_i) = mean(scores of all MC completions from boundary i)

    Reference: SPO paper Section 3.2 -- Monte Carlo value estimation.

    Args:
        mc_scores: mc_scores[i] = list of reward scores for segment boundary i.

    Returns:
        List of V(c_i) values, one per segment boundary.
    """
    values = []
    for scores in mc_scores:
        if scores:
            values.append(sum(scores) / len(scores))
        else:
            values.append(0.0)
    return values
