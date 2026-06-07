"""Reward function for Critique-GRPO baseline.

Returns the base math correctness reward. The off-policy shaping for
refinement tokens is handled by the loss function, not here.

Reference:
    Zhang et al. "Critique-GRPO: Advancing LLM Reasoning with Natural
    Language and Numerical Feedback." arXiv:2506.03106.
"""

from training.reward_fn import compute_score as tgrpo_compute_score


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

    Identical to training/reward_fn.py — reused for consistency.
    """
    return tgrpo_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        format_reward_cap=format_reward_cap,
        format_reward_steps=format_reward_steps,
        **kwargs,
    )
