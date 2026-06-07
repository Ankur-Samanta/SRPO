"""Reward function for SCoRe baseline.

Returns the base math correctness reward. The shaped reward (with
self-correction bonus) is applied by the scoring hook, NOT here.

This keeps the reward function identical to TGRPO for fair comparison —
the only difference is the shaped reward applied post-hoc.

Reference:
    Kumar et al. "Training Language Models to Self-Correct via Reinforcement
    Learning." ICLR 2025. arXiv:2409.12917.
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
    The SCoRe-specific shaped reward is applied by the scoring hook.
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
