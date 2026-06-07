"""Reward function for SDPO baseline.

Returns the base math correctness reward. The off-policy shaping for
self-distilled tokens is handled by the loss function, not here.

Reference:
    Hübotter et al. "Reinforcement Learning via Self-Distillation."
    arXiv:2601.20802.
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

    Identical to training/reward_fn.py -- reused for consistency.
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
