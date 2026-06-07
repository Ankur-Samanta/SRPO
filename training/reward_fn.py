"""Reward function for VERL with multi-dataset routing.

Routes scoring by data_source:
- Math datasets: extract \\boxed{} and check via VERL math_reward
- MC datasets (GPQA, CSQA, MathQA, MMLU-Pro): letter extraction + match
- IFEval: rule-based instruction-following checks
- SciKnowEval L3: MCQ or normalized string match
- LiveCodeBench: sandboxed code execution

Optionally adds a format reward proportional to </thought> delimiter count.

Format reward is controlled via reward_kwargs in the YAML config:
    custom_reward_function:
      path: training/reward_fn.py
      name: compute_score
      reward_kwargs:
        format_reward_cap: 0.2      # max format reward (0 = disabled)
        format_reward_steps: 10     # steps to reach cap
"""

from verl.utils.reward_score.math_reward import compute_score as math_compute_score
from verl.utils.reward_score.math_verify import compute_score as math_verify_compute_score

from training.reward_scorers import (
    mc_score,
    ifeval_score,
    sciknoweval_score,
    code_score,
    yes_no_score,
    qa_em_score,
    python_assert_score,
    retro_score,
)

# Dataset type classification
MATH_DATASETS = {
    "math500", "math_level5", "gsm8k", "amc23", "aime",
    "numinamath_olympiads", "numinamath_aops", "numinamath_amc",
    "openmath2", "polaris",
}
MATH_VERIFY_DATASETS = {"theoremqa"}
MC_DATASETS = {"gpqa", "csqa", "mathqa", "mmlu_pro", "agieval"}
IFEVAL_DATASETS = {"ifeval"}
SCIKNOWEVAL_DATASETS = {
    "sciknoweval_l3",
    "sciknoweval_chemistry", "sciknoweval_physics",
    "sciknoweval_biology", "sciknoweval_materials",
}
CODE_DATASETS = {"livecodebench", "livecodebench_medium", "livecodebench_hard"}
PYTHON_TESTS_DATASETS = {"humaneval_plus", "mbpp_plus"}
YES_NO_DATASETS = {"strategyqa"}
QA_DATASETS = {"hotpotqa"}
RETRO_DATASETS = {"retrosynthesis_uspto50k"}


def _format_reward(solution_str: str, cap: float, steps: int) -> float:
    """Count </thought> delimiters and return a reward in [0, cap]."""
    if cap <= 0:
        return 0.0
    num_thoughts = solution_str.count("</thought>")
    per_step = cap / steps
    return min(cap, num_thoughts * per_step)


def _compute_base_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> float:
    """Route to the appropriate scorer based on data_source."""
    if data_source in MC_DATASETS:
        return mc_score(solution_str, ground_truth)
    elif data_source in IFEVAL_DATASETS:
        return ifeval_score(solution_str, extra_info)
    elif data_source in SCIKNOWEVAL_DATASETS:
        return sciknoweval_score(solution_str, ground_truth, extra_info)
    elif data_source in CODE_DATASETS:
        return code_score(solution_str, extra_info, data_source=data_source)
    elif data_source in RETRO_DATASETS:
        return retro_score(solution_str, ground_truth, extra_info, data_source=data_source)
    elif data_source in PYTHON_TESTS_DATASETS:
        return python_assert_score(solution_str, extra_info)
    elif data_source in YES_NO_DATASETS:
        return yes_no_score(solution_str, ground_truth)
    elif data_source in QA_DATASETS:
        return qa_em_score(solution_str, ground_truth, exact=False)
    elif data_source in MATH_VERIFY_DATASETS:
        return float(math_verify_compute_score(solution_str, ground_truth))
    else:
        # Default: math scoring (covers MATH_DATASETS and any unknown sources)
        return math_compute_score(solution_str, ground_truth)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    format_reward_cap: float = 0.0,
    format_reward_steps: int = 10,
    **kwargs,
) -> dict:
    """Compute reward score with dataset-aware routing + optional format reward.

    Args:
        data_source: Dataset identifier used for routing.
        solution_str: Full model output string.
        ground_truth: Ground truth answer string.
        extra_info: Additional info (IFEval constraints, test cases, etc.).
        format_reward_cap: Max format reward (0 = disabled). Passed via reward_kwargs.
        format_reward_steps: Number of </thought> steps to reach the cap.

    Returns:
        Dict with "score" (total reward), "math_correct", and "format_reward".
    """
    base_score = _compute_base_score(data_source, solution_str, ground_truth, extra_info)
    fmt_reward = _format_reward(solution_str, format_reward_cap, format_reward_steps)
    return {
        "score": base_score + fmt_reward,
        "math_correct": base_score,
        "format_reward": fmt_reward,
    }
