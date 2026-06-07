"""
Dataset loading utilities for SCPO training.

Provides:
- Loading from TREE's dataset_loaders
- Train/val/test splitting with proper isolation
- Prompt formatting
"""

import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .constants import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    DEFAULT_TEST_RATIO,
    SUPPORTED_DATASETS,
)

logger = logging.getLogger(__name__)

# dataset_loaders is vendored under SRPO/vendor/ (see vendor/README.md).
TREE_DIR = Path(__file__).parent.parent / "vendor"
if str(TREE_DIR) not in sys.path:
    sys.path.insert(0, str(TREE_DIR))

try:
    from dataset_loaders import (
        load_dataset_by_name as _load_dataset_by_name,
        get_dataset_info,
        normalize_answer,
    )
    TREE_AVAILABLE = True
except ImportError:
    logger.warning("TREE dataset_loaders not available, using dummy data")
    TREE_AVAILABLE = False


@dataclass
class DataSplit:
    """Container for a data split with prompts and metadata."""
    name: str  # "train", "val", or "test"
    problems: List[Dict[str, Any]]
    prompts: List[str]

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.problems[idx]


@dataclass
class DatasetSplits:
    """Container for all data splits."""
    train: DataSplit
    val: DataSplit
    test: DataSplit

    @property
    def train_prompts(self) -> List[str]:
        return self.train.prompts

    @property
    def val_prompts(self) -> List[str]:
        return self.val.prompts

    @property
    def test_prompts(self) -> List[str]:
        return self.test.prompts


def format_prompt(
    problem: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    template: str = DEFAULT_PROMPT_TEMPLATE,
) -> str:
    """
    Format a problem into a prompt for the model.

    Args:
        problem: The problem statement
        system_prompt: System prompt to prepend
        template: Template with {system_prompt} and {problem} placeholders

    Returns:
        Formatted prompt string
    """
    return template.format(system_prompt=system_prompt, problem=problem)


def _get_dummy_problems(n_problems: int, seed: int) -> List[Dict[str, Any]]:
    """Generate dummy problems for testing when TREE is not available."""
    rng = random.Random(seed)

    dummy_templates = [
        ("What is {a} + {b}?", lambda a, b: str(a + b)),
        ("What is {a} * {b}?", lambda a, b: str(a * b)),
        ("What is {a} - {b}?", lambda a, b: str(a - b)),
        ("Solve for x: {a}x + {b} = {c}", lambda a, b, c: str((c - b) / a) if a != 0 else "undefined"),
        ("What is {a}^2?", lambda a, _: str(a ** 2)),
    ]

    problems = []
    for i in range(n_problems):
        template, answer_fn = rng.choice(dummy_templates)
        a = rng.randint(1, 20)
        b = rng.randint(1, 20)
        c = rng.randint(1, 100)

        if "{c}" in template:
            problem_text = template.format(a=a, b=b, c=c)
            answer = answer_fn(a, b, c)
        else:
            problem_text = template.format(a=a, b=b)
            answer = answer_fn(a, b)

        problems.append({
            "problem": problem_text,
            "answer": answer,
            "unique_id": f"dummy_{i}",
            "subject": "Math",
            "level": rng.randint(1, 5),
        })

    return problems


def _load_from_parquet(
    data_dir: Path,
    n_problems: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load problems from verl-format parquet files (train + test combined)."""
    import pandas as pd

    all_rows = []
    for split_file in ["train.parquet", "test.parquet"]:
        path = data_dir / split_file
        if path.exists():
            df = pd.read_parquet(path)
            all_rows.extend(df.to_dict("records"))

    # Convert verl format → standard problem dict
    problems = []
    for row in all_rows:
        prompt_msgs = row["prompt"]
        problem_text = prompt_msgs[0]["content"] if len(prompt_msgs) > 0 else ""
        answer = row.get("reward_model", {}).get("ground_truth", "")
        extra = row.get("extra_info", {})
        problems.append({
            "problem": problem_text,
            "answer": answer,
            "level": extra.get("level", 5),
            "subject": extra.get("subject", ""),
            "unique_id": f"{extra.get('source_split', 'unknown')}/{extra.get('index', len(problems))}",
        })

    # Subsample if requested
    if n_problems and n_problems < len(problems):
        rng = random.Random(seed)
        problems = rng.sample(problems, n_problems)

    return problems


def load_raw_problems(
    dataset: str,
    n_problems: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load raw problems from dataset.

    Args:
        dataset: Dataset name (math500, gsm8k, amc23, aime, gpqa, csqa, mathqa)
        n_problems: Number of problems to load (None = all)
        seed: Random seed for sampling

    Returns:
        List of problem dictionaries
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {SUPPORTED_DATASETS}")

    # Parquet-backed datasets (from full MATH dataset, not MATH-500)
    PARQUET_DATASETS = {
        "math_level5": Path.home() / "data" / "rlhf" / "math_level5",
        "mmlu_pro": Path.home() / "data" / "rlhf" / "mmlu_pro",
        "ifeval": Path.home() / "data" / "rlhf" / "ifeval",
        "sciknoweval_l3": Path.home() / "data" / "rlhf" / "sciknoweval_l3",
        "sciknoweval_chemistry": Path.home() / "data" / "rlhf" / "sciknoweval_chemistry",
        "sciknoweval_physics": Path.home() / "data" / "rlhf" / "sciknoweval_physics",
        "sciknoweval_biology": Path.home() / "data" / "rlhf" / "sciknoweval_biology",
        "sciknoweval_materials": Path.home() / "data" / "rlhf" / "sciknoweval_materials",
        "livecodebench": Path.home() / "data" / "rlhf" / "livecodebench",
        "livecodebench_medium": Path.home() / "data" / "rlhf" / "livecodebench_medium",
        "livecodebench_hard": Path.home() / "data" / "rlhf" / "livecodebench_hard",
    }

    if dataset in PARQUET_DATASETS:
        problems = _load_from_parquet(PARQUET_DATASETS[dataset], n_problems, seed)
        logger.info(f"Loaded {len(problems)} problems from {dataset} (parquet)")
    elif TREE_AVAILABLE:
        problems = _load_dataset_by_name(
            dataset_name=dataset,
            n_problems=n_problems,
            seed=seed,
        )
        logger.info(f"Loaded {len(problems)} problems from {dataset}")
    else:
        n = n_problems or 100
        problems = _get_dummy_problems(n, seed)
        logger.warning(f"Using {len(problems)} dummy problems (TREE not available)")

    return problems


def split_problems(
    problems: List[Dict[str, Any]],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split problems into train/val/test sets.

    Uses deterministic shuffling based on seed for reproducibility.

    Args:
        problems: List of problem dictionaries
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        test_ratio: Fraction for testing (default 0.1)
        seed: Random seed for shuffling

    Returns:
        Tuple of (train_problems, val_problems, test_problems)
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    # Create deterministic shuffle
    rng = random.Random(seed)
    shuffled = list(problems)
    rng.shuffle(shuffled)

    # Calculate split indices
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_problems = shuffled[:train_end]
    val_problems = shuffled[train_end:val_end]
    test_problems = shuffled[val_end:]

    logger.info(
        f"Split {n} problems: "
        f"train={len(train_problems)}, val={len(val_problems)}, test={len(test_problems)}"
    )

    return train_problems, val_problems, test_problems


def create_data_split(
    problems: List[Dict[str, Any]],
    name: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> DataSplit:
    """
    Create a DataSplit from a list of problems.

    Args:
        problems: List of problem dictionaries
        name: Split name ("train", "val", or "test")
        system_prompt: System prompt for formatting
        prompt_template: Template for formatting

    Returns:
        DataSplit object
    """
    # Add formatted prompts to each problem
    for p in problems:
        p["prompt"] = format_prompt(
            problem=p["problem"],
            system_prompt=system_prompt,
            template=prompt_template,
        )

    prompts = [p["prompt"] for p in problems]

    return DataSplit(name=name, problems=problems, prompts=prompts)


def load_dataset_splits(
    dataset: str,
    n_problems: Optional[int] = None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> DatasetSplits:
    """
    Load dataset and split into train/val/test.

    This is the main entry point for loading data with proper splits.

    Args:
        dataset: Dataset name
        n_problems: Total number of problems to load (None = all)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
        system_prompt: System prompt for formatting
        prompt_template: Template for formatting

    Returns:
        DatasetSplits containing train, val, and test splits
    """
    # Load all problems
    problems = load_raw_problems(dataset=dataset, n_problems=n_problems, seed=seed)

    # Split
    train_problems, val_problems, test_problems = split_problems(
        problems=problems,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Create DataSplit objects
    train_split = create_data_split(
        train_problems, "train", system_prompt, prompt_template
    )
    val_split = create_data_split(
        val_problems, "val", system_prompt, prompt_template
    )
    test_split = create_data_split(
        test_problems, "test", system_prompt, prompt_template
    )

    return DatasetSplits(train=train_split, val=val_split, test=test_split)


# =============================================================================
# Convenience functions
# =============================================================================

def get_train_prompts(
    dataset: str,
    n_problems: Optional[int] = None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = 42,
) -> List[str]:
    """Get just the training prompts (convenience function)."""
    splits = load_dataset_splits(
        dataset=dataset,
        n_problems=n_problems,
        train_ratio=train_ratio,
        val_ratio=(1.0 - train_ratio) / 2,
        test_ratio=(1.0 - train_ratio) / 2,
        seed=seed,
    )
    return splits.train_prompts


# =============================================================================
# Re-exports
# =============================================================================

__all__ = [
    # Main functions
    "load_dataset_splits",
    "load_raw_problems",
    "split_problems",
    "format_prompt",
    "get_train_prompts",
    # Data classes
    "DataSplit",
    "DatasetSplits",
    # Constants
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_PROMPT_TEMPLATE",
    "SUPPORTED_DATASETS",
]


if __name__ == "__main__":
    """Test the data loader."""
    logging.basicConfig(level=logging.INFO)

    print("Testing SCPO data loader with splits...\n")

    # Test loading with splits
    splits = load_dataset_splits(
        dataset="math500",
        n_problems=100,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )

    print(f"Train: {len(splits.train)} problems")
    print(f"Val:   {len(splits.val)} problems")
    print(f"Test:  {len(splits.test)} problems")

    print(f"\nSample train prompt:\n{splits.train_prompts[0][:200]}...")

    # Verify isolation - check that IDs don't overlap
    train_ids = {p["unique_id"] for p in splits.train.problems}
    val_ids = {p["unique_id"] for p in splits.val.problems}
    test_ids = {p["unique_id"] for p in splits.test.problems}

    assert train_ids.isdisjoint(val_ids), "Train and val overlap!"
    assert train_ids.isdisjoint(test_ids), "Train and test overlap!"
    assert val_ids.isdisjoint(test_ids), "Val and test overlap!"
    print("\n✓ Splits are properly isolated (no overlap)")
