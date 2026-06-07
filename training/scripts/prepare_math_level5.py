"""Prepare MATH Level 5 dataset in VERL parquet format.

Uses the original train/test splits from EleutherAI/hendrycks_math, filters to
Level 5, samples --n-train and --n-test problems from each, and writes parquets
with size-tagged filenames (e.g. train_1500.parquet, test_500.parquet).

Usage:
    python training/scripts/prepare_math_level5.py --n-train 1500 --n-test 500
    python training/scripts/prepare_math_level5.py --n-train 400 --n-test 100
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = Path.home() / "data" / "rlhf" / "math_level5"

SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def extract_boxed_answer(solution: str) -> str:
    """Extract the last \\boxed{...} answer from a solution string."""
    matches = list(re.finditer(r"\\boxed\{", solution))
    if not matches:
        return ""
    start_pos = matches[-1].end()
    brace_count = 1
    i = start_pos
    while i < len(solution) and brace_count > 0:
        if solution[i] == "{":
            brace_count += 1
        elif solution[i] == "}":
            brace_count -= 1
        i += 1
    if brace_count == 0:
        return solution[start_pos:i - 1]
    return ""


def load_level5(split: str):
    """Load Level 5 problems with valid boxed answers from a given split."""
    problems = []
    for subject in SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split=split)
        for row in ds:
            if row["level"] != "Level 5":
                continue
            answer = extract_boxed_answer(row["solution"])
            if not answer:
                continue
            problems.append({
                "problem": row["problem"],
                "answer": answer,
                "subject": subject,
            })
    return problems


def to_verl_df(problems):
    """Convert problem dicts to VERL-format DataFrame."""
    rows = []
    for idx, p in enumerate(problems):
        rows.append({
            "prompt": [{"role": "user", "content": p["problem"]}],
            "data_source": "math_level5",
            "reward_model": {"ground_truth": p["answer"]},
            "extra_info": {
                "index": idx,
                "level": 5,
                "subject": p["subject"],
            },
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load from original splits separately
    train_problems = load_level5("train")
    test_problems = load_level5("test")
    print(f"Level 5 with valid answers: train={len(train_problems)}, test={len(test_problems)}")

    # Sample from each
    train_df = pd.DataFrame(train_problems).sample(n=args.n_train, random_state=args.seed).reset_index(drop=True)
    test_df = pd.DataFrame(test_problems).sample(n=args.n_test, random_state=args.seed).reset_index(drop=True)
    print(f"Sampled: train={len(train_df)}, test={len(test_df)}")

    print("\nTrain subject distribution:")
    for subject, count in train_df["subject"].value_counts().items():
        print(f"  {subject}: {count}")

    # Convert to VERL format
    train_verl = to_verl_df(train_df.to_dict("records"))
    test_verl = to_verl_df(test_df.to_dict("records"))

    # Write with size-tagged filenames
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / f"train_{args.n_train}.parquet"
    test_path = OUTPUT_DIR / f"test_{args.n_test}.parquet"

    train_verl.to_parquet(train_path)
    test_verl.to_parquet(test_path)

    print(f"\nTrain: {len(train_verl)} samples -> {train_path}")
    print(f"Test:  {len(test_verl)} samples -> {test_path}")


if __name__ == "__main__":
    main()
