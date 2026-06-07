#!/usr/bin/env python3
"""Run post-training evaluation on a checkpoint."""
import argparse
import json
from evaluation.data_loader import load_dataset_splits
from evaluation.evaluate import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = load_dataset_splits(
        dataset_name=args.dataset,
        num_prompts=args.num_prompts,
        train_ratio=0.8, val_ratio=0.0, test_ratio=0.2,
        seed=args.seed,
    )

    results = evaluate_checkpoint(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        test_split=splits.test,
        max_steps=20,
        max_tokens_per_thought=512,
        torch_dtype="bfloat16",
        use_thought_examples=True,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
