#!/usr/bin/env python3
"""
Aggregate per-method Thought-MDP output files into a single combined file.

Usage:
    python tests/aggregate_thought_mdp.py \
        --inputs tests/outputs/thought_mdp_gpqa_qwen7b_with_ex.json \
                 tests/outputs/thought_mdp_gpqa_qwen7b_no_ex.json \
                 tests/outputs/thought_mdp_gpqa_qwen7b_cot.json \
        --output tests/outputs/thought_mdp_gpqa_qwen7b.json
"""

import json
import argparse
from pathlib import Path


def aggregate(input_files, output_file):
    """Merge per-method JSONs into one combined file.

    Each input has the same problems but different methods.
    Output merges all methods into each problem's "methods" dict.
    """
    all_data = []
    for f in input_files:
        with open(f) as fh:
            all_data.append(json.load(fh))

    # Use first file as base
    merged = all_data[0].copy()
    merged["config"]["methods"] = "aggregated"

    # Merge methods from all files into each problem
    for prob_idx in range(len(merged["results"])):
        for data in all_data[1:]:
            if prob_idx < len(data["results"]):
                for method_name, method_data in data["results"][prob_idx]["methods"].items():
                    merged["results"][prob_idx]["methods"][method_name] = method_data

    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)

    # Print summary
    methods_found = set()
    for prob in merged["results"]:
        methods_found.update(prob["methods"].keys())
    print(f"Aggregated {len(input_files)} files -> {output_file}")
    print(f"  Problems: {len(merged['results'])}")
    print(f"  Methods: {sorted(methods_found)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    aggregate(args.inputs, args.output)
