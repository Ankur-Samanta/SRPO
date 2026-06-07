#!/usr/bin/env python3
"""
Test script: Compare Thought-MDP prompt templates vs standard CoT.

Compares three prompting approaches on GPQA problems:
1. Thought-MDP with in-context examples (2 worked problems)
2. Thought-MDP without examples (generic format template)
3. Standard CoT ("Let's think step by step")

Generates 8 rollouts per problem per method.

Usage:
    python tests/test_thought_mdp_prompts.py --model llama3b --gpus 0
    python tests/test_thought_mdp_prompts.py --model llama3b --gpus 0,1 --tensor-parallel-size 2
"""

import os
os.environ['VLLM_USE_V1'] = '1'

import sys
from pathlib import Path
import json
import argparse
import logging
import re
from typing import List, Dict, Any, Optional

# Add paths
SRPO_DIR = Path(__file__).parent.parent
TREE_DIR = SRPO_DIR.parent / "TREE"
sys.path.insert(0, str(SRPO_DIR))
sys.path.insert(0, str(TREE_DIR))

# Imports
from tree_of_thought import initialize_model
from dataset_loaders import load_dataset_by_name, normalize_answer
from evaluation.thought_mdp import generate_thought_chains_batched

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

def get_standard_cot_prompt() -> str:
    """Standard Chain-of-Thought prompt template."""
    return """Solve this problem step by step. Show your reasoning clearly, then provide your final answer in \\boxed{{answer}} format.

Q: {question}

Let's think step by step.
"""


# =============================================================================
# GENERATION
# =============================================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...}."""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    return match.group(1) if match else None


def count_thoughts(text: str, delimiter: str = "</thought>") -> int:
    """Count number of thoughts in text."""
    return text.count(delimiter)


def count_cot_steps(text: str) -> int:
    """Estimate number of reasoning steps in CoT output."""
    # Count sentences that look like reasoning steps
    steps = len(re.findall(r'[.!?]\s+', text))
    return max(1, steps)


def generate_cot_rollouts(
    manager,
    question: str,
    n_rollouts: int = 8,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> List[str]:
    """
    Generate N CoT rollouts using vLLM's native n parameter.

    Single call to vLLM with n=8 for efficiency.
    """
    template = get_standard_cot_prompt()
    prompt = template.format(question=question)

    # vLLM generates n completions in parallel
    outputs = manager.generate(
        prompts=[prompt],
        max_tokens=max_tokens,
        temperature=temperature,
        min_tokens=10,
        n=n_rollouts,
    )

    # outputs is a flat list of n_rollouts completions
    return [o.strip() for o in outputs]


def analyze_rollouts(
    rollouts: List[str],
    ground_truth: str,
    is_thought_mdp: bool = False,
) -> Dict[str, Any]:
    """Analyze a set of rollouts."""
    results = []
    correct_count = 0

    for i, rollout in enumerate(rollouts):
        answer = extract_boxed_answer(rollout)
        is_correct = (
            normalize_answer(answer) == normalize_answer(ground_truth)
            if answer else False
        )
        if is_correct:
            correct_count += 1

        if is_thought_mdp:
            n_steps = count_thoughts(rollout)
        else:
            n_steps = count_cot_steps(rollout)

        results.append({
            "rollout_idx": i,
            "output": rollout,
            "extracted_answer": answer,
            "is_correct": is_correct,
            "num_steps": n_steps,
        })

    return {
        "rollouts": results,
        "n_correct": correct_count,
        "n_total": len(rollouts),
        "accuracy": correct_count / len(rollouts) if rollouts else 0,
        "avg_steps": sum(r["num_steps"] for r in results) / len(results) if results else 0,
    }


ALL_METHODS = ["thought_mdp_with_examples", "thought_mdp_no_examples", "standard_cot"]


def _thoughts_to_output(thoughts: List[str]) -> str:
    """Convert a list of thought strings back to delimited output format."""
    if not thoughts:
        return ""
    return "</thought>\n".join(thoughts) + "</thought>"


def run_comparison(
    manager,
    problems: List[Dict[str, Any]],
    n_rollouts: int = 8,
    temperature: float = 0.7,
    methods: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run prompting methods on each problem with N rollouts each.

    Uses batched generation for thought_mdp methods: all problems x rollouts
    are generated in depth-synchronized batch calls for ~100x speedup.

    Args:
        methods: List of methods to run. Default: all three.
                 Options: thought_mdp_with_examples, thought_mdp_no_examples, standard_cot
    """
    if methods is None:
        methods = ALL_METHODS

    questions = [p['problem'] for p in problems]
    ground_truths = [p['answer'] for p in problems]

    # Initialize results structure
    results = []
    for i, problem in enumerate(problems):
        results.append({
            "problem_idx": i,
            "question": problem['problem'],
            "ground_truth": problem['answer'],
            "methods": {}
        })

    # --- Batched Thought-MDP generation ---
    for method_name, use_examples in [
        ("thought_mdp_with_examples", True),
        ("thought_mdp_no_examples", False),
    ]:
        if method_name not in methods:
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"BATCHED GENERATION: {method_name} ({len(questions)} problems x {n_rollouts} rollouts)")
        logger.info(f"{'='*80}")

        try:
            # Single batched call for ALL problems x rollouts
            all_chains = generate_thought_chains_batched(
                manager=manager,
                questions=questions,
                n_rollouts=n_rollouts,
                use_examples=use_examples,
                temperature=temperature,
            )

            # Convert to output format and analyze per-problem
            for prob_idx in range(len(questions)):
                rollout_outputs = [
                    _thoughts_to_output(all_chains[prob_idx][r])
                    for r in range(n_rollouts)
                ]
                analysis = analyze_rollouts(
                    rollout_outputs, ground_truths[prob_idx], is_thought_mdp=True
                )
                results[prob_idx]["methods"][method_name] = analysis

                logger.info(f"  Problem {prob_idx+1}: {analysis['n_correct']}/{analysis['n_total']} correct, avg {analysis['avg_steps']:.1f} thoughts")

        except Exception as e:
            logger.error(f"Error in {method_name}: {e}")
            for prob_idx in range(len(questions)):
                results[prob_idx]["methods"][method_name] = {"error": str(e)}

    # --- Standard CoT (already efficient with n param) ---
    if "standard_cot" in methods:
        logger.info(f"\n{'='*80}")
        logger.info(f"STANDARD COT ({len(questions)} problems x {n_rollouts} rollouts)")
        logger.info(f"{'='*80}")

        for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
            logger.info(f"\n--- Problem {i+1}/{len(questions)} ---")
            try:
                rollouts_cot = generate_cot_rollouts(
                    manager, question,
                    n_rollouts=n_rollouts, temperature=temperature
                )
                analysis = analyze_rollouts(rollouts_cot, ground_truth, is_thought_mdp=False)
                results[i]["methods"]["standard_cot"] = analysis

                logger.info(f"  {analysis['n_correct']}/{analysis['n_total']} correct, avg {analysis['avg_steps']:.1f} steps")
            except Exception as e:
                logger.error(f"Error in standard_cot for problem {i}: {e}")
                results[i]["methods"]["standard_cot"] = {"error": str(e)}

    return results


def print_summary(results: List[Dict[str, Any]], n_rollouts: int):
    """Print summary statistics."""
    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY ({n_rollouts} rollouts per problem)")
    logger.info(f"{'='*80}")

    methods = ["thought_mdp_with_examples", "thought_mdp_no_examples", "standard_cot"]

    for method in methods:
        total_correct = 0
        total_rollouts = 0
        total_steps = 0
        n_problems = 0

        for r in results:
            if method in r["methods"] and "error" not in r["methods"][method]:
                analysis = r["methods"][method]
                total_correct += analysis["n_correct"]
                total_rollouts += analysis["n_total"]
                total_steps += analysis["avg_steps"] * analysis["n_total"]
                n_problems += 1

        if total_rollouts > 0:
            overall_acc = total_correct / total_rollouts * 100
            avg_steps = total_steps / total_rollouts
        else:
            overall_acc = 0
            avg_steps = 0

        logger.info(f"\n{method}:")
        logger.info(f"  Problems evaluated: {n_problems}")
        logger.info(f"  Overall accuracy: {total_correct}/{total_rollouts} ({overall_acc:.1f}%)")
        logger.info(f"  Avg thoughts/steps per rollout: {avg_steps:.1f}")

    logger.info(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Thought-MDP prompt templates vs standard CoT"
    )

    # Model args
    parser.add_argument("--model", type=str, default="llama3b",
                        help="Model name (default: llama3b)")
    parser.add_argument("--gpus", type=str, default="0",
                        help="GPU IDs (comma-separated, default: 0)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size (default: 1)")

    # Data args
    parser.add_argument("--dataset", type=str, default="gpqa",
                        help="Dataset name: gpqa, math500, etc. (default: gpqa)")
    parser.add_argument("--level", type=int, default=None,
                        help="For math500: difficulty level 1-5 (default: None)")
    parser.add_argument("--n-problems", type=int, default=5,
                        help="Number of problems to test (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Generation args
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        choices=["thought_mdp_with_examples", "thought_mdp_no_examples", "standard_cot"],
                        help="Methods to run (default: all three)")
    parser.add_argument("--n-rollouts", type=int, default=8,
                        help="Number of rollouts per problem per method (default: 8)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (optional)")

    args = parser.parse_args()

    # Initialize model
    logger.info("="*80)
    logger.info("INITIALIZING MODEL")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"GPUs: {args.gpus}")
    logger.info(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    logger.info(f"N Rollouts: {args.n_rollouts}")

    manager = initialize_model(
        gpu_ids=args.gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        model_name=args.model,
        model_seed=args.seed,
    )

    try:
        # Load problems
        logger.info(f"\nLoading {args.n_problems} {args.dataset} problems...")
        load_kwargs = dict(
            dataset_name=args.dataset,
            n_problems=args.n_problems,
            seed=args.seed,
        )
        if args.level is not None:
            load_kwargs["level"] = args.level
        problems = load_dataset_by_name(**load_kwargs)
        logger.info(f"Loaded {len(problems)} problems")

        # Run comparison
        results = run_comparison(
            manager=manager,
            problems=problems,
            n_rollouts=args.n_rollouts,
            temperature=args.temperature,
            methods=args.methods,
        )

        # Print summary
        print_summary(results, args.n_rollouts)

        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "config": {
                    "model": args.model,
                    "dataset": args.dataset,
                    "level": args.level,
                    "methods": args.methods or ALL_METHODS,
                    "n_problems": args.n_problems,
                    "n_rollouts": args.n_rollouts,
                    "seed": args.seed,
                    "temperature": args.temperature,
                },
                "results": results,
            }

            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            logger.info(f"\nResults saved to: {output_path}")

        return results

    finally:
        logger.info("\nUnloading model...")
        manager.unload_base_model()


if __name__ == "__main__":
    main()
