#!/usr/bin/env python3
"""
Test script: Evaluate individual thoughts using ThoughtEvaluator.

Loads Thought-MDP outputs, splits into individual thoughts, and scores each
thought using the same model that generated them.

Usage:
    python tests/test_thought_evaluator.py --model llama3b --gpus 0
    python tests/test_thought_evaluator.py --model qwen7b --gpus 0,1 --tensor-parallel-size 2
"""

import os
os.environ['VLLM_USE_V1'] = '1'

import sys
from pathlib import Path
import json
import argparse
import logging
from typing import List, Dict, Any, Optional

# Add paths
SCPO_DIR = Path(__file__).parent.parent
TREE_DIR = SCPO_DIR.parent / "TREE"
sys.path.insert(0, str(SCPO_DIR))
sys.path.insert(0, str(TREE_DIR))

from tree_of_thought import initialize_model
from evaluation.thought_evaluator import ThoughtEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_thoughts(output: str, delimiter: str = "</thought>") -> List[str]:
    """
    Parse a Thought-MDP output into individual thoughts.

    Args:
        output: Full output string with thoughts delimited by </thought>
        delimiter: Thought delimiter

    Returns:
        List of individual thoughts (without delimiters)
    """
    # Split by delimiter and filter empty
    parts = output.split(delimiter)
    thoughts = [t.strip() for t in parts if t.strip()]
    return thoughts


def load_thought_mdp_outputs(model_name: str, dataset: str = None) -> Dict[str, Any]:
    """Load the pre-generated Thought-MDP outputs for a model."""
    outputs_dir = SCPO_DIR / "tests" / "outputs"
    if dataset:
        filepath = outputs_dir / f"thought_mdp_{dataset}_{model_name}.json"
    else:
        filepath = outputs_dir / f"thought_mdp_comparison_{model_name}.json"

    if not filepath.exists():
        raise FileNotFoundError(f"No outputs found at {filepath}")

    with open(filepath, 'r') as f:
        return json.load(f)


def run_evaluation(
    manager,
    data: Dict[str, Any],
    max_problems: Optional[int] = None,
    max_rollouts_per_method: Optional[int] = None,
    problem_start: int = 0,
    problem_end: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run thought evaluation on all problems/rollouts using batched evaluation.

    Collects ALL (question, prefix, thought) tuples across all problems and
    rollouts, evaluates them in a single batch call, then organizes results.

    Args:
        manager: Model manager with generate() method
        data: Loaded thought_mdp_comparison JSON
        max_problems: Limit number of problems (for quick testing)
        max_rollouts_per_method: Limit rollouts per method (for quick testing)

    Returns:
        Evaluation results dict
    """
    evaluator = ThoughtEvaluator()

    # Create batch generate function for evaluator
    def batch_generate_fn(prompts: List[str]) -> List[str]:
        return manager.generate(
            prompts=prompts,
            max_tokens=2048,
            temperature=0.7,
            n=1,
        )

    problems = data["results"]
    if max_problems:
        problems = problems[:max_problems]
    problems = problems[problem_start:problem_end]

    # Only evaluate thought_mdp methods (not standard_cot)
    method_names = ["thought_mdp_with_examples", "thought_mdp_no_examples"]

    # --- Phase 1: Collect all (question, prefix, thought) tuples ---
    # We also track where each tuple came from so we can put results back
    all_questions = []
    all_prefixes = []
    all_thoughts = []
    # Index tracking: (prob_idx, method_name, rollout_list_idx, thought_idx)
    index_map = []

    for prob_idx, problem in enumerate(problems):
        question = problem["question"]

        for method_name in method_names:
            if method_name not in problem["methods"]:
                continue
            method_data = problem["methods"][method_name]
            if "error" in method_data:
                continue

            rollouts = method_data["rollouts"]
            if max_rollouts_per_method:
                rollouts = rollouts[:max_rollouts_per_method]

            for r_list_idx, rollout in enumerate(rollouts):
                output = rollout["output"]
                thoughts = parse_thoughts(output)

                for t_idx, thought in enumerate(thoughts):
                    prefix = thoughts[:t_idx]
                    all_questions.append(question)
                    all_prefixes.append(prefix)
                    all_thoughts.append(thought)
                    index_map.append((prob_idx, method_name, r_list_idx, t_idx))

    logger.info(f"\n{'='*60}")
    logger.info(f"BATCHED EVALUATION: {len(all_thoughts)} thoughts across {len(problems)} problems")
    logger.info(f"{'='*60}")

    # --- Phase 2: Single batched evaluation call ---
    if all_thoughts:
        eval_results = evaluator.batch_evaluate(
            batch_generate_fn=batch_generate_fn,
            questions=all_questions,
            thought_prefixes=all_prefixes,
            new_thoughts=all_thoughts,
        )
    else:
        eval_results = []

    # --- Phase 3: Organize results back into per-problem, per-method, per-rollout ---
    # Build a nested dict to collect thought evals
    # organized[prob_idx][method_name][r_list_idx] = list of thought eval dicts
    organized: Dict[int, Dict[str, Dict[int, List[Dict[str, Any]]]]] = {}

    for (prob_idx, method_name, r_list_idx, t_idx), eval_result in zip(index_map, eval_results):
        if prob_idx not in organized:
            organized[prob_idx] = {}
        if method_name not in organized[prob_idx]:
            organized[prob_idx][method_name] = {}
        if r_list_idx not in organized[prob_idx][method_name]:
            organized[prob_idx][method_name][r_list_idx] = []

        organized[prob_idx][method_name][r_list_idx].append({
            "thought_idx": t_idx,
            "thought": eval_result.thought,
            "dimension_scores": eval_result.dimension_scores,
            "composite_score": eval_result.composite_score,
            "prefix_length": t_idx,  # prefix is thoughts[:t_idx]
            "raw_evaluation": eval_result.raw_evaluation,
        })

    # --- Phase 4: Build final results structure ---
    results = []
    for prob_idx, problem in enumerate(problems):
        problem_results = {
            "problem_idx": problem["problem_idx"],
            "question": problem["question"],
            "ground_truth": problem["ground_truth"],
            "methods": {}
        }

        for method_name in method_names:
            if method_name not in problem["methods"]:
                continue
            method_data = problem["methods"][method_name]
            if "error" in method_data:
                continue

            rollouts = method_data["rollouts"]
            if max_rollouts_per_method:
                rollouts = rollouts[:max_rollouts_per_method]

            evaluated_rollouts = []
            for r_list_idx, rollout in enumerate(rollouts):
                thought_evals = (
                    organized.get(prob_idx, {}).get(method_name, {}).get(r_list_idx, [])
                )

                if thought_evals:
                    scores = [e["composite_score"] for e in thought_evals]
                    avg_score = sum(scores) / len(scores)
                    min_score = min(scores)
                else:
                    avg_score = 0.0
                    min_score = 0.0

                evaluated_rollouts.append({
                    "rollout_idx": rollout["rollout_idx"],
                    "is_correct": rollout["is_correct"],
                    "extracted_answer": rollout["extracted_answer"],
                    "num_thoughts": len(thought_evals),
                    "thoughts": thought_evals,
                    "avg_composite_score": avg_score,
                    "min_composite_score": min_score,
                })

                logger.info(
                    f"  Problem {prob_idx+1} {method_name} rollout {rollout['rollout_idx']}: "
                    f"avg={avg_score:.3f}, min={min_score:.3f}"
                )

            problem_results["methods"][method_name] = {
                "rollouts": evaluated_rollouts,
                "n_rollouts": len(evaluated_rollouts),
            }

        results.append(problem_results)

    return {
        "config": {
            "model": data["config"]["model"],
            "evaluator": "ThoughtEvaluator (batched)",
            "source_file": f"thought_mdp_comparison_{data['config']['model']}.json",
        },
        "results": results,
    }


def print_summary(eval_results: Dict[str, Any]):
    """Print summary statistics."""
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*60}")

    for method_name in ["thought_mdp_with_examples", "thought_mdp_no_examples"]:
        all_scores = []
        correct_scores = []
        incorrect_scores = []

        for problem in eval_results["results"]:
            if method_name not in problem["methods"]:
                continue

            for rollout in problem["methods"][method_name]["rollouts"]:
                avg_score = rollout["avg_composite_score"]
                all_scores.append(avg_score)

                if rollout["is_correct"]:
                    correct_scores.append(avg_score)
                else:
                    incorrect_scores.append(avg_score)

        if all_scores:
            logger.info(f"\n{method_name}:")
            logger.info(f"  Total rollouts: {len(all_scores)}")
            logger.info(f"  Avg composite score (all): {sum(all_scores)/len(all_scores):.3f}")

            if correct_scores:
                logger.info(f"  Avg score (correct answers): {sum(correct_scores)/len(correct_scores):.3f} (n={len(correct_scores)})")
            if incorrect_scores:
                logger.info(f"  Avg score (incorrect answers): {sum(incorrect_scores)/len(incorrect_scores):.3f} (n={len(incorrect_scores)})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Thought-MDP outputs using ThoughtEvaluator"
    )

    # Model args
    parser.add_argument("--model", type=str, default="llama3b",
                        help="Model name (must match existing outputs file)")
    parser.add_argument("--gpus", type=str, default="0",
                        help="GPU IDs (comma-separated)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size")

    # Data args
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g. gpqa, mathlv5). If set, loads thought_mdp_{dataset}_{model}.json")

    # Eval args
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Max problems to evaluate (for quick testing)")
    parser.add_argument("--max-rollouts", type=int, default=None,
                        help="Max rollouts per method (for quick testing)")
    parser.add_argument("--problem-start", type=int, default=0,
                        help="Start problem index for sharding (default: 0)")
    parser.add_argument("--problem-end", type=int, default=None,
                        help="End problem index for sharding (default: all)")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: auto-generated)")

    args = parser.parse_args()

    # Load pre-generated outputs
    logger.info(f"Loading outputs for model: {args.model}, dataset: {args.dataset or 'default'}")
    try:
        data = load_thought_mdp_outputs(args.model, dataset=args.dataset)
    except FileNotFoundError as e:
        logger.error(str(e))
        available = list((SCPO_DIR / "tests" / "outputs").glob("thought_mdp_*.json"))
        logger.info(f"Available files: {[f.name for f in available]}")
        return

    logger.info(f"Loaded {len(data['results'])} problems")

    # Initialize model
    logger.info(f"\nInitializing model: {args.model}")
    logger.info(f"GPUs: {args.gpus}")

    manager = initialize_model(
        gpu_ids=args.gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        model_name=args.model,
    )

    try:
        # Run evaluation
        eval_results = run_evaluation(
            manager=manager,
            data=data,
            max_problems=args.max_problems,
            max_rollouts_per_method=args.max_rollouts,
            problem_start=args.problem_start,
            problem_end=args.problem_end,
        )

        # Print summary
        print_summary(eval_results)

        # Save results
        output_path = args.output
        if output_path is None:
            output_dir = SCPO_DIR / "tests" / "outputs"
            if args.dataset:
                output_path = output_dir / f"thought_evaluations_{args.dataset}_{args.model}.json"
            else:
                output_path = output_dir / f"thought_evaluations_{args.model}.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(eval_results, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")

    finally:
        logger.info("\nUnloading model...")
        manager.unload_base_model()


if __name__ == "__main__":
    main()
