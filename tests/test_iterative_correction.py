#!/usr/bin/env python3
"""
Test script: Run single GPQA problem through TREE's Thought ICS pipeline.

Uses the same defaults as batch_eval.py scripts:
- autonomy_level: 2 (L2 Binary Feedback)
- generation_temp: 0.5
- resample_temp: 0.5
- judge_temp: 0.5
- seed: 42
- max_iterations: 10

Usage:
    python tests/test_iterative_correction.py --model llama3b --gpus 0
    python tests/test_iterative_correction.py --model llama3b --gpus 0,1 --tensor-parallel-size 2
"""

import os
os.environ['VLLM_USE_V1'] = '1'

import sys
from pathlib import Path
import json
import argparse
import logging

# Add TREE to path
TREE_DIR = Path(__file__).parent.parent.parent / "TREE"
sys.path.insert(0, str(TREE_DIR))

# Import from TREE (verbatim)
from tree_of_thought import initialize_model
from iterative_self_correction import iterative_self_correction
from dataset_loaders import load_dataset_by_name, normalize_answer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DEFAULTS FROM BATCH SCRIPTS (submit_l2_jobs.sh)
# =============================================================================
DEFAULT_MODEL = "llama3b"
DEFAULT_DATASET = "gpqa"
DEFAULT_AUTONOMY_LEVEL = 2          # L2 (Binary Feedback)
DEFAULT_GENERATION_TEMP = 0.5
DEFAULT_RESAMPLE_TEMP = 0.5
DEFAULT_JUDGE_TEMP = 0.5
DEFAULT_SEED = 42
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_ERROR_DETECTION = "batch"
DEFAULT_SHARED_PREFIX = True


def load_single_gpqa_problem(problem_idx: int = 0, seed: int = DEFAULT_SEED):
    """Load a single GPQA problem."""
    problems = load_dataset_by_name(
        dataset_name="gpqa",
        n_problems=10,  # Load a few to pick from
        seed=seed,
    )

    if problem_idx >= len(problems):
        problem_idx = 0

    problem = problems[problem_idx]
    logger.info(f"Loaded GPQA problem {problem_idx}: {problem['problem'][:100]}...")
    logger.info(f"Ground truth answer: {problem['answer']}")

    return problem


def run_single_problem_correction(
    manager,
    problem_dict: dict,
    autonomy_level: int = DEFAULT_AUTONOMY_LEVEL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    generation_temp: float = DEFAULT_GENERATION_TEMP,
    resample_temp: float = DEFAULT_RESAMPLE_TEMP,
    judge_temp: float = DEFAULT_JUDGE_TEMP,
    error_detection_method: str = DEFAULT_ERROR_DETECTION,
    shared_prefix: bool = DEFAULT_SHARED_PREFIX,
) -> dict:
    """
    Run iterative self-correction on a single problem.

    Uses TREE's iterative_self_correction() verbatim with batch script defaults.

    Returns:
        Tree result dictionary containing:
        - problem: str
        - ground_truth: str
        - success: bool
        - total_iterations: int
        - iterations: List[dict] with per-iteration data
    """
    problem = problem_dict['problem']
    ground_truth = problem_dict['answer']

    logger.info("="*80)
    logger.info("RUNNING ITERATIVE SELF-CORRECTION")
    logger.info("="*80)
    logger.info(f"Autonomy Level: L{autonomy_level}")
    logger.info(f"Max Iterations: {max_iterations}")
    logger.info(f"Temperatures: gen={generation_temp}, resample={resample_temp}, judge={judge_temp}")
    logger.info(f"Error Detection: {error_detection_method}")
    logger.info(f"Shared Prefix: {shared_prefix}")
    logger.info("="*80)

    # Run TREE's iterative correction (verbatim)
    result = iterative_self_correction(
        manager=manager,
        problem=problem,
        ground_truth=ground_truth,
        L=max_iterations,
        autonomy_level=autonomy_level,
        error_detection_method=error_detection_method,
        shared_prefix=shared_prefix,
        generation_temp=generation_temp,
        resample_temp=resample_temp,
        judge_temp=judge_temp,
        no_auto_stop=False,
        use_context=False,
        verify=False,
        mv_verify=False,
    )

    return result


def print_result_summary(result: dict):
    """Print a summary of the correction result."""
    logger.info("\n" + "="*80)
    logger.info("RESULT SUMMARY")
    logger.info("="*80)
    logger.info(f"Success: {result.get('success', False)}")
    logger.info(f"Total Iterations: {result.get('total_iterations', len(result.get('iterations', [])))}")

    for it in result.get('iterations', []):
        status = "CORRECT" if it.get('correct') else "WRONG"
        error_step = it.get('error_step', '-')
        prefix_len = it.get('prefix_length', '-')
        logger.info(f"  Iter {it['iteration']}: {it['answer'][:50]}... [{status}] (error_step={error_step}, prefix={prefix_len})")

    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Test TREE's iterative correction on a single GPQA problem"
    )

    # Model args (defaults match batch_eval.py scripts)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="GPU IDs (comma-separated, default: 0,1)")
    parser.add_argument("--tensor-parallel-size", type=int, default=2,
                        help="Tensor parallel size (default: 2)")

    # Problem selection
    parser.add_argument("--problem-idx", type=int, default=0,
                        help="Index of GPQA problem to use (default: 0)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for dataset loading (default: {DEFAULT_SEED})")

    # Correction params (match batch script defaults)
    parser.add_argument("--autonomy-level", type=int, default=DEFAULT_AUTONOMY_LEVEL,
                        help=f"Autonomy level 1-4 (default: {DEFAULT_AUTONOMY_LEVEL})")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
                        help=f"Max correction iterations (default: {DEFAULT_MAX_ITERATIONS})")
    parser.add_argument("--generation-temp", type=float, default=DEFAULT_GENERATION_TEMP,
                        help=f"Initial chain generation temperature (default: {DEFAULT_GENERATION_TEMP})")
    parser.add_argument("--resample-temp", type=float, default=DEFAULT_RESAMPLE_TEMP,
                        help=f"Correction regeneration temperature (default: {DEFAULT_RESAMPLE_TEMP})")
    parser.add_argument("--judge-temp", type=float, default=DEFAULT_JUDGE_TEMP,
                        help=f"Error detection temperature (default: {DEFAULT_JUDGE_TEMP})")
    parser.add_argument("--error-detection", type=str, default=DEFAULT_ERROR_DETECTION,
                        choices=["batch", "incremental"],
                        help=f"Error detection method (default: {DEFAULT_ERROR_DETECTION})")
    parser.add_argument("--no-shared-prefix", action="store_true",
                        help="Disable shared prefix (regenerate from scratch)")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for result (optional)")

    args = parser.parse_args()

    # Initialize model
    logger.info("="*80)
    logger.info("INITIALIZING MODEL")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"GPUs: {args.gpus}")
    logger.info(f"Tensor Parallel Size: {args.tensor_parallel_size}")

    manager = initialize_model(
        gpu_ids=args.gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        model_name=args.model,
        model_seed=args.seed,
    )

    try:
        # Load problem
        problem_dict = load_single_gpqa_problem(
            problem_idx=args.problem_idx,
            seed=args.seed,
        )

        # Run correction
        result = run_single_problem_correction(
            manager=manager,
            problem_dict=problem_dict,
            autonomy_level=args.autonomy_level,
            max_iterations=args.max_iterations,
            generation_temp=args.generation_temp,
            resample_temp=args.resample_temp,
            judge_temp=args.judge_temp,
            error_detection_method=args.error_detection,
            shared_prefix=not args.no_shared_prefix,
        )

        # Print summary
        print_result_summary(result)

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Result saved to: {output_path}")

        return result

    finally:
        # Cleanup
        logger.info("Unloading model...")
        manager.unload_base_model()


if __name__ == "__main__":
    result = main()
