#!/usr/bin/env python3
"""Analyze thought trajectories using ThoughtEvaluator.

Loads cached eval results (with thought chains), runs ThoughtEvaluator on each
trajectory using a specified model, and compares base vs trained outputs.

Usage:
  # Evaluate base vs trained Qwen trajectories using base Qwen as evaluator
  CUDA_VISIBLE_DEVICES=6 python scripts/analyze_trajectories.py \
      --base-results experiments/base_eval_qwen2_5-7b-instruct/eval_results_detailed.json \
      --trained-results experiments/thought_grpo_olmo-3-7b-instruct/eval_results_detailed.json \
      --evaluator-model allenai/OLMo-3-7B-Instruct \
      --output-dir experiments/trajectory_analysis_qwen

  # Use a trained checkpoint as evaluator instead
  CUDA_VISIBLE_DEVICES=6 python scripts/analyze_trajectories.py \
      --base-results experiments/base_eval_qwen2_5-7b-instruct/eval_results_detailed.json \
      --trained-results experiments/thought_grpo_olmo-3-7b-instruct/eval_results_detailed.json \
      --evaluator-model allenai/OLMo-3-7B-Instruct \
      --evaluator-checkpoint experiments/thought_grpo_olmo-3-7b-instruct \
      --output-dir experiments/trajectory_analysis_qwen_trained_eval
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from evaluation.thought_evaluator import ThoughtEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_evaluator_model(model_name, checkpoint_dir=None):
    """Load model for use as evaluator."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if checkpoint_dir:
        ckpt_path = Path(checkpoint_dir)
        checkpoints = sorted(ckpt_path.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        ckpt = str(checkpoints[-1]) if checkpoints else checkpoint_dir

        adapter_config = Path(ckpt) / "adapter_config.json"
        if adapter_config.exists():
            logger.info(f"Loading evaluator base model on CPU...")
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cpu")
            logger.info(f"Loading LoRA adapter from {ckpt}")
            model = PeftModel.from_pretrained(model, ckpt, device_map="cpu")
            model = model.merge_and_unload()
            model = model.to(device="cuda")
            logger.info("Merged LoRA adapter — evaluator on CUDA")
        else:
            model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        logger.info(f"Loading evaluator base model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    model.eval()
    return model, tokenizer


def make_batch_generate_fn(model, tokenizer, max_tokens=512):
    """Create a batch_generate_fn for ThoughtEvaluator."""
    device = next(model.parameters()).device

    def batch_generate(prompts):
        results = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(text)
        return results

    return batch_generate


def load_results(path):
    """Load eval results, handling both flat and nested formats."""
    data = json.loads(Path(path).read_text())
    # Handle nested format from evaluate_checkpoint ({"test_results": {..., "results": [...]}})
    if "test_results" in data:
        return data["test_results"]["results"]
    # Flat format from evaluate_split ({"results": [...]})
    if "results" in data:
        return data["results"]
    raise ValueError(f"Unexpected format in {path}")


def evaluate_all_trajectories(results, evaluator, batch_generate_fn, label, batch_size=32):
    """Run ThoughtEvaluator on all trajectories, return per-problem scores."""
    # Flatten all thoughts into one big batch
    all_questions = []
    all_prefixes = []
    all_thoughts = []
    problem_indices = []  # track which problem each thought belongs to

    for i, r in enumerate(results):
        thoughts = r.get("thoughts", [])
        question = r["problem"]
        for t_idx, thought in enumerate(thoughts):
            all_questions.append(question)
            all_prefixes.append(thoughts[:t_idx])
            all_thoughts.append(thought)
            problem_indices.append(i)

    total = len(all_thoughts)
    logger.info(f"[{label}] Evaluating {total} thoughts across {len(results)} problems")

    # Process in batches
    all_eval_results = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_results = evaluator.batch_evaluate(
            batch_generate_fn=batch_generate_fn,
            questions=all_questions[start:end],
            thought_prefixes=all_prefixes[start:end],
            new_thoughts=all_thoughts[start:end],
        )
        all_eval_results.extend(batch_results)
        logger.info(f"  [{label}] {end}/{total} thoughts evaluated")

    # Group back by problem
    per_problem = [[] for _ in range(len(results))]
    for eval_result, prob_idx in zip(all_eval_results, problem_indices):
        per_problem[prob_idx].append(eval_result)

    # Compute per-problem summary
    scored_results = []
    for i, r in enumerate(results):
        thought_scores = per_problem[i]
        if not thought_scores:
            scored_results.append({
                **r,
                "eval_scores": [],
                "mean_composite": 0.0,
                "mean_forward_progress": 0.0,
                "mean_substantiveness": 0.0,
                "mean_coherence": 0.0,
            })
            continue

        composites = [s.composite_score for s in thought_scores]
        fp = [s.dimension_scores.get("forward_progress", 0.5) for s in thought_scores]
        sub = [s.dimension_scores.get("substantiveness", 0.5) for s in thought_scores]
        coh = [s.dimension_scores.get("coherence", 0.5) for s in thought_scores]

        scored_results.append({
            **r,
            "eval_scores": [
                {
                    "thought_idx": j,
                    "composite": s.composite_score,
                    "forward_progress": s.dimension_scores.get("forward_progress", 0.5),
                    "substantiveness": s.dimension_scores.get("substantiveness", 0.5),
                    "coherence": s.dimension_scores.get("coherence", 0.5),
                    "raw_evaluation": s.raw_evaluation[:500],
                }
                for j, s in enumerate(thought_scores)
            ],
            "mean_composite": sum(composites) / len(composites),
            "mean_forward_progress": sum(fp) / len(fp),
            "mean_substantiveness": sum(sub) / len(sub),
            "mean_coherence": sum(coh) / len(coh),
        })

    return scored_results


def compare_and_report(base_scored, trained_scored, output_dir):
    """Compare base vs trained and produce analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Match problems by index (same test split, same order)
    assert len(base_scored) == len(trained_scored), \
        f"Mismatched problem counts: {len(base_scored)} vs {len(trained_scored)}"

    categories = {
        "stayed_correct": [],      # correct -> correct
        "stayed_incorrect": [],    # incorrect -> incorrect
        "improved": [],            # incorrect -> correct
        "regressed": [],           # correct -> incorrect
    }

    for i, (b, t) in enumerate(zip(base_scored, trained_scored)):
        entry = {
            "idx": i,
            "problem": b["problem"],
            "ground_truth": b["ground_truth"],
            "base_correct": b["correct"],
            "trained_correct": t["correct"],
            "base_extracted": b["extracted"],
            "trained_extracted": t["extracted"],
            "base_n_thoughts": b["n_thoughts"],
            "trained_n_thoughts": t["n_thoughts"],
            "base_composite": b["mean_composite"],
            "trained_composite": t["mean_composite"],
            "base_forward_progress": b["mean_forward_progress"],
            "trained_forward_progress": t["mean_forward_progress"],
            "base_substantiveness": b["mean_substantiveness"],
            "trained_substantiveness": t["mean_substantiveness"],
            "base_coherence": b["mean_coherence"],
            "trained_coherence": t["mean_coherence"],
            "delta_composite": t["mean_composite"] - b["mean_composite"],
            "delta_forward_progress": t["mean_forward_progress"] - b["mean_forward_progress"],
            "delta_substantiveness": t["mean_substantiveness"] - b["mean_substantiveness"],
            "delta_coherence": t["mean_coherence"] - b["mean_coherence"],
        }

        if b["correct"] and t["correct"]:
            categories["stayed_correct"].append(entry)
        elif not b["correct"] and not t["correct"]:
            categories["stayed_incorrect"].append(entry)
        elif not b["correct"] and t["correct"]:
            categories["improved"].append(entry)
        else:
            categories["regressed"].append(entry)

    # Print summary
    n = len(base_scored)
    print(f"\n{'='*70}")
    print(f"TRAJECTORY ANALYSIS: {n} problems")
    print(f"{'='*70}")
    print(f"  Stayed correct:   {len(categories['stayed_correct']):3d}")
    print(f"  Stayed incorrect: {len(categories['stayed_incorrect']):3d}")
    print(f"  Improved (->correct):   {len(categories['improved']):3d}")
    print(f"  Regressed (->wrong):    {len(categories['regressed']):3d}")

    # Per-category dimension analysis
    for cat_name, entries in categories.items():
        if not entries:
            continue
        print(f"\n--- {cat_name.upper()} ({len(entries)} problems) ---")
        for dim in ["composite", "forward_progress", "substantiveness", "coherence"]:
            base_vals = [e[f"base_{dim}"] for e in entries]
            trained_vals = [e[f"trained_{dim}"] for e in entries]
            deltas = [e[f"delta_{dim}"] for e in entries]
            base_mean = sum(base_vals) / len(base_vals)
            trained_mean = sum(trained_vals) / len(trained_vals)
            delta_mean = sum(deltas) / len(deltas)
            print(f"  {dim:25s}: base={base_mean:.3f}  trained={trained_mean:.3f}  delta={delta_mean:+.3f}")

    # Save full results
    (output_dir / "base_scored.json").write_text(json.dumps(base_scored, indent=2))
    (output_dir / "trained_scored.json").write_text(json.dumps(trained_scored, indent=2))
    (output_dir / "comparison.json").write_text(json.dumps(categories, indent=2))

    # Save improved problems with full thought chains for inspection
    improved_detail = []
    for entry in categories["improved"]:
        idx = entry["idx"]
        improved_detail.append({
            **entry,
            "base_thoughts": base_scored[idx].get("thoughts", []),
            "trained_thoughts": trained_scored[idx].get("thoughts", []),
            "base_eval_scores": base_scored[idx].get("eval_scores", []),
            "trained_eval_scores": trained_scored[idx].get("eval_scores", []),
        })
    (output_dir / "improved_detail.json").write_text(json.dumps(improved_detail, indent=2))

    # Same for regressed
    regressed_detail = []
    for entry in categories["regressed"]:
        idx = entry["idx"]
        regressed_detail.append({
            **entry,
            "base_thoughts": base_scored[idx].get("thoughts", []),
            "trained_thoughts": trained_scored[idx].get("thoughts", []),
            "base_eval_scores": base_scored[idx].get("eval_scores", []),
            "trained_eval_scores": trained_scored[idx].get("eval_scores", []),
        })
    (output_dir / "regressed_detail.json").write_text(json.dumps(regressed_detail, indent=2))

    logger.info(f"Saved analysis to {output_dir}/")
    return categories


def main():
    parser = argparse.ArgumentParser(description="Analyze thought trajectories with ThoughtEvaluator")
    parser.add_argument("--base-results", required=True, help="Path to base model eval_results_detailed.json")
    parser.add_argument("--trained-results", required=True, help="Path to trained model eval_results_detailed.json")
    parser.add_argument("--evaluator-model", required=True, help="HF model name for evaluator")
    parser.add_argument("--evaluator-checkpoint", default=None, help="Checkpoint dir for evaluator (default: use base model)")
    parser.add_argument("--output-dir", required=True, help="Output directory for analysis")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()

    # Load cached trajectories
    logger.info(f"Loading base results from {args.base_results}")
    base_results = load_results(args.base_results)
    logger.info(f"Loading trained results from {args.trained_results}")
    trained_results = load_results(args.trained_results)

    logger.info(f"Base: {len(base_results)} problems, Trained: {len(trained_results)} problems")

    # Verify thoughts are present
    for label, results in [("base", base_results), ("trained", trained_results)]:
        n_with_thoughts = sum(1 for r in results if r.get("thoughts"))
        if n_with_thoughts == 0:
            raise ValueError(f"{label} results have no thought chains — re-run eval with updated evaluate.py")
        logger.info(f"  {label}: {n_with_thoughts}/{len(results)} have thought chains")

    # Load evaluator model
    eval_label = "checkpoint" if args.evaluator_checkpoint else "base"
    logger.info(f"Loading evaluator model ({eval_label})...")
    model, tokenizer = load_evaluator_model(args.evaluator_model, args.evaluator_checkpoint)
    batch_generate_fn = make_batch_generate_fn(model, tokenizer)

    # Create evaluator
    evaluator = ThoughtEvaluator()

    # Score both sets of trajectories
    base_scored = evaluate_all_trajectories(base_results, evaluator, batch_generate_fn, "base", args.batch_size)
    trained_scored = evaluate_all_trajectories(trained_results, evaluator, batch_generate_fn, "trained", args.batch_size)

    # Compare and report
    compare_and_report(base_scored, trained_scored, args.output_dir)


if __name__ == "__main__":
    main()
