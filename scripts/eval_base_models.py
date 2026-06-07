#!/usr/bin/env python3
"""Evaluate base or trained model on the MATH-500 test split.

Usage:
  # Base model eval
  python scripts/eval_base_models.py allenai/OLMo-3-7B-Instruct

  # Trained checkpoint eval (loads LoRA adapter + merges)
  python scripts/eval_base_models.py allenai/OLMo-3-7B-Instruct --checkpoint experiments/thought_grpo_olmo-3-7b-instruct
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from evaluation.data_loader import load_dataset_splits
from evaluation.evaluate import evaluate_split


def load_model(model_name, checkpoint_dir=None):
    """Load base model or trained checkpoint (with LoRA merge)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if checkpoint_dir:
        # Find latest checkpoint subfolder
        ckpt_path = Path(checkpoint_dir)
        checkpoints = sorted(ckpt_path.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        ckpt = str(checkpoints[-1]) if checkpoints else checkpoint_dir

        adapter_config = Path(ckpt) / "adapter_config.json"
        if adapter_config.exists():
            logger.info(f"Loading base model on CPU for LoRA merge...")
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cpu")
            logger.info(f"Loading LoRA adapter from {ckpt}")
            model = PeftModel.from_pretrained(model, ckpt, device_map="cpu")
            model = model.merge_and_unload()
            model = model.to(device="cuda")
            logger.info("Merged LoRA adapter into base model (on CUDA)")
        else:
            logger.info(f"No adapter_config.json in {ckpt}, loading as full model")
            model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        logger.info(f"Loading base model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="HF model name (e.g. allenai/OLMo-3-7B-Instruct)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint dir (omit for base model eval)")
    parser.add_argument("--dataset", default="math500", help="Dataset name (math500, aime, math_level5, etc.)")
    parser.add_argument("--n-problems", type=int, default=None, help="Number of problems (None = all)")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    # Determine output dir
    short_name = args.model_name.split("/")[-1].lower().replace(".", "_")
    dataset_tag = args.dataset
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.checkpoint:
        output_dir = Path(args.checkpoint)
    else:
        output_dir = Path(f"experiments/base_eval_{short_name}_{dataset_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test split
    n_problems = args.n_problems or (500 if args.dataset == "math500" else None)
    data_splits = load_dataset_splits(
        dataset=args.dataset, n_problems=n_problems,
        train_ratio=0.8, val_ratio=0.0, test_ratio=0.2, seed=42,
    )
    test_split = data_splits.test
    logger.info(f"Test split: {len(test_split)} problems")

    # Load model
    model, tokenizer = load_model(args.model_name, args.checkpoint)

    # Evaluate
    results = evaluate_split(
        model=model, tokenizer=tokenizer, split=test_split,
        max_steps=8, max_tokens_per_thought=512, use_thought_examples=True,
    )

    # Save
    summary = {k: v for k, v in results.items() if k != "results"}
    (output_dir / "eval_results.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "eval_results_detailed.json").write_text(json.dumps(results, indent=2))

    mode = "TRAINED" if args.checkpoint else "BASE"
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"{mode} {args.model_name}: {results['n_correct']}/{results['n_total']} = {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
