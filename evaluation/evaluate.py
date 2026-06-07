"""
Post-training evaluation for ThoughtGRPOTrainer.

Loads a trained model checkpoint and evaluates on val/test splits using
batched depth-synchronized thought generation:
1. Wrap HF model in HFGenerateAdapter
2. Call generate_thought_chains_batched for all problems at once
3. Extract answers, check against ground truth
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StopStringCriteria, StoppingCriteriaList
from peft import PeftModel

from .data_loader import DataSplit
from .thought_mdp import generate_thought_chains_batched

logger = logging.getLogger(__name__)

THOUGHT_DELIMITER = "</thought>"


def _load_math_reward():
    """Load verl's math_reward module directly, bypassing verl's heavy __init__."""
    import importlib.util
    import sysconfig
    site_packages = sysconfig.get_path("purelib")
    math_reward_path = Path(site_packages) / "verl/utils/reward_score/math_reward.py"
    spec = importlib.util.spec_from_file_location("math_reward", math_reward_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_math_reward = None

def _mr():
    global _math_reward
    if _math_reward is None:
        _math_reward = _load_math_reward()
    return _math_reward


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} answer from text, handling nested braces."""
    boxed_str = _mr().last_boxed_only_string(text)
    if boxed_str is not None:
        return _mr().remove_boxed(boxed_str)
    return None


def check_answer(extracted: Optional[str], ground_truth: str) -> bool:
    """Check if extracted answer matches ground truth using verl's math equivalence."""
    if extracted is None:
        return False
    return _mr().is_equiv(extracted, ground_truth)


class HFGenerateAdapter:
    """Wraps HF model to match manager.generate() interface for generate_thought_chains_batched."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def generate(self, prompts, max_tokens, temperature, stop, min_tokens=None, n=1):
        results = []
        stop_criteria = StoppingCriteriaList([
            StopStringCriteria(tokenizer=self.tokenizer, stop_strings=stop),
        ])
        do_sample = temperature > 0
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            stopping_criteria=stop_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            for s in stop:
                if s in text:
                    text = text[:text.index(s)]
            text = text.strip()
            for special in [self.tokenizer.eos_token, self.tokenizer.pad_token]:
                if special and text.endswith(special):
                    text = text[:-len(special)].strip()
            results.append(text)
        return results


def evaluate_split(
    model,
    tokenizer,
    split: DataSplit,
    max_steps: int = 8,
    max_tokens_per_thought: int = 512,
    use_thought_examples: bool = True,
) -> Dict:
    """
    Evaluate a model on a data split using batched depth-synchronized generation.

    Args:
        model: HF model (already on device)
        tokenizer: Tokenizer
        split: DataSplit with problems containing "problem" and "answer"
        max_steps: Max thought steps per problem
        max_tokens_per_thought: Max tokens per thought
        use_thought_examples: Use prompt template with in-context examples

    Returns:
        Dict with accuracy, n_correct, n_total, per-problem results
    """
    questions, ground_truths = [], []
    for problem in split.problems:
        q, a = problem.get("problem", ""), str(problem.get("answer", ""))
        if q and a:
            questions.append(q)
            ground_truths.append(a)

    if not questions:
        return {"split": split.name, "accuracy": 0.0, "n_correct": 0, "n_total": 0, "results": []}

    model.eval()
    adapter = HFGenerateAdapter(model, tokenizer)
    all_chains = generate_thought_chains_batched(
        manager=adapter, questions=questions, n_rollouts=1,
        use_examples=use_thought_examples, max_thoughts=max_steps,
        max_tokens_per_thought=max_tokens_per_thought, temperature=0.0,
    )

    results, n_correct = [], 0
    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        thoughts = all_chains[i][0]
        full_traj = f" {THOUGHT_DELIMITER} ".join(thoughts)
        extracted = extract_boxed_answer(full_traj)
        correct = check_answer(extracted, gt)
        if correct:
            n_correct += 1
        results.append({"problem": q[:200], "ground_truth": gt, "extracted": extracted,
                        "correct": correct, "n_thoughts": len(thoughts),
                        "has_boxed": any("\\boxed{" in t for t in thoughts),
                        "thoughts": thoughts})

    accuracy = n_correct / len(questions)
    logger.info(f"  [{split.name}] {n_correct}/{len(questions)} = {accuracy:.4f}")
    return {"split": split.name, "accuracy": accuracy, "n_correct": n_correct,
            "n_total": len(questions), "results": results}


def evaluate_checkpoint(
    model_name_or_path: str,
    output_dir: str,
    val_split: Optional[DataSplit] = None,
    test_split: Optional[DataSplit] = None,
    max_steps: int = 8,
    max_tokens_per_thought: int = 512,
    torch_dtype: str = "bfloat16",
    use_thought_examples: bool = True,
) -> Dict:
    """
    Load a trained checkpoint and evaluate on val/test splits.

    Uses batched depth-synchronized generation to produce trajectories,
    then checks if the final \\boxed{} answer is correct.

    Args:
        model_name_or_path: Base model name (for tokenizer + base weights)
        output_dir: Training output dir (contains checkpoint-* and/or adapter)
        val_split: Validation DataSplit (skipped if None or empty)
        test_split: Test DataSplit (skipped if None or empty)
        max_steps: Max thought steps per problem
        max_tokens_per_thought: Max tokens per thought
        torch_dtype: Torch dtype string
        use_thought_examples: Use prompt template with in-context examples

    Returns:
        Dict with val_results and test_results
    """
    logger.info("=" * 60)
    logger.info("POST-TRAINING EVALUATION (batched thought generation)")
    logger.info("=" * 60)

    # Find the latest checkpoint
    output_path = Path(output_dir)
    checkpoints = sorted(output_path.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)

    if checkpoints:
        checkpoint_dir = str(checkpoints[-1])
        logger.info(f"Loading checkpoint: {checkpoint_dir}")
    else:
        checkpoint_dir = output_dir
        logger.info(f"No checkpoint-* found, using output_dir: {output_dir}")

    # Resolve dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model on CPU first to avoid DTensor issues from prior distributed training
    adapter_config = Path(checkpoint_dir) / "adapter_config.json"
    if adapter_config.exists():
        logger.info(f"Loading base model on CPU for LoRA merge...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="cpu",
        )
        logger.info(f"Loading LoRA adapter from {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir, device_map="cpu")
        model = model.merge_and_unload()
        model = model.to(device="cuda")
        logger.info("Merged LoRA adapter into base model (on CUDA)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
        )

    all_results = {}

    for name, split in [("val", val_split), ("test", test_split)]:
        if split is None or len(split) == 0:
            logger.info(f"Skipping {name} evaluation (no data)")
            continue

        logger.info(f"Evaluating {name} split ({len(split)} problems)...")
        split_results = evaluate_split(
            model=model,
            tokenizer=tokenizer,
            split=split,
            max_steps=max_steps,
            max_tokens_per_thought=max_tokens_per_thought,
            use_thought_examples=use_thought_examples,
        )
        all_results[f"{name}_results"] = split_results

        logger.info(
            f"{name.upper()}: {split_results['n_correct']}/{split_results['n_total']} "
            f"= {split_results['accuracy']:.4f}"
        )

    # Save results to disk
    results_path = output_path / "eval_results.json"
    serializable = {
        k: {kk: vv for kk, vv in v.items() if kk != "results"}
        for k, v in all_results.items()
    }
    results_path.write_text(json.dumps(serializable, indent=2))
    logger.info(f"Saved eval summary to {results_path}")

    # Save detailed per-problem results
    detailed_path = output_path / "eval_results_detailed.json"
    detailed_path.write_text(json.dumps(all_results, indent=2))
    logger.info(f"Saved detailed results to {detailed_path}")

    return all_results
