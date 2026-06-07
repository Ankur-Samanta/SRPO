"""Evaluate a thought-GRPO / SCPO checkpoint with pass@k.

Loads a base model + optional LoRA adapter into vLLM, generates k rollouts
per test problem using the same thought-by-thought loop as training, and
computes pass@1, pass@5, pass@10.

Usage:
    python scripts/eval_pass_at_k.py \
        --base-model allenai/OLMo-3-7B-Instruct \
        --lora-adapter checkpoints/thought_grpo/mathlvl5_olmo7b_ics/global_step_24/actor/lora_adapter \
        --test-data ~/data/rlhf/math_level5/test.parquet \
        --k 10 \
        --temperature 0.7 \
        --output-dir outputs/eval/mathlvl5_olmo7b_ics \
        --use-examples false
"""

import argparse
import asyncio
import json
import math
import os
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ── Answer grading (identical to verl's math_reward.py) ─────────────────────

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a_str, b_str = string.split("/")
    try:
        a = int(a_str)
        b = int(b_str)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_substr = "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def is_equiv(str1, str2):
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return strip_string(str1) == strip_string(str2)
    except Exception:
        return str1 == str2


def grade_answer(solution_str, ground_truth):
    """Returns 1.0 if correct, 0.0 otherwise. Same as verl's compute_score."""
    try:
        boxed = last_boxed_only_string(solution_str)
        if boxed is not None:
            answer = remove_boxed(boxed)
            if is_equiv(answer, ground_truth):
                return 1.0
    except Exception:
        pass
    return 0.0


# ── pass@k estimator (unbiased, from Chen et al. 2021) ──────────────────────

def pass_at_k(n, c, k):
    """Unbiased estimator for pass@k given n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ── Prompt templates (same as training) ─────────────────────────────────────

def get_prompt_template(use_examples: bool) -> str:
    if use_examples:
        return """You are solving a problem step-by-step.

Instructions:
1. State your next reasoning step (one observation, calculation, or deduction)
2. End each thought with </thought>
3. Continue until you reach the final answer, then write it in \\boxed{{answer}} format

Examples:

Q: In how many ways can 5 distinct books be arranged on a shelf if 2 specific books must not be adjacent?
Total arrangements without restrictions is 5! = 120</thought>
I need to subtract arrangements where the 2 specific books ARE adjacent</thought>
If I treat the 2 books as a single unit, I have 4 units to arrange: 4! = 24 ways</thought>
The 2 books within their unit can be arranged in 2! = 2 ways</thought>
So arrangements with the books adjacent = 24 x 2 = 48</thought>
Therefore, arrangements where they are NOT adjacent = 120 - 48 = \\boxed{{72}}</thought>

Q: A rectangle has area 48 and perimeter 28. What is the length of its diagonal?
Let length = l and width = w. From the area: lw = 48</thought>
From the perimeter: 2l + 2w = 28, so l + w = 14</thought>
From l + w = 14, we get w = 14 - l. Substituting into lw = 48: l(14 - l) = 48</thought>
Expanding: 14l - l^2 = 48, so l^2 - 14l + 48 = 0. Factoring: (l - 6)(l - 8) = 0</thought>
So l = 8 and w = 6 (or vice versa). Using the Pythagorean theorem: d^2 = 8^2 + 6^2 = 64 + 36 = 100</thought>
Therefore d = 10, so the answer is \\boxed{{10}}</thought>

Q: {question}
"""
    else:
        return """You are solving a problem by producing one reasoning step at a time.

Do not try to solve the entire problem at once. Given the previously taken steps, think about what the single next step should be, then articulate it clearly and conclude just that step with </thought>.

Each step should be a complete, self-contained thought — one observation, calculation, or deduction that:
- Makes forward progress toward the solution
- Contains substantive reasoning (not filler like "let me think" or restating the problem)
- Coheres logically with the previous steps

When your next step arrives at the final answer, include \\boxed{{answer}} and end with </thought>.

Q: {question}
"""


# ── Thought-by-thought generation ───────────────────────────────────────────

def generate_thought_chain(
    llm: LLM,
    prompts: list[str],
    temperature: float,
    max_thoughts: int = 20,
    max_tokens_per_thought: int = 256,
    response_length: int = 5120,
    thought_delimiter: str = "</thought>",
    lora_request=None,
) -> list[str]:
    """Generate thought chains for a batch of prompts, one thought at a time.

    Mirrors ThoughtAgentLoop._generate_thoughts_core() from training.
    Returns the full concatenated response for each prompt.
    """
    # Track state per prompt
    n = len(prompts)
    current_texts = list(prompts)  # running text so far
    responses = [""] * n           # accumulated response tokens
    active = [True] * n            # still generating?
    prompt_lens = [len(p) for p in prompts]

    for thought_idx in range(max_thoughts):
        # Collect active prompts
        active_indices = [i for i in range(n) if active[i]]
        if not active_indices:
            break

        active_texts = [current_texts[i] for i in active_indices]

        params = SamplingParams(
            temperature=temperature,
            top_p=0.95 if temperature > 0 else 1.0,
            max_tokens=max_tokens_per_thought,
            stop=[thought_delimiter],
            include_stop_str_in_output=True,
        )

        outputs = llm.generate(active_texts, params, use_tqdm=False, lora_request=lora_request)

        for idx, out in zip(active_indices, outputs):
            generated = out.outputs[0].text
            responses[idx] += generated
            current_texts[idx] += generated

            resp_len = len(responses[idx])

            # Check termination conditions
            if "\\boxed{" in generated:
                active[idx] = False
            elif resp_len >= response_length:
                active[idx] = False
            elif out.outputs[0].finish_reason == "abort":
                active[idx] = False

    return responses


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate pass@k for thought-GRPO/SCPO checkpoints")
    parser.add_argument("--base-model", required=True, help="HuggingFace model ID or path")
    parser.add_argument("--lora-adapter", default=None, help="Path to LoRA adapter dir (omit for base model eval)")
    parser.add_argument("--test-data", required=True, help="Path to test parquet file")
    parser.add_argument("--k", type=int, default=10, help="Number of samples per problem")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-thoughts", type=int, default=20, help="Max thoughts per chain")
    parser.add_argument("--max-tokens-per-thought", type=int, default=256, help="Max tokens per thought")
    parser.add_argument("--response-length", type=int, default=5120, help="Max total response length in chars")
    parser.add_argument("--use-examples", type=str, default="false", help="Use in-context examples in prompt")
    parser.add_argument("--output-dir", required=True, help="Directory for results")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vLLM GPU memory utilization")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of prompts per vLLM batch")
    parser.add_argument("--max-problems", type=int, default=None, help="Limit evaluation to first N problems (for smoke tests)")
    args = parser.parse_args()

    use_examples = args.use_examples.lower() in ("true", "1", "yes")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config["use_examples_parsed"] = use_examples
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Load model ──────────────────────────────────────────────────────
    print(f"Loading model: {args.base_model}")
    if args.lora_adapter:
        print(f"  LoRA adapter: {args.lora_adapter}")

    print(f"Loading vLLM with model: {args.base_model}")
    llm = LLM(
        model=args.base_model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
        enable_prefix_caching=True,
        enable_lora=args.lora_adapter is not None,
        max_lora_rank=64,
    )

    if args.lora_adapter:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("adapter", 1, args.lora_adapter)
    else:
        lora_request = None

    # ── Load test data ──────────────────────────────────────────────────
    df = pd.read_parquet(args.test_data)
    print(f"Loaded {len(df)} test problems from {args.test_data}")

    template = get_prompt_template(use_examples)

    problems = []
    for _, row in df.iterrows():
        question = row["prompt"][0]["content"]  # chat format: [{"role": "user", "content": ...}]
        ground_truth = row["reward_model"]["ground_truth"]
        problems.append({"question": question, "ground_truth": ground_truth})

    if args.max_problems is not None:
        problems = problems[:args.max_problems]
        print(f"Limiting to first {len(problems)} problems (--max-problems)")

    # ── Generate k rollouts per problem ─────────────────────────────────
    print(f"\nGenerating {args.k} rollouts per problem (temp={args.temperature})...")
    all_results = []
    t0 = time.time()

    for sample_idx in range(args.k):
        print(f"\n  Sample {sample_idx + 1}/{args.k}")

        # Format prompts
        prompts = [template.format(question=p["question"]) for p in problems]

        # Generate in batches
        all_responses = []
        for batch_start in range(0, len(prompts), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            batch_responses = generate_thought_chain(
                llm=llm,
                prompts=batch_prompts,
                temperature=args.temperature,
                max_thoughts=args.max_thoughts,
                max_tokens_per_thought=args.max_tokens_per_thought,
                response_length=args.response_length,
                lora_request=lora_request,
            )
            all_responses.extend(batch_responses)
            print(f"    Batch {batch_start//args.batch_size + 1}: "
                  f"{batch_end - batch_start} prompts done")

        # Grade
        for prob_idx, (prob, response) in enumerate(zip(problems, all_responses)):
            correct = grade_answer(response, prob["ground_truth"])
            all_results.append({
                "problem_idx": prob_idx,
                "sample_idx": sample_idx,
                "question": prob["question"],
                "ground_truth": prob["ground_truth"],
                "response": response,
                "correct": correct,
            })

    elapsed = time.time() - t0
    print(f"\nGeneration complete in {elapsed:.0f}s")

    # ── Save raw results ────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "rollouts.jsonl")
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(all_results)} rollouts to {results_path}")

    # ── Compute pass@k ──────────────────────────────────────────────────
    n_problems = len(problems)
    k_values = [1, 5, 10]

    # Count correct per problem
    correct_counts = [0] * n_problems
    for r in all_results:
        correct_counts[r["problem_idx"]] += int(r["correct"])

    metrics = {}
    for k in k_values:
        if k > args.k:
            continue
        scores = [pass_at_k(args.k, c, k) for c in correct_counts]
        metrics[f"pass@{k}"] = sum(scores) / len(scores)

    # Also compute raw accuracy (fraction of all rollouts that are correct)
    total_correct = sum(correct_counts)
    metrics["raw_accuracy"] = total_correct / len(all_results)
    metrics["n_problems"] = n_problems
    metrics["k"] = args.k
    metrics["total_rollouts"] = len(all_results)

    # ── Print and save metrics ──────────────────────────────────────────
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Model:    {args.base_model}")
    print(f"Adapter:  {args.lora_adapter or 'none (base model)'}")
    print(f"Test set: {args.test_data} ({n_problems} problems)")
    print(f"Samples:  {args.k} per problem, temp={args.temperature}")
    print(f"Examples: {'yes' if use_examples else 'no'}")
    print("-" * 50)
    for k in k_values:
        if f"pass@{k}" in metrics:
            print(f"  pass@{k:>2}: {metrics[f'pass@{k}']:.4f}")
    print(f"  raw_acc: {metrics['raw_accuracy']:.4f}")
    print("=" * 50)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
