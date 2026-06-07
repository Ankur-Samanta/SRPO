"""Probe different localization prompts on saved wrong rollouts.

Loads examples from loc_prompt_examples.json, formats each under every
registered prompt variant, runs offline vLLM inference (matching the
training-time call -- raw completion, add_special_tokens=True, no chat
template), parses the predicted error step with the same parser used
in training, and writes results to JSONL for inspection.

The "l2_default" variant is the L2 prompt used in training verbatim
(see thought_ics_agent_loop.py:1070-1083).

Usage:
    python training/tests/probe_localization_prompts.py \\
        --model allenai/OLMo-3-7B-Instruct \\
        --variants l2_default l2_terse l2_selfcheck \\
        --out loc_probe_olmo7b.jsonl

Note: run as a plain script (not `python -m ...`). The training
package `__init__.py` imports ThoughtAgentLoop, which pulls in verl and
its tensordict native extension; running as a script skips that.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Callable

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


# Inlined from ThoughtICSAgentLoop._parse_error_step (thought_ics_agent_loop.py:1546).
# Duplicated here so the probe doesn't drag in the full verl import chain.
def parse_error_step(response_text: str, chain_length: int) -> int:
    matches = list(re.finditer(r"\\boxed\{", response_text))
    if matches:
        start_pos = matches[-1].end()
        brace_count = 1
        i = start_pos
        while i < len(response_text) and brace_count > 0:
            if response_text[i] == "{":
                brace_count += 1
            elif response_text[i] == "}":
                brace_count -= 1
            i += 1
        if brace_count == 0:
            boxed = response_text[start_pos : i - 1].strip()
            try:
                step_num = int(boxed)
                if step_num == 0:
                    return 0
                if 1 <= step_num <= chain_length:
                    return step_num
            except (ValueError, TypeError):
                pass
    for num_str in re.findall(r"\b(\d+)\b", response_text):
        num = int(num_str)
        if 1 <= num <= chain_length:
            return num
    return max(1, chain_length // 2)


THOUGHT_DELIMITER = "</thought>"
LOC_TEMPERATURE = 0.3
LOC_MAX_TOKENS = 2048


def build_chain_text(decoded_thoughts: list[str]) -> str:
    chain_text = ""
    for i, thought in enumerate(decoded_thoughts, 1):
        clean = thought.replace(THOUGHT_DELIMITER, "").strip()
        chain_text += f"\nStep {i}: {clean}"
    return chain_text


# --------------------------------------------------------------------------
# Prompt variants. Each takes (question, chain_text, n_steps) -> prompt str.
# "l2_default" MUST match thought_ics_agent_loop.py:1070-1083 verbatim.
# --------------------------------------------------------------------------

def l2_default(question: str, chain_text: str, n_steps: int) -> str:
    return (
        f"Problem: {question}\n\n"
        f"Current reasoning chain (WRONG - got incorrect answer):\n"
        f"{chain_text}\n\n"
        f"Your answer is incorrect. Analyze the reasoning chain step "
        f"by step to identify where the error occurred. Which step "
        f"number (1 to {n_steps}) contains the first critical error "
        f"(logical flaw, arithmetic error, or incorrect assumption)?"
        f"\n\n"
        f"Do NOT solve the problem again. Your ONLY task is to "
        f"identify the first erroneous step. Provide your reasoning, "
        f"then put ONLY the step number (an integer from 1 to "
        f"{n_steps}) in the format: \\boxed{{step_number}}\n"
    )


def l2_terse(question: str, chain_text: str, n_steps: int) -> str:
    return (
        f"Problem: {question}\n\n"
        f"Wrong reasoning:\n{chain_text}\n\n"
        f"Find the first step (1..{n_steps}) that is wrong. "
        f"Answer as \\boxed{{N}}.\n"
    )


def l2_selfcheck(question: str, chain_text: str, n_steps: int) -> str:
    return (
        f"Problem: {question}\n\n"
        f"A student produced this reasoning, which is incorrect:\n"
        f"{chain_text}\n\n"
        f"For each step from 1 to {n_steps}, briefly state whether it is "
        f"correct or where it deviates. Then pick the FIRST step that "
        f"contains a critical error and output it as \\boxed{{N}}. "
        f"Do not re-solve the problem.\n"
    )


VARIANTS: dict[str, Callable[[str, str, int], str]] = {
    "l2_default": l2_default,
    "l2_terse": l2_terse,
    "l2_selfcheck": l2_selfcheck,
}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--examples", default=str(Path(__file__).parent / "loc_prompt_examples.json"))
    p.add_argument("--model", default="allenai/OLMo-3-7B-Instruct")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    p.add_argument("--temperature", type=float, default=LOC_TEMPERATURE)
    p.add_argument("--max-tokens", type=int, default=LOC_MAX_TOKENS)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="loc_probe_results.jsonl")
    p.add_argument(
        "--raw-completion",
        action="store_true",
        help="Skip chat template and feed prompts as raw completion (matches training invocation verbatim). "
             "For chat-tuned models like OLMo-Instruct this often yields empty outputs.",
    )
    args = p.parse_args()

    for v in args.variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; available: {list(VARIANTS)}")

    with open(args.examples) as f:
        examples = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    use_chat = (not args.raw_completion) and bool(getattr(tokenizer, "chat_template", None))
    print(f"prompt mode: {'chat_template' if use_chat else 'raw_completion'}")

    # Build all (example, variant) prompts up front, batch them through vLLM
    jobs = []
    for ex in examples:
        chain_text = build_chain_text(ex["decoded_thoughts"])
        n_steps = ex["n_steps"]
        for variant_name in args.variants:
            prompt_text = VARIANTS[variant_name](ex["question"], chain_text, n_steps)
            if use_chat:
                prompt_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            else:
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
            jobs.append({
                "example_id": ex["id"],
                "variant": variant_name,
                "n_steps": n_steps,
                "ground_truth": ex["ground_truth"],
                "question": ex["question"],
                "prompt_ids": prompt_ids,
                "prompt_len": len(prompt_ids),
            })

    print(f"Loaded {len(examples)} examples × {len(args.variants)} variants = {len(jobs)} generations")
    print(f"Prompt-length range: {min(j['prompt_len'] for j in jobs)} - {max(j['prompt_len'] for j in jobs)} tokens")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=j["prompt_ids"]) for j in jobs],
        sampling_params=sp,
    )

    with open(args.out, "w") as fout:
        for job, out in zip(jobs, outputs):
            gen = out.outputs[0]
            response_text = gen.text
            predicted_step = parse_error_step(response_text, job["n_steps"])
            rec = {
                "example_id": job["example_id"],
                "variant": job["variant"],
                "n_steps": job["n_steps"],
                "ground_truth": job["ground_truth"],
                "predicted_step": predicted_step,
                "finish_reason": gen.finish_reason,
                "n_output_tokens": len(gen.token_ids),
                "prompt_len": job["prompt_len"],
                "response_text": response_text,
            }
            fout.write(json.dumps(rec) + "\n")
            print(
                f"[ex {job['example_id']:>2d} | {job['variant']:<14s}] "
                f"step {predicted_step}/{job['n_steps']}  "
                f"(out_tokens={len(gen.token_ids)}, finish={gen.finish_reason})"
            )

    print(f"\nWrote {len(jobs)} rows to {args.out}")


if __name__ == "__main__":
    main()
