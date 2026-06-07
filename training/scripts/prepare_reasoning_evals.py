"""Prepare reasoning eval datasets as VERL parquet files.

Rules:
  - Prefer the dedicated test split; fall back to validation, then train.
  - If the chosen split has <= 500 rows, use all; otherwise random-sample 500 (seed=42).

Benchmarks:
  - MMLU-Pro          (MC, 10-option)            TIGER-Lab/MMLU-Pro
  - TheoremQA         (short answer)             TIGER-Lab/TheoremQA
  - StrategyQA        (yes/no)                   ChilleD/StrategyQA
  - AGIEval           (20 subtasks, 25 each)     hails/agieval-* [per subtask]
  - HotpotQA          (short answer, distractor) hotpot_qa
  - PutnamBench-Lean  (FLAG: needs Lean verifier) trishullab/PutnamBench
  - HumanEval+        (code)                     evalplus/humanevalplus
  - MBPP+             (code)                     evalplus/mbppplus

Schema (matches existing eval parquets):
  prompt:       [{"role": "user", "content": <problem>}]
  data_source:  str
  reward_model: {"ground_truth": <answer>}
  extra_info:   {"index": int, "unique_id": str, ...}

Usage:
    python training/scripts/prepare_reasoning_evals.py
"""

from pathlib import Path
import pandas as pd
from datasets import load_dataset

SEED = 42
CAP = 500
OUTPUT_DIR = Path.home() / "data" / "rlhf" / "eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- helpers

def _write(rows, out_name):
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / f"{out_name}.parquet"
    df.to_parquet(out)
    print(f"  {out_name}: {len(df)} rows -> {out}")


def _load_test_split(repo_id, name=None, split_candidates=("test", "validation", "train")):
    """Load the first available split from `split_candidates`."""
    last_err = None
    for split in split_candidates:
        try:
            return load_dataset(repo_id, name=name, split=split) if name else load_dataset(repo_id, split=split)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"No split available for {repo_id} (name={name}): {last_err}")


def _cap(ds, cap=CAP, seed=SEED):
    if len(ds) <= cap:
        return ds
    return ds.shuffle(seed=seed).select(range(cap))


# ---------------------------------------------------------------- preparers

def prepare_mmlu_pro():
    ds = _load_test_split("TIGER-Lab/MMLU-Pro")
    ds = _cap(ds)
    rows = []
    for i, ex in enumerate(ds):
        options = ex["options"]
        letters = "ABCDEFGHIJ"[: len(options)]
        opts_str = "\n".join(f"({letters[j]}) {opt}" for j, opt in enumerate(options))
        content = f"{ex['question']}\n\n{opts_str}"
        rows.append({
            "prompt": [{"role": "user", "content": content}],
            "data_source": "mmlu_pro",
            "reward_model": {"ground_truth": str(ex.get("answer", ""))},
            "extra_info": {
                "index": i,
                "unique_id": f"eval_mmlupro_{i}",
                "category": ex.get("category", ""),
                "answer_index": ex.get("answer_index", -1),
            },
        })
    _write(rows, "mmlu_pro")


def prepare_theoremqa():
    ds = _load_test_split("TIGER-Lab/TheoremQA")
    ds = _cap(ds)
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "prompt": [{"role": "user", "content": ex["Question"]}],
            "data_source": "theoremqa",
            "reward_model": {"ground_truth": str(ex["Answer"])},
            "extra_info": {
                "index": i,
                "unique_id": f"eval_theoremqa_{i}",
                "answer_type": ex.get("Answer_type", ""),
                "theorem": ex.get("theorem", ""),
            },
        })
    _write(rows, "theoremqa")


def prepare_strategyqa():
    candidates = ["ChilleD/StrategyQA", "voidful/StrategyQA", "wics/strategy-qa"]
    ds = None
    for repo in candidates:
        try:
            ds = _load_test_split(repo)
            print(f"    loaded strategyqa from {repo}")
            break
        except Exception:
            continue
    if ds is None:
        print("  ! strategyqa: no HF id matched; FLAG — verify dataset id")
        return
    ds = _cap(ds)
    rows = []
    for i, ex in enumerate(ds):
        q = ex.get("question") or ex.get("Question") or ""
        a = ex.get("answer")
        if isinstance(a, bool):
            gt = "yes" if a else "no"
        else:
            gt = "yes" if str(a).lower() in ("true", "yes", "1") else "no"
        rows.append({
            "prompt": [{"role": "user", "content": q}],
            "data_source": "strategyqa",
            "reward_model": {"ground_truth": gt},
            "extra_info": {"index": i, "unique_id": f"eval_strategyqa_{i}"},
        })
    _write(rows, "strategyqa")


def prepare_agieval():
    """AGIEval English subtasks, 25 problems each.

    Uses `baber/agieval` which only ships the English configs. Converts
    digit ground truth (0-4) to letter (A-E) so the existing MC verifier
    works. `math_agieval` is dropped (free-form numeric, not MC).
    """
    subtasks = [
        "aqua_rat", "sat_en", "sat_en_wop", "sat_math",
        "lsat_ar", "lsat_lr", "lsat_rc", "logiqa",
    ]
    rows = []
    idx = 0
    for sub in subtasks:
        try:
            from datasets import load_dataset
            ds = load_dataset("baber/agieval", sub, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"    ! agieval-{sub}: not found ({type(e).__name__}), skipping")
            continue
        ds = ds.shuffle(seed=SEED).select(range(min(25, len(ds))))
        for ex in ds:
            q = ex.get("question") or ex.get("query") or ""
            passage = ex.get("passage", "")
            if passage:
                q = f"{passage}\n\n{q}"
            # Options come as a stringified list of pre-labeled choices.
            options = ex.get("options", "")
            if options:
                if isinstance(options, list):
                    opts_str = "\n".join(str(o) for o in options)
                else:
                    opts_str = str(options)
                q = f"{q}\n\n{opts_str}"
            a = ex.get("label") or ex.get("gold") or ex.get("answer") or ""
            if isinstance(a, list) and a:
                a = a[0]
            try:
                gt_letter = "ABCDEFGHIJ"[int(a)]
            except (ValueError, TypeError, IndexError):
                gt_letter = str(a).upper()
            rows.append({
                "prompt": [{"role": "user", "content": str(q)}],
                "data_source": "agieval",
                "reward_model": {"ground_truth": gt_letter},
                "extra_info": {
                    "index": idx,
                    "unique_id": f"eval_agieval_{idx}",
                    "subtask": sub,
                },
            })
            idx += 1
    _write(rows, "agieval")


def prepare_hotpotqa():
    # HotpotQA test split is unlabeled (leaderboard-held); use validation (distractor).
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    ds = _cap(ds)
    rows = []
    for i, ex in enumerate(ds):
        ctx = ex["context"]
        titles = ctx.get("title", [])
        sents = ctx.get("sentences", [])
        context = "\n\n".join(f"{t}: {' '.join(s)}" for t, s in zip(titles, sents))
        content = f"{context}\n\nQuestion: {ex['question']}"
        rows.append({
            "prompt": [{"role": "user", "content": content}],
            "data_source": "hotpotqa",
            "reward_model": {"ground_truth": str(ex["answer"])},
            "extra_info": {
                "index": i,
                "unique_id": f"eval_hotpotqa_{i}",
                "type": ex.get("type", ""),
                "level": ex.get("level", ""),
            },
        })
    _write(rows, "hotpotqa")


def prepare_humaneval_plus():
    ds = _load_test_split("evalplus/humanevalplus")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "data_source": "humaneval_plus",
            "reward_model": {"ground_truth": ""},
            "extra_info": {
                "index": i,
                "unique_id": f"eval_humanevalplus_{i}",
                "task_id": ex.get("task_id", ""),
                "entry_point": ex.get("entry_point", ""),
                "canonical_solution": ex.get("canonical_solution", ""),
                "test": ex.get("test", ""),
            },
        })
    _write(rows, "humaneval_plus")


def prepare_mbpp_plus():
    ds = _load_test_split("evalplus/mbppplus")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "data_source": "mbpp_plus",
            "reward_model": {"ground_truth": ""},
            "extra_info": {
                "index": i,
                "unique_id": f"eval_mbppplus_{i}",
                "task_id": ex.get("task_id", ""),
                "code": ex.get("code", ""),
                "test": ex.get("test", ""),
            },
        })
    _write(rows, "mbpp_plus")


def prepare_putnambench_lean():
    """FLAG: PutnamBench formal Lean version.

    Ground truth is a Lean proof, not a string-matchable answer. Verification
    requires running the Lean compiler on each candidate proof.

    Options:
      1. Skip PutnamBench entirely.
      2. Integrate a Lean verifier into the reward pipeline.
      3. Use a natural-language Putnam subset with numeric answers (if desired,
         change this function to load that instead).

    Stub left as placeholder; uncomment + adapt once verification strategy is set.
    """
    print("  ! putnambench_lean: SKIPPED (needs Lean verifier; see function docstring)")
    return

    # Example scaffold if Lean integration becomes available:
    # ds = _load_test_split("trishullab/PutnamBench")
    # ds = _cap(ds)
    # rows = []
    # for i, ex in enumerate(ds):
    #     rows.append({
    #         "prompt": [{"role": "user", "content": ex["informal_statement"]}],
    #         "data_source": "putnambench_lean",
    #         "reward_model": {"ground_truth": ex["formal_statement"]},
    #         "extra_info": {
    #             "index": i,
    #             "unique_id": f"eval_putnam_{i}",
    #             "formal_statement": ex["formal_statement"],
    #         },
    #     })
    # _write(rows, "putnambench_lean")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    print("Preparing reasoning eval datasets...")
    prepare_mmlu_pro()
    prepare_theoremqa()
    prepare_strategyqa()
    prepare_agieval()
    prepare_hotpotqa()
    prepare_humaneval_plus()
    prepare_mbpp_plus()
    prepare_putnambench_lean()
