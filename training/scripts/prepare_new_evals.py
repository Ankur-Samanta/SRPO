"""Prepare 4 new eval datasets from HuggingFace as VERL parquet files.

Datasets (all short-answer math, no MCQ):
  - IMO-AnswerBench (OpenEvals/IMO-AnswerBench, 400 problems)
  - HMMT Nov 2025   (MathArena/hmmt_nov_2025, 30 problems)
  - AMO-Bench       (meituan-longcat/AMO-Bench, 50 problems)
  - Apex Shortlist  (MathArena/apex-shortlist, 48 problems)

Output schema matches existing eval parquets (aime, acereason, etc.):
  prompt:       [{"role": "user", "content": <problem>}]
  data_source:  str
  reward_model: {"ground_truth": <answer>}
  extra_info:   {"index": int, "unique_id": str}

Usage:
    python training/scripts/prepare_new_evals.py
"""

from pathlib import Path
import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = Path.home() / "data" / "rlhf" / "eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _write(rows, out_name):
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / f"{out_name}.parquet"
    df.to_parquet(out)
    print(f"  {out_name}: {len(df)} rows -> {out}")


def prepare_imo_answerbench():
    ds = load_dataset("OpenEvals/IMO-AnswerBench", split="train")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "prompt": [{"role": "user", "content": ex["Problem"]}],
            "data_source": "imo_answerbench",
            "reward_model": {"ground_truth": str(ex["Short Answer"])},
            "extra_info": {"index": i, "unique_id": f"eval_imo_ab_{i}"},
        })
    _write(rows, "imo_answerbench")


def prepare_hmmt_nov_2025():
    ds = load_dataset("MathArena/hmmt_nov_2025", split="train")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "prompt": [{"role": "user", "content": ex["problem"]}],
            "data_source": "hmmt_nov_2025",
            "reward_model": {"ground_truth": str(ex["answer"])},
            "extra_info": {"index": i, "unique_id": f"eval_hmmt_nov25_{i}"},
        })
    _write(rows, "hmmt_nov_2025")


def prepare_amo_bench():
    ds = load_dataset("meituan-longcat/AMO-Bench", split="test")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "data_source": "amo_bench",
            "reward_model": {"ground_truth": str(ex["answer"])},
            "extra_info": {
                "index": i,
                "unique_id": f"eval_amo_{i}",
                "answer_type": ex.get("answer_type", ""),
            },
        })
    _write(rows, "amo_bench")


def prepare_apex_shortlist():
    ds = load_dataset("MathArena/apex-shortlist", split="train")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "prompt": [{"role": "user", "content": ex["problem"]}],
            "data_source": "apex_shortlist",
            "reward_model": {"ground_truth": str(ex["answer"])},
            "extra_info": {"index": i, "unique_id": f"eval_apex_{i}"},
        })
    _write(rows, "apex_shortlist")


if __name__ == "__main__":
    print("Preparing eval datasets...")
    prepare_imo_answerbench()
    prepare_hmmt_nov_2025()
    prepare_amo_bench()
    prepare_apex_shortlist()
