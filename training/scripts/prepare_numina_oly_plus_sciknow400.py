"""Build a mixed train.parquet that combines numinamath_olympiads (400) with
SciKnowEval (100 per subject × 4 subjects = 400), giving 800 training problems.

The 100 sciknow problems per subject are taken from L3 rows that are NOT in
the existing 500-problem eval split — produced by replicating the same
seed-42 shuffle used in prepare_datasets.py and slicing rows 500:600.

Outputs:
    ~/data/rlhf/numina_oly_plus_sciknow400/train.parquet  (800 rows)
    ~/data/rlhf/numina_oly_plus_sciknow400/test.parquet   (= numina_oly test)
"""
import os
from pathlib import Path

import pandas as pd


def _load_sciknoweval_l3(domain: str | None = None) -> list[dict]:
    """Inlined from prepare_datasets._load_sciknoweval_l3 (avoids triggering
    the training package __init__, which fails on this host's tensordict
    install). Loads SciKnowEval v2 test split, filters to L3 + optional domain."""
    from datasets import load_dataset
    label = f"L3/{domain}" if domain else "L3"
    print(f"  Loading hicai-zju/SciKnowEval v2, filtering to {label}...")
    ds = load_dataset("hicai-zju/SciKnowEval", "v2", split="test")
    problems = []
    for row in ds:
        details = row.get("details", {})
        if details.get("level") != "L3":
            continue
        if domain and row.get("domain") != domain:
            continue
        qtype = row.get("type", "")
        answer_key = row.get("answerKey", "")
        answer = row.get("answer", "")
        problem_text = row["question"]
        choices = row.get("choices", {})
        if "mcq" in qtype.lower() and choices:
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            option_lines = [f"{l}. {t}" for l, t in zip(labels, texts)]
            if option_lines:
                problem_text = f"{problem_text}\n\n" + "\n".join(option_lines)
        gt = answer_key if ("mcq" in qtype.lower() and answer_key) else answer
        problems.append({
            "problem": problem_text,
            "answer": gt,
            "unique_id": f"sciknoweval_l3_{len(problems)}",
            "subject": row.get("domain", ""),
            "level": "L3",
            "type": qtype,
            "answerKey": answer_key,
            "domain": row.get("domain", ""),
        })
    print(f"  {len(problems)} L3 problems loaded")
    return problems

DATA_BASE = Path(os.path.expanduser("~/data/rlhf"))
OUT_DIR = DATA_BASE / "numina_oly_plus_sciknow400"
SEED = 42
SUBJECTS = {
    "Chemistry": "sciknoweval_chemistry",
    "Physics":   "sciknoweval_physics",
    "Biology":   "sciknoweval_biology",
    "Material":  "sciknoweval_materials",
}
PER_SUBJECT_TRAIN = 100
EVAL_PREFIX = 500   # eval already used the first 500 of each shuffle


def build_sciknow_train_rows() -> list[dict]:
    """Pull 100 disjoint L3 problems per subject and emit verl-format rows."""
    rows = []
    for domain, ds_name in SUBJECTS.items():
        problems = _load_sciknoweval_l3(domain=domain)
        # Replicate the shuffle/slice from prepare_datasets.to_verl_parquet
        df = pd.DataFrame([
            {
                "prompt": [{"role": "user", "content": p["problem"]}],
                "data_source": ds_name,
                "reward_model": {"ground_truth": p["answer"]},
                "extra_info": {
                    "answerKey": p.get("answerKey", ""),
                    "domain": p.get("domain", ""),
                    "level": p.get("level", "L3"),
                    "subject": p.get("subject", ""),
                    "type": p.get("type", ""),
                    "unique_id": p["unique_id"],
                },
            }
            for p in problems
        ])
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        train_slice = df.iloc[EVAL_PREFIX : EVAL_PREFIX + PER_SUBJECT_TRAIN]
        if len(train_slice) < PER_SUBJECT_TRAIN:
            raise RuntimeError(
                f"{ds_name}: only {len(train_slice)} L3 problems available after "
                f"the 500-row eval prefix (need {PER_SUBJECT_TRAIN})"
            )
        print(f"  {ds_name}: pulled {len(train_slice)} disjoint L3 problems "
              f"(rows {EVAL_PREFIX}:{EVAL_PREFIX+PER_SUBJECT_TRAIN})")
        rows.extend(train_slice.to_dict(orient="records"))
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading numinamath_olympiads train.parquet...")
    numina_train = pd.read_parquet(DATA_BASE / "numinamath_olympiads" / "train.parquet")
    print(f"  {len(numina_train)} rows")

    print("[2/3] Pulling 100×4 disjoint SciKnowEval-L3 problems...")
    sciknow_rows = build_sciknow_train_rows()
    sciknow_train = pd.DataFrame(sciknow_rows)
    print(f"  {len(sciknow_train)} rows total")

    combined = pd.concat([numina_train, sciknow_train], ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"[3/3] Combined train: {len(combined)} rows (interleaved, seed={SEED})")

    train_out = OUT_DIR / "train.parquet"
    combined.to_parquet(train_out)
    print(f"  wrote {train_out}")

    # Test = numinamath olympiads test, copied unchanged (eval stays math-only)
    test_src = DATA_BASE / "numinamath_olympiads" / "test.parquet"
    test_out = OUT_DIR / "test.parquet"
    pd.read_parquet(test_src).to_parquet(test_out)
    print(f"  wrote {test_out} (= numinamath_olympiads test, unchanged)")


if __name__ == "__main__":
    main()
