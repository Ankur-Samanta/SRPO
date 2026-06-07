#!/usr/bin/env python3
"""Extract shared scalar training metrics from wandb runs for the 6
NuminaMath-Olympiads ep2 checkpoints (TGRPO / SRPO / SRPO-Rand x
olmo7b). Writes per-run CSVs + a combined long-form CSV.
"""
from __future__ import annotations

import csv
from pathlib import Path

try:
    from wandb.sdk.lib import runid  # noqa: F401
    from wandb.proto import wandb_internal_pb2  # noqa: F401
    from wandb.sdk.internal.datastore import DataStore
except Exception as e:
    raise SystemExit(f"wandb not importable: {e}")

REPO = Path(__file__).resolve().parents[1]
WANDB = REPO / "wandb"

RUNS = {
    "tgrpo_olmo7b":      "run-20260328_020425-h31df6ui",
    "srpo_olmo7b":     "run-20260328_023854-o3j4inwh",
    "srpo_rand_olmo7b":"run-20260410_214057-an8ss5yz",
    "tgrpo_olmo7b_s420":       "run-20260415_161907-6ax7m84u",
    "srpo_olmo7b_s420":      "run-20260415_162230-7bet0toa",
    "srpo_rand_olmo7b_s420": "run-20260415_162140-6zozwlxy",
}

KEYS = [
    "training/global_step",
    "training/epoch",
    "critic/rewards/mean",
    "critic/rewards/max",
    "critic/rewards/min",
    "critic/advantages/mean",
    "critic/score/mean",
    "actor/pg_loss",
    "actor/entropy",
    "actor/kl_loss",
    "actor/ppo_kl",
    "actor/grad_norm",
    "actor/lr",
    "response_length/mean",
    "response_length/max",
    "prompt_length/mean",
    "num_turns/mean",
    "response/aborted_ratio",
    "val-core/numinamath_olympiads/reward/mean@1",
    "val-aux/numinamath_olympiads/math_correct/mean@1",
    "val-aux/numinamath_olympiads/format_reward/mean@1",
    "perf/throughput",
    "perf/time_per_step",
]


def read_run_history(run_dir: Path):
    """Stream history rows from the .wandb file via wandb's DataStore.

    Uses leveldb-framed scan_data() to handle block padding and multi-chunk
    records. Each data blob is a serialized wandb_internal.Record.
    """
    wandb_file = next(run_dir.glob("*.wandb"))
    ds = DataStore()
    ds.open_for_scan(str(wandb_file))
    rows = []
    from wandb.proto import wandb_internal_pb2 as pb
    while True:
        data = ds.scan_data()
        if data is None:
            break
        r = pb.Record()
        try:
            r.ParseFromString(data)
        except Exception:
            continue
        if r.WhichOneof("record_type") == "history":
            item = {}
            for i in r.history.item:
                key = i.key or "/".join(i.nested_key)
                item[key] = i.value_json
            rows.append(item)
    return rows


def main(out_dir: Path):
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = out_dir / "training_stats_long.csv"
    with combined.open("w", newline="") as cf:
        cw = csv.writer(cf)
        cw.writerow(["run", "step", "metric", "value"])
        for label, rdir in RUNS.items():
            path = WANDB / rdir
            if not path.exists():
                print(f"[skip] {label}: {path} missing")
                continue
            rows = read_run_history(path)
            per_run = out_dir / f"{label}.csv"
            with per_run.open("w", newline="") as pf:
                pw = csv.writer(pf)
                pw.writerow(["step"] + KEYS)
                for row in rows:
                    step = json.loads(row.get("_step", "null"))
                    vals = []
                    for k in KEYS:
                        v = row.get(k)
                        vals.append(json.loads(v) if v is not None else "")
                    pw.writerow([step] + vals)
                    for k in KEYS:
                        if k in row:
                            cw.writerow([label, step, k, json.loads(row[k])])
            print(f"[ok] {label}: {len(rows)} history rows -> {per_run}")
    print(f"[ok] combined -> {combined}")


def extract_branches(out_dir: Path):
    """Extract per-rollout ICS stats from logs/srpo_branches/*.jsonl."""
    import json
    bdir = REPO / "logs" / "srpo_branches"
    if not bdir.exists():
        print(f"[skip] branches: {bdir} missing")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    per_rollout = out_dir / "branches_rollouts.csv"
    per_prompt = out_dir / "branches_prompts.csv"
    agg = out_dir / "branches_summary.csv"

    rows_prompt = []
    rows_rollout = []
    for jf in sorted(bdir.glob("branches_pid*.jsonl")):
        with jf.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ics = d.get("ics_stats") or {}
                rows_prompt.append({
                    "file": jf.name,
                    "ts": d.get("ts"),
                    "question_hash": hash(d.get("question", "")),
                    "ground_truth": d.get("ground_truth"),
                    "ics_triggered": ics.get("ics_triggered"),
                    "ics_iterations": ics.get("ics_iterations"),
                    "ics_corrected": ics.get("ics_corrected"),
                    "ics_triggers": ics.get("ics_triggers"),
                    "fresh_chains": ics.get("fresh_chains"),
                    "num_error_steps": len(ics.get("ics_error_steps") or []),
                    "iter_oracle_correct_any": any(ics.get("iter_oracle_correct") or []),
                    "iter_oracle_correct_last": (ics.get("iter_oracle_correct") or [None])[-1],
                    "n_rollouts": len(d.get("rollouts") or []),
                })
                for r in d.get("rollouts") or []:
                    rows_rollout.append({
                        "file": jf.name,
                        "ts": d.get("ts"),
                        "slot": r.get("slot"),
                        "num_thoughts": r.get("num_thoughts"),
                        "found_answer": r.get("found_answer"),
                        "oracle_correct": r.get("oracle_correct"),
                    })
        print(f"[ok] branches: {jf.name}")

    def dump(path, rows):
        if not rows:
            return
        keys = list(rows[0].keys())
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    dump(per_prompt, rows_prompt)
    dump(per_rollout, rows_rollout)

    # aggregate per file
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows_prompt:
        buckets[r["file"]].append(r)
    summary_rows = []
    for fname, rs in buckets.items():
        n = len(rs)
        trig = sum(1 for r in rs if r["ics_triggered"])
        corr = sum(1 for r in rs if r["ics_corrected"])
        mean_iters = (sum(r["ics_iterations"] or 0 for r in rs) / n) if n else 0
        summary_rows.append({
            "file": fname,
            "n_prompts": n,
            "ics_trigger_rate": trig / n if n else 0,
            "ics_correction_rate": corr / trig if trig else 0,
            "mean_ics_iterations": mean_iters,
            "mean_rollouts_per_prompt": (sum(r["n_rollouts"] for r in rs) / n) if n else 0,
        })
    dump(agg, summary_rows)
    print(f"[ok] branches summary -> {agg}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "outputs" / "training_stats"))
    args = ap.parse_args()
    out = Path(args.out)
    main(out)
    extract_branches(out)
