#!/usr/bin/env python3
"""Reconstruct loc_grader prompts from srpo branch dumps.

For each ICS-triggered record in <dump_dir>/branches_*.jsonl, emits one
JSONL line per localization in the same schema as srpo_ep1_s42_prompts.jsonl:
    {id, rec_idx, sub_idx, prompt, local_step, local_n_steps, local_reasoning}

Usage:
    python scripts/build_grader_prompts.py \
        --dump-dir logs/srpo_localizations/lcb_medium_olmo7b_srpo_l2new_ep1_s0 \
        --out      logs/loc_grader/srpo_ep1_s0_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_emitted = 0
    n_records = 0
    n_triggered = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for jsonl in sorted(dump_dir.glob("branches_*.jsonl")):
            for rec_idx, line in enumerate(jsonl.read_text(errors="ignore").splitlines()):
                n_records += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                ics = rec.get("ics_stats") or {}
                if not ics.get("ics_triggered"):
                    continue
                n_triggered += 1
                prompts = ics.get("ics_loc_prompts") or []
                steps = ics.get("ics_loc_error_steps") or []
                ns = ics.get("ics_loc_n_steps") or []
                reasonings = ics.get("ics_loc_reasonings") or []
                for sub_idx, prompt in enumerate(prompts):
                    fp.write(json.dumps({
                        "id": f"{rec_idx:04d}_{sub_idx}",
                        "rec_idx": rec_idx,
                        "sub_idx": sub_idx,
                        "prompt": prompt,
                        "local_step": steps[sub_idx] if sub_idx < len(steps) else None,
                        "local_n_steps": ns[sub_idx] if sub_idx < len(ns) else None,
                        "local_reasoning": reasonings[sub_idx] if sub_idx < len(reasonings) else None,
                    }) + "\n")
                    n_emitted += 1

    print(f"records scanned: {n_records}  ics_triggered: {n_triggered}  prompts emitted: {n_emitted}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
