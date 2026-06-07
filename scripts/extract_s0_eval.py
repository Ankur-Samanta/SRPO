#!/usr/bin/env python3
"""Extract math_correct/mean@1 for each (model, dataset) pair from eval .out files.

The eval jobs use verl val_only mode. The wandb summary (copied to metrics.json)
only contains one aggregate key, but the .out file prints full validation metrics.
"""
from __future__ import annotations

import re
from pathlib import Path

MODELS = [
    "tgrpo_oly_olmo7b_s0",
    "scgrpo_oly_olmo7b_s0",
    "scgrpo_rand_oly_olmo7b_s0",
]
DATASETS = [
    ("polaris_d2", "d2"),
    ("numinamath_olympiads", "oly"),
    ("aime", "aime"),
    ("acereason_math", "ace"),
    ("sciknoweval_chemistry", "sk_chem"),
    ("sciknoweval_physics", "sk_phys"),
    ("sciknoweval_biology", "sk_bio"),
    ("sciknoweval_materials", "sk_mat"),
]
LOGDIR = Path("batch_scripts/logs")

# Use the single-line "step:0 - val-aux/<ds>/math_correct/mean@1:<v> - ..." format
PAT = re.compile(r"val-aux/([a-z0-9_]+)/math_correct/mean@1:([0-9.]+)")


def main():
    results = {m: {} for m in MODELS}
    for m in MODELS:
        for ds, short in DATASETS:
            out = LOGDIR / f"eval_{m}_{ds}.out"
            if not out.exists():
                continue
            txt = out.read_text(errors="ignore")
            matches = PAT.findall(txt)
            for k, v in matches:
                if k == ds:
                    results[m][short] = float(v)
                    break
    # print table
    shorts = [s for _, s in DATASETS]
    hdr = "| Method | " + " | ".join(shorts) + " |"
    sep = "|" + "|".join(["---"] * (len(shorts) + 1)) + "|"
    print(hdr); print(sep)
    for m in MODELS:
        row = [m] + [f"{results[m].get(s, '—'):.3f}" if isinstance(results[m].get(s), float) else "—" for s in shorts]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
