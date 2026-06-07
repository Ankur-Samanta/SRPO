#!/usr/bin/env python3
"""Per-step trajectory of the paper's per-token gradient signal g_{i,t}.

Inputs:  scripts/grad_dump_analysis_full_out/per_step.csv
Outputs: figures/grad_signal_trajectory.pdf
         figures/grad_signal_factors.pdf
         figures/grad_signal_outcomes.pdf

Two figures:

(1) grad_signal_trajectory.pdf — headline trajectory of g̅ per group over the
    11-step LCB-medium ep1 training arc, plus their ratio. This is the
    direct training-arc generalization of the single-update visualization in
    the paper (App. H, Fig. grad_tree_p2).

(2) grad_signal_factors.pdf — the multiplicative decomposition
        g̅_i = (|A_i| / T_i) · overline{(1-π)}_i
    showing each factor (|A|, T, (1-π)) per group across training, plus the
    derived per-rollout floor |A|/T and the rate at which group-relative
    advantage normalization collapses to zero (degenerate-group rate).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRPO_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = SRPO_DIR / "scripts" / "grad_dump_analysis_full_out" / "per_step.csv"
FIG_DIR = SRPO_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent colors with paper convention: SP = orange-ish, base = blue-ish.
COL_BASE = "#1f77b4"
COL_SP   = "#d62728"
COL_RATIO = "#444444"


def load_per_step() -> list[dict]:
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in list(r.keys()):
            if k == "step":
                r[k] = int(r[k])
            else:
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
    # Add option-(b) conditional ratio: restrict to qualifying prompts (both
    # groups deliver gradient), but average over ALL 4 rollouts within each
    # group on those prompts (rather than nonzero-adv rollouts only). This
    # makes the conditional and unconditional ratios use the same within-prompt
    # averaging structure; only the prompt-filter differs.
    pp_path = SRPO_DIR / "scripts" / "grad_dump_analysis_full_out" / "per_prompt.csv"
    with open(pp_path) as f:
        pp = list(csv.DictReader(f))
    for r in pp:
        for k in list(r.keys()):
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                pass
    for step_row in rows:
        s = step_row["step"]
        qual = [p for p in pp if int(p["step"]) == s
                and (p["both_nonzero"] in ("True", 1, 1.0) or p["both_nonzero"] is True)]
        if qual:
            sp = sum(p["g_sp"] for p in qual) / len(qual)
            base = sum(p["g_base"] for p in qual) / len(qual)
            step_row["g_sp_over_base_qual"] = (sp / base) if base > 0 else float("nan")
        else:
            step_row["g_sp_over_base_qual"] = float("nan")
    return rows


def plot_trajectory(rows: list[dict]) -> None:
    """Conditional per-token signal ratio across the ep1 training arc.

    Single panel: $\\bar g^{SP}/\\bar g^{base}$ on prompts where both groups
    deliver gradient (the population on which the prefix mask can act).
    """
    steps = [r["step"] + 1 for r in rows]
    ratio_cond = [r["g_sp_over_base_qual"] for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 2.6))
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.plot(steps, ratio_cond, "D-", color=COL_SP, lw=1.5, ms=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient Signal Ratio SP/Base")
    ax.set_xticks(steps)

    fig.tight_layout()
    out = FIG_DIR / "grad_signal_trajectory.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def plot_factors(rows: list[dict]) -> None:
    """Deprecated: factor decomposition is now in plot_trajectory's right panel.

    Kept for backward compatibility / debugging; not rendered in the appendix.
    """
    steps = [r["step"] + 1 for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(8.5, 4.2))
    panels = [
        ("abs_adv",       r"$|\hat A_i|$",                    "advantage magnitude"),
        ("T",             r"$T_i$ (active tokens)",           "active-region length"),
        ("aT",            r"$|\hat A_i| / T_i$",              "per-rollout floor"),
        ("ompi",          r"$\overline{(1-\pi)}_i$",          "self-confidence-complement"),
        ("zero_adv_frac", r"frac. $|\hat A_i| = 0$",          "degenerate-group rate"),
        ("g",             r"$\bar g_i$",                      "per-token signal (recap)"),
    ]
    for ax, (key, ylabel, title) in zip(axes.flat, panels):
        base_vals = [r[f"{key}_base"] for r in rows]
        sp_vals   = [r[f"{key}_sp"]   for r in rows]
        ax.plot(steps, base_vals, "o-", color=COL_BASE, label="base", lw=1.3, ms=3)
        ax.plot(steps, sp_vals,   "s-", color=COL_SP,   label="SP",   lw=1.3, ms=3)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(steps)
        ax.tick_params(labelsize=7)
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 1].set_xlabel("Training Step")
    axes[1, 2].set_xlabel("Training Step")
    axes[0, 0].legend(loc="best", frameon=False, fontsize=8)

    fig.tight_layout()
    out = FIG_DIR / "grad_signal_factors.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def plot_outcomes(rows: list[dict]) -> None:
    """Per-step self-correction pass rate (G2-corr only).

    G2-corr rollouts are conditional on having identified an error in a
    parent rollout's prefix and attempting to repair from the cut point,
    so this is a measurement of self-correction skill specifically — not
    comparable to the unconditional base (G1) pass rate.
    """
    steps   = [r["step"] + 1 for r in rows]  # display as 1..11 (matches verl global_step naming)
    g2_pct  = [r["g2_pass_rate"] * 100 for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 2.6))
    ax.plot(steps, g2_pct, "s-", color=COL_SP, lw=1.5, ms=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Self-Correction %")
    ax.set_xticks(steps)
    ax.set_ylim(0, max(g2_pct) * 1.2)

    fig.tight_layout()
    out = FIG_DIR / "grad_signal_outcomes.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def main():
    rows = load_per_step()
    plot_trajectory(rows)
    plot_factors(rows)
    plot_outcomes(rows)


if __name__ == "__main__":
    main()
