#!/usr/bin/env python3
"""Plot val accuracy vs step for SRPO (SRPO-l2new) / RRPO (SRPO-rand) / GRPO (TGRPO) on LCB medium.

Reads val curves from wandb output.log files (one per run/seed), aggregates across
seeds (mean + std), and saves a PNG + PDF.

Usage:
    python scripts/plot_val_curves.py
    python scripts/plot_val_curves.py --out plots/val_curves_lcbm
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WANDB_DIR = Path(__file__).resolve().parents[1] / "wandb"
PAPER_FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

# (method, seed) -> wandb run id (suffix after the timestamp)
RUNS = {
    # GRPO = vanilla TGRPO (3 seeds, 2-epoch runs; we slice steps 1..11)
    ("grpo", 42):  "b81579ti",
    ("grpo", 0):   "r9qi5u1q",
    ("grpo", 420): "up38atb3",
    # SRPO = SRPO + L2new self-localization (1-epoch with logging fix)
    ("srpo", 42):  None,  # filled at runtime by job id 86327
    ("srpo", 0):   None,  # job 86415
    ("srpo", 420): None,  # job 86416
    # RRPO = SRPO + random parent (1-epoch with logging fix)
    ("rrpo", 42):  None,  # job 86328
    ("rrpo", 0):   None,  # job 86417
    ("rrpo", 420): None,  # job 86418
}

# Slurm job-name → (method, seed). We look up wandb run id by reading the
# matching <jobname>.err file in batch_scripts/logs/.
JOBNAME_TO_KEY = {
    "srpo_olmo7b_lcbm_l2n_ep1":      ("srpo", 42),
    "srpo_olmo7b_lcbm_l2n_ep1_s0":   ("srpo", 0),
    "srpo_olmo7b_lcbm_l2n_ep1_s420": ("srpo", 420),
    "srpo_olmo7b_lcbm_rand_ep1":     ("rrpo", 42),
    "srpo_olmo7b_lcbm_rand_ep1_s0":  ("rrpo", 0),
    "srpo_olmo7b_lcbm_rand_ep1_s420":("rrpo", 420),
}

LOG_DIR = Path(__file__).resolve().parents[1] / "batch_scripts" / "logs"
WANDB_URL_RE = re.compile(r"wandb\.ai/[a-z\-]+/[a-z_0-9]+/runs/([a-z0-9]+)")


def resolve_run_ids():
    """Read the wandb run id from each <jobname>.err file."""
    for jobname, key in JOBNAME_TO_KEY.items():
        if RUNS[key] is not None:
            continue
        err = LOG_DIR / f"{jobname}.err"
        if not err.exists():
            continue
        try:
            txt = err.read_text(errors="ignore")
        except Exception:
            continue
        m = WANDB_URL_RE.search(txt)
        if m:
            RUNS[key] = m.group(1)


def find_run_dir(run_id: str) -> Path | None:
    matches = list(WANDB_DIR.glob(f"run-*-{run_id}"))
    return matches[0] if matches else None


METRIC_RES = {
    "val":          re.compile(r"val-core/livecodebench_medium/reward/mean@1:([0-9.]+)"),
    "train_reward": re.compile(r"critic/score/mean:([0-9.]+)"),
}
METRIC_YLABEL = {
    "val":          "Test cases passed (%)",
    "train_reward": "Mean reward — test cases passed (%)",
}
STEP_LINE_RE = re.compile(r"^step:(\d+) ")


def load_curve(run_id: str, metric: str) -> dict[int, float]:
    """Return {step: value} for a single run, for the chosen metric."""
    run_dir = find_run_dir(run_id)
    if run_dir is None:
        print(f"  ! no wandb dir for {run_id}")
        return {}
    log = run_dir / "files" / "output.log"
    if not log.exists():
        print(f"  ! no output.log for {run_id}")
        return {}

    rx = METRIC_RES[metric]
    out = {}
    initial_val = None
    for line in log.read_text(errors="ignore").splitlines():
        if line.startswith("step:0 "):
            # Pre-training validation row (no training reward yet, only val)
            if metric == "val":
                vals = rx.findall(line)
                if vals:
                    initial_val = float(vals[-1])
            continue
        m = STEP_LINE_RE.match(line)
        if not m:
            continue
        step = int(m.group(1))
        vals = rx.findall(line)
        if not vals:
            continue
        out[step] = float(vals[-1])
    if initial_val is not None and 0 not in out:
        out[0] = initial_val
    return out


def aggregate(curves: list[dict[int, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, mean, std, n) — per-step union across seeds.

    At each step we average over the seeds that have data there. n[i] tells you
    how many seeds contributed to the value at step[i]. This handles in-flight
    runs cleanly: a step with only 1 seed shows up but the std band collapses.
    """
    if not curves:
        return np.array([]), np.array([]), np.array([]), np.array([])
    all_steps = sorted({s for c in curves for s in c})
    means, stds, ns = [], [], []
    for s in all_steps:
        vals = [c[s] for c in curves if s in c]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        ns.append(len(vals))
    return np.array(all_steps), np.array(means), np.array(stds), np.array(ns)


METHOD_STYLE = {
    "srpo": {"color": "#1f6feb", "marker": "o", "label": "SRPO"},
    "rrpo": {"color": "#f0883e", "marker": "s", "label": "RRPO"},
    "grpo": {"color": "#6e7681", "marker": "^", "label": "GRPO"},
}


def _set_publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=list(METRIC_RES), default="val",
                    help="Which curve to plot (val pass@1 or train mean reward)")
    ap.add_argument("--out", default=None,
                    help="Output basename (without extension); default: plots/<metric>_curves_lcb_medium")
    ap.add_argument("--max-step", type=int, default=11,
                    help="Cap x-axis at this step (default: end of epoch 1)")
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated seeds to include (default: all). e.g. --seeds 42")
    ap.add_argument("--title", default="",
                    help="Plot title (default: empty — set explicitly to add one)")
    args = ap.parse_args()
    seed_filter = (
        {int(s) for s in args.seeds.split(",")} if args.seeds else None
    )
    if args.out is None:
        args.out = f"plots/{args.metric}_curves_lcb_medium"

    _set_publication_style()
    resolve_run_ids()

    by_method = defaultdict(list)
    for (method, seed), rid in RUNS.items():
        if seed_filter is not None and seed not in seed_filter:
            continue
        if rid is None:
            print(f"[skip] {method} s{seed}: no run id yet")
            continue
        curve = load_curve(rid, args.metric)
        if not curve:
            print(f"[skip] {method} s{seed} ({rid}): empty curve")
            continue
        print(f"[ok]   {method} s{seed} ({rid}): {len(curve)} points, "
              f"steps {min(curve)}..{max(curve)}")
        by_method[method].append(curve)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    # Stash plotted curves so we can draw "speed-up" annotations after the loop.
    plotted = {}  # method -> (steps_pct_x, mean_pct_y)

    # All methods start from the same base checkpoint at step 0; any per-method
    # difference there is eval noise. Anchor every method to a shared baseline.
    step0_vals = []
    for curves in by_method.values():
        for curve in curves:
            if 0 in curve:
                step0_vals.append(curve[0])
    shared_step0_pct = (float(np.mean(step0_vals)) * 100) if step0_vals else None

    # plot grpo first so srpo (the headline method) lands on top
    for method in ("grpo", "rrpo", "srpo"):
        curves = by_method.get(method, [])
        if not curves:
            continue
        steps, mean, std, n = aggregate(curves)
        if len(steps) == 0:
            continue
        mask = steps <= args.max_step
        steps, mean, std, n = steps[mask], mean[mask], std[mask], n[mask]
        s = METHOD_STYLE[method]
        n_max = int(n.max())
        label = s["label"]
        # Reward is in [0,1] per problem; render as percentage for readability.
        mean_pct = mean * 100
        sem_pct = (std / np.sqrt(np.maximum(n, 1))) * 100
        if shared_step0_pct is not None and len(steps) and steps[0] == 0:
            mean_pct[0] = shared_step0_pct
            sem_pct[0] = 0.0
        ax.plot(
            steps, mean_pct,
            color=s["color"], linewidth=1.8, label=label, zorder=3,
        )
        # SE band only where ≥2 seeds contributed; band collapses where n=1.
        band_mask = n > 1
        if band_mask.any():
            ax.fill_between(
                steps, mean_pct - sem_pct, mean_pct + sem_pct,
                where=band_mask, color=s["color"], alpha=0.16, linewidth=0, zorder=2,
            )
        plotted[method] = (steps.astype(float), mean_pct.astype(float))

    # Speed-up annotations: from each comparator's final point, draw a horizontal
    # dashed line back to where SRPO first reaches that score (linear interp).
    if "srpo" in plotted:
        srpo_x, srpo_y = plotted["srpo"]
        for comp in ("grpo", "rrpo"):
            if comp not in plotted:
                continue
            cx, cy = plotted[comp]
            target_y = float(cy[-1])
            target_x_end = float(cx[-1])
            # First SRPO step where the curve crosses target_y
            cross_x = None
            for i in range(1, len(srpo_x)):
                y0, y1 = srpo_y[i - 1], srpo_y[i]
                if (y0 < target_y <= y1) or (y0 >= target_y > y1):
                    # linear interpolation between (srpo_x[i-1], y0) and (srpo_x[i], y1)
                    if y1 == y0:
                        cross_x = srpo_x[i]
                    else:
                        cross_x = srpo_x[i - 1] + (target_y - y0) * (srpo_x[i] - srpo_x[i - 1]) / (y1 - y0)
                    break
            if cross_x is None or cross_x >= target_x_end:
                continue
            comp_color = METHOD_STYLE[comp]["color"]
            # horizontal "score-matched at this step" connector at the target score
            ax.hlines(target_y, cross_x, target_x_end,
                      colors=comp_color, linestyles=(0, (4, 3)),
                      linewidth=1.2, alpha=0.85, zorder=2.5)
            # small marker at the SRPO crossing on the dashed connector
            ax.plot([cross_x], [target_y], marker="o", markersize=4,
                    markerfacecolor="white",
                    markeredgecolor=comp_color,
                    markeredgewidth=1.2, zorder=4)
            # vertical dotted droppers from the crossing down to the bracket level
            y_bracket = 7.0 if comp == "rrpo" else 2.0
            ax.vlines(cross_x, y_bracket, target_y,
                      colors=comp_color, linestyles=(0, (1, 2)),
                      linewidth=1.0, alpha=0.7, zorder=2.4)
            ax.vlines(target_x_end, y_bracket, target_y,
                      colors=comp_color, linestyles=(0, (1, 2)),
                      linewidth=1.0, alpha=0.7, zorder=2.4)
            # |--| bracket below: short caps + horizontal connector
            cap = 0.55  # vertical half-height of the bracket caps in data units
            ax.vlines(cross_x,        y_bracket - cap, y_bracket + cap,
                      colors=comp_color, linewidth=1.4, alpha=0.95, zorder=4)
            ax.vlines(target_x_end,   y_bracket - cap, y_bracket + cap,
                      colors=comp_color, linewidth=1.4, alpha=0.95, zorder=4)
            ax.hlines(y_bracket, cross_x, target_x_end,
                      colors=comp_color, linewidth=1.4, alpha=0.95, zorder=4)
            # speedup label sits just above the bracket
            speedup = target_x_end / cross_x if cross_x > 0 else float("inf")
            label_x = (cross_x + target_x_end) / 2
            ax.annotate(
                f"~{speedup:.1f}× speedup",
                xy=(label_x, y_bracket),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10, color=comp_color, fontweight="bold", zorder=5,
            )

    ax.set_xlabel("Training step")
    ax.set_ylabel(METRIC_YLABEL[args.metric])
    if args.title:
        ax.set_title(args.title, pad=8)
    ax.set_xlim(-0.3, args.max_step + 0.3)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, args.max_step + 1, 1))
    ax.minorticks_off()
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    leg = ax.legend(
        handles, labels,
        loc="upper left", frameon=True, framealpha=0.95,
        edgecolor="#cccccc", fancybox=False, borderpad=0.6,
    )
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300)
        print(f"saved {path}")

    # Mirror the val PDF into the paper's figures dir so \includegraphics
    # finds the latest version automatically. Only val plots go to the paper —
    # train_reward and other diagnostics stay under plots/.
    if args.metric == "val":
        PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        paper_pdf = PAPER_FIGURES_DIR / f"{out_base.name}.pdf"
        fig.savefig(paper_pdf, dpi=300)
        print(f"saved {paper_pdf}")


if __name__ == "__main__":
    main()
