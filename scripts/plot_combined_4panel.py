#!/usr/bin/env python3
"""Combined 4-panel figure: loc dist | loc deviation | correction by deviation | correction pooled.

Pass --thr T to use effective deviation (meaningful-step adjusted) for panels 2-4;
omit --thr for raw deviation. Panel 1 (loc distribution SRPO vs RRPO) is the same
in both versions.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
DUMP_ROOT = ROOT / "logs/srpo_localizations"
PAPER_FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

# (seed -> (prompts_path, grader_path, dump_dir)) for SRPO panels (b)-(d).
SEED_SOURCES = {
    42:  (ROOT / "logs/loc_grader/srpo_ep1_s42_prompts.jsonl",
          ROOT / "logs/loc_grader/srpo_ep1_s42_opus.jsonl",
          ROOT / "logs/srpo_localizations/lcb_medium_olmo7b_srpo_l2new_ep1"),
    0:   (ROOT / "logs/loc_grader/srpo_ep1_s0_prompts.jsonl",
          ROOT / "logs/loc_grader/srpo_ep1_s0_opus.jsonl",
          ROOT / "logs/srpo_localizations/lcb_medium_olmo7b_srpo_l2new_ep1_s0"),
    420: (ROOT / "logs/loc_grader/srpo_ep1_s420_prompts.jsonl",
          ROOT / "logs/loc_grader/srpo_ep1_s420_opus.jsonl",
          ROOT / "logs/srpo_localizations/lcb_medium_olmo7b_srpo_l2new_ep1_s420"),
}

EXP = {
    ("srpo", 42):  "lcb_medium_olmo7b_srpo_l2new_ep1",
    ("srpo", 0):   "lcb_medium_olmo7b_srpo_l2new_ep1_s0",
    ("srpo", 420): "lcb_medium_olmo7b_srpo_l2new_ep1_s420",
    ("rrpo", 42):  "lcb_medium_olmo7b_srpo_rand_ep1",
    ("rrpo", 0):   "lcb_medium_olmo7b_srpo_rand_ep1_s0",
    ("rrpo", 420): "lcb_medium_olmo7b_srpo_rand_ep1_s420",
}
METHOD_STYLE = {
    "srpo": {"color": "#1f6feb", "label": "SRPO"},
    "rrpo": {"color": "#f0883e", "label": "RRPO"},
}
SIDE_COLOR = {"clean": "#1f6feb", "exact": "#2ca02c", "erroneous": "#cc3344"}

BIN_CONFIGS = {
    "a": [
        (-99, -3, "≤ −3", "clean"),
        (-2,  -2, "−2",   "clean"),
        (-1,  -1, "−1",   "clean"),
        ( 0,   0, "0",    "exact"),
        ( 1,   1, "+1",   "erroneous"),
        ( 2,   2, "+2",   "erroneous"),
        ( 3,  99, "≥ +3", "erroneous"),
    ],
    "fine": [
        (-99, -4, "≤ −4", "clean"),
        *[(d, d, f"−{abs(d)}", "clean") for d in range(-3, 0)],
        ( 0,   0, "0", "exact"),
        *[(d, d, f"+{d}", "erroneous") for d in range(1, 4)],
        ( 4,  99, "≥ +4", "erroneous"),
    ],
}


step_re = re.compile(r"\nStep (\d+)[:\.](.+?)(?=\nStep \d+[:\.]|\nNow,? )", re.DOTALL)
SPECIAL = re.compile(r"<\|[^|]+\|>|<answer>|</answer>|<compute[^>]*>|</compute[^>]*>|</?think>")


def cleaned_words(text):
    return len(re.findall(r"\S+", SPECIAL.sub(" ", text).strip()))


def _set_publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def collect_error_steps(method, relative=True):
    values = []
    for (m, _seed), exp_name in EXP.items():
        if m != method:
            continue
        exp_dir = DUMP_ROOT / exp_name
        if not exp_dir.exists():
            continue
        for jsonl in exp_dir.glob("branches_*.jsonl"):
            for line in jsonl.read_text(errors="ignore").splitlines():
                try: rec = json.loads(line)
                except Exception: continue
                ics = rec.get("ics_stats") or {}
                steps = ics.get("ics_loc_error_steps") or ics.get("ics_error_steps") or []
                n_steps = ics.get("ics_loc_n_steps") or []
                for i, s in enumerate(steps):
                    if relative:
                        if i < len(n_steps) and n_steps[i] > 0:
                            values.append(float(s) / float(n_steps[i]))
                    else:
                        values.append(float(s))
    return values


def load_records(thr, seeds=None):
    """Returns pooled list of dicts with dev, outcomes, ok across the requested seeds.

    Each row also carries `seed` so callers can break down per seed if needed.
    """
    if seeds is None:
        seeds = list(SEED_SOURCES.keys())

    rows = []
    for seed in seeds:
        if seed not in SEED_SOURCES:
            continue
        prompts_path, grader_path, dump_dir = SEED_SOURCES[seed]
        if not (prompts_path.exists() and grader_path.exists() and dump_dir.exists()):
            print(f"  [skip] seed {seed}: missing prompts/grader/dump files")
            continue
        prompts = {(p['rec_idx'], p['sub_idx']): p for p in
                   (json.loads(l) for l in prompts_path.read_text().splitlines() if l)}
        grader = {(g['rec_idx'], g['sub_idx']): g for g in
                  (json.loads(l) for l in grader_path.read_text().splitlines() if l)
                  if g.get('frontier_step') is not None}
        dumps = {}
        for f in dump_dir.glob("branches_*.jsonl"):
            for i, line in enumerate(f.read_text(errors="ignore").splitlines()):
                try: rec = json.loads(line)
                except Exception: continue
                ics = rec.get("ics_stats") or {}
                if not ics.get("ics_triggered"): continue
                ioc = ics.get("iter_oracle_correct") or []
                if len(ioc) < 8: continue
                dumps[i] = ioc[4:8]

        seed_rows = 0
        for (i, sub), p in prompts.items():
            g = grader.get((i, sub))
            if g is None or i not in dumps: continue
            L, F = p['local_step'], g['frontier_step']
            if thr is None:
                dev = L - F
            else:
                steps = {int(m.group(1)): m.group(2).strip() for m in step_re.finditer(p['prompt'])}
                word = {s: cleaned_words(steps.get(s, "")) for s in range(1, p['local_n_steps']+1)}
                Lkm = sum(1 for s in range(1, L) if word.get(s, 0) >= thr)
                Fkm = sum(1 for s in range(1, F) if word.get(s, 0) >= thr)
                dev = Lkm - Fkm
            rows.append({"dev": dev, "outcomes": dumps[i], "ok": any(dumps[i]), "seed": seed})
            seed_rows += 1
        print(f"  [ok]   seed {seed}: {seed_rows} rows")
    return rows


def panel_loc_dist(ax):
    bins = np.linspace(0.0, 1.0, 21)
    for method in ("srpo", "rrpo"):
        vals = collect_error_steps(method, relative=True)
        if not vals: continue
        v = np.array(vals)
        s = METHOD_STYLE[method]
        ax.hist(v, bins=bins, density=True,
                histtype="stepfilled", color=s["color"], alpha=0.32,
                edgecolor=s["color"], linewidth=1.4, zorder=2,
                label=s["label"])
    ax.set_xlabel("(a) Normalized Error Localization")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="#cccccc", fancybox=False, borderpad=0.5).get_frame().set_linewidth(0.6)


def _annotate_rates(ax, x, rates, err_hi):
    """Place rotated '%' labels just above each error bar's upper whisker."""
    ymax = max(r + e for r, e in zip(rates, err_hi))
    pad = ymax * 0.025
    for xi, r, eh in zip(x, rates, err_hi):
        ax.text(xi, r + eh + pad, f"{r:.1f}%",
                ha="center", va="bottom", rotation=90,
                fontsize=13, color="#222", fontweight="bold")


def panel_loc_deviation(ax, rows, buckets, bar_width):
    devs = np.array([r['dev'] for r in rows])
    n = len(devs)
    pcts, labels, colors = [], [], []
    err_lo, err_hi = [], []
    for lo, hi, lbl, side in buckets:
        mask = (devs >= lo) & (devs <= hi)
        c = int(mask.sum())
        rate = c / n * 100
        ci = binomtest(c, n).proportion_ci(method="wilson", confidence_level=0.95)
        pcts.append(rate)
        err_lo.append(rate - ci.low * 100); err_hi.append(ci.high * 100 - rate)
        labels.append(lbl); colors.append(SIDE_COLOR[side])

    x = np.arange(len(buckets))
    ax.bar(x, pcts, width=bar_width, color=colors, alpha=0.85,
           edgecolor="white", linewidth=1.0, zorder=2)
    for xi, p, lo, hi, c in zip(x, pcts, err_lo, err_hi, colors):
        ax.errorbar(xi, p, yerr=[[lo], [hi]], fmt="none", ecolor=c,
                    capsize=3, capthick=1.0, elinewidth=1.0, zorder=4)
    _annotate_rates(ax, x, pcts, err_hi)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("(b) Localization Deviation (thought steps)")
    ax.set_ylabel("Share of Localizations (%)")
    ymax = max(p + e for p, e in zip(pcts, err_hi))
    ax.set_ylim(0, ymax * 1.32)


def panel_correction_by_dev(ax, rows, buckets, bar_width):
    devs = np.array([r['dev'] for r in rows])
    rates, labels, colors = [], [], []
    err_lo, err_hi = [], []
    for lo, hi, lbl, side in buckets:
        mask = (devs >= lo) & (devs <= hi)
        bucket = [r for r, m in zip(rows, mask) if m]
        n_att = 4 * len(bucket)
        k = sum(o for r in bucket for o in r['outcomes'])
        rate = (k / n_att * 100) if n_att else 0.0
        if n_att:
            ci = binomtest(k, n_att).proportion_ci(method="wilson", confidence_level=0.95)
            err_lo.append(rate - ci.low * 100); err_hi.append(ci.high * 100 - rate)
        else:
            err_lo.append(0); err_hi.append(0)
        rates.append(rate)
        labels.append(lbl); colors.append(SIDE_COLOR[side])
    overall = sum(o for r in rows for o in r['outcomes']) / (4 * len(rows)) * 100

    x = np.arange(len(buckets))
    ax.bar(x, rates, width=bar_width, color=colors, alpha=0.85,
           edgecolor="white", linewidth=1.0, zorder=2)
    for xi, r, lo, hi, c in zip(x, rates, err_lo, err_hi, colors):
        ax.errorbar(xi, r, yerr=[[lo], [hi]], fmt="none", ecolor=c,
                    capsize=3, capthick=1.0, elinewidth=1.0, zorder=4)
    _annotate_rates(ax, x, rates, err_hi)
    ax.axhline(overall, color="#444", linestyle=":", linewidth=1.0, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("(c) Localization Deviation (thought steps)")
    ax.set_ylabel("Correction Rate (%)")
    ymax = max(r + e for r, e in zip(rates, err_hi))
    ax.set_ylim(0, ymax * 1.32)


def panel_correction_pooled(ax, rows):
    sides = {
        "Clean":     [r for r in rows if r['dev'] <= 0],
        "Erroneous": [r for r in rows if r['dev']  > 0],
    }
    colors = [SIDE_COLOR["clean"], SIDE_COLOR["erroneous"]]
    labels, rates, lows, highs = [], [], [], []
    for label, bucket in sides.items():
        n = len(bucket)
        k = sum(r['ok'] for r in bucket)
        rate = k / n * 100
        ci = binomtest(k, n).proportion_ci(method="wilson", confidence_level=0.95)
        labels.append(label); rates.append(rate)
        lows.append(rate - ci.low * 100); highs.append(ci.high * 100 - rate)

    x = np.arange(2)
    ax.bar(x, rates, width=0.4, color=colors, alpha=0.85,
           edgecolor="white", linewidth=1.0, zorder=2)
    for xi, r, lo, hi, c in zip(x, rates, lows, highs, colors):
        ax.errorbar(xi, r, yerr=[[lo], [hi]], fmt="none", ecolor=c,
                    capsize=5, capthick=1.2, elinewidth=1.2, zorder=4)
    _annotate_rates(ax, x, rates, highs)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("(d) Prefix Quality")
    ax.set_ylabel("Suffix Group Pass@4 (%)")
    ymax = max(r + h for r, h in zip(rates, highs))
    ax.set_ylim(0, ymax * 1.35)
    ax.set_xlim(-0.7, 1.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=int, default=None,
                    help="Meaningful-step threshold (e.g. 5). Omit for raw deviation.")
    ap.add_argument("--bins", choices=("a", "fine"), default="a",
                    help="Bucket scheme for panels (b)/(c). a=7 buckets ≤−3..≥+3; "
                         "fine=11 buckets ≤−5..≥+5.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        bin_tag = "" if args.bins == "a" else f"_{args.bins}"
        eff_tag = (f"_eff{args.thr}" if args.thr is not None else "")
        args.out = f"plots/combined_4panel{eff_tag}_lcb_medium{bin_tag}"

    rows = load_records(args.thr)
    print(f"loaded {len(rows)} rows  (thr={args.thr})")

    buckets = BIN_CONFIGS[args.bins]
    # Wider middle panels and skinnier bars when there are more buckets.
    if args.bins == "fine":
        figsize = (22, 5.0)
        width_ratios = [0.95, 1.2, 1.2, 0.6]
        bar_width = 0.6
    else:
        figsize = (20, 5.0)
        width_ratios = [1.0, 1.0, 1.0, 0.6]
        bar_width = 0.65

    _set_publication_style()
    fig, axes = plt.subplots(
        1, 4, figsize=figsize,
        gridspec_kw={"width_ratios": width_ratios},
    )
    panel_loc_dist(axes[0])
    panel_loc_deviation(axes[1], rows, buckets, bar_width)
    panel_correction_by_dev(axes[2], rows, buckets, bar_width)
    panel_correction_pooled(axes[3], rows)

    # Reserve bottom space for the global legend.
    fig.tight_layout(w_pad=2.4, rect=(0, 0.10, 1, 1))

    # Global legend at the very bottom: clean (blue+green) + erroneous (red).
    handles = [
        (Patch(color=SIDE_COLOR["clean"], alpha=0.85),
         Patch(color=SIDE_COLOR["exact"], alpha=0.85)),
        Patch(color=SIDE_COLOR["erroneous"], alpha=0.85),
    ]
    leg_labels = [
        "Clean Prefix (Self ≤ Oracle)",
        "Erroneous Prefix (Self > Oracle)",
    ]
    fig.legend(
        handles, leg_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=2, frameon=False, handlelength=2.6, columnspacing=2.4, fontsize=17,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
        print(f"saved {out.with_suffix(f'.{ext}')}")
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paper_pdf = PAPER_FIGURES_DIR / f"{out.name}.pdf"
    fig.savefig(paper_pdf, dpi=300)
    print(f"saved {paper_pdf}")


if __name__ == "__main__":
    main()
