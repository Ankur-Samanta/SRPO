#!/usr/bin/env python3
"""
Structured SRPO gradient visualization for one prompt.

Layout (prompt 2 by default, the apple-market problem):

  G2-corr slot 4 :                             [suffix thoughts...]
  G2-corr slot 5 :                             [suffix thoughts...]
                  [shared 6-thought prefix box, centered between rows 2&3]
  G2-corr slot 6 :                             [suffix thoughts...]
  G2-corr slot 7 :                             [suffix thoughts...]
  ─────────────────────────────────────────
  G1 slot 0 :     [full 11-thought sequence...]
  G1 slot 1 :     [full 20-thought sequence...]
  G1 slot 2 :     [full 14-thought sequence...]
  G1 slot 3 :     [full 20-thought sequence...]

Each box = 1 thought. Color = mean per-token |gradient| over that thought's tokens.
Gray = zero or near-zero (e.g. shared prefix, which is masked out by SRPO).
Red = high per-token gradient signal.

Style: rounded boxes, FancyBboxPatch.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, to_hex
import torch

# ─── Paths ──────────────────────────────────────────────────────────────────
SCPO_DIR = Path(__file__).resolve().parents[1]
LOSS_DIR = SCPO_DIR / "logs" / "srpo_per_token" / "grad_dump_l2n_ep1"
BRANCH_FILE = next((SCPO_DIR / "logs" / "srpo_localizations" / "grad_dump_l2n_ep1").glob("branches_pid*.jsonl"))
OUT_DIR = SCPO_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_IDX = 2  # default; can override via CLI
PAIR_SIZE = 1   # combine every PAIR_SIZE consecutive thoughts into one box

# Per-token gradient metric:
#   "coeff"     — |A|/T_suf · ratio_t, the literal scalar coefficient (uniform within a suffix)
#   "effective" — coeff × √(1 − π_θ(y_t)), proxy for actual parameter-space update size
#                 (score-function gradient norm is ‖∇_z log π_θ‖ ∝ √(1 − π_θ(y_t)))
METRIC = "coeff"  # overridden by CLI arg 2

# ─── Layout constants (mirrors render_tree_viz.py style) ────────────────────
COL_SPACING = 0.55
BOX_WIDTH = 0.45
BOX_HEIGHT = 0.45
ROW_SPACING = 0.65
SEPARATOR_PAD = 0.20  # extra vertical gap between G2 group and G1 group
EDGE_LW = 1.4
LABEL_X = 0.0    # data starts at x=0; no left label gutter

# Colors
PREFIX_GRAY = "#B4B4B4"   # shared prefix (gray)
SEPARATOR_COLOR = "#888888"
LABEL_COLOR = "#333333"

# Custom gradient colormap: gray (0) → light yellow → red → dark red (max)
# Stops chosen so low (pale) and high (saturated) values are clearly separated:
# values around 15% of vmax sit in the pale-cream region, ~35% sits firmly in
# the warm-orange region, and values above ~70% saturate toward deep red.
GRAD_CMAP = LinearSegmentedColormap.from_list(
    "grad",
    [
        (0.00, "#E0E0E0"),  # near-zero: light gray
        (0.08, "#FAECC8"),  # very low: very pale cream
        (0.25, "#F4C078"),  # low-mid: peach
        (0.45, "#EF7E40"),  # mid: orange
        (0.70, "#C7392A"),  # high: red
        (1.00, "#7A0F0F"),  # max: deep red
    ],
)

# Saturation percentile for vmax. Setting < 100 clips the top-N% of per-thought
# values to saturate at deep red, compressing the colormap into the bulk of the
# data range so typical values land farther up the cmap.
VMAX_PCTL = 98.0


# ─── Data joining (same logic as analysis script, condensed) ────────────────

def infer_ssi(rollouts_in_prompt):
    g2 = sorted([r for r in rollouts_in_prompt if r["slot"] >= 4], key=lambda r: r["slot"])
    if len(g2) < 2: return 0
    shared = 0
    min_len = min(len(r["decoded_thoughts"]) for r in g2)
    for k in range(min_len):
        ref = g2[0]["decoded_thoughts"][k]
        if all(r["decoded_thoughts"][k] == ref for r in g2):
            shared += 1
        else:
            break
    return sum(g2[0]["segment_lengths"][:shared]), shared


def per_token_grad(lr, metric: str):
    """Per-token gradient magnitude tensor for a loss row, under chosen metric.

    "coeff": |A|/T_suf · ratio_t · suffix_mask_t (the scalar PG coefficient)
    "effective": coeff_t × √(1 − π_θ(y_t)), score-function-weighted update size
    """
    ratio = (lr["log_prob"] - lr["old_log_prob"]).exp()
    suff_count = max(int(lr["suffix_mask"].sum().item()), 1)
    coeff = (-lr["adv"] * ratio * lr["suffix_mask"].float() / suff_count).abs()
    if metric == "coeff":
        return coeff
    if metric == "effective":
        # Score-function-norm proxy for the per-token pre-Jacobian PG contribution:
        #   ‖∇_z log π_θ(y_t)‖² = (1 − π_y)² + Σ_{k≠y} π_k²  ≈  (1 − π_y)² for large V
        # so ‖score‖ ≈ (1 − π_y) (linear, NOT √).
        pi_t = lr["log_prob"].exp().clamp(min=0.0, max=1.0)
        score_norm = (1.0 - pi_t).clamp(min=0.0)
        return coeff * score_norm
    raise ValueError(f"unknown metric: {metric}")


def load_data(prompt_idx: int, metric: str = "coeff"):
    # Branch dump for chosen prompt
    rec = None
    for i, line in enumerate(open(BRANCH_FILE)):
        if i == prompt_idx:
            rec = json.loads(line); break
    assert rec is not None, f"prompt {prompt_idx} not in branch dump"

    rollouts_raw = sorted(rec["rollouts"], key=lambda r: r["slot"])
    ssi, n_prefix_thoughts = infer_ssi(rollouts_raw)

    rollouts = []
    for ro in rollouts_raw:
        role = "G1" if ro["slot"] < 4 else "G2-corr"
        rollouts.append({
            "slot": ro["slot"],
            "role": role,
            "ssi": ssi if role == "G2-corr" else 0,
            "response_len": ro["response_len"],
            "decoded_thoughts": ro["decoded_thoughts"],
            "thought_boundaries": [tuple(b) for b in ro["thought_boundaries"]],
            "segment_lengths": ro["segment_lengths"],
            "oracle_correct": ro.get("oracle_correct"),
        })

    # Loss rows
    loss_rows = []
    for fp in sorted(LOSS_DIR.glob("loss_pid*_call*.pt")):
        d = torch.load(fp, weights_only=False)
        rl_t = d["response_mask"].sum(dim=-1)
        for i in range(d["log_prob"].shape[0]):
            loss_rows.append({
                "file": fp.name,
                "row_in_file": i,
                "rl": int(rl_t[i]),
                "ssi": int(d["suffix_start_idx"][i]),
                "adv": float(d["reset_advantage"][i]),
                "log_prob": d["log_prob"][i].float(),
                "old_log_prob": d["old_log_prob"][i].float(),
                "response_mask": d["response_mask"][i].bool(),
                "suffix_mask": d["suffix_mask"][i].bool(),
            })

    # Match each rollout to its loss row by (rl, ssi); disambiguate ties by
    # oracle_correct (positive adv ↔ correct, in the typical case).
    used = set()
    for ro in rollouts:
        cands = [(j, lr) for j, lr in enumerate(loss_rows)
                 if lr["rl"] == ro["response_len"] and lr["ssi"] == ro["ssi"] and j not in used]
        if not cands:
            ro["loss_row"] = None
            continue
        if len(cands) == 1:
            j, lr = cands[0]
            used.add(j)
            ro["loss_row"] = lr
            continue
        # Multi-candidate: pair by oracle vs sign of adv
        prefer_pos = (ro["oracle_correct"] is True)
        cands.sort(key=lambda c: c[1]["adv"], reverse=prefer_pos)
        j, lr = cands[0]
        used.add(j)
        ro["loss_row"] = lr

    # Per-thought |grad| under chosen metric
    for ro in rollouts:
        lr = ro["loss_row"]
        ro["thought_grads"] = []
        if lr is None:
            ro["thought_grads"] = [0.0] * len(ro["thought_boundaries"])
            continue
        pg_token = per_token_grad(lr, metric)
        for s, e in ro["thought_boundaries"]:
            mask = lr["suffix_mask"][s:e]
            n_active = int(mask.sum().item())
            if n_active == 0:
                ro["thought_grads"].append(0.0)
            else:
                ro["thought_grads"].append(float(pg_token[s:e][mask].mean().item()))

    return rollouts, n_prefix_thoughts, ssi, rec["question"]


# ─── Pairing ───────────────────────────────────────────────────────────────

def pair_in_n(grads, n=PAIR_SIZE, offset=0):
    """Pair `grads` in chunks of `n`. Returns [(label, mean_grad)] where label is
    e.g. '1-2' or '5' for a half-pair tail. `offset` shifts the 1-based labels."""
    out = []
    for i in range(0, len(grads), n):
        chunk = grads[i:i + n]
        first = i + offset + 1
        last = first + len(chunk) - 1
        label = f"{first}-{last}" if len(chunk) > 1 else f"{first}"
        out.append((label, sum(chunk) / len(chunk)))
    return out


# ─── Render ────────────────────────────────────────────────────────────────

def draw_box(ax, x, y, label, value, vmax, edge="#444444", w=BOX_WIDTH, h=BOX_HEIGHT,
             label_size=8.5, force_gray=False):
    if force_gray or value <= 0:
        face = "#E0E0E0"
        edge = "#888888"
    else:
        norm = min(value / vmax, 1.0) if vmax > 0 else 0.0
        rgba = GRAD_CMAP(norm)
        face = to_hex(rgba)
        edge = "#444444"
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=face, edgecolor=edge, linewidth=EDGE_LW,
    )
    ax.add_patch(box)
    if label:
        # White or dark text depending on background luminance
        text_color = "white" if (not force_gray and value > 0 and value / vmax > 0.5) else "#222222"
        ax.text(x, y, label, ha="center", va="center",
                fontsize=label_size, fontweight="bold", color=text_color)


def render(prompt_idx: int = PROMPT_IDX, metric: str = METRIC):
    rollouts, n_prefix, ssi, question = load_data(prompt_idx, metric=metric)

    # Order: G2-corrections (slots 4-7) on top, G1 (slots 0-3) on bottom
    g2 = [r for r in rollouts if r["role"] == "G2-corr"]
    g1 = [r for r in rollouts if r["role"] == "G1"]
    assert len(g2) == 4 and len(g1) == 4, f"expected 4+4, got {len(g2)}+{len(g1)}"

    # ─── Pair every PAIR_SIZE thoughts (respecting prefix/suffix boundary) ─────
    # G2: pair the suffix portion separately so we don't merge a masked prefix
    # thought with an unmasked suffix one.
    n_prefix_pairs = math.ceil(n_prefix / PAIR_SIZE)
    for r in g2:
        r["paired_suffix"] = pair_in_n(r["thought_grads"][n_prefix:], offset=n_prefix)
    for r in g1:
        r["paired_full"] = pair_in_n(r["thought_grads"])
    paired_prefix = pair_in_n([0.0] * n_prefix)  # all zero, just for labels

    # Compute layout dimensions in PAIRED units
    g2_suffix_pair_n = [len(r["paired_suffix"]) for r in g2]
    g1_pair_n = [len(r["paired_full"]) for r in g1]
    max_total_cols = max(
        n_prefix_pairs + max(g2_suffix_pair_n),
        max(g1_pair_n),
    )

    # Global vmax for color scale. Clip to VMAX_PCTL percentile of nonzero
    # per-thought |grad| values so the bulk of the data spans most of the
    # colormap (the top tail saturates at deep red).
    all_grads = []
    for r in g2:
        all_grads.extend(g for _, g in r["paired_suffix"])
    for r in g1:
        all_grads.extend(g for _, g in r["paired_full"])
    nonzero = [g for g in all_grads if g > 0]
    if nonzero:
        import numpy as np
        vmax = float(np.percentile(nonzero, VMAX_PCTL))
    else:
        vmax = 1.0

    # Y positions (top → bottom). 4 G2-corr rows + space for prefix + separator + 4 G1 rows
    n_rows_visible = 4 + 4  # 8 trajectory rows
    y_g2 = [0.0, -ROW_SPACING, -2 * ROW_SPACING, -3 * ROW_SPACING]
    y_prefix = (y_g2[1] + y_g2[2]) / 2  # midpoint between G2 row 1 and row 2
    y_separator = y_g2[-1] - ROW_SPACING - SEPARATOR_PAD
    y_g1 = [y_separator - ROW_SPACING * (i + 1) for i in range(4)]

    # Figure size: ~0.5 inch per column, ~0.5 inch per row. Extra horizontal
    # padding so the larger header fonts fit cleanly in the empty x-region
    # to the left of the suffix boxes.
    fig_w = max(14.0, 0.55 * max_total_cols + 5.0)
    fig_h = 0.55 * (n_rows_visible + 1) + 2.0

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # ─── Per-group mean per-token |grad| (mass-weighted across rollouts) ──
    def group_mean_grad(rs):
        total_mass = 0.0
        total_tokens = 0
        for r in rs:
            lr = r["loss_row"]
            if lr is None:
                continue
            pg_token = per_token_grad(lr, metric)
            mask = lr["suffix_mask"]
            total_mass += float(pg_token[mask].sum().item())
            total_tokens += int(mask.sum().item())
        return (total_mass / total_tokens) if total_tokens > 0 else 0.0

    g2_mean = group_mean_grad(g2)
    g1_mean = group_mean_grad(g1)
    ratio = (g2_mean / g1_mean) if g1_mean > 0 else float("nan")

    # ─── Section headers ────────────────────────────────────────────────────
    # Both groups: 2-line header (name, mean per-token |g|). Tucked into the
    # empty x-region to the left of the suffix boxes.
    cbar_top_y = y_g2[0] + BOX_HEIGHT / 2  # top edge of colorbar
    g2_header_y = cbar_top_y - 0.10
    g1_header_y = y_g1[0] + ROW_SPACING + 0.05
    header_x = -0.15
    ax.text(header_x, g2_header_y, "shared-prefix group",
            ha="left", va="center", fontsize=19, fontweight="bold", color="#333333")
    ax.text(header_x, g2_header_y - 0.34,
            rf"mean per-token $|g| = {g2_mean:.2e}$",
            ha="left", va="center", fontsize=15, fontstyle="italic", color="#666666")
    ax.text(header_x, g1_header_y, "base group",
            ha="left", va="center", fontsize=19, fontweight="bold", color="#333333")
    ax.text(header_x, g1_header_y - 0.34,
            rf"mean per-token $|g| = {g1_mean:.2e}$",
            ha="left", va="center", fontsize=15, fontstyle="italic", color="#666666")

    # ─── Draw G2-corrections (paired suffix thoughts, starting at n_prefix_pairs) ──
    for row_idx, ro in enumerate(g2):
        y = y_g2[row_idx]
        oracle_mark = "✓" if ro["oracle_correct"] is True else "✗"
        oracle_color = "#2E8B57" if ro["oracle_correct"] is True else "#DC3545"
        last_col = n_prefix_pairs + len(ro["paired_suffix"])
        ax.text(last_col * COL_SPACING + 0.25, y, oracle_mark,
                ha="left", va="center", fontsize=24, fontweight="bold", color=oracle_color)
        for j, (label, grad) in enumerate(ro["paired_suffix"]):
            x = (n_prefix_pairs + j) * COL_SPACING
            draw_box(ax, x, y, "", grad, vmax)

    # ─── Draw shared prefix row (centered between G2 rows 1 and 2) ────────────
    for j, (label, _) in enumerate(paired_prefix):
        x = j * COL_SPACING
        draw_box(ax, x, y_prefix, "", 0.0, vmax, force_gray=True)
    # Connecting dashed lines from prefix-right to suffix-left of each G2 row
    for row_idx, ro in enumerate(g2):
        y = y_g2[row_idx]
        x_right = (n_prefix_pairs - 1) * COL_SPACING + BOX_WIDTH / 2 + 0.02
        x_left = n_prefix_pairs * COL_SPACING - BOX_WIDTH / 2 - 0.02
        ax.plot([x_right, x_left], [y_prefix, y], color="#999999",
                linewidth=1.1, linestyle=(0, (3, 2)), alpha=0.65, zorder=1)

    # ─── Separator between G2 group and G1 group ─────────────────────────────
    sep_x_left = LABEL_X - 0.5
    sep_x_right = max_total_cols * COL_SPACING + 0.5
    sep_y = y_separator + ROW_SPACING / 2
    ax.plot([sep_x_left, sep_x_right], [sep_y, sep_y],
            color=SEPARATOR_COLOR, linewidth=0.8, linestyle="--", alpha=0.55)

    # ─── Draw G1 trajectories (paired full sequence) ────────────────────────
    for row_idx, ro in enumerate(g1):
        y = y_g1[row_idx]
        oracle_mark = "✓" if ro["oracle_correct"] is True else "✗"
        oracle_color = "#2E8B57" if ro["oracle_correct"] is True else "#DC3545"
        n_pairs = len(ro["paired_full"])
        ax.text(n_pairs * COL_SPACING + 0.25, y, oracle_mark,
                ha="left", va="center", fontsize=24, fontweight="bold", color=oracle_color)
        for j, (label, grad) in enumerate(ro["paired_full"]):
            x = j * COL_SPACING
            draw_box(ax, x, y, "", grad, vmax)

    # ─── Color bar (vertical, tight against the left edge of the plot) ────
    cbar_w = 0.22
    cbar_x = -0.55  # close to the data on the left
    cbar_top = y_g2[0] + BOX_HEIGHT / 2
    cbar_bot = y_g1[-1] - BOX_HEIGHT / 2
    n_cbar = 100
    for i in range(n_cbar):
        frac = i / (n_cbar - 1)
        rect_y = cbar_bot + frac * (cbar_top - cbar_bot)
        rect_h = (cbar_top - cbar_bot) / n_cbar + 0.01
        ax.add_patch(Rectangle(
            (cbar_x, rect_y), cbar_w, rect_h,
            facecolor=to_hex(GRAD_CMAP(frac)), edgecolor="none",
        ))
    ax.add_patch(Rectangle(
        (cbar_x, cbar_bot), cbar_w, cbar_top - cbar_bot,
        facecolor="none", edgecolor="#444444", linewidth=0.8,
    ))
    # Tick labels: all at the same x, all centered on their own y; the
    # end-tick labels are nudged INTO the plot (away from the edges) so
    # the rotated text stays comfortably within the colorbar's vertical
    # extent. Tick marks remain at the true 0/half/max positions.
    label_x = cbar_x - 0.18
    end_inset = 0.55  # data-units to push end labels inward from the bar edge
    ticks = [
        (0.0, 0.0,       cbar_bot + end_inset),                           # bottom
        (0.5, vmax / 2,  (cbar_bot + cbar_top) / 2),                      # middle
        (1.0, vmax,      cbar_top - end_inset),                           # top
    ]
    for tick_frac, tick_val, ty_label in ticks:
        ty_tick = cbar_bot + tick_frac * (cbar_top - cbar_bot)
        ax.plot([cbar_x - 0.04, cbar_x], [ty_tick, ty_tick],
                color="#444444", linewidth=0.7)
        ax.text(label_x, ty_label, f"{tick_val:.1e}",
                ha="center", va="center", fontsize=16, color="#333333",
                rotation=90)

    # ─── Axes setup ─────────────────────────────────────────────────────────
    ax.set_xlim(cbar_x - 0.5, max_total_cols * COL_SPACING + 0.6)
    ax.set_ylim(y_g1[-1] - ROW_SPACING / 2, g2_header_y + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    suffix = "" if metric == "coeff" else f"_{metric}"
    out_pdf = OUT_DIR / f"grad_tree_p{prompt_idx}{suffix}.pdf"
    out_png = OUT_DIR / f"grad_tree_p{prompt_idx}{suffix}.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    print(f"\nPrompt {prompt_idx} (metric={metric}): {question[:120]!r}")
    print(f"  shared prefix: {n_prefix} thoughts ({ssi} tokens)")
    print(f"  G2-corrections suffix lengths (paired): {g2_suffix_pair_n} (raw: {[len(r['thought_boundaries']) - n_prefix for r in g2]})")
    print(f"  G1 thought counts (paired): {g1_pair_n} (raw: {[len(r['thought_boundaries']) for r in g1]})")
    print(f"  vmax for color scale: {vmax:.3e}")
    print(f"  group means: G2={g2_mean:.3e}  G1={g1_mean:.3e}  ratio={ratio:.2f}")
    print(f"  PAIR_SIZE = {PAIR_SIZE}, total cols = {max_total_cols}")


if __name__ == "__main__":
    import sys
    p = int(sys.argv[1]) if len(sys.argv) > 1 else PROMPT_IDX
    m = sys.argv[2] if len(sys.argv) > 2 else METRIC
    render(p, metric=m)
