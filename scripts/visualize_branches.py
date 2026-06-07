"""Visualize SCGRPO rollout branching structure + per-thought credit signal.

Reads the JSONL dumps produced by training.branch_logger and renders a
tree diagram per group of 8 rollouts, where:
  - each edge = one thought (labeled with truncated text)
  - branching points = where rollouts diverge after a shared prefix
  - leaf color = oracle correctness (green=correct, red=wrong)
  - edge thickness ∝ number of rollouts passing through that thought
    (shared prefixes are thick, unique suffixes are thin)
  - annotations at branch points show the advantage delta between siblings
    (approximated from oracle correctness since advantages aren't in the dump)

Usage:
  python scripts/visualize_branches.py <jsonl_path> [--index N] [--out PNG]
  python scripts/visualize_branches.py <dir> --latest [--out PNG]

The viz uses matplotlib (no graphviz dep). For a trie with hash-equal thoughts,
thoughts with IDENTICAL decoded text are treated as the same branch node — so
TGRPO-style rollouts (no prefix sharing) render as 8 disjoint chains from the
root, while SCGRPO rollouts (ICS corrections share a prefix with their parent)
render as a trunk-with-branches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────

def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_loss_dumps(path: Path) -> dict[str, list[dict]]:
    """Load all loss JSONL files under path; return dict fp -> list of records.

    If path is a file, loads just that file. If path is a directory, loads all
    *.jsonl files under it (typically one per DP worker PID).
    """
    files: list[Path]
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        return {}

    fp_to_records: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fp_to_records[rec["fp"]].append(rec)
    return dict(fp_to_records)


def match_losses_to_rollouts(rollouts: list[dict],
                             loss_index: dict[str, list[dict]]) -> tuple[dict, dict]:
    """Attach per-thought losses to each rollout via fingerprint matching.

    Returns:
        rollout_to_loss: slot → loss record (or None if no match)
        stats: {'matched': int, 'ambiguous': int, 'missed': int, 'total': int}
    """
    rollout_to_loss: dict[int, dict | None] = {}
    stats = {"matched": 0, "ambiguous": 0, "missed": 0, "total": 0}

    # Pre-compute fp collision count on the rollout side. If 2+ rollouts in
    # this group share a fp, ANY match to that fp is inherently ambiguous —
    # we cannot know which rollout the loss record came from.
    fp_rollout_count: dict[str, int] = defaultdict(int)
    for r in rollouts:
        fp = r.get("segment_fp")
        if fp:
            fp_rollout_count[fp] += 1

    # Track per-fp "consumption" across the group so we don't reuse the same
    # loss record for multiple rollouts.
    fp_consumed: dict[str, int] = defaultdict(int)

    for r in rollouts:
        slot = r["slot"]
        fp = r.get("segment_fp")
        stats["total"] += 1
        if fp is None:
            rollout_to_loss[slot] = None
            stats["missed"] += 1
            continue
        candidates = loss_index.get(fp, [])
        consumed = fp_consumed[fp]

        if consumed >= len(candidates):
            # no record left (either fp absent, or all records already assigned)
            rollout_to_loss[slot] = None
            stats["missed"] += 1
        elif fp_rollout_count[fp] == 1 and len(candidates) == 1:
            # unique fp on both sides: unambiguous 1:1 match
            rollout_to_loss[slot] = candidates[0]
            fp_consumed[fp] += 1
            stats["matched"] += 1
        else:
            # Collision: multiple rollouts share this fp, OR multiple loss
            # records share this fp. Either way we cannot attribute the record
            # to a specific rollout. Assign in order (best-effort) and flag.
            rollout_to_loss[slot] = {**candidates[consumed], "_ambiguous": True}
            fp_consumed[fp] += 1
            stats["ambiguous"] += 1

    return rollout_to_loss, stats


def select_record(records: list[dict], index: int) -> dict:
    if not records:
        raise SystemExit("no records found in dump")
    return records[index % len(records)]


# ─────────────────────────────────────────────────────────────────────────
# Trie construction
# ─────────────────────────────────────────────────────────────────────────

def _hash_thought(text: str) -> str:
    """8-char hash used as trie key. Exact-match semantics."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


class TrieNode:
    __slots__ = ("key", "text", "depth", "rollouts", "children")

    def __init__(self, key: str, text: str, depth: int):
        self.key = key
        self.text = text
        self.depth = depth
        self.rollouts: list[int] = []  # rollout slots that traverse this node
        self.children: dict[str, "TrieNode"] = {}


def build_trie(rollouts: list[dict]) -> TrieNode:
    """Build a trie over decoded_thoughts sequences.

    Two rollouts share a node iff their thoughts up to that depth are byte-identical.
    """
    root = TrieNode(key="<root>", text="", depth=0)
    for r in rollouts:
        slot = r["slot"]
        thoughts = r.get("decoded_thoughts") or []
        node = root
        node.rollouts.append(slot)
        for depth, text in enumerate(thoughts, start=1):
            k = _hash_thought(text)
            if k not in node.children:
                node.children[k] = TrieNode(k, text, depth)
            node = node.children[k]
            node.rollouts.append(slot)
    return root


# ─────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────

def _count_leaves(node: TrieNode) -> int:
    if not node.children:
        return 1
    return sum(_count_leaves(c) for c in node.children.values())


def layout(node: TrieNode, x_start: float, x_end: float, depth: int,
           positions: dict[int, tuple[float, float]]) -> None:
    """Assign (x, y) to each node. x = horizontal slot, y = -depth."""
    positions[id(node)] = ((x_start + x_end) / 2, -depth)
    if not node.children:
        return
    # Allocate horizontal space proportional to leaf counts
    children = list(node.children.values())
    total = sum(_count_leaves(c) for c in children)
    cursor = x_start
    for c in children:
        w = (x_end - x_start) * _count_leaves(c) / max(total, 1)
        layout(c, cursor, cursor + w, depth + 1, positions)
        cursor += w


# ─────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────

def _rollout_color(rollouts: list[dict], slot_set: list[int]) -> str:
    """Color a path by the correctness of the rollouts passing through it.

    - all correct → green
    - all wrong → red
    - mixed → orange
    - unknown → gray
    """
    verdicts = [rollouts[s].get("oracle_correct") for s in slot_set
                if 0 <= s < len(rollouts)]
    verdicts = [v for v in verdicts if v is not None]
    if not verdicts:
        return "#888888"
    n_correct = sum(1 for v in verdicts if v)
    if n_correct == len(verdicts):
        return "#2a9d4a"  # green
    if n_correct == 0:
        return "#c0392b"  # red
    return "#e67e22"  # orange


def _truncate(text: str, n: int = 60) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _edge_loss_for_thought(rollouts: list[dict],
                           rollout_to_loss: dict | None,
                           slot_set: list[int],
                           depth: int) -> float | None:
    """Return mean per-thought loss across the rollouts traversing this edge.

    depth==k means this edge is thought k (1-indexed); looks up
    rollout_to_loss[slot]['thought_losses'][k-1] across slot_set.
    None if no loss data available.
    """
    if rollout_to_loss is None or depth < 1:
        return None
    vals = []
    for s in slot_set:
        lr = rollout_to_loss.get(s)
        if lr is None:
            continue
        losses = lr.get("thought_losses") or []
        if depth - 1 < len(losses):
            vals.append(losses[depth - 1])
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _sign_color(loss: float) -> str:
    """Sign-encoded color: blue=reinforced (neg loss), red=penalized (pos loss).

    (Recall: pg_loss = -advantage*ratio → negative means gradient PUSHES probability UP.)
    """
    if loss < 0:
        return "#1f77b4"  # blue
    return "#d62728"  # red


def render(record: dict, out_path: Path | None = None, figsize: tuple = (20, 12),
           rollout_to_loss: dict | None = None,
           loss_match_stats: dict | None = None) -> None:
    rollouts = record["rollouts"]
    root = build_trie(rollouts)

    positions: dict[int, tuple[float, float]] = {}
    layout(root, 0.0, float(len(rollouts)), 0, positions)

    # If loss data available, compute global |loss| scale for edge width
    loss_abs_max = None
    if rollout_to_loss is not None:
        all_losses = []
        for slot, lr in rollout_to_loss.items():
            if lr is None:
                continue
            all_losses.extend(abs(x) for x in (lr.get("thought_losses") or []))
        if all_losses:
            loss_abs_max = max(all_losses)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges + labels
    def draw(node: TrieNode) -> None:
        x0, y0 = positions[id(node)]
        for c in node.children.values():
            x1, y1 = positions[id(c)]

            # Edge styling
            if rollout_to_loss is not None and loss_abs_max and loss_abs_max > 0:
                # Gradient-mode: edge color = sign(loss), width ∝ |loss|
                edge_loss = _edge_loss_for_thought(rollouts, rollout_to_loss,
                                                   c.rollouts, c.depth)
                if edge_loss is None:
                    color = "#cccccc"
                    lw = 0.5
                    linestyle = ":"  # dashed = unmatched
                else:
                    color = _sign_color(edge_loss)
                    lw = 0.5 + 5.0 * abs(edge_loss) / loss_abs_max
                    # Dashed if any rollout here had ambiguous match
                    amb = any(
                        (rollout_to_loss.get(s) or {}).get("_ambiguous")
                        for s in c.rollouts
                    )
                    linestyle = "--" if amb else "-"
            else:
                # Structural-mode (no loss data): color by correctness, width by count
                lw = 0.5 + 1.4 * len(c.rollouts)
                color = _rollout_color(rollouts, c.rollouts)
                linestyle = "-"
                edge_loss = None

            ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                    linestyle=linestyle, alpha=0.85, zorder=2)

            # Thought label near the edge midpoint
            label = _truncate(c.text, 70)
            suffix = ""
            if edge_loss is not None:
                suffix = f"\nℓ={edge_loss:+.3f}"
            ax.annotate(
                f"t{c.depth}: {label}{suffix}",
                xy=((x0 + x1) / 2, (y0 + y1) / 2),
                fontsize=6, color="#333333",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                zorder=3,
            )
            draw(c)

    draw(root)

    # Draw nodes
    for nid, (x, y) in positions.items():
        ax.scatter([x], [y], s=40, color="#333333", zorder=4)

    # Leaf markers: show slot IDs + correctness
    def mark_leaves(node: TrieNode) -> None:
        if not node.children:
            x, y = positions[id(node)]
            slots = ",".join(str(s) for s in node.rollouts)
            verdicts = [rollouts[s].get("oracle_correct")
                        for s in node.rollouts if 0 <= s < len(rollouts)]
            mark = "✓" if all(v for v in verdicts if v is not None) else (
                "✗" if all(v is False for v in verdicts if v is not None) else "?"
            )
            ax.annotate(
                f"[{slots}] {mark}",
                xy=(x, y - 0.3), fontsize=8, ha="center", va="top",
                color=_rollout_color(rollouts, node.rollouts), weight="bold",
                zorder=5,
            )
            return
        for c in node.children.values():
            mark_leaves(c)

    mark_leaves(root)

    # Title + ICS stats
    ics = record.get("ics_stats", {})
    question = (record.get("question") or "").replace("\n", " ")
    if len(question) > 120:
        question = question[:119] + "…"
    n_correct = sum(1 for r in rollouts if r.get("oracle_correct"))

    mode_line = ""
    if rollout_to_loss is not None:
        if loss_match_stats:
            s = loss_match_stats
            mode_line = (
                f"Gradient mode: matched={s['matched']}/{s['total']}, "
                f"ambiguous={s['ambiguous']}, missed={s['missed']}  |  "
                "width ∝ |per-thought loss|, blue=reinforced, red=penalized\n"
            )
        else:
            mode_line = "Gradient mode (no stats)\n"

    title = (
        f"{mode_line}"
        f"SCGRPO rollout branches — {n_correct}/{len(rollouts)} correct\n"
        f"ICS: triggered={ics.get('ics_triggered', False)}, "
        f"iters={ics.get('ics_iterations', 0)}, "
        f"corrected={ics.get('ics_corrected', False)}, "
        f"error_steps={ics.get('ics_error_steps', [])}\n"
        f"Q: {question}"
    )
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("rollout slot")
    ax.set_ylabel("thought depth (0 = root)")
    ax.axis("off")
    ax.margins(0.02)

    # Footer: summary metrics
    max_depth = max((p[1] for p in positions.values()), default=0)
    total_nodes = len(positions) - 1  # exclude root
    total_thoughts = sum(r.get("num_thoughts") or 0 for r in rollouts)
    shared_nodes = sum(1 for n_id in positions if _node_shared(root, n_id))
    ax.text(
        0.01, 0.01,
        f"depth={-int(max_depth)}  nodes={total_nodes}  "
        f"avg_thoughts/rollout={total_thoughts / max(len(rollouts), 1):.1f}  "
        f"shared_nodes={shared_nodes}",
        transform=ax.transAxes, fontsize=8, color="#555555",
    )

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"wrote {out_path}")
    else:
        plt.show()


def _node_shared(root: TrieNode, target_id: int) -> bool:
    """True if any node with id==target_id has >1 rollout passing through it."""
    if id(root) == target_id:
        return len(root.rollouts) > 1
    for c in root.children.values():
        if _node_shared(c, target_id):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", type=Path, help="JSONL file or directory of JSONL files")
    ap.add_argument("--index", type=int, default=-1,
                    help="record index (default: -1 = last)")
    ap.add_argument("--latest", action="store_true",
                    help="if path is a dir, pick the most recent JSONL")
    ap.add_argument("--out", type=Path, default=None, help="output PNG path")
    ap.add_argument("--loss-dumps", type=Path, default=None,
                    help="optional path to loss dump JSONL file or directory; "
                         "enables gradient-overlay mode")
    args = ap.parse_args()

    path = args.path
    if path.is_dir():
        candidates = sorted(path.glob("*.jsonl"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            print(f"no .jsonl files in {path}", file=sys.stderr)
            sys.exit(1)
        path = candidates[-1] if args.latest else candidates[0]

    records = load_records(path)
    if not records:
        print(f"no records in {path}", file=sys.stderr)
        sys.exit(1)

    idx = args.index if args.index >= 0 else len(records) + args.index
    record = select_record(records, idx)
    print(f"rendering record {idx}/{len(records) - 1} from {path.name}")

    rollout_to_loss: dict | None = None
    stats: dict | None = None
    if args.loss_dumps is not None:
        loss_index = load_loss_dumps(args.loss_dumps)
        if not loss_index:
            print(f"warning: no loss records loaded from {args.loss_dumps}",
                  file=sys.stderr)
        else:
            rollout_to_loss, stats = match_losses_to_rollouts(
                record["rollouts"], loss_index
            )
            print(f"loss join: matched={stats['matched']}/{stats['total']}, "
                  f"ambiguous={stats['ambiguous']}, missed={stats['missed']}")

    render(record, out_path=args.out,
           rollout_to_loss=rollout_to_loss, loss_match_stats=stats)


if __name__ == "__main__":
    main()
