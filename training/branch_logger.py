"""Opt-in branch structure logger for SCGRPO analysis.

Dumps one JSONL record per rollout group (8 trajectories per prompt) capturing
the data needed to reconstruct the branching tree offline:
  - decoded_thoughts per trajectory (for prefix hashing)
  - thought_boundaries (for later correlating to per-token gradient signal)
  - ICS metadata (error steps, corrections, per-iter oracle correctness)

Enable with env vars (all optional — no-op if SCGRPO_BRANCH_DUMP_DIR unset):
  SCGRPO_BRANCH_DUMP_DIR=/path/to/dir       # output directory (required to enable)
  SCGRPO_BRANCH_DUMP_EVERY=50               # dump every Nth prompt (default: 50)
  SCGRPO_BRANCH_DUMP_MAX=1000               # stop after N total dumps (default: unlimited)

Output: one JSONL file per process (PID-scoped), appended as training progresses.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional


def _fingerprint_segment_lengths(segment_lengths: list[int]) -> str:
    """Stable fingerprint over the per-thought token-count pattern.

    Two rollouts collide iff they produced the same number of thoughts with
    the same token count in each corresponding position. Used to join
    rollout-time branch dumps to loss-time per-thought loss dumps offline.
    """
    payload = json.dumps(segment_lengths, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]

logger = logging.getLogger(__name__)

# Module-level state (single-threaded asyncio — no lock needed)
_call_count: int = 0
_dump_count: int = 0
_output_path: Optional[Path] = None


def _resolve_output_path() -> Optional[Path]:
    """Lazily resolve the output path on first call. Returns None if disabled."""
    global _output_path
    if _output_path is not None:
        return _output_path

    dump_dir = os.environ.get("SCGRPO_BRANCH_DUMP_DIR", "").strip()
    if not dump_dir:
        return None

    try:
        out_dir = Path(dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _output_path = out_dir / f"branches_pid{os.getpid()}_{int(time.time())}.jsonl"
        logger.warning(f"[BranchLogger] Dumping rollout groups to {_output_path}")
    except Exception as e:
        logger.error(f"[BranchLogger] Failed to init output dir: {e}")
        return None

    return _output_path


def dump_branch_group(
    question: str,
    trajectories: list,
    ics_stats: dict,
    ground_truth: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Dump one rollout group. Silent no-op if disabled or on any error.

    Args:
        question: the prompt question text.
        trajectories: list of _ThoughtChainResult (buffer.trajectories).
        ics_stats: the ics_stats dict from the ICS coordinator.
        ground_truth: optional ground-truth answer (for reference).
        extra: optional dict of extra fields to attach (e.g. step number).
    """
    global _call_count, _dump_count

    path = _resolve_output_path()
    if path is None:
        return

    _call_count += 1

    try:
        every = int(os.environ.get("SCGRPO_BRANCH_DUMP_EVERY", "50"))
        max_dumps = int(os.environ.get("SCGRPO_BRANCH_DUMP_MAX", "0") or "0")
    except ValueError:
        every = 50
        max_dumps = 0

    if every < 1:
        every = 1
    if _call_count % every != 0:
        return
    if max_dumps > 0 and _dump_count >= max_dumps:
        return

    try:
        iter_oracle = ics_stats.get("iter_oracle_correct", []) if ics_stats else []

        rollouts: list[dict[str, Any]] = []
        for i, traj in enumerate(trajectories):
            oracle = None
            if i < len(iter_oracle):
                oracle = bool(iter_oracle[i])

            boundaries = [
                list(b) for b in (getattr(traj, "thought_boundaries", []) or [])
            ]
            segment_lengths = [int(end - start) for start, end in boundaries]

            rollouts.append(
                {
                    "slot": i,
                    "num_thoughts": getattr(traj, "num_thoughts", None),
                    "found_answer": getattr(traj, "found_answer", None),
                    "oracle_correct": oracle,
                    "decoded_thoughts": list(getattr(traj, "decoded_thoughts", []) or []),
                    "thought_boundaries": boundaries,
                    "segment_lengths": segment_lengths,
                    "segment_fp": _fingerprint_segment_lengths(segment_lengths),
                    "response_len": len(getattr(traj, "response_ids", []) or []),
                }
            )

        record = {
            "ts": time.time(),
            "question": question[:500],
            "ground_truth": ground_truth[:200] if ground_truth else None,
            "ics_stats": dict(ics_stats) if ics_stats else {},
            "rollouts": rollouts,
        }
        if extra:
            record["extra"] = extra

        # Append as one line. Python I/O on append is atomic for small writes
        # on POSIX; this is single-threaded asyncio anyway.
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        _dump_count += 1

    except Exception as e:
        logger.error(f"[BranchLogger] Dump failed: {e}")
