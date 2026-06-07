#!/usr/bin/env python3
"""Grade SRPO localization prompts with Claude Opus.

Reads prompts from logs/loc_grader/srpo_ep1_s42_prompts.jsonl, sends each
verbatim to Claude Opus, parses the \\boxed{N} answer, and appends to
logs/loc_grader/srpo_ep1_s42_opus.jsonl.

Resumable — skips ids already in the output file. Async with bounded
concurrency.

Usage:
    ANTHROPIC_API_KEY=sk-... python scripts/grade_localizations.py
    ANTHROPIC_API_KEY=sk-... python scripts/grade_localizations.py --concurrency 20
    ANTHROPIC_API_KEY=sk-... python scripts/grade_localizations.py --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "logs/loc_grader/srpo_ep1_s42_prompts.jsonl"
OUTPUT = ROOT / "logs/loc_grader/srpo_ep1_s42_opus.jsonl"

MODEL = "claude-opus-4-5"  # Opus 4.7 if available; fall back to 4.5
BOXED_RE = re.compile(r"\\boxed\{(\d+)\}")


def load_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(errors="ignore").splitlines():
        try:
            done.add(json.loads(line)["id"])
        except Exception:
            continue
    return done


def parse_boxed(text: str) -> int | None:
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


async def grade_one(client, sem, rec: dict, max_retries: int = 4) -> dict:
    async with sem:
        prompt = rec["prompt"]
        attempt = 0
        last_err = None
        while attempt < max_retries:
            try:
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text"
                )
                return {
                    "id": rec["id"],
                    "rec_idx": rec["rec_idx"],
                    "sub_idx": rec["sub_idx"],
                    "local_step": rec["local_step"],
                    "local_n_steps": rec["local_n_steps"],
                    "frontier_step": parse_boxed(text),
                    "frontier_raw": text,
                    "model": MODEL,
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                }
            except anthropic.RateLimitError as e:
                last_err = e
                wait = 2 ** attempt + 1
                await asyncio.sleep(wait)
                attempt += 1
            except (anthropic.APIError, anthropic.APIConnectionError) as e:
                last_err = e
                wait = 2 ** attempt + 1
                await asyncio.sleep(wait)
                attempt += 1
        return {"id": rec["id"], "error": str(last_err)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--input", default=str(INPUT))
    ap.add_argument("--output", default=str(OUTPUT))
    args = ap.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_recs = [json.loads(l) for l in inp.read_text().splitlines() if l]
    done_ids = load_done_ids(out)
    todo = [r for r in all_recs if r["id"] not in done_ids]
    if args.limit:
        todo = todo[: args.limit]
    print(f"total={len(all_recs)} done={len(done_ids)} todo={len(todo)} model={MODEL}")

    if not todo:
        print("nothing to do")
        return

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)

    t0 = time.time()
    completed = 0
    parsed_ok = 0
    in_tok = 0
    out_tok = 0

    # Append-only as we go (resumable)
    out_fp = open(out, "a", encoding="utf-8")

    async def task(rec):
        nonlocal completed, parsed_ok, in_tok, out_tok
        result = await grade_one(client, sem, rec)
        out_fp.write(json.dumps(result) + "\n")
        out_fp.flush()
        completed += 1
        if result.get("frontier_step") is not None:
            parsed_ok += 1
        in_tok += result.get("input_tokens", 0)
        out_tok += result.get("output_tokens", 0)
        if completed % 25 == 0 or completed == len(todo):
            dt = time.time() - t0
            rate = completed / dt
            eta = (len(todo) - completed) / max(rate, 1e-6)
            print(
                f"[{completed:>4}/{len(todo)}] "
                f"parsed={parsed_ok}  in={in_tok/1e6:.2f}M out={out_tok/1e6:.3f}M  "
                f"{rate:.1f}/s  ETA {eta/60:.1f}m"
            )

    await asyncio.gather(*(task(r) for r in todo))
    out_fp.close()
    print(f"done. wrote {completed} records to {out}")


if __name__ == "__main__":
    asyncio.run(main())
