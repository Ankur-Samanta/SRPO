#!/usr/bin/env python3
"""
Parse ICS eval logs to compute comprehensive self-correction statistics.

Reads the log lines emitted by thought_ics_agent_loop and computes:

  Verifier confusion matrix (all steps — fresh + corrections):
    TP: self_verify=False, oracle=False  (correctly triggered correction)
    TN: self_verify=True,  oracle=True   (correctly skipped)
    FP: self_verify=False, oracle=True   (unnecessary correction)
    FN: self_verify=True,  oracle=False  (missed, no correction)
    → Accuracy, Precision, Recall, F1

  Fresh-chain-only verifier stats (same confusion matrix, only iter 0)

  Accuracy over iterations:
    iter 0 = fresh chain oracle accuracy
    iter k = oracle accuracy after k-th correction (only over triggered prompts)
    "oracle-best@k": fraction of prompts where ANY of the first k iters was correct

  Correction effectiveness (over triggered prompts):
    - Trigger rate: fraction of prompts where ICS fired ≥1 time
    - Fix rate: fraction of triggered prompts where final oracle=True
    - Before/after accuracy: fresh acc vs final acc on triggered subset
    - Flip rates: wrong→right, right→wrong, no-change

  Per-file table suitable for copy-paste into paper.

Usage:
    python parse_ics_eval.py <logfile.out> [<logfile2.out> ...]
    python parse_ics_eval.py batch_scripts/logs/eval_ics_scgrpo_oly_qwen7b_*.out
"""

import re
import sys
from collections import defaultdict

# ─── Log line patterns ────────────────────────────────────────────────────────

# [ICS] Fresh #N: K thoughts, self_verify=True/False, oracle=True/False
# [ICS] Fresh #N: K thoughts, correct=True/False          (oracle-gated mode)
FRESH_RE = re.compile(
    r"\[ICS\] Fresh #(\d+): (\d+) thoughts"
    r"(?:, self_verify=(\w+), oracle=(\w+))?"   # verifier mode
    r"(?:, correct=(\w+))?"                      # oracle mode
)

# [ICS] Trigger N iter K: M thoughts, self_verify=X, oracle=Y
# [ICS] Trigger N iter K: M thoughts, correct=Y
CORR_RE = re.compile(
    r"\[ICS\] Trigger (\d+) iter (\d+): (\d+) thoughts"
    r"(?:, self_verify=(\w+), oracle=(\w+))?"
    r"(?:, correct=(\w+))?"
)

# [ICS] Trigger N iter K: no error found, stopping
NO_ERR_RE = re.compile(
    r"\[ICS\] Trigger (\d+) iter (\d+): no error found, stopping"
)


def _parse_bool(s):
    return s == "True"


# ─── Core parser ─────────────────────────────────────────────────────────────

def parse_log(path: str) -> list:
    """
    Parse one log file.

    Returns list of per-prompt dicts, each with:
        steps: list of dicts — one per iteration (index 0 = fresh):
            {self_verify: bool|None, oracle: bool, no_error_stop: bool}
        triggered: bool — ICS fired at least once
        n_thoughts_fresh: int
    """
    per_prompt = []
    current = None  # list of step dicts for the current prompt

    with open(path) as f:
        for line in f:
            # Fresh chain → start new prompt
            m = FRESH_RE.search(line)
            if m:
                if current is not None:
                    per_prompt.append(current)
                n_thoughts = int(m.group(2))
                if m.group(3) is not None:  # verifier mode
                    sv = _parse_bool(m.group(3))
                    oracle = _parse_bool(m.group(4))
                elif m.group(5) is not None:  # oracle mode
                    oracle = _parse_bool(m.group(5))
                    sv = None
                else:
                    continue  # malformed
                current = {
                    "n_thoughts_fresh": n_thoughts,
                    "steps": [{"self_verify": sv, "oracle": oracle, "no_error_stop": False}],
                }
                continue

            # No-error stop (localization gave up)
            m = NO_ERR_RE.search(line)
            if m and current is not None:
                k = int(m.group(2))
                # Mark the step as a no-error stop (no oracle change — use last)
                while len(current["steps"]) <= k:
                    last = current["steps"][-1]["oracle"]
                    current["steps"].append({"self_verify": None, "oracle": last, "no_error_stop": True})
                current["steps"][k]["no_error_stop"] = True
                continue

            # Correction step
            m = CORR_RE.search(line)
            if m and current is not None:
                k = int(m.group(2))           # 1-indexed
                if m.group(4) is not None:    # verifier mode
                    sv = _parse_bool(m.group(4))
                    oracle = _parse_bool(m.group(5))
                elif m.group(6) is not None:  # oracle mode
                    oracle = _parse_bool(m.group(6))
                    sv = None
                else:
                    continue
                # Pad if gaps exist (shouldn't happen normally)
                while len(current["steps"]) < k:
                    last = current["steps"][-1]["oracle"]
                    current["steps"].append({"self_verify": None, "oracle": last, "no_error_stop": False})
                if len(current["steps"]) == k:
                    current["steps"].append({"self_verify": sv, "oracle": oracle, "no_error_stop": False})
                else:
                    current["steps"][k] = {"self_verify": sv, "oracle": oracle, "no_error_stop": False}

    if current is not None:
        per_prompt.append(current)

    return per_prompt


# ─── Stat helpers ─────────────────────────────────────────────────────────────

def confusion(steps_list, include_corrections=True):
    """
    Compute TP/TN/FP/FN treating self_verify as prediction of 'correct'.
    ICS fires when self_verify=False (model thinks it's wrong).
    Oracle=True means actually correct.

    Convention:
      Positive class = WRONG (oracle=False), ICS should fire.
      self_verify=False → model predicts WRONG → ICS fires → True if oracle=False.

      TP: self_verify=False & oracle=False  (fired correctly)
      TN: self_verify=True  & oracle=True   (held back correctly)
      FP: self_verify=False & oracle=True   (fired unnecessarily)
      FN: self_verify=True  & oracle=False  (missed a wrong answer)
    """
    tp = tn = fp = fn = 0
    for steps in steps_list:
        idxs = range(len(steps)) if include_corrections else [0]
        for i in idxs:
            s = steps[i]
            if s["self_verify"] is None or s["no_error_stop"]:
                continue
            sv = s["self_verify"]
            oracle = s["oracle"]
            if not sv and not oracle:   # predict wrong, is wrong → TP
                tp += 1
            elif sv and oracle:         # predict right, is right → TN
                tn += 1
            elif not sv and oracle:     # predict wrong, is right → FP
                fp += 1
            else:                       # predict right, is wrong → FN
                fn += 1
    return tp, tn, fp, fn


def verifier_metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return acc, prec, rec, f1


def accuracy_curve(per_prompt):
    """Oracle accuracy at each iteration index (all prompts; missing = last known)."""
    max_iters = max((len(p["steps"]) for p in per_prompt), default=0)
    correct_at = [[] for _ in range(max_iters)]
    for p in per_prompt:
        last_oracle = p["steps"][-1]["oracle"]
        for k in range(max_iters):
            if k < len(p["steps"]):
                correct_at[k].append(p["steps"][k]["oracle"])
            else:
                correct_at[k].append(last_oracle)
    return [sum(v) / len(v) if v else float("nan") for v in correct_at]


def oracle_best_at_k(per_prompt, k):
    """Fraction of prompts where ANY of the first k+1 iters was correct."""
    if not per_prompt:
        return float("nan")
    n_correct = 0
    for p in per_prompt:
        steps = p["steps"][:k + 1]
        if any(s["oracle"] for s in steps):
            n_correct += 1
    return n_correct / len(per_prompt)


def flip_stats(per_prompt):
    """Among triggered prompts: wrong→right, right→wrong, no-change."""
    triggered = [p for p in per_prompt if len(p["steps"]) > 1]
    if not triggered:
        return 0, 0, 0, 0
    wrong_to_right = sum(
        1 for p in triggered
        if not p["steps"][0]["oracle"] and p["steps"][-1]["oracle"]
    )
    right_to_wrong = sum(
        1 for p in triggered
        if p["steps"][0]["oracle"] and not p["steps"][-1]["oracle"]
    )
    no_change_correct = sum(
        1 for p in triggered
        if p["steps"][0]["oracle"] and p["steps"][-1]["oracle"]
    )
    no_change_wrong = sum(
        1 for p in triggered
        if not p["steps"][0]["oracle"] and not p["steps"][-1]["oracle"]
    )
    return wrong_to_right, right_to_wrong, no_change_correct, no_change_wrong


# ─── Report ──────────────────────────────────────────────────────────────────

def report(path: str):
    per_prompt = parse_log(path)
    n = len(per_prompt)

    print(f"\n{'='*70}")
    print(f"File: {path}")
    print(f"{'='*70}")

    if n == 0:
        print("  No ICS log lines found.")
        return

    print(f"  Prompts: {n}")

    # ── Verifier confusion matrix ──────────────────────────────────────────
    has_verifier = any(
        s["self_verify"] is not None
        for p in per_prompt for s in p["steps"]
    )
    steps_list = [p["steps"] for p in per_prompt]

    if has_verifier:
        print("\n  ── Verifier (self_verify vs oracle) ──────────────────────────")
        print(f"  {'':30s}  {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}  "
              f"{'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")

        # Fresh chains only
        tp, tn, fp, fn = confusion(steps_list, include_corrections=False)
        acc, prec, rec, f1 = verifier_metrics(tp, tn, fp, fn)
        print(f"  {'Fresh chains only':30s}  {tp:>5} {tn:>5} {fp:>5} {fn:>5}  "
              f"{acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")

        # All steps
        tp, tn, fp, fn = confusion(steps_list, include_corrections=True)
        acc, prec, rec, f1 = verifier_metrics(tp, tn, fp, fn)
        print(f"  {'All steps (fresh+corrections)':30s}  {tp:>5} {tn:>5} {fp:>5} {fn:>5}  "
              f"{acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")

        # Note on convention
        print("  (Positive=WRONG: TP=correctly triggered, TN=correctly skipped,")
        print("   FP=fired on correct answer, FN=missed wrong answer)")

    # ── Accuracy over iterations ───────────────────────────────────────────
    print("\n  ── Accuracy over iterations (oracle) ─────────────────────────")
    curve = accuracy_curve(per_prompt)
    max_iters = len(curve)
    print(f"  {'Iter':>6}  {'Oracle Acc':>10}  {'Best@k':>8}  {'N with data':>12}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*12}")
    for k, acc in enumerate(curve):
        label = "fresh" if k == 0 else f"corr {k}"
        n_data = sum(1 for p in per_prompt if k < len(p["steps"]))
        best_k = oracle_best_at_k(per_prompt, k)
        print(f"  {label:>6}  {acc:>10.4f}  {best_k:>8.4f}  {n_data:>12}")

    print(f"\n  Initial (iter 0):   {curve[0]:.4f}")
    if max_iters > 1:
        final_acc = curve[-1]
        print(f"  Final (last iter):  {final_acc:.4f}")
        print(f"  Delta:              {final_acc - curve[0]:+.4f}")
        print(f"  Best@{max_iters-1}:            {oracle_best_at_k(per_prompt, max_iters-1):.4f}")

    # ── Trigger + correction effectiveness ────────────────────────────────
    triggered = [p for p in per_prompt if len(p["steps"]) > 1]
    n_triggered = len(triggered)
    trigger_rate = n_triggered / n if n else float("nan")
    print(f"\n  ── Trigger & correction stats ────────────────────────────────")
    print(f"  Trigger rate:    {n_triggered}/{n} = {trigger_rate:.3f}")

    if triggered:
        wrong_to_right, right_to_wrong, nc_correct, nc_wrong = flip_stats(per_prompt)
        print(f"  Among triggered prompts ({n_triggered}):")
        print(f"    wrong→right:   {wrong_to_right:>4}  ({wrong_to_right/n_triggered:.3f})")
        print(f"    right→wrong:   {right_to_wrong:>4}  ({right_to_wrong/n_triggered:.3f})")
        print(f"    no-change✓:    {nc_correct:>4}  ({nc_correct/n_triggered:.3f})")
        print(f"    no-change✗:    {nc_wrong:>4}  ({nc_wrong/n_triggered:.3f})")

        fresh_acc_triggered = sum(
            1 for p in triggered if p["steps"][0]["oracle"]
        ) / n_triggered
        final_acc_triggered = sum(
            1 for p in triggered if p["steps"][-1]["oracle"]
        ) / n_triggered
        print(f"  Fresh acc (triggered subset):  {fresh_acc_triggered:.4f}")
        print(f"  Final acc (triggered subset):  {final_acc_triggered:.4f}")
        print(f"  Delta (triggered subset):      {final_acc_triggered - fresh_acc_triggered:+.4f}")

        avg_iters = sum(len(p["steps"]) - 1 for p in triggered) / n_triggered
        print(f"  Avg corrections per triggered: {avg_iters:.2f}")

    # ── Mean thoughts ──────────────────────────────────────────────────────
    avg_thoughts = sum(p["n_thoughts_fresh"] for p in per_prompt) / n
    print(f"\n  Avg thoughts (fresh chain):  {avg_thoughts:.2f}")


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        report(path)
