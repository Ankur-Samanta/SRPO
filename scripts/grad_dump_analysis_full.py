#!/usr/bin/env python3
"""
Per-step / per-thought gradient analysis for the FULL-EPOCH dump run.

Inputs:
    logs/srpo_per_token/grad_dump_l2n_ep1_full/loss_pid*_call*.pt
    logs/srpo_localizations/grad_dump_l2n_ep1_full/branches_pid*.jsonl

Forked from grad_dump_analysis.py. The single-step version lumped all microbatch
dumps together; here each .pt file's training step is derived from its call_idx,
each branch jsonl line's training step from its line number, and v1's match /
aggregate logic runs per-step. Output CSVs add a `step` column, plus a new
per_step.csv with concentration metrics aggregated across the 32 prompts in
each step.

Step grouping:
    branches: line i → step i // PROMPTS_PER_STEP
    loss:     call_idx c → step c // CALLS_PER_RANK_PER_STEP
    Both inferred from data (n_steps = len(branches) // PROMPTS_PER_STEP).

Outputs (CSVs in scripts/grad_dump_analysis_full_out/):
    per_trajectory.csv   one row per (step, prompt, slot)
    per_thought.csv      one row per (step, prompt, slot, thought_idx)
    per_prompt.csv       one row per (step, prompt) — concentration metrics
    per_step.csv         one row per step — cross-prompt aggregates
    summary.txt          per-step + cross-step tables
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch

# ─── Paths ──────────────────────────────────────────────────────────────────
SCPO_DIR = Path(__file__).resolve().parents[1]
RUN_NAME = "grad_dump_l2n_ep1_full"
LOSS_DIR = SCPO_DIR / "logs" / "srpo_per_token" / RUN_NAME
BRANCH_FILE = next((SCPO_DIR / "logs" / "srpo_localizations" / RUN_NAME).glob("branches_pid*.jsonl"))
OUT_DIR = SCPO_DIR / "scripts" / "grad_dump_analysis_full_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS_PER_STEP = 32  # config default data.train_batch_size


# ═══════════════════════════════════════════════════════════════════════════
# 1. Loaders (with step tagging)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LossRow:
    file: str
    row_in_file: int
    pid: str
    call_idx: int
    step: int
    rl: int
    ssi: int
    adv: float
    prompt_fp: str   # sha1 of the active prompt token sequence — uniquely identifies the prompt
    log_prob: torch.Tensor
    old_log_prob: torch.Tensor
    response_mask: torch.Tensor
    suffix_mask: torch.Tensor


def _prompt_fingerprint(input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        response_mask_short: torch.Tensor) -> str:
    """Hash the active prompt tokens.

    input_ids/attention_mask are (seq_len_total,); response_mask is (response_len,)
    covering only the response portion at the end. Active prompt tokens are
    attention=1 AND not in the response.
    """
    pl = input_ids.shape[0] - response_mask_short.shape[0]
    full_rm = torch.zeros_like(attention_mask)
    full_rm[pl:] = response_mask_short
    is_prompt = (attention_mask == 1) & (full_rm == 0)
    prompt_tokens = input_ids[is_prompt]
    return hashlib.sha1(prompt_tokens.cpu().numpy().tobytes()).hexdigest()[:16]


def load_loss_rows() -> tuple[list[LossRow], int, int]:
    """Return (rows, n_steps, calls_per_rank_per_step)."""
    files = sorted(LOSS_DIR.glob("loss_pid*_call*.pt"))
    pids = sorted({re.search(r"pid(\d+)", f.name).group(1) for f in files})
    calls_per_rank = {p: sum(1 for f in files if f"pid{p}_" in f.name) for p in pids}
    n_branches = sum(1 for _ in open(BRANCH_FILE))
    n_steps = n_branches // PROMPTS_PER_STEP
    cps = next(iter(calls_per_rank.values())) // n_steps
    assert all(c // n_steps == cps for c in calls_per_rank.values()), \
        f"uneven calls per rank: {calls_per_rank}"

    rows: list[LossRow] = []
    for fp in files:
        m = re.search(r"pid(\d+)_call(\d+)", fp.name)
        pid, call_idx = m.group(1), int(m.group(2))
        step = call_idx // cps
        d = torch.load(fp, weights_only=False)
        rl_t = d["response_mask"].sum(dim=-1)
        ssi_t = d["suffix_start_idx"]
        adv_t = d["reset_advantage"]
        for i in range(d["log_prob"].shape[0]):
            pfp = _prompt_fingerprint(d["input_ids"][i], d["attention_mask"][i], d["response_mask"][i])
            rows.append(LossRow(
                file=fp.name, row_in_file=i, pid=pid, call_idx=call_idx, step=step,
                rl=int(rl_t[i]), ssi=int(ssi_t[i]), adv=float(adv_t[i]),
                prompt_fp=pfp,
                log_prob=d["log_prob"][i].float(),
                old_log_prob=d["old_log_prob"][i].float(),
                response_mask=d["response_mask"][i].bool(),
                suffix_mask=d["suffix_mask"][i].bool(),
            ))
    return rows, n_steps, cps


@dataclass
class BranchRollout:
    step: int
    prompt_idx: int          # global line index
    prompt_in_step: int      # 0..PROMPTS_PER_STEP-1
    slot: int
    response_len: int
    decoded_thoughts: list[str]
    thought_boundaries: list[tuple[int, int]]
    segment_lengths: list[int]
    oracle_correct: Optional[bool]
    question: str
    inferred_ssi: int = 0
    role: str = "G1"


def load_branch_rollouts() -> list[BranchRollout]:
    rollouts: list[BranchRollout] = []
    for prompt_idx, line in enumerate(open(BRANCH_FILE)):
        rec = json.loads(line)
        step = prompt_idx // PROMPTS_PER_STEP
        pis = prompt_idx % PROMPTS_PER_STEP
        for ro in rec["rollouts"]:
            rollouts.append(BranchRollout(
                step=step, prompt_idx=prompt_idx, prompt_in_step=pis,
                slot=ro["slot"], response_len=ro["response_len"],
                decoded_thoughts=ro.get("decoded_thoughts", []),
                thought_boundaries=[tuple(b) for b in ro.get("thought_boundaries", [])],
                segment_lengths=ro.get("segment_lengths", []),
                oracle_correct=ro.get("oracle_correct"),
                question=rec["question"][:200],
            ))
    return rollouts


def load_prompt_meta() -> dict[int, dict]:
    """Per-prompt outcome / localization fields from branches.jsonl.

    K_g1 / K_g2 are the count of oracle-correct rollouts in each group of 4.
    ics_corrected is the agent-reported flag for "at least one G2 correction
    landed the right answer" on this prompt (matches K_g2 > 0 by construction
    in our srpo setup, but we read what the agent wrote).
    """
    out: dict[int, dict] = {}
    for prompt_idx, line in enumerate(open(BRANCH_FILE)):
        rec = json.loads(line)
        ics = rec.get("ics_stats", {}) or {}
        loc_n = ics.get("ics_loc_n_steps") or []
        out[prompt_idx] = {
            "step": prompt_idx // PROMPTS_PER_STEP,
            "K_g1": sum(int(bool(r.get("oracle_correct"))) for r in rec["rollouts"] if r["slot"] < 4),
            "K_g2": sum(int(bool(r.get("oracle_correct"))) for r in rec["rollouts"] if r["slot"] >= 4),
            "ics_triggered": bool(ics.get("ics_triggered", False)),
            "ics_corrected": bool(ics.get("ics_corrected", False)),
            "ics_iterations": int(ics.get("ics_iterations", 0) or 0),
            "ics_loc_n_steps": int(loc_n[0]) if loc_n else math.nan,
        }
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. SSI inference (per prompt — same logic as v1)
# ═══════════════════════════════════════════════════════════════════════════

def infer_ssi(rollouts_by_prompt: list[list[BranchRollout]]) -> None:
    for ro_list in rollouts_by_prompt:
        g2 = sorted([r for r in ro_list if r.slot >= 4], key=lambda r: r.slot)
        if len(g2) >= 2:
            shared = 0
            min_len = min(len(r.decoded_thoughts) for r in g2)
            for k in range(min_len):
                ref = g2[0].decoded_thoughts[k]
                if all(r.decoded_thoughts[k] == ref for r in g2):
                    shared += 1
                else:
                    break
            ssi = sum(g2[0].segment_lengths[:shared])
            for r in g2:
                r.inferred_ssi = ssi
                r.role = "G2-corr"
        for r in ro_list:
            if r.slot < 4:
                r.role = "G1"
                r.inferred_ssi = 0


# ═══════════════════════════════════════════════════════════════════════════
# 3. Match loss rows ↔ branch rollouts (scoped per prompt cluster)
# ═══════════════════════════════════════════════════════════════════════════
#
# Per step:
#   1. Cluster loss rows by prompt_fp → 32 groups of 8 (all 8 rolls of a prompt
#      share identical prompt tokens by construction).
#   2. Align clusters to the 32 branch lines by sorted rl multiset (8 ints) —
#      vanishingly unlikely to collide across prompts. ssi is NOT used for
#      alignment because the branch-side inference is off-by-one from the
#      loss-side suffix_start_idx (branch counts the boundary token, loss
#      uses an exclusive index).
#   3. Within each (cluster, branch_line) pair, split each side by group
#      (G1: ssi==0, G2-corr: ssi>0) and match by rl exact + adv-rank. Within a
#      (rl, group) bucket, sort loss rows by adv descending and branch rollouts
#      by oracle_correct. Truly tied entries (same rl, same group, same
#      predicted adv) are interchangeable for any adv-weighted aggregation.

def _rl_multiset(items, rl_attr) -> tuple:
    return tuple(sorted(getattr(r, rl_attr) for r in items))


def match_rollouts_to_loss_per_step(
    rollouts: list[BranchRollout],
    loss_rows: list[LossRow],
) -> dict[tuple[int, int, int], LossRow]:
    """Return (step, prompt_idx, slot) → LossRow. Deterministic; no None values."""
    rollouts_by_step_prompt: dict[tuple[int, int], list[BranchRollout]] = defaultdict(list)
    for r in rollouts:
        rollouts_by_step_prompt[(r.step, r.prompt_idx)].append(r)
    loss_by_step_fp: dict[tuple[int, str], list[LossRow]] = defaultdict(list)
    for lr in loss_rows:
        loss_by_step_fp[(lr.step, lr.prompt_fp)].append(lr)

    out: dict[tuple[int, int, int], LossRow] = {}

    prompts_by_step: dict[int, list[int]] = defaultdict(list)
    fps_by_step: dict[int, list[str]] = defaultdict(list)
    for (step, p_idx) in rollouts_by_step_prompt:
        prompts_by_step[step].append(p_idx)
    for (step, fp_) in loss_by_step_fp:
        fps_by_step[step].append(fp_)

    for step in sorted(prompts_by_step):
        prompt_keys = {
            p_idx: _rl_multiset(rollouts_by_step_prompt[(step, p_idx)], "response_len")
            for p_idx in prompts_by_step[step]
        }
        cluster_keys = {
            fp_: _rl_multiset(loss_by_step_fp[(step, fp_)], "rl")
            for fp_ in fps_by_step[step]
        }

        prompts_by_key: dict[tuple, list[int]] = defaultdict(list)
        for p_idx, k in prompt_keys.items():
            prompts_by_key[k].append(p_idx)
        clusters_by_key: dict[tuple, list[str]] = defaultdict(list)
        for fp_, k in cluster_keys.items():
            clusters_by_key[k].append(fp_)

        fp_to_prompt: dict[str, int] = {}
        for k, p_list in prompts_by_key.items():
            fp_list = clusters_by_key.get(k, [])
            assert len(p_list) == len(fp_list), (
                f"step {step}: rl-multiset {k} matches {len(p_list)} prompts "
                f"but {len(fp_list)} loss clusters"
            )
            for p_idx, fp_ in zip(sorted(p_list), sorted(fp_list)):
                fp_to_prompt[fp_] = p_idx

        for fp_, p_idx in fp_to_prompt.items():
            cluster_rows = loss_by_step_fp[(step, fp_)]
            prompt_ros = rollouts_by_step_prompt[(step, p_idx)]

            # Match within cluster by rl exact. (rl) is on both sides exactly;
            # ssi has known off-by-one issues and edge cases where G2 corrections
            # collapse to ssi=0, so we don't split by group up front.
            by_rl_lr: dict[int, list[LossRow]] = defaultdict(list)
            for lr in cluster_rows:
                by_rl_lr[lr.rl].append(lr)
            by_rl_ro: dict[int, list[BranchRollout]] = defaultdict(list)
            for ro in prompt_ros:
                by_rl_ro[ro.response_len].append(ro)

            for rl_, ro_group in by_rl_ro.items():
                lr_group = by_rl_lr.get(rl_, [])
                assert len(lr_group) == len(ro_group), (
                    f"step {step}, prompt {p_idx}, rl={rl_}: "
                    f"{len(ro_group)} branch vs {len(lr_group)} loss"
                )
                # Within an (rl) bucket on a single prompt, sort by (adv on loss,
                # oracle on branch). Sign convention: oracle_correct True →
                # positive group-relative adv. Within identical (rl, adv, oracle)
                # sub-buckets the assignment is arbitrary but the trajectories
                # are interchangeable for any adv-weighted aggregation.
                ro_sorted = sorted(ro_group,
                                   key=lambda r: (r.oracle_correct is True, r.slot),
                                   reverse=True)
                lr_sorted = sorted(lr_group, key=lambda r: r.adv, reverse=True)
                for ro, lr in zip(ro_sorted, lr_sorted):
                    out[(step, ro.prompt_idx, ro.slot)] = lr

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. Per-token / per-thought metrics — implements the appendix's g_{i,t}
# ═══════════════════════════════════════════════════════════════════════════
#
# Per-token signal (paper App. H, eq. per_token_signal):
#     g_{i,t} = |A_i| / T_i * (1 - π_θ(y_{i,t}|·)) * 1[t ∈ active_i]
#
# Implementation notes:
#   - T_i = active-token count = sum of suffix_mask (= response_mask for base
#     since suffix_start_idx=0 there; = suffix-only region for SP).
#   - π_θ(y_t|·) = exp(log_prob[t]); clamp (1-π) to [0, 1] for FP safety.
#   - This is the chosen-token logit-gradient magnitude. The full per-position
#     gradient norm also includes off-token components proportional to π_v
#     (v ≠ y_t), which we cannot recover from the dump (full logits aren't
#     dumped). The appendix is explicit that g_{i,t} is the chosen-token entry.

def per_token_signal(lr: LossRow) -> torch.Tensor:
    """Paper's g_{i,t} as a (response_length,) tensor (zeros outside active region)."""
    T = max(int(lr.suffix_mask.sum().item()), 1)
    one_minus_pi = (1.0 - lr.log_prob.exp()).clamp(min=0.0, max=1.0)
    return (abs(lr.adv) / T) * one_minus_pi * lr.suffix_mask.float()


def per_thought_grad_signal(ro: BranchRollout, lr: LossRow) -> list[dict]:
    """Per-thought aggregates of g_{i,t} (paper's bar g_{i,h})."""
    g_tok = per_token_signal(lr)
    one_minus_pi = (1.0 - lr.log_prob.exp()).clamp(min=0.0, max=1.0)
    out = []
    for t_idx, (s, e) in enumerate(ro.thought_boundaries):
        if e <= s: continue
        thought_g = g_tok[s:e]
        thought_omp = one_minus_pi[s:e]
        thought_mask = lr.suffix_mask[s:e].bool()
        active = int(thought_mask.sum().item())
        out.append({
            "step": ro.step, "prompt": ro.prompt_idx, "slot": ro.slot, "role": ro.role,
            "thought_idx": t_idx, "tok_start": s, "tok_end": e, "n_tok": e - s,
            "n_active_tok": active,
            "thought_in_suffix": s >= ro.inferred_ssi,
            "sum_g": float(thought_g.sum().item()),
            "mean_g_active": (
                float(thought_g[thought_mask].mean().item()) if active > 0 else 0.0
            ),
            "mean_one_minus_pi_active": (
                float(thought_omp[thought_mask].mean().item()) if active > 0 else 0.0
            ),
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 5. Concentration metrics + factor decomposition (per prompt)
# ═══════════════════════════════════════════════════════════════════════════
#
# Per-rollout signal: bar g_i = (|A_i|/T_i) * mean_t∈active (1 - π_t)
#   This decomposes g multiplicatively into three per-rollout factors:
#     |A_i|             — advantage magnitude
#     1/T_i             — inverse active length (1/T = how the mask concentrates)
#     overline{(1-π)}_i — mean self-confidence-complement over active tokens
#
# Cross-group ratio per step:
#     g_ratio = E_SP[bar g_i] / E_base[bar g_i]
#   factors multiplicatively (in expectation) as:
#     g_ratio ≈ E_SP[|A|/T] * E_SP[(1-π)]    /    E_base[|A|/T] * E_base[(1-π)]

def compute_concentration_per_prompt(
    per_traj_rows: list[dict],
    prompt_meta: dict[int, dict] | None = None,
) -> dict[tuple[int, int], dict]:
    """Per (step, prompt): C_arch, C_w_g2, C_signal + per-group factor means.

    If prompt_meta is provided, merges per-prompt outcome / localization fields
    (K_g1, K_g2, ics_corrected, ics_iterations, ics_loc_n_steps) into the
    per-prompt record so they can be aggregated per-step downstream.
    """
    by_sp = defaultdict(list)
    for r in per_traj_rows:
        by_sp[(r["step"], r["prompt"])].append(r)

    out: dict[tuple[int, int], dict] = {}
    for (step, p_idx), rows in by_sp.items():
        g1 = [r for r in rows if r["role"] == "G1"]
        gc = [r for r in rows if r["role"] == "G2-corr"]

        g1_lens = [r["rl"] for r in g1]
        gc_suffix_lens = [r["rl"] - r["ssi"] for r in gc]
        gc_total_lens = [r["rl"] for r in gc]

        c_arch = (
            (sum(g1_lens) / len(g1_lens)) / (sum(gc_suffix_lens) / len(gc_suffix_lens))
            if g1_lens and gc_suffix_lens else math.nan
        )
        c_within_g2 = (
            (sum(gc_total_lens) / len(gc_total_lens)) / (sum(gc_suffix_lens) / len(gc_suffix_lens))
            if gc_total_lens and gc_suffix_lens else math.nan
        )

        # C_signal: per-rollout bar g_i ratio (uses paper's exact formula)
        g1_g = [r["mean_g_active"] for r in g1]
        gc_g = [r["mean_g_active"] for r in gc]
        # Use rollouts with nonzero adv only for the ratio (zero-adv rollouts
        # have g_i = 0 by construction; including them just dilutes both sides
        # equally for the *mean*, but they would dominate any geometric mean).
        g1_g_nz = [g for g, r in zip(g1_g, g1) if abs(r["adv"]) > 0]
        gc_g_nz = [g for g, r in zip(gc_g, gc) if abs(r["adv"]) > 0]
        c_sig = (
            (sum(gc_g_nz) / len(gc_g_nz)) / (sum(g1_g_nz) / len(g1_g_nz))
            if g1_g_nz and gc_g_nz else math.nan
        )

        ssi_norm = (gc[0]["ssi"] / (sum(gc_total_lens) / len(gc_total_lens))
                    if gc and gc_total_lens else math.nan)

        # ─── Per-group factor means (for App. H decomposition) ─────────────
        # Per-rollout factors (rollout-weighted means):
        #   |A_i|, T_i, |A_i|/T_i, overline{(1-π)}_i, bar g_i
        # Plus zero_adv_frac: rate at which group-relative outcome normalization
        # collapses to zero (group all-same-outcome → std=0 → A=0 → g=0).
        # Computed both unconditionally and conditionally on nonzero-adv
        # (the "pairwise nonzero" subpopulation that isolates rollouts that
        # actually pass gradient through the policy).
        def _mean(xs):
            return sum(xs) / len(xs) if xs else math.nan
        def _factors(group):
            return {
                "abs_adv": _mean([abs(r["adv"]) for r in group]),
                "T":       _mean([r["active_tok_count"] for r in group]),
                "aT":      _mean([abs(r["adv"]) / max(r["active_tok_count"], 1) for r in group]),
                "ompi":    _mean([r["mean_one_minus_pi_active"] for r in group]),
                "g":       _mean([r["mean_g_active"] for r in group]),
                "zero_adv_frac": (
                    sum(1 for r in group if r["adv"] == 0) / len(group) if group else math.nan
                ),
            }
        f_base = _factors(g1)
        f_sp   = _factors(gc)

        # ─── Conditional factor means (nonzero-adv rollouts only) ──────────
        g1_nz = [r for r in g1 if abs(r["adv"]) > 0]
        gc_nz = [r for r in gc if abs(r["adv"]) > 0]
        f_base_nz = _factors(g1_nz)
        f_sp_nz   = _factors(gc_nz)
        # Both-nonzero indicator: True only if each group has ≥1 nonzero rollout
        # in this prompt. The conditional analysis is restricted to these prompts
        # at the step level — those are the only prompts where both groups
        # contribute nonzero per-token gradient signal in the same update.
        both_nonzero = bool(g1_nz) and bool(gc_nz)

        out[(step, p_idx)] = {
            "step": step, "prompt": p_idx,
            "ssi": gc[0]["ssi"] if gc else 0,
            "ssi_norm": ssi_norm,
            "g1_mean_len": sum(g1_lens) / len(g1_lens) if g1_lens else 0,
            "gc_mean_total_len": sum(gc_total_lens) / len(gc_total_lens) if gc_total_lens else 0,
            "gc_mean_suffix_len": sum(gc_suffix_lens) / len(gc_suffix_lens) if gc_suffix_lens else 0,
            "c_arch": c_arch,
            "c_within_g2": c_within_g2,
            "c_signal": c_sig,
            "n_g1_nonzero": len(g1_g_nz),
            "n_gc_nonzero": len(gc_g_nz),
            # Unconditional factor decomposition (per-prompt rollout-weighted means)
            "abs_adv_base": f_base["abs_adv"], "abs_adv_sp": f_sp["abs_adv"],
            "T_base":       f_base["T"],       "T_sp":       f_sp["T"],
            "aT_base":      f_base["aT"],      "aT_sp":      f_sp["aT"],
            "ompi_base":    f_base["ompi"],    "ompi_sp":    f_sp["ompi"],
            "g_base":       f_base["g"],       "g_sp":       f_sp["g"],
            "zero_adv_frac_base": f_base["zero_adv_frac"],
            "zero_adv_frac_sp":   f_sp["zero_adv_frac"],
            # Conditional (nonzero-adv only) factors; NaN if no nonzero rollouts in that group
            "abs_adv_base_nz": f_base_nz["abs_adv"], "abs_adv_sp_nz": f_sp_nz["abs_adv"],
            "T_base_nz":       f_base_nz["T"],       "T_sp_nz":       f_sp_nz["T"],
            "aT_base_nz":      f_base_nz["aT"],      "aT_sp_nz":      f_sp_nz["aT"],
            "ompi_base_nz":    f_base_nz["ompi"],    "ompi_sp_nz":    f_sp_nz["ompi"],
            "g_base_nz":       f_base_nz["g"],       "g_sp_nz":       f_sp_nz["g"],
            "both_nonzero":    both_nonzero,
        }
        if prompt_meta is not None and p_idx in prompt_meta:
            meta = prompt_meta[p_idx]
            out[(step, p_idx)].update({
                "K_g1": meta["K_g1"],
                "K_g2": meta["K_g2"],
                "ics_corrected": meta["ics_corrected"],
                "ics_iterations": meta["ics_iterations"],
                "ics_loc_n_steps": meta["ics_loc_n_steps"],
            })
    return out


def compute_per_step_aggregates(per_prompt: dict[tuple[int, int], dict]) -> list[dict]:
    """Aggregate concentration across prompts within each step."""
    by_step = defaultdict(list)
    for (step, _), d in per_prompt.items():
        by_step[step].append(d)

    def agg(vals):
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        if not vals:
            return {"mean": math.nan, "median": math.nan, "geomean": math.nan,
                    "min": math.nan, "max": math.nan, "n": 0}
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "geomean": statistics.geometric_mean([max(v, 1e-6) for v in vals]),
            "min": min(vals), "max": max(vals), "n": len(vals),
        }

    rows = []
    for step in sorted(by_step):
        ds = by_step[step]
        row = {"step": step, "n_prompts": len(ds)}
        for key in ("c_arch", "c_within_g2", "c_signal", "ssi_norm",
                    "g1_mean_len", "gc_mean_total_len", "gc_mean_suffix_len"):
            a = agg([d[key] for d in ds])
            row[f"{key}_mean"] = a["mean"]
            row[f"{key}_median"] = a["median"]
            row[f"{key}_geomean"] = a["geomean"]
            row[f"{key}_min"] = a["min"]
            row[f"{key}_max"] = a["max"]
        # ── Unconditional factor decomposition ──────────────────────────────
        # Step-level value = mean across all 32 prompts (each contributes one
        # per-prompt rollout-weighted mean of the factor).
        for fac in ("abs_adv", "T", "aT", "ompi", "g", "zero_adv_frac"):
            for grp in ("base", "sp"):
                col = f"{fac}_{grp}"
                row[col] = sum(d[col] for d in ds) / len(ds) if ds else math.nan
        for fac in ("abs_adv", "T", "aT", "ompi", "g"):
            num = row[f"{fac}_sp"]
            den = row[f"{fac}_base"]
            row[f"{fac}_sp_over_base"] = (num / den) if den else math.nan

        # ── Conditional factor decomposition (both-nonzero prompts only) ────
        # Average per-prompt conditional factors across the subset of prompts
        # where both groups have ≥1 nonzero-adv rollout. This isolates "what
        # do the per-token gradients look like when both groups are actually
        # contributing signal in the same update?"
        ds_nz = [d for d in ds if d["both_nonzero"]]
        row["n_prompts_both_nonzero"] = len(ds_nz)
        row["frac_prompts_both_nonzero"] = len(ds_nz) / len(ds) if ds else math.nan
        for fac in ("abs_adv", "T", "aT", "ompi", "g"):
            for grp in ("base", "sp"):
                col_nz = f"{fac}_{grp}_nz"
                vals = [d[col_nz] for d in ds_nz
                        if not (isinstance(d[col_nz], float) and math.isnan(d[col_nz]))]
                row[col_nz] = (sum(vals) / len(vals)) if vals else math.nan
            num = row[f"{fac}_sp_nz"]
            den = row[f"{fac}_base_nz"]
            row[f"{fac}_sp_over_base_nz"] = (num / den) if den else math.nan

        # ── Outcome rates per step (from prompt_meta if present) ────────────
        # G1/G2 pass rates are rollout-level (K out of 4 per prompt → averaged
        # over 4 rolls × n_prompts rollouts). ics_corrected, any_g1, any_g2,
        # any_either, and rescue are prompt-level (one number per prompt).
        if all("K_g1" in d for d in ds):
            n_prompts = len(ds)
            row["g1_pass_rate"] = sum(d["K_g1"] for d in ds) / (4 * n_prompts)
            row["g2_pass_rate"] = sum(d["K_g2"] for d in ds) / (4 * n_prompts)
            row["any_g1_rate"]  = sum(1 for d in ds if d["K_g1"] > 0) / n_prompts
            row["any_g2_rate"]  = sum(1 for d in ds if d["K_g2"] > 0) / n_prompts
            row["any_either_rate"] = sum(1 for d in ds if d["K_g1"] > 0 or d["K_g2"] > 0) / n_prompts
            row["ics_corrected_rate"] = sum(1 for d in ds if d["ics_corrected"]) / n_prompts
            # Rescue: among prompts where G1 group failed (K_g1=0), what fraction
            # had at least one successful G2-correction? Conditional probability.
            failed_g1 = [d for d in ds if d["K_g1"] == 0]
            row["rescue_rate"] = (
                sum(1 for d in failed_g1 if d["K_g2"] > 0) / len(failed_g1)
                if failed_g1 else math.nan
            )
            row["n_failed_g1"] = len(failed_g1)
            # Lift: P(any pass | G1 ∪ G2) − P(any pass | G1) — incremental yield
            # from including the SP group's outcomes.
            row["lift"] = row["any_either_rate"] - row["any_g1_rate"]
            # Mean localization depth (# thoughts in the chain at localization time)
            loc_steps = [d["ics_loc_n_steps"] for d in ds
                         if isinstance(d["ics_loc_n_steps"], (int, float))
                         and not math.isnan(d["ics_loc_n_steps"])]
            row["ics_loc_n_steps_mean"] = (sum(loc_steps) / len(loc_steps)) if loc_steps else math.nan
        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Gradient-dump analysis: FULL-EPOCH LCB-medium OLMo-7B ep1")
    print("=" * 70)
    print(f"Loss dumps:   {LOSS_DIR}")
    print(f"Branch dump:  {BRANCH_FILE}")
    print(f"Output:       {OUT_DIR}")

    loss_rows, n_steps, cps = load_loss_rows()
    rollouts = load_branch_rollouts()
    print(f"\nLoaded {len(loss_rows)} loss rows × {len(rollouts)} branch rollouts")
    print(f"  n_steps={n_steps}, calls/rank/step={cps}, prompts/step={PROMPTS_PER_STEP}")

    # Per-prompt SSI inference (prompt_idx is unique globally)
    by_prompt: dict[int, list[BranchRollout]] = defaultdict(list)
    for ro in rollouts:
        by_prompt[ro.prompt_idx].append(ro)
    infer_ssi(list(by_prompt.values()))

    # Match per step
    matched = match_rollouts_to_loss_per_step(rollouts, loss_rows)
    n_matched = sum(1 for v in matched.values() if v is not None)
    print(f"Matched: {n_matched}/{len(rollouts)} rollouts")

    # ─── Per-trajectory ─────────────────────────────────────────────────────
    # Carries the paper's g_{i,t} aggregates per rollout:
    #   mean_g_active     — bar g_i, the per-rollout mean over active tokens
    #   sum_g             — total signal mass over the rollout
    #   mean_one_minus_pi_active — overline{(1-π)}_i factor (model self-confidence-compl)
    per_traj = []
    for ro in rollouts:
        lr = matched.get((ro.step, ro.prompt_idx, ro.slot))
        if lr is None:
            continue
        active = int(lr.suffix_mask.sum().item())
        active_mask = lr.suffix_mask.bool()
        g_tok = per_token_signal(lr)
        one_minus_pi = (1.0 - lr.log_prob.exp()).clamp(min=0.0, max=1.0)
        per_traj.append({
            "step": ro.step, "prompt": ro.prompt_idx, "slot": ro.slot, "role": ro.role,
            "rl": lr.rl, "ssi": lr.ssi, "adv": lr.adv,
            "active_tok_count": active,
            "n_thoughts": len(ro.thought_boundaries),
            "oracle_correct": ro.oracle_correct,
            "mean_g_active": float(g_tok[active_mask].mean().item()) if active else 0.0,
            "sum_g": float(g_tok.sum().item()),
            "mean_one_minus_pi_active": (
                float(one_minus_pi[active_mask].mean().item()) if active else 0.0
            ),
            "loss_file": lr.file,
        })

    # ─── Per-thought ────────────────────────────────────────────────────────
    per_thought = []
    for ro in rollouts:
        lr = matched.get((ro.step, ro.prompt_idx, ro.slot))
        if lr is None:
            continue
        per_thought.extend(per_thought_grad_signal(ro, lr))

    # ─── Per-prompt + per-step concentration ────────────────────────────────
    prompt_meta = load_prompt_meta()
    per_prompt = compute_concentration_per_prompt(per_traj, prompt_meta=prompt_meta)
    per_step = compute_per_step_aggregates(per_prompt)

    # ─── Write CSVs ─────────────────────────────────────────────────────────
    import csv
    def write_csv(path: Path, rows: list[dict]):
        if not rows:
            path.write_text(""); return
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    write_csv(OUT_DIR / "per_trajectory.csv", per_traj)
    write_csv(OUT_DIR / "per_thought.csv", per_thought)
    write_csv(OUT_DIR / "per_prompt.csv",
              [d for _, d in sorted(per_prompt.items())])
    write_csv(OUT_DIR / "per_step.csv", per_step)
    print(f"  wrote per_trajectory.csv  ({len(per_traj)} rows)")
    print(f"  wrote per_thought.csv     ({len(per_thought)} rows)")
    print(f"  wrote per_prompt.csv      ({len(per_prompt)} rows)")
    print(f"  wrote per_step.csv        ({len(per_step)} rows)")

    # ─── Summary ────────────────────────────────────────────────────────────
    summary = []
    summary.append("=" * 78)
    summary.append("FULL-EPOCH gradient dump — LCB-medium OLMo-7B ep1")
    summary.append("=" * 78)
    summary.append(f"  n_steps={n_steps}, prompts/step={PROMPTS_PER_STEP}, "
                   f"calls/rank/step={cps}")
    summary.append(f"  Trajectories analyzed: {len(per_traj)} / {len(rollouts)} "
                   f"({len(rollouts) - len(per_traj)} unmatched)")
    summary.append("")
    summary.append("─" * 78)
    summary.append("Table 1: Per-step concentration aggregates (across 32 prompts)")
    summary.append("─" * 78)
    summary.append(f"  {'step':>4s}  {'n':>3s}  "
                   f"{'C_arch_med':>10s}  {'C_arch_geo':>10s}  "
                   f"{'C_w_g2_med':>10s}  {'C_w_g2_geo':>10s}  "
                   f"{'C_sig_med':>10s}  {'C_sig_geo':>10s}  "
                   f"{'ssi/L_med':>10s}")
    for r in per_step:
        summary.append(
            f"  {r['step']:>4d}  {r['n_prompts']:>3d}  "
            f"{r['c_arch_median']:>10.3f}  {r['c_arch_geomean']:>10.3f}  "
            f"{r['c_within_g2_median']:>10.3f}  {r['c_within_g2_geomean']:>10.3f}  "
            f"{r['c_signal_median']:>10.3f}  {r['c_signal_geomean']:>10.3f}  "
            f"{r['ssi_norm_median']:>10.3f}"
        )
    summary.append("")
    summary.append("─" * 78)
    summary.append("Table 2: Per-step group means of g and its factors (paper App. H)")
    summary.append("─" * 78)
    summary.append(f"  {'step':>4s}  "
                   f"{'g_base':>9s}  {'g_sp':>9s}  {'g_ratio':>8s}  "
                   f"{'|A|_base':>8s}  {'|A|_sp':>8s}  "
                   f"{'T_base':>7s}  {'T_sp':>7s}  "
                   f"{'aT_base':>9s}  {'aT_sp':>9s}  "
                   f"{'1-π_b':>6s}  {'1-π_s':>6s}")
    for r in per_step:
        summary.append(
            f"  {r['step']:>4d}  "
            f"{r['g_base']:>9.3e}  {r['g_sp']:>9.3e}  {r['g_sp_over_base']:>8.3f}  "
            f"{r['abs_adv_base']:>8.3f}  {r['abs_adv_sp']:>8.3f}  "
            f"{r['T_base']:>7.0f}  {r['T_sp']:>7.0f}  "
            f"{r['aT_base']:>9.3e}  {r['aT_sp']:>9.3e}  "
            f"{r['ompi_base']:>6.3f}  {r['ompi_sp']:>6.3f}"
        )
    summary.append("")
    summary.append("─" * 78)
    summary.append("Table 1b: Outcome rates per step (verifier pass + self-correction yield)")
    summary.append("─" * 78)
    summary.append(f"  {'step':>4s}  {'G1_pass':>8s}  {'G2_pass':>8s}  "
                   f"{'any_G1':>7s}  {'any_G2':>7s}  {'any_eit':>7s}  "
                   f"{'ics_corr':>8s}  {'rescue':>7s}  {'n_failG1':>8s}  {'lift':>6s}  "
                   f"{'loc_steps':>9s}")
    for r in per_step:
        if "g1_pass_rate" not in r:
            continue
        rescue_str = f"{r['rescue_rate']:.3f}" if not math.isnan(r['rescue_rate']) else "  nan"
        summary.append(
            f"  {r['step']:>4d}  {r['g1_pass_rate']:>8.3f}  {r['g2_pass_rate']:>8.3f}  "
            f"{r['any_g1_rate']:>7.3f}  {r['any_g2_rate']:>7.3f}  {r['any_either_rate']:>7.3f}  "
            f"{r['ics_corrected_rate']:>8.3f}  {rescue_str:>7s}  {r['n_failed_g1']:>8d}  "
            f"{r['lift']:>+6.3f}  {r['ics_loc_n_steps_mean']:>9.1f}"
        )

    summary.append("")
    summary.append("─" * 78)
    summary.append("Table 2b: Conditional on both groups having nonzero advantage in the same prompt")
    summary.append("─" * 78)
    summary.append(f"  {'step':>4s}  {'n_both':>6s}  {'frac':>5s}  "
                   f"{'g_base_nz':>10s}  {'g_sp_nz':>10s}  {'g_ratio_nz':>10s}  "
                   f"{'|A|_b_nz':>8s}  {'|A|_s_nz':>8s}  "
                   f"{'T_b_nz':>7s}  {'T_s_nz':>7s}  "
                   f"{'aT_ratio_nz':>11s}")
    for r in per_step:
        summary.append(
            f"  {r['step']:>4d}  {r['n_prompts_both_nonzero']:>6d}  "
            f"{r['frac_prompts_both_nonzero']:>5.2f}  "
            f"{r['g_base_nz']:>10.3e}  {r['g_sp_nz']:>10.3e}  "
            f"{r['g_sp_over_base_nz']:>10.3f}  "
            f"{r['abs_adv_base_nz']:>8.3f}  {r['abs_adv_sp_nz']:>8.3f}  "
            f"{r['T_base_nz']:>7.0f}  {r['T_sp_nz']:>7.0f}  "
            f"{r['aT_sp_over_base_nz']:>11.3f}"
        )

    summary.append("")
    summary.append("─" * 78)
    summary.append("Table 3: Trend across training steps (first vs last step values)")
    summary.append("─" * 78)

    def first_last_trend(key, label, fmt=".3f"):
        vals = [r[key] for r in per_step if not math.isnan(r[key])]
        if not vals:
            return
        summary.append(f"  {label}: first={vals[0]:{fmt}}, last={vals[-1]:{fmt}}, "
                       f"min={min(vals):{fmt}}, max={max(vals):{fmt}}")

    first_last_trend("g_base",         "g̅           (base)", ".3e")
    first_last_trend("g_sp",           "g̅           (SP)  ", ".3e")
    first_last_trend("g_sp_over_base", "g̅  SP/base ratio  ")
    first_last_trend("abs_adv_base",   "|A|         (base)")
    first_last_trend("abs_adv_sp",     "|A|         (SP)  ")
    first_last_trend("T_base",         "T           (base)", ".0f")
    first_last_trend("T_sp",           "T           (SP)  ", ".0f")
    first_last_trend("aT_base",        "|A|/T       (base)", ".3e")
    first_last_trend("aT_sp",          "|A|/T       (SP)  ", ".3e")
    first_last_trend("aT_sp_over_base","|A|/T SP/base     ")
    first_last_trend("zero_adv_frac_base","zero-|A| frac (base)")
    first_last_trend("zero_adv_frac_sp",  "zero-|A| frac (SP)  ")
    summary.append("")
    summary.append("  Conditional on nonzero advantage in BOTH groups (the pairwise subset):")
    first_last_trend("frac_prompts_both_nonzero", "frac prompts qualifying ")
    first_last_trend("g_sp_over_base_nz", "g̅  SP/base ratio (NZ)  ")
    first_last_trend("aT_sp_over_base_nz","|A|/T SP/base    (NZ)  ")
    first_last_trend("abs_adv_sp_over_base_nz","|A|  SP/base    (NZ)  ")
    first_last_trend("ompi_base",      "(1-π)       (base)")
    first_last_trend("ompi_sp",        "(1-π)       (SP)  ")
    first_last_trend("ssi_norm_median","ssi/L     (median)")
    summary.append("")
    summary.append("  Quantities (paper App. H):")
    summary.append("    g_{i,t} = |A_i|/T_i · (1 - π_θ(y_t|·)) on active tokens")
    summary.append("    g̅_i    = per-rollout mean over active tokens")
    summary.append("    g̅       = rollout-weighted mean of g̅_i across the group at this step")
    summary.append("    |A|, T, (1-π) are rollout-weighted means of the three factors")
    summary.append("    base = G1 rollouts (full-response gradient)")
    summary.append("    SP   = G2-corrections (suffix-only gradient)")

    summary_txt = "\n".join(summary)
    (OUT_DIR / "summary.txt").write_text(summary_txt + "\n")
    print(f"  wrote summary.txt")
    print()
    print(summary_txt)


if __name__ == "__main__":
    main()
