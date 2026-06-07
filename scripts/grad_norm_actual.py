"""
Compute exact per-thought logit-gradient L2 norm for the SRPO grad-tree analysis.

Per-token logit-gradient (SRPO loss, ratio≈1 with ppo_epochs=1):
  ∇_z L_t = -(Â_i / T_i) · (e_{y_t} - π_θ(·|x_{<t}))
  ‖∇_z L_t‖₂² = (Â_i / T_i)² · [ (1 - π_y)² + Σ_{k ≠ y_t} π_k² ]
              = (Â_i / T_i)² · [ 1 - 2π_y + Σ_k π_k² ]

This is exact — no proxy, no Jacobian assumption. Only requires a forward
pass (no backward), so memory is bounded by the LM-head softmax and fits
easily on a 40GB GPU.

Pipeline (per microbatch dump):
  1. Load base model + LoRA adapter (forward only, no grad)
  2. For each row, forward(input_ids) → logits → response-token softmax
  3. Compute per-token Σ_k π_k² via softmax→squared-sum (chunked over positions)
  4. Compute ‖∇_z L_t‖₂ = (|Â_i|/T_i) · sqrt(1 - 2π_y + Σ π_k²) at suffix tokens
  5. Match microbatch row to a branch rollout via (response_len, ssi, sign(adv))
  6. Aggregate to per-thought mean → CSV row

Output CSV columns:
  file, row_in_file, prompt_idx, slot, role, thought_idx, n_tokens,
  advantage, T_i, logit_grad_norm
"""

from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import PeftModel


# ─── Paths ──────────────────────────────────────────────────────────────────
SRPO_DIR    = Path(__file__).resolve().parents[1]
LOSS_DIR    = SRPO_DIR / "logs" / "srpo_per_token" / "grad_dump_l2n_ep1_v2"
BRANCH_DIR  = SRPO_DIR / "logs" / "srpo_localizations" / "grad_dump_l2n_ep1_v2"
OUT_CSV     = SRPO_DIR / "scripts" / "grad_norm_actual.csv"
BASE_MODEL  = "allenai/OLMo-3-7B-Instruct"
LORA_PATH   = SRPO_DIR / "checkpoints" / "srpo_grad_dump" / "lora_init" / "lora_adapter"

DEVICE = "cuda"
DTYPE  = torch.bfloat16
TEMPERATURE = 0.7       # VeRL applies this to logits before softmax (training default)
LOG_PROB_TOL = 1e-2     # bf16 noise tolerance for forward-pass verification
SOFTMAX_CHUNK = 256     # chunk positions when computing softmax sum-of-squares


# ─── Model loading (forward-only) ──────────────────────────────────────────
def load_model():
    print(f"Loading base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=DTYPE, attn_implementation="sdpa",
    ).to(DEVICE)
    print(f"Loading LoRA adapter: {LORA_PATH}")
    model = PeftModel.from_pretrained(base, str(LORA_PATH), is_trainable=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ─── Branch-dump loading + matching ────────────────────────────────────────
def load_branch_rollouts(branch_dir: Path):
    files = sorted(branch_dir.glob("branches_pid*.jsonl"))
    rollouts = []
    for fp in files:
        for prompt_idx, line in enumerate(open(fp)):
            rec = json.loads(line)
            ros = sorted(rec["rollouts"], key=lambda r: r["slot"])
            g2 = [r for r in ros if r["slot"] >= 4]
            ssi_tok = 0
            if len(g2) >= 2:
                min_n = min(len(r["decoded_thoughts"]) for r in g2)
                shared = 0
                for k in range(min_n):
                    ref = g2[0]["decoded_thoughts"][k]
                    if all(r["decoded_thoughts"][k] == ref for r in g2):
                        shared += 1
                    else:
                        break
                ssi_tok = sum(g2[0]["segment_lengths"][:shared])
            for r in ros:
                role = "G1" if r["slot"] < 4 else "G2-corr"
                rollouts.append({
                    "prompt_idx": prompt_idx,
                    "slot": r["slot"],
                    "role": role,
                    "ssi": ssi_tok if role == "G2-corr" else 0,
                    "response_len": r["response_len"],
                    "thought_boundaries": [tuple(b) for b in r["thought_boundaries"]],
                    "oracle_correct": r.get("oracle_correct"),
                })
    return rollouts


def match_rollout(branches, rl: int, ssi: int, adv: float, used: set):
    cands = [(j, r) for j, r in enumerate(branches)
             if r["response_len"] == rl and r["ssi"] == ssi and j not in used]
    if not cands:
        return None
    if len(cands) == 1:
        j, r = cands[0]; used.add(j); return r
    prefer_pos = adv > 0
    cands.sort(key=lambda c: (c[1]["oracle_correct"] is True), reverse=prefer_pos)
    j, r = cands[0]; used.add(j); return r


# ─── Forward pass + per-token logit-gradient L2 norm ───────────────────────
@torch.no_grad()
def per_token_logit_grad(model, input_ids, attention_mask, responses, position_ids=None):
    """Run forward, return (response_log_prob, response_pi_sumsq) per response token.

    response_log_prob[t] = log π_θ(y_t)
    response_pi_sumsq[t] = Σ_k π_θ(k | x_<t)²  (in fp32 for numerical safety)

    The L2 norm of the per-token logit-gradient is then
      |Â/T| · sqrt(1 - 2·exp(response_log_prob[t]) + response_pi_sumsq[t]).
    """
    fwd_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if position_ids is not None:
        fwd_kwargs["position_ids"] = position_ids
    out = model(**fwd_kwargs)
    seq_len      = input_ids.shape[1]
    response_len = responses.shape[1]
    prompt_len   = seq_len - response_len
    response_logits = out.logits[:, prompt_len - 1 : seq_len - 1, :]   # (1, R, V)
    R = response_logits.shape[1]

    # log π and π² sum, computed in chunks of positions to bound peak memory
    response_log_prob = torch.empty(1, R, device=DEVICE, dtype=torch.float32)
    response_pi_sumsq = torch.empty(1, R, device=DEVICE, dtype=torch.float32)
    for s in range(0, R, SOFTMAX_CHUNK):
        e = min(s + SOFTMAX_CHUNK, R)
        chunk_logits = response_logits[:, s:e, :].float() / TEMPERATURE  # (1, c, V)
        log_probs    = F.log_softmax(chunk_logits, dim=-1)               # (1, c, V)
        # log π_y at sampled token
        sampled = log_probs.gather(-1, responses[:, s:e].unsqueeze(-1)).squeeze(-1)
        response_log_prob[:, s:e] = sampled
        # Σ_k π² = sum exp(2 · log π) over vocab
        pi = log_probs.exp()
        response_pi_sumsq[:, s:e] = (pi * pi).sum(dim=-1)
    return response_log_prob, response_pi_sumsq


def per_thought_logit_norms(model, lr, branches, used):
    response_mask_b  = lr["response_mask"]
    suffix_mask_b    = lr["suffix_mask"]
    reset_adv_b       = lr["reset_advantage"].float()
    suffix_start_b   = lr.get("suffix_start_idx")
    dumped_lp_b      = lr["log_prob"]

    bs = response_mask_b.shape[0]
    rows = []
    for i in range(bs):
        A = float(reset_adv_b[i].item())
        if A == 0.0:
            continue
        if suffix_mask_b[i].sum().item() == 0:
            continue

        rl  = int(response_mask_b[i].sum().item())
        ssi = int(suffix_start_b[i].item()) if suffix_start_b is not None else 0
        m   = match_rollout(branches, rl, ssi, A, used)
        if m is None:
            print(f"  row {i}: no branch match (rl={rl}, A={A:+.2f})")
            continue

        input_ids_i      = lr["input_ids"][i:i+1].to(DEVICE)
        attention_mask_i = lr["attention_mask"][i:i+1].to(DEVICE)
        responses_i      = lr["responses"][i:i+1].to(DEVICE)
        position_ids_b   = lr.get("position_ids")
        position_ids_i   = position_ids_b[i:i+1].to(DEVICE) if position_ids_b is not None else None
        suffix_mask_i    = suffix_mask_b[i:i+1].to(DEVICE).bool()
        response_mask_i  = response_mask_b[i:i+1].to(DEVICE).bool()
        dumped_lp_i      = dumped_lp_b[i:i+1].to(DEVICE)

        log_prob, pi_sumsq = per_token_logit_grad(
            model, input_ids_i, attention_mask_i, responses_i, position_ids_i)

        with torch.no_grad():
            active = response_mask_i[0]
            err = (log_prob[0][active] - dumped_lp_i[0][active]).abs()
            max_err  = err.max().item() if active.any() else 0.0
            mean_err = err.mean().item() if active.any() else 0.0
        print(f"  row {i}: log_prob err max={max_err:.3e} mean={mean_err:.3e}  A={A:+.3f}")

        # ‖∇_z log π‖² = 1 - 2·π_y + Σ_k π_k²   (per token)
        pi_y = log_prob[0].exp()                               # (R,)
        score_sq = (1.0 - 2.0 * pi_y + pi_sumsq[0]).clamp(min=0.0)
        score = score_sq.sqrt()                                # (R,) = ‖∇_z log π_t‖
        coeff = abs(A) / max(int(suffix_mask_i[0].sum().item()), 1)
        per_token = coeff * score                              # exact logit-grad L2

        T_i = int(suffix_mask_i[0].sum().item())
        suff_i_cpu = suffix_mask_i[0].cpu()

        for k_idx, (s, e) in enumerate(m["thought_boundaries"]):
            mask_k = torch.zeros_like(suff_i_cpu)
            mask_k[s:e] = True
            mask_k = mask_k & suff_i_cpu
            n_tok = int(mask_k.sum().item())
            if n_tok == 0:
                continue
            mean_norm = per_token[mask_k.to(DEVICE)].mean().item()
            rows.append({
                "row_in_file": i,
                "prompt_idx":  m["prompt_idx"],
                "slot":        m["slot"],
                "role":        m["role"],
                "thought_idx": k_idx,
                "n_tokens":    n_tok,
                "advantage":   A,
                "T_i":         T_i,
                "logit_grad_norm": mean_norm,
            })

        # Free per-row tensors before moving on
        del log_prob, pi_sumsq, score, per_token
        torch.cuda.empty_cache()

    return rows


# ─── Driver ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss-dir", type=Path, default=LOSS_DIR)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    branches = load_branch_rollouts(BRANCH_DIR)
    print(f"Loaded {len(branches)} branch rollouts from {BRANCH_DIR}")
    used = set()

    model = load_model()

    files = sorted(args.loss_dir.glob("loss_pid*_call*.pt"))
    if args.limit:
        files = files[: args.limit]
    print(f"\nProcessing {len(files)} loss dumps from {args.loss_dir}")

    fields = ["file", "row_in_file", "prompt_idx", "slot", "role",
              "thought_idx", "n_tokens", "advantage", "T_i", "logit_grad_norm"]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for fp in files:
            print(f"\n[{fp.name}]")
            lr = torch.load(fp, weights_only=False, map_location="cpu")
            if lr.get("input_ids") is None:
                print("  SKIP: no input_ids (v1 dump?)")
                continue
            rows = per_thought_logit_norms(model, lr, branches, used)
            print(f"  {len(rows)} thought rows")
            for r in rows:
                r["file"] = fp.name
                writer.writerow(r)
            fh.flush()

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
