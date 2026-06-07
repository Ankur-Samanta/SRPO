#!/bin/bash
# Submit verl val_only evaluation jobs for baseline checkpoints
#
# Evaluates trained adapters (SRPO/RRPO, GRPO, and the SCoRe / Critique-GRPO /
# SPO-Tree baselines) across the benchmark suite in ALL_DATASETS.
#
# Uses the exact same verl evaluation pipeline as training (same agent loop,
# same reward function, temp=0, n=1 pass@1).
#
# All baselines use thought_agent for eval (all correction/branching agent
# loops fall back to thought_agent at temp=0).
#
# Usage:
#   ./submit_eval_baselines.sh <model> <dataset>     # single job
#   ./submit_eval_baselines.sh <model> all            # all datasets for one model
#   ./submit_eval_baselines.sh all all                # all jobs
#   ./submit_eval_baselines.sh list                   # show available jobs

set -e

SRPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH="${SRPO_DIR}/training/scripts/launch_training.sh"
CONDA_PATH="/home/${USER}/miniconda3"
CONDA_ENV="srpo"
EVAL_DATA="${HOME}/data/rlhf/eval"

mkdir -p "${SRPO_DIR}/batch_scripts/logs"

# ─── Base models ────────────────────────────────────────────────────────────
QWEN14B="Qwen/Qwen2.5-14B-Instruct"
OLMO7B="allenai/OLMo-3-7B-Instruct"

# ─── Eval datasets ──────────────────────────────────────────────────────────
ALL_DATASETS=(numinamath_olympiads acereason_math csqa sciknoweval_chemistry sciknoweval_physics sciknoweval_biology sciknoweval_materials hmmt_nov_2025 math_level5 strategyqa)

# ─── Shared overrides (match training config) ──────────────────────────────
SHARED_OVERRIDES=(
    actor_rollout_ref.model.external_lib=training
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=true
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    trainer.n_gpus_per_node=2
    trainer.val_only=true
    trainer.val_before_train=true
    'trainer.logger=["console"]'
)

OLMO_OVERRIDES=(
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
)

# Small models (≤3B): single GPU, TP=1
SMALL_MODEL_OVERRIDES=(
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    trainer.n_gpus_per_node=1
)

# ─── Model registry ────────────────────────────────────────────────────────
# Format: MODEL_HF, MODEL_CKPT (adapter path), MODEL_IS_OLMO
# All use thought_agent for eval (temp=0 → no correction/branching)

declare -A MODEL_HF
declare -A MODEL_CKPT
declare -A MODEL_IS_OLMO
declare -A MODEL_IS_SMALL

# --- Base models (no adapter)
MODEL_HF[base_olmo7b]="$OLMO7B"; MODEL_CKPT[base_olmo7b]=""; MODEL_IS_OLMO[base_olmo7b]=true
MODEL_HF[base_qwen14b]="$QWEN14B"; MODEL_CKPT[base_qwen14b]=""; MODEL_IS_OLMO[base_qwen14b]=false

# --- TGRPO (trained on oly)
MODEL_HF[tgrpo_oly_olmo7b]="$OLMO7B"; MODEL_CKPT[tgrpo_oly_olmo7b]="${SRPO_DIR}/checkpoints/thought_grpo/numina_oly_olmo7b_thought/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[tgrpo_oly_olmo7b]=true

# --- SRPO (trained on oly)

# --- SCoRe (trained on oly)
MODEL_HF[score_oly_olmo7b]="$OLMO7B"; MODEL_CKPT[score_oly_olmo7b]="${SRPO_DIR}/checkpoints/score_baseline/numina_oly_olmo7b_score/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[score_oly_olmo7b]=true
MODEL_HF[score_oly_olmo7b_s0]="$OLMO7B";   MODEL_CKPT[score_oly_olmo7b_s0]="${SRPO_DIR}/checkpoints/score_baseline/numina_oly_olmo7b_score_s0/global_step_24/actor/lora_adapter";   MODEL_IS_OLMO[score_oly_olmo7b_s0]=true
MODEL_HF[score_oly_olmo7b_s420]="$OLMO7B"; MODEL_CKPT[score_oly_olmo7b_s420]="${SRPO_DIR}/checkpoints/score_baseline/numina_oly_olmo7b_score_s420/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[score_oly_olmo7b_s420]=true

# --- Critique-GRPO (trained on oly)
MODEL_HF[crgrpo_oly_olmo7b]="$OLMO7B"; MODEL_CKPT[crgrpo_oly_olmo7b]="${SRPO_DIR}/checkpoints/critique_grpo_baseline/numina_oly_olmo7b_cgrpo/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[crgrpo_oly_olmo7b]=true
# Critique-GRPO 3-seed retrains (qwen14b s42/s0/s420 + olmo7b s0/s420)
MODEL_HF[crgrpo_oly_qwen14b]="$QWEN14B";       MODEL_CKPT[crgrpo_oly_qwen14b]="${SRPO_DIR}/checkpoints/critique_grpo_baseline/numina_oly_qwen14b_cgrpo/global_step_24/actor/lora_adapter";       MODEL_IS_OLMO[crgrpo_oly_qwen14b]=false
MODEL_HF[crgrpo_oly_qwen14b_s0]="$QWEN14B";    MODEL_CKPT[crgrpo_oly_qwen14b_s0]="${SRPO_DIR}/checkpoints/critique_grpo_baseline/numina_oly_qwen14b_cgrpo_s0/global_step_24/actor/lora_adapter";    MODEL_IS_OLMO[crgrpo_oly_qwen14b_s0]=false
MODEL_HF[crgrpo_oly_qwen14b_s420]="$QWEN14B";  MODEL_CKPT[crgrpo_oly_qwen14b_s420]="${SRPO_DIR}/checkpoints/critique_grpo_baseline/numina_oly_qwen14b_cgrpo_s420/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[crgrpo_oly_qwen14b_s420]=false
MODEL_HF[crgrpo_oly_olmo7b_s0]="$OLMO7B";      MODEL_CKPT[crgrpo_oly_olmo7b_s0]="${SRPO_DIR}/checkpoints/critique_grpo_baseline/numina_oly_olmo7b_cgrpo_s0/global_step_24/actor/lora_adapter";      MODEL_IS_OLMO[crgrpo_oly_olmo7b_s0]=true
MODEL_HF[crgrpo_oly_olmo7b_s420]="$OLMO7B";    MODEL_CKPT[crgrpo_oly_olmo7b_s420]="${SRPO_DIR}/checkpoints/critique_grpo_baseline/numina_oly_olmo7b_cgrpo_s420/global_step_24/actor/lora_adapter";    MODEL_IS_OLMO[crgrpo_oly_olmo7b_s420]=true

# --- SRPO (trained on oly, olmo7b)
MODEL_HF[srpo_oly_olmo7b]="$OLMO7B";          MODEL_CKPT[srpo_oly_olmo7b]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo/global_step_24/actor/lora_adapter";               MODEL_IS_OLMO[srpo_oly_olmo7b]=true
MODEL_HF[srpo_oly_olmo7b_s0]="$OLMO7B";       MODEL_CKPT[srpo_oly_olmo7b_s0]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo_s0/global_step_24/actor/lora_adapter";        MODEL_IS_OLMO[srpo_oly_olmo7b_s0]=true
MODEL_HF[srpo_oly_olmo7b_s420]="$OLMO7B";     MODEL_CKPT[srpo_oly_olmo7b_s420]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo_s420/global_step_24/actor/lora_adapter";    MODEL_IS_OLMO[srpo_oly_olmo7b_s420]=true
MODEL_HF[srpo_rand_oly_olmo7b]="$OLMO7B";     MODEL_CKPT[srpo_rand_oly_olmo7b]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo_rand/global_step_24/actor/lora_adapter";    MODEL_IS_OLMO[srpo_rand_oly_olmo7b]=true
MODEL_HF[srpo_rand_oly_olmo7b_s0]="$OLMO7B";  MODEL_CKPT[srpo_rand_oly_olmo7b_s0]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo_rand_s0/global_step_24/actor/lora_adapter";  MODEL_IS_OLMO[srpo_rand_oly_olmo7b_s0]=true
MODEL_HF[srpo_rand_oly_olmo7b_s420]="$OLMO7B"; MODEL_CKPT[srpo_rand_oly_olmo7b_s420]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo_rand_s420/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[srpo_rand_oly_olmo7b_s420]=true

# --- SRPO no-mask ablation (srpo_nomask, trained on oly, seed 42, 2 epochs)
MODEL_HF[srpo_nomask_oly_olmo7b]="$OLMO7B";   MODEL_CKPT[srpo_nomask_oly_olmo7b]="${SRPO_DIR}/checkpoints/srpo_nomask/numina_oly_olmo7b_srpo_nomask/global_step_24/actor/lora_adapter";  MODEL_IS_OLMO[srpo_nomask_oly_olmo7b]=true

# --- QWEN14B v4 entries (post chat-template patches, Apr 24)
MODEL_HF[tgrpo_oly_qwen14b]="$QWEN14B";       MODEL_CKPT[tgrpo_oly_qwen14b]="${SRPO_DIR}/checkpoints/thought_grpo/numina_oly_qwen14b_thought/global_step_24/actor/lora_adapter";       MODEL_IS_OLMO[tgrpo_oly_qwen14b]=false
MODEL_HF[srpo_oly_qwen14b]="$QWEN14B";       MODEL_CKPT[srpo_oly_qwen14b]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo/global_step_24/actor/lora_adapter";                 MODEL_IS_OLMO[srpo_oly_qwen14b]=false
MODEL_HF[srpo_rand_oly_qwen14b]="$QWEN14B";  MODEL_CKPT[srpo_rand_oly_qwen14b]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo_rand/global_step_24/actor/lora_adapter";         MODEL_IS_OLMO[srpo_rand_oly_qwen14b]=false
MODEL_HF[srpo_nomask_oly_qwen14b]="$QWEN14B";    MODEL_CKPT[srpo_nomask_oly_qwen14b]="${SRPO_DIR}/checkpoints/srpo_nomask/numina_oly_qwen14b_srpo_nomask/global_step_24/actor/lora_adapter";         MODEL_IS_OLMO[srpo_nomask_oly_qwen14b]=false

# --- SPO-Tree (trained on oly)
MODEL_HF[spotree_oly_olmo7b]="$OLMO7B";         MODEL_CKPT[spotree_oly_olmo7b]="${SRPO_DIR}/checkpoints/spo_baseline/numina_oly_olmo7b_spo_tree/global_step_24/actor/lora_adapter";             MODEL_IS_OLMO[spotree_oly_olmo7b]=true
MODEL_HF[spotree_oly_olmo7b_s0]="$OLMO7B";      MODEL_CKPT[spotree_oly_olmo7b_s0]="${SRPO_DIR}/checkpoints/spo_baseline/numina_oly_olmo7b_spo_tree_s0/global_step_24/actor/lora_adapter";      MODEL_IS_OLMO[spotree_oly_olmo7b_s0]=true
MODEL_HF[spotree_oly_olmo7b_s420]="$OLMO7B";    MODEL_CKPT[spotree_oly_olmo7b_s420]="${SRPO_DIR}/checkpoints/spo_baseline/numina_oly_olmo7b_spo_tree_s420/global_step_24/actor/lora_adapter";    MODEL_IS_OLMO[spotree_oly_olmo7b_s420]=true

# --- SRPO-rand (random localization, trained on oly)

# --- SRPO-reset (random loc + reset all, trained on oly)

# --- Seed 0 reruns on oly (with loss/branch dumps + variance/pass@k metrics)
MODEL_HF[tgrpo_oly_olmo7b_s0]="$OLMO7B";       MODEL_CKPT[tgrpo_oly_olmo7b_s0]="${SRPO_DIR}/checkpoints/thought_grpo/numina_oly_olmo7b_thought_s0/global_step_24/actor/lora_adapter";    MODEL_IS_OLMO[tgrpo_oly_olmo7b_s0]=true

# qwen14b seed-0 retrains (tgrpo / srpo l2new T=0.3 / srpo-rand)
MODEL_HF[tgrpo_oly_qwen14b_s0]="$QWEN14B";       MODEL_CKPT[tgrpo_oly_qwen14b_s0]="${SRPO_DIR}/checkpoints/thought_grpo/numina_oly_qwen14b_thought_s0/global_step_24/actor/lora_adapter";       MODEL_IS_OLMO[tgrpo_oly_qwen14b_s0]=false
MODEL_HF[srpo_oly_qwen14b_l2n_s0]="$QWEN14B";   MODEL_CKPT[srpo_oly_qwen14b_l2n_s0]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo_l2new_s0/global_step_24/actor/lora_adapter";    MODEL_IS_OLMO[srpo_oly_qwen14b_l2n_s0]=false
MODEL_HF[srpo_rand_oly_qwen14b_s0]="$QWEN14B";  MODEL_CKPT[srpo_rand_oly_qwen14b_s0]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo_rand_s0/global_step_24/actor/lora_adapter";    MODEL_IS_OLMO[srpo_rand_oly_qwen14b_s0]=false

# olmo7b l2n s0 + srpo2x4 (SRPO_2x4) variant
MODEL_HF[srpo_oly_olmo7b_l2n_s0]="$OLMO7B"; MODEL_CKPT[srpo_oly_olmo7b_l2n_s0]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo_l2new_s0/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[srpo_oly_olmo7b_l2n_s0]=true
MODEL_HF[srpo2x4_oly_olmo7b]="$OLMO7B";       MODEL_CKPT[srpo2x4_oly_olmo7b]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo2x4/global_step_24/actor/lora_adapter";       MODEL_IS_OLMO[srpo2x4_oly_olmo7b]=true
MODEL_HF[srpo2x4_oly_qwen14b]="$QWEN14B";     MODEL_CKPT[srpo2x4_oly_qwen14b]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo2x4/global_step_24/actor/lora_adapter";     MODEL_IS_OLMO[srpo2x4_oly_qwen14b]=false

# spotree qwen14b retrains
MODEL_HF[spotree_oly_qwen14b]="$QWEN14B";       MODEL_CKPT[spotree_oly_qwen14b]="${SRPO_DIR}/checkpoints/spo_baseline/numina_oly_qwen14b_spo_tree/global_step_24/actor/lora_adapter";          MODEL_IS_OLMO[spotree_oly_qwen14b]=false
MODEL_HF[spotree_oly_qwen14b_s0]="$QWEN14B";    MODEL_CKPT[spotree_oly_qwen14b_s0]="${SRPO_DIR}/checkpoints/spo_baseline/numina_oly_qwen14b_spo_tree_s0/global_step_24/actor/lora_adapter";    MODEL_IS_OLMO[spotree_oly_qwen14b_s0]=false
MODEL_HF[spotree_oly_qwen14b_s420]="$QWEN14B";  MODEL_CKPT[spotree_oly_qwen14b_s420]="${SRPO_DIR}/checkpoints/spo_baseline/numina_oly_qwen14b_spo_tree_s420/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[spotree_oly_qwen14b_s420]=false

# srpo_clip (clipped GRPO loss + srpo advantages + suffix mask) on s42-clip bug in srpo_loss
MODEL_HF[srpo_clip_l2n_oly_qwen14b]="$QWEN14B";   MODEL_CKPT[srpo_clip_l2n_oly_qwen14b]="${SRPO_DIR}/checkpoints/srpo_clip/numina_oly_qwen14b_srpo_clip_l2new/global_step_24/actor/lora_adapter";   MODEL_IS_OLMO[srpo_clip_l2n_oly_qwen14b]=false
MODEL_HF[srpo_clip_rand_oly_qwen14b]="$QWEN14B";  MODEL_CKPT[srpo_clip_rand_oly_qwen14b]="${SRPO_DIR}/checkpoints/srpo_clip/numina_oly_qwen14b_srpo_clip_rand/global_step_24/actor/lora_adapter";  MODEL_IS_OLMO[srpo_clip_rand_oly_qwen14b]=false

# srpo_clip_sm (srpo_clip loss + seq-mean aggregation) on s42
MODEL_HF[srpo_clip_sm_l2n_oly_qwen14b]="$QWEN14B";   MODEL_CKPT[srpo_clip_sm_l2n_oly_qwen14b]="${SRPO_DIR}/checkpoints/srpo_clip_sm/numina_oly_qwen14b_srpo_clip_sm_l2new/global_step_24/actor/lora_adapter";   MODEL_IS_OLMO[srpo_clip_sm_l2n_oly_qwen14b]=false
MODEL_HF[srpo_clip_sm_rand_oly_qwen14b]="$QWEN14B";  MODEL_CKPT[srpo_clip_sm_rand_oly_qwen14b]="${SRPO_DIR}/checkpoints/srpo_clip_sm/numina_oly_qwen14b_srpo_clip_sm_rand/global_step_24/actor/lora_adapter";  MODEL_IS_OLMO[srpo_clip_sm_rand_oly_qwen14b]=false
# srpo_clip_sm olmo7b s42 (ablation: w/ vs w/o clip)
MODEL_HF[srpo_clip_sm_l2n_oly_olmo7b]="$OLMO7B";     MODEL_CKPT[srpo_clip_sm_l2n_oly_olmo7b]="${SRPO_DIR}/experiments/numina_oly_olmo7b_srpo_clip_sm_l2new_20260514_052259/checkpoint/lora_adapter";          MODEL_IS_OLMO[srpo_clip_sm_l2n_oly_olmo7b]=true
# srpo l2n s420 for qwen14b and olmo7b
MODEL_HF[srpo_l2n_oly_qwen14b_s420]="$QWEN14B";  MODEL_CKPT[srpo_l2n_oly_qwen14b_s420]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo_l2new_s420/global_step_24/actor/lora_adapter";  MODEL_IS_OLMO[srpo_l2n_oly_qwen14b_s420]=false
MODEL_HF[srpo_rand_oly_qwen14b_s420]="$QWEN14B"; MODEL_CKPT[srpo_rand_oly_qwen14b_s420]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo_rand_s420/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[srpo_rand_oly_qwen14b_s420]=false
MODEL_HF[tgrpo_oly_qwen14b_s420]="$QWEN14B";      MODEL_CKPT[tgrpo_oly_qwen14b_s420]="${SRPO_DIR}/checkpoints/thought_grpo/numina_oly_qwen14b_thought_s420/global_step_24/actor/lora_adapter";  MODEL_IS_OLMO[tgrpo_oly_qwen14b_s420]=false
MODEL_HF[srpo_l2n_oly_olmo7b_s420]="$OLMO7B";    MODEL_CKPT[srpo_l2n_oly_olmo7b_s420]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo_l2new_s420/global_step_24/actor/lora_adapter";   MODEL_IS_OLMO[srpo_l2n_oly_olmo7b_s420]=true
MODEL_HF[srpo_rand_oly_olmo7b_s420]="$OLMO7B";   MODEL_CKPT[srpo_rand_oly_olmo7b_s420]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo_rand_s420/global_step_24/actor/lora_adapter";  MODEL_IS_OLMO[srpo_rand_oly_olmo7b_s420]=true
MODEL_HF[tgrpo_oly_olmo7b_s420]="$OLMO7B";        MODEL_CKPT[tgrpo_oly_olmo7b_s420]="${SRPO_DIR}/checkpoints/thought_grpo/numina_oly_olmo7b_thought_s420/global_step_24/actor/lora_adapter";   MODEL_IS_OLMO[tgrpo_oly_olmo7b_s420]=true
# srpo2x4 + srpo_1x8 with l2new prompt (s42)
MODEL_HF[srpo2x4_l2n_oly_olmo7b]="$OLMO7B";    MODEL_CKPT[srpo2x4_l2n_oly_olmo7b]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo2x4_l2new/global_step_24/actor/lora_adapter";    MODEL_IS_OLMO[srpo2x4_l2n_oly_olmo7b]=true
MODEL_HF[srpo2x4_l2n_oly_qwen14b]="$QWEN14B";  MODEL_CKPT[srpo2x4_l2n_oly_qwen14b]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo2x4_l2new/global_step_24/actor/lora_adapter";  MODEL_IS_OLMO[srpo2x4_l2n_oly_qwen14b]=false
MODEL_HF[srpo1x8_l2n_oly_olmo7b]="$OLMO7B";    MODEL_CKPT[srpo1x8_l2n_oly_olmo7b]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo_1x8_l2new/global_step_24/actor/lora_adapter";   MODEL_IS_OLMO[srpo1x8_l2n_oly_olmo7b]=true
MODEL_HF[srpo1x8_l2n_oly_qwen14b]="$QWEN14B";  MODEL_CKPT[srpo1x8_l2n_oly_qwen14b]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo_1x8_l2new/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[srpo1x8_l2n_oly_qwen14b]=false
# srpo-l2n sampling-ablation seed sweep (2x4 / 1x8 × s0 / s420)
MODEL_HF[srpo2x4_l2n_oly_qwen14b_s0]="$QWEN14B";   MODEL_CKPT[srpo2x4_l2n_oly_qwen14b_s0]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo2x4_l2new_s0/global_step_24/actor/lora_adapter";   MODEL_IS_OLMO[srpo2x4_l2n_oly_qwen14b_s0]=false
MODEL_HF[srpo2x4_l2n_oly_qwen14b_s420]="$QWEN14B"; MODEL_CKPT[srpo2x4_l2n_oly_qwen14b_s420]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo2x4_l2new_s420/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[srpo2x4_l2n_oly_qwen14b_s420]=false
MODEL_HF[srpo1x8_l2n_oly_qwen14b_s0]="$QWEN14B";   MODEL_CKPT[srpo1x8_l2n_oly_qwen14b_s0]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo_1x8_l2new_s0/global_step_24/actor/lora_adapter";  MODEL_IS_OLMO[srpo1x8_l2n_oly_qwen14b_s0]=false
MODEL_HF[srpo1x8_l2n_oly_qwen14b_s420]="$QWEN14B"; MODEL_CKPT[srpo1x8_l2n_oly_qwen14b_s420]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo_1x8_l2new_s420/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[srpo1x8_l2n_oly_qwen14b_s420]=false
MODEL_HF[srpo2x4_l2n_oly_olmo7b_s0]="$OLMO7B";     MODEL_CKPT[srpo2x4_l2n_oly_olmo7b_s0]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo2x4_l2new_s0/global_step_24/actor/lora_adapter";    MODEL_IS_OLMO[srpo2x4_l2n_oly_olmo7b_s0]=true
MODEL_HF[srpo2x4_l2n_oly_olmo7b_s420]="$OLMO7B";   MODEL_CKPT[srpo2x4_l2n_oly_olmo7b_s420]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo2x4_l2new_s420/global_step_24/actor/lora_adapter";  MODEL_IS_OLMO[srpo2x4_l2n_oly_olmo7b_s420]=true
MODEL_HF[srpo1x8_l2n_oly_olmo7b_s0]="$OLMO7B";     MODEL_CKPT[srpo1x8_l2n_oly_olmo7b_s0]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo_1x8_l2new_s0/global_step_24/actor/lora_adapter";   MODEL_IS_OLMO[srpo1x8_l2n_oly_olmo7b_s0]=true
MODEL_HF[srpo1x8_l2n_oly_olmo7b_s420]="$OLMO7B";   MODEL_CKPT[srpo1x8_l2n_oly_olmo7b_s420]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo_1x8_l2new_s420/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[srpo1x8_l2n_oly_olmo7b_s420]=true
# SCoRe qwen14b all 3 seeds
MODEL_HF[score_oly_qwen14b]="$QWEN14B";     MODEL_CKPT[score_oly_qwen14b]="${SRPO_DIR}/checkpoints/score_baseline/numina_oly_qwen14b_score/global_step_24/actor/lora_adapter";     MODEL_IS_OLMO[score_oly_qwen14b]=false
MODEL_HF[score_oly_qwen14b_s0]="$QWEN14B";  MODEL_CKPT[score_oly_qwen14b_s0]="${SRPO_DIR}/checkpoints/score_baseline/numina_oly_qwen14b_score_s0/global_step_24/actor/lora_adapter";  MODEL_IS_OLMO[score_oly_qwen14b_s0]=false
MODEL_HF[score_oly_qwen14b_s420]="$QWEN14B"; MODEL_CKPT[score_oly_qwen14b_s420]="${SRPO_DIR}/checkpoints/score_baseline/numina_oly_qwen14b_score_s420/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[score_oly_qwen14b_s420]=false
# srpo_clip olmo7b s42 and s420
MODEL_HF[srpo_clip_l2n_oly_olmo7b]="$OLMO7B";    MODEL_CKPT[srpo_clip_l2n_oly_olmo7b]="${SRPO_DIR}/checkpoints/srpo_clip/numina_oly_olmo7b_srpo_clip_l2new/global_step_24/actor/lora_adapter";     MODEL_IS_OLMO[srpo_clip_l2n_oly_olmo7b]=true
MODEL_HF[srpo_clip_rand_oly_olmo7b]="$OLMO7B";   MODEL_CKPT[srpo_clip_rand_oly_olmo7b]="${SRPO_DIR}/checkpoints/srpo_clip/numina_oly_olmo7b_srpo_clip_rand/global_step_24/actor/lora_adapter";    MODEL_IS_OLMO[srpo_clip_rand_oly_olmo7b]=true
MODEL_HF[srpo_clip_l2n_oly_olmo7b_s420]="$OLMO7B";  MODEL_CKPT[srpo_clip_l2n_oly_olmo7b_s420]="${SRPO_DIR}/checkpoints/srpo_clip/numina_oly_olmo7b_srpo_clip_l2new_s420/global_step_24/actor/lora_adapter";  MODEL_IS_OLMO[srpo_clip_l2n_oly_olmo7b_s420]=true
MODEL_HF[srpo_clip_rand_oly_olmo7b_s420]="$OLMO7B"; MODEL_CKPT[srpo_clip_rand_oly_olmo7b_s420]="${SRPO_DIR}/checkpoints/srpo_clip/numina_oly_olmo7b_srpo_clip_rand_s420/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[srpo_clip_rand_oly_olmo7b_s420]=true

# --- Seed 420 reruns (tgrpo, srpo, srpo_rand)
MODEL_HF[tgrpo_oly_olmo7b_s420]="$OLMO7B";           MODEL_CKPT[tgrpo_oly_olmo7b_s420]="${SRPO_DIR}/checkpoints/thought_grpo/numina_oly_olmo7b_thought_s420/global_step_24/actor/lora_adapter";        MODEL_IS_OLMO[tgrpo_oly_olmo7b_s420]=true

# SRPO variants on numina_oly (l2new prompt; l2n=T0.3, l2ng=greedy T0.0)
MODEL_HF[srpo_oly_olmo7b_l2n]="$OLMO7B";    MODEL_CKPT[srpo_oly_olmo7b_l2n]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo_l2new/global_step_24/actor/lora_adapter";          MODEL_IS_OLMO[srpo_oly_olmo7b_l2n]=true
MODEL_HF[srpo_oly_qwen14b_l2n]="$QWEN14B";  MODEL_CKPT[srpo_oly_qwen14b_l2n]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo_l2new/global_step_24/actor/lora_adapter";        MODEL_IS_OLMO[srpo_oly_qwen14b_l2n]=false

# SRPO_1x8 (8 corrections, no parent) on numina_oly
MODEL_HF[srpo1x8_oly_olmo7b]="$OLMO7B";         MODEL_CKPT[srpo1x8_oly_olmo7b]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo_1x8/global_step_24/actor/lora_adapter";              MODEL_IS_OLMO[srpo1x8_oly_olmo7b]=true
MODEL_HF[srpo1x8_oly_qwen14b]="$QWEN14B";       MODEL_CKPT[srpo1x8_oly_qwen14b]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo_1x8/global_step_24/actor/lora_adapter";            MODEL_IS_OLMO[srpo1x8_oly_qwen14b]=false
MODEL_HF[srpo1x8_oly_olmo7b_rand]="$OLMO7B";    MODEL_CKPT[srpo1x8_oly_olmo7b_rand]="${SRPO_DIR}/checkpoints/srpo/numina_oly_olmo7b_srpo_1x8_rand/global_step_24/actor/lora_adapter";    MODEL_IS_OLMO[srpo1x8_oly_olmo7b_rand]=true
MODEL_HF[srpo1x8_oly_qwen14b_rand]="$QWEN14B";  MODEL_CKPT[srpo1x8_oly_qwen14b_rand]="${SRPO_DIR}/checkpoints/srpo/numina_oly_qwen14b_srpo_1x8_rand/global_step_24/actor/lora_adapter";  MODEL_IS_OLMO[srpo1x8_oly_qwen14b_rand]=false

# SRPO + TGRPO on sk4 (numina_oly + sciknow400 mix)

# ─── All models list ────────────────────────────────────────────────────────
ALL_MODELS=(
    base_olmo7b
    tgrpo_oly_olmo7b score_oly_olmo7b crgrpo_oly_olmo7b
    spotree_oly_olmo7b spotree_oly_olmo7b_s0 spotree_oly_olmo7b_s420
    tgrpo_oly_qwen14b srpo_oly_qwen14b srpo_rand_oly_qwen14b srpo_nomask_oly_qwen14b
    srpo_oly_olmo7b_l2n srpo_oly_qwen14b_l2n
    srpo1x8_oly_olmo7b srpo1x8_oly_qwen14b srpo1x8_oly_olmo7b_rand srpo1x8_oly_qwen14b_rand
)

# ─── Run a single eval ─────────────────────────────────────────────────────

run_eval() {
    local model_key=$1
    local dataset=$2

    local base_model="${MODEL_HF[$model_key]}"
    local ckpt_dir="${MODEL_CKPT[$model_key]}"
    local is_olmo="${MODEL_IS_OLMO[$model_key]}"
    local test_data="${EVAL_DATA}/${dataset}.parquet"

    if [ ! -f "$test_data" ]; then
        echo "ERROR: test data not found: $test_data"
        exit 1
    fi

    # Optional run tag: when RUN_TAG=foo is set in the env, every artifact path
    # for this job gets `_foo` appended so re-runs don't collide with prior results.
    local tag_suffix=""
    [ -n "${RUN_TAG:-}" ] && tag_suffix="_${RUN_TAG}"

    local overrides=(
        "${SHARED_OVERRIDES[@]}"
        actor_rollout_ref.model.path="$base_model"
        data.val_files="$test_data"
        data.train_files="$test_data"
        trainer.experiment_name="eval_${model_key}_${dataset}${tag_suffix}"
        trainer.project_name=eval_baselines
        actor_rollout_ref.rollout.agent.default_agent_loop=thought_agent
    )

    # Load adapter or run base model
    overrides+=(
        trainer.resume_mode=disable
        "trainer.default_local_dir='checkpoints/eval_baselines/eval_${model_key}_${dataset}${tag_suffix}'"
        "trainer.validation_data_dir='eval_generations/${model_key}_${dataset}${tag_suffix}'"
    )
    if [ -n "$ckpt_dir" ]; then
        overrides+=(actor_rollout_ref.model.lora_adapter_path="${ckpt_dir}")
    fi

    if [ "$is_olmo" = "true" ]; then
        overrides+=("${OLMO_OVERRIDES[@]}")
    fi

    local is_small="${MODEL_IS_SMALL[$model_key]}"
    if [ "$is_small" = "true" ]; then
        overrides+=("${SMALL_MODEL_OVERRIDES[@]}")
    fi

    # 14B models: use TP=4, 4 GPUs (matches training config; 2 GPUs isn't enough for inference KV cache)
    if [[ "$model_key" == *"_qwen14b"* || "$model_key" == *"_qwen14b_"* ]]; then
        overrides+=(
            actor_rollout_ref.rollout.tensor_model_parallel_size=4
            trainer.n_gpus_per_node=4
        )
    fi

    # GPQA / hotpotqa prompts exceed 2048 tokens — extend context window
    if [ "$dataset" = "gpqa" ] || [ "$dataset" = "hotpotqa" ]; then
        overrides+=(
            data.max_prompt_length=4096
            actor_rollout_ref.rollout.prompt_length=4096
        )
    fi

    # Small eval datasets (< default train_batch_size=32) need a reduced batch size
    # to avoid "Train dataloader is empty!" assertion in verl
    case "$dataset" in
        hmmt_nov_2025|amo_bench|apex_shortlist)
            overrides+=(data.train_batch_size=24)
            ;;
    esac

    bash "$LAUNCH" "${overrides[@]}"
}

# ─── SLURM submission ──────────────────────────────────────────────────────

submit() {
    local job_name=$1
    local model_key=$2
    local dataset=$3
    local n_gpus=2
    [ "${MODEL_IS_SMALL[$model_key]}" = "true" ] && n_gpus=1
    # 14B models need 4 GPUs for inference (TP=4, matches training)
    [[ "$model_key" == *"_qwen14b"* ]] && n_gpus=4

    sbatch --partition="${SLURM_PARTITION:-q1}" \
           --nodes=1 \
           --gpus-per-node=${n_gpus} \
           --cpus-per-gpu=10 \
           --time=4:00:00 \
           --job-name="${job_name}" \
           --output="batch_scripts/logs/${job_name}.out" \
           --error="batch_scripts/logs/${job_name}.err" \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && export TRANSFORMERS_OFFLINE=1 && export RUN_TAG=\"${RUN_TAG:-}\" && cd ${SRPO_DIR} && bash ${SRPO_DIR}/batch_scripts/submit_eval_baselines.sh ${model_key} ${dataset} --local'"

    echo "Submitted: ${job_name}"
}

# ─── Entrypoint ─────────────────────────────────────────────────────────────

MODEL_KEY=${1:?Usage: $0 <model> <dataset> [--local]  (run '$0 list' for help)}

if [ "$MODEL_KEY" = "list" ]; then
    echo "Usage: $0 <model> <dataset> [--local]"
    echo ""
    echo "Models (${#ALL_MODELS[@]} total):"
    for m in "${ALL_MODELS[@]}"; do
        ckpt="${MODEL_CKPT[$m]}"
        echo "  $m  (${MODEL_HF[$m]}${ckpt:+ + adapter})"
    done
    echo ""
    echo "Datasets (${#ALL_DATASETS[@]} total):"
    for d in "${ALL_DATASETS[@]}"; do
        echo "  $d"
    done
    echo ""
    echo "Special:"
    echo "  $0 <model> all    — all 5 datasets for one model"
    echo "  $0 all all         — all ${#ALL_MODELS[@]} x 5 = $((${#ALL_MODELS[@]} * 5)) jobs"
    exit 0
fi

DATASET=${2:?Usage: $0 <model> <dataset> [--local]}
MODE=${3:-slurm}

# Expand model/dataset groups
expand_models() {
    case $1 in
        all) echo "${ALL_MODELS[@]}" ;;
        *)   echo "$1" ;;
    esac
}

expand_datasets() {
    case $1 in
        all) echo "${ALL_DATASETS[@]}" ;;
        *)   echo "$1" ;;
    esac
}

MODELS=$(expand_models "$MODEL_KEY")
DATASETS=$(expand_datasets "$DATASET")

for m in $MODELS; do
    if [ -z "${MODEL_HF[$m]}" ]; then
        echo "Unknown model: $m (run '$0 list')"
        exit 1
    fi

    for d in $DATASETS; do
        tag_suffix=""
        [ -n "${RUN_TAG:-}" ] && tag_suffix="_${RUN_TAG}"
        job_name="eval_${m}_${d}${tag_suffix}"

        if [ "$MODE" = "--local" ]; then
            echo "=== Running: ${m} on ${d} ==="
            run_eval "$m" "$d"
        else
            submit "$job_name" "$m" "$d"
        fi
    done
done
