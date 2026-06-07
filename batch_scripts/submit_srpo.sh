#!/bin/bash
# Submit SRPO training jobs (two-group GRPO: 4 fresh i.i.d. + 4 oracle-gated corrections)
#
# Usage:
#   ./submit_srpo.sh <job>          # submit to SLURM
#   ./submit_srpo.sh <job> --local   # run locally (set CUDA_VISIBLE_DEVICES first)
#
# Primary jobs (SRPO = self-localization, SRPO-rand = random localization):
#   olmo7b_numina_oly       OLMo 7B      + SRPO      on NuminaMath Olympiads  (2 GPUs)
#   qwen14b_numina_oly      Qwen 2.5 14B + SRPO      on NuminaMath Olympiads  (4 GPUs)
#   olmo7b_numina_oly_rand  OLMo 7B      + SRPO-rand on NuminaMath Olympiads  (2 GPUs)
#   qwen14b_numina_oly_rand Qwen 2.5 14B + SRPO-rand on NuminaMath Olympiads  (4 GPUs)
#   all_oly                 Submit all 4 Olympiads jobs above (7B only)
#
# Additional datasets:
#
# Seed variants: append _s0, _s420 to any numina_oly job

set -e

SCPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH="${SCPO_DIR}/training/scripts/launch_srpo_training.sh"
CONDA_PATH="/home/${USER}/miniconda3"
CONDA_ENV="scpo"

mkdir -p "${SCPO_DIR}/batch_scripts/logs"

# ─── Shared overrides for 7-8B models (TP=2, 2 GPUs) ───────────────────────
SHARED_OVERRIDES=(
    actor_rollout_ref.model.external_lib=training
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=true
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    trainer.n_gpus_per_node=2
    "trainer.default_local_dir='checkpoints/\${trainer.project_name}/\${trainer.experiment_name}'"
    trainer.project_name=srpo
    # SRPO: self-localization agent loop
    actor_rollout_ref.rollout.agent.default_agent_loop=srpo_agent
    # SRPO loss: two-group GRPO with pre-computed advantages + suffix masking
    actor_rollout_ref.actor.policy_loss.loss_mode=srpo
    # 2 epochs (consistent with SCGRPO/SCPO runs)
    trainer.total_epochs=2
)

export VERL_LOGGING_LEVEL=INFO
# Dump every coordinator call (one record per prompt) instead of the default 1/50 sample.
# ~5KB per record × ~1600 prompts/run ≈ 8MB per run — negligible.
export SCGRPO_BRANCH_DUMP_EVERY=1

# ─── Model shortcuts ──────────────────────────────────────────────────────
QWEN14B="Qwen/Qwen2.5-14B-Instruct"
OLMO7B="allenai/OLMo-3-7B-Instruct"

OLMO_OVERRIDES=(
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
)

SMALL_MODEL_OVERRIDES=(
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    trainer.n_gpus_per_node=1
)


QWEN14B_OVERRIDES=(
    trainer.n_gpus_per_node=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
)

# Native context is 4096 tokens (max_position_embeddings=4096). At 1024/3072
# we truncate ~2.7% of LCB prompts and ~19% of long chains; at any other
# split one side or the other gets hit harder.

# SRPO-rand: override agent loop to random localization variant
RAND_OVERRIDES=(
    actor_rollout_ref.rollout.agent.default_agent_loop=rrpo_agent
)

# SRPO-NM (no shared-prefix masking on G2 counterfactuals)
NM_OVERRIDES=(
    actor_rollout_ref.rollout.agent.default_agent_loop=srpo_nomask_agent
    trainer.project_name=srpo_nomask
    trainer.test_freq=6
)

# SRPO-NM-rand (no shared-prefix masking + random localization)
NM_RAND_OVERRIDES=(
    actor_rollout_ref.rollout.agent.default_agent_loop=rrpo_nomask_agent
    trainer.project_name=srpo_nomask
    trainer.test_freq=6
)

# SRPO_clip: standard PPO/GRPO clipped surrogate + srpo advantages + suffix masking
SRPO_CLIP_OVERRIDES=(
    actor_rollout_ref.actor.policy_loss.loss_mode=srpo_clip
    trainer.project_name=srpo_clip
)

# SRPO_clip-SM: srpo_clip with sequence-mean batch aggregation (matches srpo's old aggregation)
SRPO_CLIP_SM_OVERRIDES=(
    actor_rollout_ref.actor.policy_loss.loss_mode=srpo_clip
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    trainer.project_name=srpo_clip_sm
)

# SRPO_clip-KL: srpo_clip + KL-to-ref penalty in the actor loss (verl handles the addition)
SRPO_CLIP_KL_OVERRIDES=(
    actor_rollout_ref.actor.policy_loss.loss_mode=srpo_clip
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    trainer.project_name=srpo_clip_kl
)

# 1-epoch variant — early-stop ablation for runs that overfit in epoch 2
EP1_OVERRIDES=(
    trainer.total_epochs=1
)

# Qwen 7B on long-sequence datasets (Oly/AIME): halve log_prob micro batch to avoid OOM

# ─── Data overrides ───────────────────────────────────────────────────────
MATHLVL5_DATA=(
    data.train_files=/home/${USER}/data/rlhf/math_level5/train.parquet
    data.val_files=/home/${USER}/data/rlhf/math_level5/test.parquet
)
NUMINA_OLYMPIADS_DATA=(
    data.train_files=/home/${USER}/data/rlhf/numinamath_olympiads/train.parquet
    data.val_files=/home/${USER}/data/rlhf/numinamath_olympiads/test.parquet
)
# numinamath_olympiads (400) + sciknoweval L3 disjoint (100×4 = 400) = 800 mixed train.
# Eval = numinamath_olympiads test (sciknow eval untouched). Use with EP1_OVERRIDES
# so 1 epoch matches the trajectory count of 2 epochs on numina_oly alone.
NUMINA_OLY_PLUS_SCIKNOW400_DATA=(
    data.train_files=/home/${USER}/data/rlhf/numina_oly_plus_sciknow400/train.parquet
    data.val_files=/home/${USER}/data/rlhf/numina_oly_plus_sciknow400/test.parquet
)
LCB_MEDIUM_DATA=(
    data.train_files=/home/${USER}/data/rlhf/livecodebench_medium/train.parquet
    data.val_files=/home/${USER}/data/rlhf/livecodebench_medium/test.parquet
    trainer.test_freq=11
)
LCB_HARD_DATA=(
    data.train_files=/home/${USER}/data/rlhf/livecodebench_hard/train.parquet
    data.val_files=/home/${USER}/data/rlhf/livecodebench_hard/test.parquet
    trainer.test_freq=9
)
# ─── Generic job builder ───────────────────────────────────────────────────
_run() {
    local model_path=$1; shift
    local experiment_name=$1; shift
    local extra_overrides=("$@")

    local cmd_args=(
        "${SHARED_OVERRIDES[@]}"
        actor_rollout_ref.model.path="$model_path"
        trainer.experiment_name="$experiment_name"
        "${extra_overrides[@]}"
    )

    bash "$LAUNCH" "${cmd_args[@]}"
}

# ─── Job definitions ────────────────────────────────────────────────────────

# --- MATH Level 5 ---
cmd_olmo7b_mathlvl5() { _run "$OLMO7B" mathlvl5_olmo7b_srpo "${MATHLVL5_DATA[@]}" "${OLMO_OVERRIDES[@]}" ; }

# --- NuminaMath Olympiads: SRPO ---
cmd_olmo7b_numina_oly()   { _run "$OLMO7B"   numina_oly_olmo7b_srpo   "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly()      { _run "$QWEN14B"      numina_oly_qwen14b_srpo      "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" trainer.test_freq=6 ; }

# --- NuminaMath Olympiads: SRPO with L2new prompt (early-biasing localization variant) ---
cmd_olmo7b_numina_oly_l2new()  { _run "$OLMO7B"  numina_oly_olmo7b_srpo_l2new  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_l2new() { _run "$QWEN14B" numina_oly_qwen14b_srpo_l2new "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }

# --- LiveCodeBench Medium: SRPO with L2new prompt ---
cmd_qwen14b_lcb_medium_l2new() { _run "$QWEN14B" lcb_medium_qwen14b_srpo_l2new "${LCB_MEDIUM_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" trainer.resume_mode=disable ; }
cmd_olmo7b_lcb_medium_l2new()  { _run "$OLMO7B"  lcb_medium_olmo7b_srpo_l2new  "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_medium_rand()   { _run "$OLMO7B"  lcb_medium_olmo7b_srpo_rand   "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_medium_l2new_ep1()       { _run "$OLMO7B"  lcb_medium_olmo7b_srpo_l2new_ep1       "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}"                       "${EP1_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_medium_rand_ep1()        { _run "$OLMO7B"  lcb_medium_olmo7b_srpo_rand_ep1        "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" "${EP1_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_medium_l2new_ep1_s0()    { _run "$OLMO7B"  lcb_medium_olmo7b_srpo_l2new_ep1_s0    "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}"                       "${EP1_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_medium_l2new_ep1_s420()  { _run "$OLMO7B"  lcb_medium_olmo7b_srpo_l2new_ep1_s420  "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}"                       "${EP1_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_medium_rand_ep1_s0()     { _run "$OLMO7B"  lcb_medium_olmo7b_srpo_rand_ep1_s0     "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" "${EP1_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_medium_rand_ep1_s420()   { _run "$OLMO7B"  lcb_medium_olmo7b_srpo_rand_ep1_s420   "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" "${EP1_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=1 ; }

# --- LiveCodeBench Medium: SRPO + L2new + localization offset = 3 (l2n-o3) ---
# SRPO_LOC_OFFSET=3 pulls the localizer's chosen step back by 3 boundaries
# (floor at empty prefix). Same training schedule as the standard l2n_ep1 jobs.
cmd_olmo7b_lcb_medium_l2new_o3_ep1()      { export SRPO_LOC_OFFSET=3; _run "$OLMO7B" lcb_medium_olmo7b_srpo_l2new_o3_ep1      "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${EP1_OVERRIDES[@]}"                              trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_medium_l2new_o3_ep1_s0()   { export SRPO_LOC_OFFSET=3; _run "$OLMO7B" lcb_medium_olmo7b_srpo_l2new_o3_ep1_s0   "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${EP1_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_medium_l2new_o3_ep1_s420() { export SRPO_LOC_OFFSET=3; _run "$OLMO7B" lcb_medium_olmo7b_srpo_l2new_o3_ep1_s420 "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${EP1_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=1 ; }

# --- LiveCodeBench Hard: SRPO with L2new prompt ---
cmd_qwen14b_lcb_hard_l2new() { _run "$QWEN14B" lcb_hard_qwen14b_srpo_l2new "${LCB_HARD_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" trainer.resume_mode=disable ; }
cmd_olmo7b_lcb_hard_l2new()  { _run "$OLMO7B"  lcb_hard_olmo7b_srpo_l2new  "${LCB_HARD_DATA[@]}" "${OLMO_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_hard_rand()   { _run "$OLMO7B"  lcb_hard_olmo7b_srpo_rand   "${LCB_HARD_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=1 ; }

# --- NuminaMath Olympiads: SRPO_clip (proper GRPO/PPO clipped surrogate + srpo advantages + suffix mask) with rand and l2new variants ---
cmd_olmo7b_numina_oly_srpo_clip_rand()   { _run "$OLMO7B"  numina_oly_olmo7b_srpo_clip_rand   "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SRPO_CLIP_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_srpo_clip_rand()  { _run "$QWEN14B" numina_oly_qwen14b_srpo_clip_rand  "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_CLIP_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_olmo7b_numina_oly_srpo_clip_l2new()  { _run "$OLMO7B"  numina_oly_olmo7b_srpo_clip_l2new  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SRPO_CLIP_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_srpo_clip_l2new() { _run "$QWEN14B" numina_oly_qwen14b_srpo_clip_l2new "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_CLIP_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }

# SRPO_clip seed-420 variants (distinct experiment_name to avoid checkpoint/wandb collision with s42 default)
cmd_olmo7b_numina_oly_srpo_clip_rand_s420()   { _run "$OLMO7B"  numina_oly_olmo7b_srpo_clip_rand_s420   "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SRPO_CLIP_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_srpo_clip_rand_s420()  { _run "$QWEN14B" numina_oly_qwen14b_srpo_clip_rand_s420  "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_CLIP_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_olmo7b_numina_oly_srpo_clip_l2new_s420()  { _run "$OLMO7B"  numina_oly_olmo7b_srpo_clip_l2new_s420  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SRPO_CLIP_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_srpo_clip_l2new_s420() { _run "$QWEN14B" numina_oly_qwen14b_srpo_clip_l2new_s420 "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_CLIP_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }

# SRPO_clip-SM (sequence-mean aggregation) — default seed
cmd_qwen14b_numina_oly_srpo_clip_sm_rand()  { _run "$QWEN14B" numina_oly_qwen14b_srpo_clip_sm_rand  "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_CLIP_SM_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_srpo_clip_sm_l2new() { _run "$QWEN14B" numina_oly_qwen14b_srpo_clip_sm_l2new "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_CLIP_SM_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_olmo7b_numina_oly_srpo_clip_sm_l2new()  { _run "$OLMO7B"  numina_oly_olmo7b_srpo_clip_sm_l2new  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SRPO_CLIP_SM_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }

# SRPO_clip-KL seed-420 variants (qwen14b only — test whether KL anchoring further stabilizes srpo_clip)
cmd_qwen14b_numina_oly_srpo_clip_kl_rand_s420()  { _run "$QWEN14B" numina_oly_qwen14b_srpo_clip_kl_rand_s420  "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_CLIP_KL_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_srpo_clip_kl_l2new_s420() { _run "$QWEN14B" numina_oly_qwen14b_srpo_clip_kl_l2new_s420 "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_CLIP_KL_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }

# --- NuminaMath Olympiads: SRPO with L2new prompt + GREEDY localization (temp=0.0 via --loc_temp=0.0) ---
cmd_olmo7b_numina_oly_l2new_greedy()  { _run "$OLMO7B"  numina_oly_olmo7b_srpo_l2new_greedy  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_l2new_greedy() { _run "$QWEN14B" numina_oly_qwen14b_srpo_l2new_greedy "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }

# --- NuminaMath Olympiads: SRPO_2x4 ("SRPO doubled" — 2 groups of 4 corrections, no parents) ---
SRPO_2X4_OVERRIDES=(actor_rollout_ref.rollout.agent.default_agent_loop=srpo_2x4_agent)
cmd_olmo7b_numina_oly_srpo2x4()       { _run "$OLMO7B"  numina_oly_olmo7b_srpo2x4       "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SRPO_2X4_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_srpo2x4()      { _run "$QWEN14B" numina_oly_qwen14b_srpo2x4      "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_2X4_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_olmo7b_numina_oly_srpo2x4_l2new() { _run "$OLMO7B"  numina_oly_olmo7b_srpo2x4_l2new "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SRPO_2X4_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_srpo2x4_l2new(){ _run "$QWEN14B" numina_oly_qwen14b_srpo2x4_l2new "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_2X4_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }

# --- NuminaMath Olympiads: SRPO_1x8 (8 corrections, no parent) — uses srpo_1x8_agent (loc temp=0.0 default) ---
SRPO_1X8_OVERRIDES=(actor_rollout_ref.rollout.agent.default_agent_loop=srpo_1x8_agent)
SRPO_1X8_RAND_OVERRIDES=(actor_rollout_ref.rollout.agent.default_agent_loop=rrpo_1x8_agent)
cmd_olmo7b_numina_oly_srpo_1x8()          { _run "$OLMO7B"  numina_oly_olmo7b_srpo_1x8          "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SRPO_1X8_OVERRIDES[@]}"      trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_srpo_1x8()         { _run "$QWEN14B" numina_oly_qwen14b_srpo_1x8         "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_1X8_OVERRIDES[@]}"      trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_olmo7b_numina_oly_srpo_1x8_rand()     { _run "$OLMO7B"  numina_oly_olmo7b_srpo_1x8_rand     "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SRPO_1X8_RAND_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_srpo_1x8_rand()    { _run "$QWEN14B" numina_oly_qwen14b_srpo_1x8_rand    "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_1X8_RAND_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_olmo7b_numina_oly_srpo_1x8_l2new()    { _run "$OLMO7B"  numina_oly_olmo7b_srpo_1x8_l2new    "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SRPO_1X8_OVERRIDES[@]}"      trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_srpo_1x8_l2new()   { _run "$QWEN14B" numina_oly_qwen14b_srpo_1x8_l2new   "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_1X8_OVERRIDES[@]}"      trainer.resume_mode=disable trainer.test_freq=6 ; }

# --- NumOly + SciKnowEval L3 (100×4 = 400) mixed: SRPO self-loc, 1-epoch ---
cmd_olmo7b_numina_oly_sk400()  { _run "$OLMO7B"  numina_oly_sk400_olmo7b_srpo  "${NUMINA_OLY_PLUS_SCIKNOW400_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${EP1_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_sk400() { _run "$QWEN14B" numina_oly_sk400_qwen14b_srpo "${NUMINA_OLY_PLUS_SCIKNOW400_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${EP1_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }

# --- NumOly + SciKnowEval mixed: SRPO-rand baseline (random localization), 1-epoch ---
cmd_olmo7b_numina_oly_sk400_rand()  { _run "$OLMO7B"  numina_oly_sk400_olmo7b_srpo_rand  "${NUMINA_OLY_PLUS_SCIKNOW400_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${RAND_OVERRIDES[@]}" "${EP1_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_sk400_rand() { _run "$QWEN14B" numina_oly_sk400_qwen14b_srpo_rand "${NUMINA_OLY_PLUS_SCIKNOW400_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" "${EP1_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }

# --- NuminaMath Olympiads: SRPO-NM (no shared-prefix masking) ---
cmd_olmo7b_numina_oly_nm()  { _run "$OLMO7B"  numina_oly_olmo7b_srpo_nomask  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${NM_OVERRIDES[@]}" trainer.resume_mode=disable ; }
cmd_qwen14b_numina_oly_nm() { _run "$QWEN14B" numina_oly_qwen14b_srpo_nomask "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${NM_OVERRIDES[@]}" trainer.resume_mode=disable ; }

# --- NuminaMath Olympiads: SRPO-NM-rand (no masking + random localization) ---
cmd_olmo7b_numina_oly_nm_rand()  { _run "$OLMO7B"  numina_oly_olmo7b_srpo_nomask_rand  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${NM_RAND_OVERRIDES[@]}" trainer.resume_mode=disable ; }
cmd_qwen14b_numina_oly_nm_rand() { _run "$QWEN14B" numina_oly_qwen14b_srpo_nomask_rand "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${NM_RAND_OVERRIDES[@]}" trainer.resume_mode=disable ; }

# --- NuminaMath Olympiads: SRPO-NM, 1-epoch (early-stop ablation) ---
cmd_olmo7b_numina_oly_nm_ep1()  { _run "$OLMO7B"  numina_oly_olmo7b_srpo_nomask_ep1  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${NM_OVERRIDES[@]}" "${EP1_OVERRIDES[@]}" ; }

# --- NuminaMath Olympiads: SRPO-rand ---
cmd_olmo7b_numina_oly_rand()   { _run "$OLMO7B"   numina_oly_olmo7b_srpo_rand   "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_rand()      { _run "$QWEN14B"      numina_oly_qwen14b_srpo_rand      "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" trainer.test_freq=6 ; }

# ─── Seed variants ────────────────────────────────────────────────────────────
SEED_0_OVERRIDES=(actor_rollout_ref.actor.fsdp_config.seed=0   actor_rollout_ref.ref.fsdp_config.seed=0   critic.model.fsdp_config.seed=0)
SEED_420_OVERRIDES=(actor_rollout_ref.actor.fsdp_config.seed=420 actor_rollout_ref.ref.fsdp_config.seed=420 critic.model.fsdp_config.seed=420)

# SRPO seed variants
cmd_olmo7b_numina_oly_s0()    { _run "$OLMO7B" numina_oly_olmo7b_srpo_s0    "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}" ; }
cmd_olmo7b_numina_oly_s420()  { _run "$OLMO7B" numina_oly_olmo7b_srpo_s420  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" ; }

# SRPO-rand seed variants
cmd_olmo7b_numina_oly_rand_s0()    { _run "$OLMO7B" numina_oly_olmo7b_srpo_rand_s0    "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}" ; }
cmd_olmo7b_numina_oly_rand_s420()  { _run "$OLMO7B" numina_oly_olmo7b_srpo_rand_s420  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" ; }

# qwen14b seed variants: srpo self-loc (l2new + default temp 0.3) and srpo-rand
cmd_qwen14b_numina_oly_l2new_s0()    { _run "$QWEN14B" numina_oly_qwen14b_srpo_l2new_s0    "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_l2new_s420()  { _run "$QWEN14B" numina_oly_qwen14b_srpo_l2new_s420  "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_rand_s0()     { _run "$QWEN14B" numina_oly_qwen14b_srpo_rand_s0     "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_rand_s420()   { _run "$QWEN14B" numina_oly_qwen14b_srpo_rand_s420   "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }

# olmo7b seed variants: srpo self-loc (l2new + default temp 0.3)
cmd_olmo7b_numina_oly_l2new_s0()    { _run "$OLMO7B"  numina_oly_olmo7b_srpo_l2new_s0    "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SEED_0_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_olmo7b_numina_oly_l2new_s420()  { _run "$OLMO7B"  numina_oly_olmo7b_srpo_l2new_s420  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }

# olmo7b LCB seed variants: srpo l2new and srpo-rand on medium + hard
cmd_olmo7b_lcb_medium_l2new_s0()    { _run "$OLMO7B"  lcb_medium_olmo7b_srpo_l2new_s0    "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}"                       "${SEED_0_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_medium_l2new_s420()  { _run "$OLMO7B"  lcb_medium_olmo7b_srpo_l2new_s420  "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}"                       "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_medium_rand_s0()     { _run "$OLMO7B"  lcb_medium_olmo7b_srpo_rand_s0     "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_medium_rand_s420()   { _run "$OLMO7B"  lcb_medium_olmo7b_srpo_rand_s420   "${LCB_MEDIUM_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_hard_l2new_s0()      { _run "$OLMO7B"  lcb_hard_olmo7b_srpo_l2new_s0      "${LCB_HARD_DATA[@]}"   "${OLMO_OVERRIDES[@]}"                       "${SEED_0_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_hard_l2new_s420()    { _run "$OLMO7B"  lcb_hard_olmo7b_srpo_l2new_s420    "${LCB_HARD_DATA[@]}"   "${OLMO_OVERRIDES[@]}"                       "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_hard_rand_s0()       { _run "$OLMO7B"  lcb_hard_olmo7b_srpo_rand_s0       "${LCB_HARD_DATA[@]}"   "${OLMO_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=1 ; }
cmd_olmo7b_lcb_hard_rand_s420()     { _run "$OLMO7B"  lcb_hard_olmo7b_srpo_rand_s420     "${LCB_HARD_DATA[@]}"   "${OLMO_OVERRIDES[@]}" "${RAND_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=1 ; }

# SRPO-2x4 (l2new) seed variants — sampling-ablation: 2 fresh prefixes × 4 counterfactuals
cmd_qwen14b_numina_oly_srpo2x4_l2new_s0()    { _run "$QWEN14B" numina_oly_qwen14b_srpo2x4_l2new_s0    "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_2X4_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_srpo2x4_l2new_s420()  { _run "$QWEN14B" numina_oly_qwen14b_srpo2x4_l2new_s420  "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_2X4_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_olmo7b_numina_oly_srpo2x4_l2new_s0()     { _run "$OLMO7B"  numina_oly_olmo7b_srpo2x4_l2new_s0     "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SRPO_2X4_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_olmo7b_numina_oly_srpo2x4_l2new_s420()   { _run "$OLMO7B"  numina_oly_olmo7b_srpo2x4_l2new_s420   "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SRPO_2X4_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }

# SRPO-1x8 (l2new) seed variants — sampling-ablation: 1 fresh prefix + 8 counterfactuals
cmd_qwen14b_numina_oly_srpo_1x8_l2new_s0()   { _run "$QWEN14B" numina_oly_qwen14b_srpo_1x8_l2new_s0   "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_1X8_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_srpo_1x8_l2new_s420() { _run "$QWEN14B" numina_oly_qwen14b_srpo_1x8_l2new_s420 "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SRPO_1X8_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_olmo7b_numina_oly_srpo_1x8_l2new_s0()    { _run "$OLMO7B"  numina_oly_olmo7b_srpo_1x8_l2new_s0    "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SRPO_1X8_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}"   trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_olmo7b_numina_oly_srpo_1x8_l2new_s420()  { _run "$OLMO7B"  numina_oly_olmo7b_srpo_1x8_l2new_s420  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}"   "${SRPO_1X8_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }

# ─── SLURM submission helper ────────────────────────────────────────────────

submit() {
    local job_name=$1
    local cmd=$2
    local n_gpus=${3:-2}
    local time_limit=${4:-24:00:00}

    sbatch --partition=q1 \
           --nodes=1 \
           --gpus-per-node=${n_gpus} \
           --cpus-per-gpu=10 \
           --time=${time_limit} \
           --job-name="${job_name}" \
           --output="batch_scripts/logs/${job_name}.out" \
           --error="batch_scripts/logs/${job_name}.err" \
           --export="ALL,VERL_LOGGING_LEVEL=INFO${ICS_LOC_TEMP_VAL:+,ICS_LOC_TEMP=${ICS_LOC_TEMP_VAL}}" \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${SCPO_DIR} && ${cmd}'"

    echo "Submitted: ${job_name}"
}

# ─── Entrypoint ─────────────────────────────────────────────────────────────

TRAINING_SEED=42
ICS_LOC_TEMP_VAL=""
POSITIONAL=()
for arg in "$@"; do
    case $arg in
        --training_seed=*) TRAINING_SEED="${arg#*=}" ;;
        --training_seed)   shift_next=true ;;
        --loc_temp=*)      ICS_LOC_TEMP_VAL="${arg#*=}" ;;
        *)
            if [ "${shift_next:-}" = true ]; then
                TRAINING_SEED="$arg"
                shift_next=false
            else
                POSITIONAL+=("$arg")
            fi
            ;;
    esac
done
[ -n "$ICS_LOC_TEMP_VAL" ] && export ICS_LOC_TEMP="$ICS_LOC_TEMP_VAL"

SEED_OVERRIDES=(
    actor_rollout_ref.actor.fsdp_config.seed="${TRAINING_SEED}"
    actor_rollout_ref.ref.fsdp_config.seed="${TRAINING_SEED}"
    critic.model.fsdp_config.seed="${TRAINING_SEED}"
)
SHARED_OVERRIDES+=("${SEED_OVERRIDES[@]}")

JOB=${POSITIONAL[0]:?Usage: $0 <job> [--local] [--training_seed N]  (run with 'list' to see all jobs)}
MODE=${POSITIONAL[1]:-slurm}

run_or_submit() {
    local job_name=$1
    local func=$2
    local case_key=${3:-$JOB}
    local n_gpus=${4:-2}
    local time_limit=${5:-24:00:00}

    if [ "$MODE" = "--local" ]; then
        echo "=== Running locally: ${job_name} ==="
        $func
    else
        submit "$job_name" "bash ${SCPO_DIR}/batch_scripts/submit_srpo.sh ${case_key} --local --training_seed=${TRAINING_SEED}${ICS_LOC_TEMP_VAL:+ --loc_temp=${ICS_LOC_TEMP_VAL}}" "$n_gpus" "$time_limit"
    fi
}

case $JOB in
    # --- MATH Level 5 ---
    olmo7b_mathlvl5) run_or_submit "srpo_olmo7b_mlvl5" cmd_olmo7b_mathlvl5 ;;

    # --- NuminaMath Olympiads: SRPO ---
    olmo7b_numina_oly)        run_or_submit "srpo_olmo7b_noly"        cmd_olmo7b_numina_oly ;;
    qwen14b_numina_oly)       run_or_submit "srpo_qwen14b_noly"       cmd_qwen14b_numina_oly      qwen14b_numina_oly      4  48:00:00 ;;

    # SRPO self-loc with L2new (early-biasing) localization prompt
    olmo7b_numina_oly_l2new)   run_or_submit "srpo_olmo7b_noly_l2n"   cmd_olmo7b_numina_oly_l2new   olmo7b_numina_oly_l2new   2 ;;
    qwen14b_numina_oly_l2new)  run_or_submit "srpo_qwen14b_noly_l2n" cmd_qwen14b_numina_oly_l2new qwen14b_numina_oly_l2new  4 48:00:00 ;;

    # LiveCodeBench Medium: SRPO + L2new
    qwen14b_lcb_medium_l2new)  run_or_submit "srpo_qwen14b_lcbm_l2n" cmd_qwen14b_lcb_medium_l2new qwen14b_lcb_medium_l2new  4 48:00:00 ;;
    olmo7b_lcb_medium_l2new)   run_or_submit "srpo_olmo7b_lcbm_l2n"  cmd_olmo7b_lcb_medium_l2new  olmo7b_lcb_medium_l2new   2 48:00:00 ;;
    olmo7b_lcb_medium_rand)    run_or_submit "srpo_olmo7b_lcbm_rand" cmd_olmo7b_lcb_medium_rand   olmo7b_lcb_medium_rand    2 48:00:00 ;;
    olmo7b_lcb_medium_l2new_ep1)      run_or_submit "srpo_olmo7b_lcbm_l2n_ep1"      cmd_olmo7b_lcb_medium_l2new_ep1      olmo7b_lcb_medium_l2new_ep1      2 24:00:00 ;;
    olmo7b_lcb_medium_rand_ep1)       run_or_submit "srpo_olmo7b_lcbm_rand_ep1"     cmd_olmo7b_lcb_medium_rand_ep1       olmo7b_lcb_medium_rand_ep1       2 24:00:00 ;;
    olmo7b_lcb_medium_l2new_ep1_s0)   run_or_submit "srpo_olmo7b_lcbm_l2n_ep1_s0"   cmd_olmo7b_lcb_medium_l2new_ep1_s0   olmo7b_lcb_medium_l2new_ep1_s0   2 24:00:00 ;;
    olmo7b_lcb_medium_l2new_ep1_s420) run_or_submit "srpo_olmo7b_lcbm_l2n_ep1_s420" cmd_olmo7b_lcb_medium_l2new_ep1_s420 olmo7b_lcb_medium_l2new_ep1_s420 2 24:00:00 ;;
    olmo7b_lcb_medium_rand_ep1_s0)    run_or_submit "srpo_olmo7b_lcbm_rand_ep1_s0"  cmd_olmo7b_lcb_medium_rand_ep1_s0    olmo7b_lcb_medium_rand_ep1_s0    2 24:00:00 ;;
    olmo7b_lcb_medium_rand_ep1_s420)  run_or_submit "srpo_olmo7b_lcbm_rand_ep1_s420" cmd_olmo7b_lcb_medium_rand_ep1_s420 olmo7b_lcb_medium_rand_ep1_s420  2 24:00:00 ;;

    # SRPO + L2new + localization offset 3 (SRPO_LOC_OFFSET=3)
    olmo7b_lcb_medium_l2new_o3_ep1)      run_or_submit "srpo_olmo7b_lcbm_l2n_o3_ep1"      cmd_olmo7b_lcb_medium_l2new_o3_ep1      olmo7b_lcb_medium_l2new_o3_ep1      2 24:00:00 ;;
    olmo7b_lcb_medium_l2new_o3_ep1_s0)   run_or_submit "srpo_olmo7b_lcbm_l2n_o3_ep1_s0"   cmd_olmo7b_lcb_medium_l2new_o3_ep1_s0   olmo7b_lcb_medium_l2new_o3_ep1_s0   2 24:00:00 ;;
    olmo7b_lcb_medium_l2new_o3_ep1_s420) run_or_submit "srpo_olmo7b_lcbm_l2n_o3_ep1_s420" cmd_olmo7b_lcb_medium_l2new_o3_ep1_s420 olmo7b_lcb_medium_l2new_o3_ep1_s420 2 24:00:00 ;;

    # LiveCodeBench Hard: SRPO + L2new
    qwen14b_lcb_hard_l2new)    run_or_submit "srpo_qwen14b_lcbh_l2n" cmd_qwen14b_lcb_hard_l2new   qwen14b_lcb_hard_l2new    4 48:00:00 ;;
    olmo7b_lcb_hard_l2new)     run_or_submit "srpo_olmo7b_lcbh_l2n"  cmd_olmo7b_lcb_hard_l2new    olmo7b_lcb_hard_l2new     2 48:00:00 ;;
    olmo7b_lcb_hard_rand)      run_or_submit "srpo_olmo7b_lcbh_rand" cmd_olmo7b_lcb_hard_rand     olmo7b_lcb_hard_rand      2 48:00:00 ;;

    # LiveCodeBench olmo7b seed variants (s0, s420) for SRPO + L2new and SRPO + rand
    olmo7b_lcb_medium_l2new_s0)   run_or_submit "srpo_olmo7b_lcbm_l2n_s0"   cmd_olmo7b_lcb_medium_l2new_s0   olmo7b_lcb_medium_l2new_s0   2 48:00:00 ;;
    olmo7b_lcb_medium_l2new_s420) run_or_submit "srpo_olmo7b_lcbm_l2n_s420" cmd_olmo7b_lcb_medium_l2new_s420 olmo7b_lcb_medium_l2new_s420 2 48:00:00 ;;
    olmo7b_lcb_medium_rand_s0)    run_or_submit "srpo_olmo7b_lcbm_rand_s0"  cmd_olmo7b_lcb_medium_rand_s0    olmo7b_lcb_medium_rand_s0    2 48:00:00 ;;
    olmo7b_lcb_medium_rand_s420)  run_or_submit "srpo_olmo7b_lcbm_rand_s420" cmd_olmo7b_lcb_medium_rand_s420 olmo7b_lcb_medium_rand_s420  2 48:00:00 ;;
    olmo7b_lcb_hard_l2new_s0)     run_or_submit "srpo_olmo7b_lcbh_l2n_s0"   cmd_olmo7b_lcb_hard_l2new_s0     olmo7b_lcb_hard_l2new_s0     2 48:00:00 ;;
    olmo7b_lcb_hard_l2new_s420)   run_or_submit "srpo_olmo7b_lcbh_l2n_s420" cmd_olmo7b_lcb_hard_l2new_s420   olmo7b_lcb_hard_l2new_s420   2 48:00:00 ;;
    olmo7b_lcb_hard_rand_s0)      run_or_submit "srpo_olmo7b_lcbh_rand_s0"  cmd_olmo7b_lcb_hard_rand_s0      olmo7b_lcb_hard_rand_s0      2 48:00:00 ;;
    olmo7b_lcb_hard_rand_s420)    run_or_submit "srpo_olmo7b_lcbh_rand_s420" cmd_olmo7b_lcb_hard_rand_s420   olmo7b_lcb_hard_rand_s420    2 48:00:00 ;;

    # SRPO_clip (proper GRPO/PPO clipped surrogate + srpo advantages + suffix mask): rand and l2new variants on default seed
    olmo7b_numina_oly_srpo_clip_rand)    run_or_submit "srpo_clip_olmo7b_noly_rand"    cmd_olmo7b_numina_oly_srpo_clip_rand    olmo7b_numina_oly_srpo_clip_rand    2 ;;
    qwen14b_numina_oly_srpo_clip_rand)   run_or_submit "srpo_clip_qwen14b_noly_rand"   cmd_qwen14b_numina_oly_srpo_clip_rand   qwen14b_numina_oly_srpo_clip_rand   4 48:00:00 ;;
    olmo7b_numina_oly_srpo_clip_l2new)   run_or_submit "srpo_clip_olmo7b_noly_l2n"     cmd_olmo7b_numina_oly_srpo_clip_l2new   olmo7b_numina_oly_srpo_clip_l2new   2 ;;
    qwen14b_numina_oly_srpo_clip_l2new)  run_or_submit "srpo_clip_qwen14b_noly_l2n"    cmd_qwen14b_numina_oly_srpo_clip_l2new  qwen14b_numina_oly_srpo_clip_l2new  4 48:00:00 ;;

    # SRPO_clip seed-420 variants
    olmo7b_numina_oly_srpo_clip_rand_s420)    run_or_submit "srpo_clip_olmo7b_noly_rand_s420"    cmd_olmo7b_numina_oly_srpo_clip_rand_s420    olmo7b_numina_oly_srpo_clip_rand_s420    2 ;;
    qwen14b_numina_oly_srpo_clip_rand_s420)   run_or_submit "srpo_clip_qwen14b_noly_rand_s420"   cmd_qwen14b_numina_oly_srpo_clip_rand_s420   qwen14b_numina_oly_srpo_clip_rand_s420   4 48:00:00 ;;
    olmo7b_numina_oly_srpo_clip_l2new_s420)   run_or_submit "srpo_clip_olmo7b_noly_l2n_s420"     cmd_olmo7b_numina_oly_srpo_clip_l2new_s420   olmo7b_numina_oly_srpo_clip_l2new_s420   2 ;;
    qwen14b_numina_oly_srpo_clip_l2new_s420)  run_or_submit "srpo_clip_qwen14b_noly_l2n_s420"    cmd_qwen14b_numina_oly_srpo_clip_l2new_s420  qwen14b_numina_oly_srpo_clip_l2new_s420  4 48:00:00 ;;

    # SRPO_clip-SM (seq-mean) qwen14b only, default seed
    qwen14b_numina_oly_srpo_clip_sm_rand)   run_or_submit "srpo_clip_sm_qwen14b_noly_rand"   cmd_qwen14b_numina_oly_srpo_clip_sm_rand   qwen14b_numina_oly_srpo_clip_sm_rand   4 48:00:00 ;;
    qwen14b_numina_oly_srpo_clip_sm_l2new)  run_or_submit "srpo_clip_sm_qwen14b_noly_l2n"    cmd_qwen14b_numina_oly_srpo_clip_sm_l2new  qwen14b_numina_oly_srpo_clip_sm_l2new  4 48:00:00 ;;
    olmo7b_numina_oly_srpo_clip_sm_l2new)   run_or_submit "srpo_clip_sm_olmo7b_noly_l2n"     cmd_olmo7b_numina_oly_srpo_clip_sm_l2new   olmo7b_numina_oly_srpo_clip_sm_l2new   2 48:00:00 ;;

    # SRPO_clip-KL seed-420 (qwen14b only — test KL anchoring on top of srpo_clip clipping)
    qwen14b_numina_oly_srpo_clip_kl_rand_s420)   run_or_submit "srpo_clip_kl_qwen14b_noly_rand_s420"   cmd_qwen14b_numina_oly_srpo_clip_kl_rand_s420   qwen14b_numina_oly_srpo_clip_kl_rand_s420   4 48:00:00 ;;
    qwen14b_numina_oly_srpo_clip_kl_l2new_s420)  run_or_submit "srpo_clip_kl_qwen14b_noly_l2n_s420"    cmd_qwen14b_numina_oly_srpo_clip_kl_l2new_s420  qwen14b_numina_oly_srpo_clip_kl_l2new_s420  4 48:00:00 ;;

    # SRPO self-loc with L2new prompt + greedy localization (loc temp=0.0)
    olmo7b_numina_oly_l2new_greedy)  run_or_submit "srpo_olmo7b_noly_l2ng"  cmd_olmo7b_numina_oly_l2new_greedy  olmo7b_numina_oly_l2new_greedy  2 ;;
    qwen14b_numina_oly_l2new_greedy) run_or_submit "srpo_qwen14b_noly_l2ng" cmd_qwen14b_numina_oly_l2new_greedy qwen14b_numina_oly_l2new_greedy 4 48:00:00 ;;

    # SRPO_2x4: 2 groups of 4 corrections (no parents) — "srpo doubled"
    olmo7b_numina_oly_srpo2x4)       run_or_submit "srpo2x4_olmo7b_noly"      cmd_olmo7b_numina_oly_srpo2x4       olmo7b_numina_oly_srpo2x4       2 ;;
    qwen14b_numina_oly_srpo2x4)      run_or_submit "srpo2x4_qwen14b_noly"     cmd_qwen14b_numina_oly_srpo2x4      qwen14b_numina_oly_srpo2x4      4 48:00:00 ;;
    olmo7b_numina_oly_srpo2x4_l2new)  run_or_submit "srpo2x4l2n_olmo7b_noly"  cmd_olmo7b_numina_oly_srpo2x4_l2new  olmo7b_numina_oly_srpo2x4_l2new  2 ;;
    qwen14b_numina_oly_srpo2x4_l2new) run_or_submit "srpo2x4l2n_qwen14b_noly" cmd_qwen14b_numina_oly_srpo2x4_l2new qwen14b_numina_oly_srpo2x4_l2new 4 48:00:00 ;;
    olmo7b_numina_oly_srpo2x4_l2new_s0)    run_or_submit "srpo2x4l2n_olmo7b_noly_s0"    cmd_olmo7b_numina_oly_srpo2x4_l2new_s0    olmo7b_numina_oly_srpo2x4_l2new_s0    2 ;;
    olmo7b_numina_oly_srpo2x4_l2new_s420)  run_or_submit "srpo2x4l2n_olmo7b_noly_s420"  cmd_olmo7b_numina_oly_srpo2x4_l2new_s420  olmo7b_numina_oly_srpo2x4_l2new_s420  2 ;;
    qwen14b_numina_oly_srpo2x4_l2new_s0)   run_or_submit "srpo2x4l2n_qwen14b_noly_s0"   cmd_qwen14b_numina_oly_srpo2x4_l2new_s0   qwen14b_numina_oly_srpo2x4_l2new_s0   4 48:00:00 ;;
    qwen14b_numina_oly_srpo2x4_l2new_s420) run_or_submit "srpo2x4l2n_qwen14b_noly_s420" cmd_qwen14b_numina_oly_srpo2x4_l2new_s420 qwen14b_numina_oly_srpo2x4_l2new_s420 4 48:00:00 ;;

    # SRPO_1x8: 8 corrections from one localized prefix, no parent (loc temp=0.0 default)
    olmo7b_numina_oly_srpo_1x8)       run_or_submit "srpo1x8_olmo7b_noly"       cmd_olmo7b_numina_oly_srpo_1x8       olmo7b_numina_oly_srpo_1x8       2 ;;
    qwen14b_numina_oly_srpo_1x8)      run_or_submit "srpo1x8_qwen14b_noly"      cmd_qwen14b_numina_oly_srpo_1x8      qwen14b_numina_oly_srpo_1x8      4 48:00:00 ;;
    olmo7b_numina_oly_srpo_1x8_rand)  run_or_submit "srpo1x8r_olmo7b_noly"      cmd_olmo7b_numina_oly_srpo_1x8_rand  olmo7b_numina_oly_srpo_1x8_rand  2 ;;
    qwen14b_numina_oly_srpo_1x8_rand) run_or_submit "srpo1x8r_qwen14b_noly"     cmd_qwen14b_numina_oly_srpo_1x8_rand qwen14b_numina_oly_srpo_1x8_rand 4 48:00:00 ;;
    olmo7b_numina_oly_srpo_1x8_l2new)  run_or_submit "srpo1x8l2n_olmo7b_noly"  cmd_olmo7b_numina_oly_srpo_1x8_l2new  olmo7b_numina_oly_srpo_1x8_l2new  2 ;;
    qwen14b_numina_oly_srpo_1x8_l2new) run_or_submit "srpo1x8l2n_qwen14b_noly" cmd_qwen14b_numina_oly_srpo_1x8_l2new qwen14b_numina_oly_srpo_1x8_l2new 4 48:00:00 ;;
    olmo7b_numina_oly_srpo_1x8_l2new_s0)    run_or_submit "srpo1x8l2n_olmo7b_noly_s0"    cmd_olmo7b_numina_oly_srpo_1x8_l2new_s0    olmo7b_numina_oly_srpo_1x8_l2new_s0    2 ;;
    olmo7b_numina_oly_srpo_1x8_l2new_s420)  run_or_submit "srpo1x8l2n_olmo7b_noly_s420"  cmd_olmo7b_numina_oly_srpo_1x8_l2new_s420  olmo7b_numina_oly_srpo_1x8_l2new_s420  2 ;;
    qwen14b_numina_oly_srpo_1x8_l2new_s0)   run_or_submit "srpo1x8l2n_qwen14b_noly_s0"   cmd_qwen14b_numina_oly_srpo_1x8_l2new_s0   qwen14b_numina_oly_srpo_1x8_l2new_s0   4 48:00:00 ;;
    qwen14b_numina_oly_srpo_1x8_l2new_s420) run_or_submit "srpo1x8l2n_qwen14b_noly_s420" cmd_qwen14b_numina_oly_srpo_1x8_l2new_s420 qwen14b_numina_oly_srpo_1x8_l2new_s420 4 48:00:00 ;;

    # SRPO self-loc on numinaoly+sciknow400 mixed (1-epoch, equiv trajectories to 2-epoch numina_oly)
    olmo7b_numina_oly_sk400)   run_or_submit "srpo_olmo7b_noly_sk4"   cmd_olmo7b_numina_oly_sk400   olmo7b_numina_oly_sk400   2 ;;
    qwen14b_numina_oly_sk400)  run_or_submit "srpo_qwen14b_noly_sk4" cmd_qwen14b_numina_oly_sk400 qwen14b_numina_oly_sk400  4 48:00:00 ;;

    # SRPO-rand baseline on mixed dataset
    olmo7b_numina_oly_sk400_rand)   run_or_submit "srpo_rand_olmo7b_noly_sk4"   cmd_olmo7b_numina_oly_sk400_rand   olmo7b_numina_oly_sk400_rand   2 ;;
    qwen14b_numina_oly_sk400_rand)  run_or_submit "srpo_rand_qwen14b_noly_sk4" cmd_qwen14b_numina_oly_sk400_rand qwen14b_numina_oly_sk400_rand 4 48:00:00 ;;

    # --- NuminaMath Olympiads: SRPO-NM ---
    olmo7b_numina_oly_nm)  run_or_submit "srpo_nomask_olmo7b_noly"  cmd_olmo7b_numina_oly_nm  olmo7b_numina_oly_nm  2 ;;
    qwen14b_numina_oly_nm) run_or_submit "srpo_nomask_qwen14b_noly" cmd_qwen14b_numina_oly_nm qwen14b_numina_oly_nm 4 48:00:00 ;;
    olmo7b_numina_oly_nm_rand)  run_or_submit "srpo_nomask_olmo7b_noly_rand"  cmd_olmo7b_numina_oly_nm_rand  olmo7b_numina_oly_nm_rand  2 ;;
    qwen14b_numina_oly_nm_rand) run_or_submit "srpo_nomask_qwen14b_noly_rand" cmd_qwen14b_numina_oly_nm_rand qwen14b_numina_oly_nm_rand 4 48:00:00 ;;

    # --- NuminaMath Olympiads: SRPO-NM 1-epoch ---
    olmo7b_numina_oly_nm_ep1)  run_or_submit "srpo_nomask_olmo7b_noly_ep1"  cmd_olmo7b_numina_oly_nm_ep1  olmo7b_numina_oly_nm_ep1  2 ;;

    # --- NuminaMath Olympiads: SRPO-rand ---
    olmo7b_numina_oly_rand)        run_or_submit "srpo_olmo7b_noly_rand"        cmd_olmo7b_numina_oly_rand ;;
    qwen14b_numina_oly_rand)       run_or_submit "srpo_qwen14b_noly_rand"       cmd_qwen14b_numina_oly_rand      qwen14b_numina_oly_rand      4  48:00:00 ;;

    # --- Seed variants: SRPO ---
    olmo7b_numina_oly_s0)    run_or_submit "srpo_olmo7b_noly_s0"    cmd_olmo7b_numina_oly_s0 ;;
    olmo7b_numina_oly_s420)  run_or_submit "srpo_olmo7b_noly_s420"  cmd_olmo7b_numina_oly_s420 ;;

    # --- Seed variants: SRPO-rand ---
    olmo7b_numina_oly_rand_s0)    run_or_submit "srpo_olmo7b_noly_rand_s0"    cmd_olmo7b_numina_oly_rand_s0 ;;
    olmo7b_numina_oly_rand_s420)  run_or_submit "srpo_olmo7b_noly_rand_s420"  cmd_olmo7b_numina_oly_rand_s420 ;;

    # qwen14b seed variants (srpo self-loc l2new + srpo-rand) on regular numina_oly
    qwen14b_numina_oly_l2new_s0)   run_or_submit "srpo_qwen14b_noly_l2n_s0"  cmd_qwen14b_numina_oly_l2new_s0  qwen14b_numina_oly_l2new_s0  4 48:00:00 ;;
    qwen14b_numina_oly_l2new_s420) run_or_submit "srpo_qwen14b_noly_l2n_s420" cmd_qwen14b_numina_oly_l2new_s420 qwen14b_numina_oly_l2new_s420 4 48:00:00 ;;
    qwen14b_numina_oly_rand_s0)    run_or_submit "srpo_rand_qwen14b_noly_s0"    cmd_qwen14b_numina_oly_rand_s0    qwen14b_numina_oly_rand_s0    4 48:00:00 ;;
    qwen14b_numina_oly_rand_s420)  run_or_submit "srpo_rand_qwen14b_noly_s420"  cmd_qwen14b_numina_oly_rand_s420  qwen14b_numina_oly_rand_s420  4 48:00:00 ;;
    olmo7b_numina_oly_l2new_s0)    run_or_submit "srpo_olmo7b_noly_l2n_s0"   cmd_olmo7b_numina_oly_l2new_s0   olmo7b_numina_oly_l2new_s0   2 ;;
    olmo7b_numina_oly_l2new_s420)  run_or_submit "srpo_olmo7b_noly_l2n_s420" cmd_olmo7b_numina_oly_l2new_s420 olmo7b_numina_oly_l2new_s420 2 ;;

    # --- Batch ---
    all_oly)
        run_or_submit "srpo_olmo7b_noly"      cmd_olmo7b_numina_oly      olmo7b_numina_oly
        run_or_submit "srpo_olmo7b_noly_rand" cmd_olmo7b_numina_oly_rand olmo7b_numina_oly_rand
        ;;

    list)
        echo "Primary jobs (NuminaMath Olympiads, seed 42):"
 echo " SRPO: olmo7b_numina_oly qwen14b_numina_oly"
 echo " SRPO-rand: olmo7b_numina_oly_rand qwen14b_numina_oly_rand"
        echo "  Batch:     all_oly  (7B models only)"
        echo ""
        echo "Other datasets:"
 echo " olmo7b_mathlvl5"
        echo ""
        echo "Small models (1 GPU, TP=1) on NuminaMath Olympiads:"
 echo " "
        echo "  (append _rand for SRPO-rand variant)"
        echo ""
        echo "Seed variants (append to any numina_oly job):"
        echo "  _s0  _s420"
        exit 0
        ;;
    *)
        echo "Unknown job: $JOB"
        echo "Run '$0 list' to see all available jobs"
        exit 1
        ;;
esac
