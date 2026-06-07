#!/bin/bash
# Submit thought-level GRPO training jobs (verl pipeline)
#
# Usage:
#   ./submit_thought_grpo.sh <job>          # submit to SLURM
#   ./submit_thought_grpo.sh <job> --local   # run locally (set CUDA_VISIBLE_DEVICES first)
#
# Jobs (token-mean baseline; verl default loss_agg_mode):
#
#   all                Submit all baseline jobs

set -e

SCPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH="${SCPO_DIR}/training/scripts/launch_training.sh"
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
)

# OLMo 7B: lower micro batches to avoid OOM on deep chains
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



# ─── Job definitions (thought_grpo) ───────────────────────────────────────




cmd_olmo7b_mathlvl5() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/math_level5/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/math_level5/test.parquet \
        trainer.experiment_name=mathlvl5_olmo7b_thought \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}"
}

# --- MATH Level 5 + format reward ---





# --- LiveCodeBench Medium (eval once per epoch: test_freq=11) ---

cmd_olmo7b_lcb_medium() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/livecodebench_medium/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/livecodebench_medium/test.parquet \
        trainer.experiment_name=lcb_medium_olmo7b_thought \
        trainer.test_freq=1 \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}"
}


cmd_qwen14b_lcb_medium() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-14B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/livecodebench_medium/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/livecodebench_medium/test.parquet \
        trainer.experiment_name=lcb_medium_qwen14b_thought \
        trainer.test_freq=11 \
        "${SHARED_OVERRIDES[@]}" \
        "${QWEN14B_OVERRIDES[@]}"
}

cmd_qwen14b_lcb_hard() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-14B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/livecodebench_hard/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/livecodebench_hard/test.parquet \
        trainer.experiment_name=lcb_hard_qwen14b_thought \
        trainer.test_freq=9 \
        "${SHARED_OVERRIDES[@]}" \
        "${QWEN14B_OVERRIDES[@]}"
}

# --- LiveCodeBench Hard (eval once per epoch: test_freq=9) ---

cmd_olmo7b_lcb_hard() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/livecodebench_hard/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/livecodebench_hard/test.parquet \
        trainer.experiment_name=lcb_hard_olmo7b_thought \
        trainer.test_freq=1 \
        trainer.resume_mode=disable \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}"
}

# olmo7b LCB seed variants (seed already set via --training_seed → SHARED_OVERRIDES)
cmd_olmo7b_lcb_medium_s0() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/livecodebench_medium/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/livecodebench_medium/test.parquet \
        trainer.experiment_name=lcb_medium_olmo7b_thought_s0 \
        trainer.test_freq=1 \
        trainer.resume_mode=disable \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}"
}

cmd_olmo7b_lcb_medium_s420() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/livecodebench_medium/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/livecodebench_medium/test.parquet \
        trainer.experiment_name=lcb_medium_olmo7b_thought_s420 \
        trainer.test_freq=1 \
        trainer.resume_mode=disable \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}"
}

# olmo7b LCB-medium TGRPO ep1 (1-epoch, test_freq=1) — matches the schedule
# of the srpo *_ep1 jobs for fair side-by-side comparison.
cmd_olmo7b_lcb_medium_ep1() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/livecodebench_medium/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/livecodebench_medium/test.parquet \
        trainer.experiment_name=lcb_medium_olmo7b_thought_ep1 \
        trainer.total_epochs=1 \
        trainer.test_freq=1 \
        trainer.resume_mode=disable \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}"
}

cmd_olmo7b_lcb_medium_ep1_s0() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/livecodebench_medium/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/livecodebench_medium/test.parquet \
        trainer.experiment_name=lcb_medium_olmo7b_thought_ep1_s0 \
        trainer.total_epochs=1 \
        trainer.test_freq=1 \
        trainer.resume_mode=disable \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}" \
        "${SEED_0_OVERRIDES[@]}"
}

cmd_olmo7b_lcb_medium_ep1_s420() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/livecodebench_medium/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/livecodebench_medium/test.parquet \
        trainer.experiment_name=lcb_medium_olmo7b_thought_ep1_s420 \
        trainer.total_epochs=1 \
        trainer.test_freq=1 \
        trainer.resume_mode=disable \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}" \
        "${SEED_420_OVERRIDES[@]}"
}

cmd_olmo7b_lcb_hard_s0() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/livecodebench_hard/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/livecodebench_hard/test.parquet \
        trainer.experiment_name=lcb_hard_olmo7b_thought_s0 \
        trainer.test_freq=1 \
        trainer.resume_mode=disable \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}"
}

cmd_olmo7b_lcb_hard_s420() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/livecodebench_hard/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/livecodebench_hard/test.parquet \
        trainer.experiment_name=lcb_hard_olmo7b_thought_s420 \
        trainer.test_freq=1 \
        trainer.resume_mode=disable \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}"
}



QWEN14B_OVERRIDES=(
    trainer.n_gpus_per_node=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
)

cmd_qwen14b_numina_oly() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-14B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/numinamath_olympiads/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/numinamath_olympiads/test.parquet \
        trainer.experiment_name=numina_oly_qwen14b_thought \
        "${SHARED_OVERRIDES[@]}" \
        "${QWEN14B_OVERRIDES[@]}"
}













cmd_olmo7b_csqa() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/csqa/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/csqa/test.parquet \
        trainer.experiment_name=csqa_olmo7b_thought \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}"
}


cmd_olmo7b_numina_oly() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/numinamath_olympiads/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/numinamath_olympiads/test.parquet \
        trainer.experiment_name=numina_oly_olmo7b_thought \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}"
}

# tgrpo_sm: vanilla GRPO loss with seq-mean-token-mean aggregation (vs verl's default token-mean).
# Apples-to-apples with srpo/srpo (sequence-mean) on aggregation.
SM_OVERRIDES=(
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    trainer.project_name=thought_grpo_sm
)

cmd_olmo7b_numina_oly_sm() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/numinamath_olympiads/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/numinamath_olympiads/test.parquet \
        trainer.experiment_name=numina_oly_olmo7b_thought_sm \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}" \
        "${SM_OVERRIDES[@]}"
}

cmd_qwen14b_numina_oly_sm() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-14B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/numinamath_olympiads/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/numinamath_olympiads/test.parquet \
        trainer.experiment_name=numina_oly_qwen14b_thought_sm \
        "${SHARED_OVERRIDES[@]}" \
        "${QWEN14B_OVERRIDES[@]}" \
        "${SM_OVERRIDES[@]}"
}

SEED_0_OVERRIDES=(actor_rollout_ref.actor.fsdp_config.seed=0   actor_rollout_ref.ref.fsdp_config.seed=0   critic.model.fsdp_config.seed=0)
SEED_420_OVERRIDES=(actor_rollout_ref.actor.fsdp_config.seed=420 actor_rollout_ref.ref.fsdp_config.seed=420 critic.model.fsdp_config.seed=420)



cmd_olmo7b_numina_oly_s0() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/numinamath_olympiads/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/numinamath_olympiads/test.parquet \
        trainer.experiment_name=numina_oly_olmo7b_thought_s0 \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}" \
        "${SEED_0_OVERRIDES[@]}"
}

cmd_olmo7b_numina_oly_s420() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/numinamath_olympiads/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/numinamath_olympiads/test.parquet \
        trainer.experiment_name=numina_oly_olmo7b_thought_s420 \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}" \
        "${SEED_420_OVERRIDES[@]}"
}

cmd_qwen14b_numina_oly_s0() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-14B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/numinamath_olympiads/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/numinamath_olympiads/test.parquet \
        trainer.experiment_name=numina_oly_qwen14b_thought_s0 \
        "${SHARED_OVERRIDES[@]}" \
        "${QWEN14B_OVERRIDES[@]}" \
        "${SEED_0_OVERRIDES[@]}"
}

cmd_qwen14b_numina_oly_s420() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-14B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/numinamath_olympiads/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/numinamath_olympiads/test.parquet \
        trainer.experiment_name=numina_oly_qwen14b_thought_s420 \
        "${SHARED_OVERRIDES[@]}" \
        "${QWEN14B_OVERRIDES[@]}" \
        "${SEED_420_OVERRIDES[@]}"
}







# ─── Polaris d2 (all models) ─────────────────────────────────────────────



# ─── Polaris d3 ──────────────────────────────────────────────────────────



# ─── Polaris d4 ──────────────────────────────────────────────────────────



# ─── AceReason-Math ──────────────────────────────────────────────────────


cmd_olmo7b_acereason() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/acereason_math/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/acereason_math/test.parquet \
        trainer.experiment_name=acereason_olmo7b_thought \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}"
}

# ─── Omni-MATH ───────────────────────────────────────────────────────────



# ─── IMO-AnswerBench ─────────────────────────────────────────────────────






# ─── Job definitions (seq-mean-token-mean baseline) ───────────────────────



# ─── Job definitions (larger splits) ─────────────────────────────────────────



# ─── ep1 / ep1_kl ablations (1 epoch, test_freq=4) ──────────────────────────

EP1_OVERRIDES=(trainer.total_epochs=1 trainer.test_freq=4)
ADAPTIVE_KL_OVERRIDES=(
    algorithm.use_kl_in_reward=true
    algorithm.kl_ctrl.type=adaptive
    algorithm.kl_ctrl.kl_coef=0.01
    algorithm.kl_ctrl.target_kl=0.01
    algorithm.kl_ctrl.horizon=500
)



cmd_olmo7b_numina_oly_ep1() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/numinamath_olympiads/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/numinamath_olympiads/test.parquet \
        trainer.experiment_name=numina_oly_olmo7b_thought_ep1 \
        "${EP1_OVERRIDES[@]}" \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}"
}

# numinamath_olympiads (400) + sciknoweval L3 disjoint (400) mixed, 1-epoch TGRPO
cmd_olmo7b_numina_oly_sk400() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/numina_oly_plus_sciknow400/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/numina_oly_plus_sciknow400/test.parquet \
        trainer.experiment_name=numina_oly_sk400_olmo7b_thought \
        "${EP1_OVERRIDES[@]}" \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}"
}

cmd_qwen14b_numina_oly_sk400() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-14B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/numina_oly_plus_sciknow400/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/numina_oly_plus_sciknow400/test.parquet \
        trainer.experiment_name=numina_oly_sk400_qwen14b_thought \
        "${EP1_OVERRIDES[@]}" \
        "${SHARED_OVERRIDES[@]}" \
        "${QWEN14B_OVERRIDES[@]}"
}

cmd_olmo7b_numina_oly_ep1_kl() {
    bash "$LAUNCH" \
        actor_rollout_ref.model.path=allenai/OLMo-3-7B-Instruct \
        data.train_files=/home/${USER}/data/rlhf/numinamath_olympiads/train.parquet \
        data.val_files=/home/${USER}/data/rlhf/numinamath_olympiads/test.parquet \
        trainer.experiment_name=numina_oly_olmo7b_thought_ep1_kl \
        "${EP1_OVERRIDES[@]}" \
        "${ADAPTIVE_KL_OVERRIDES[@]}" \
        "${SHARED_OVERRIDES[@]}" \
        "${OLMO_OVERRIDES[@]}"
}





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
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${SCPO_DIR} && ${cmd}'"

    echo "Submitted: ${job_name}"
}

# ─── Entrypoint ─────────────────────────────────────────────────────────────

# Parse --training_seed from any position in args
TRAINING_SEED=42
POSITIONAL=()
for arg in "$@"; do
    case $arg in
        --training_seed=*) TRAINING_SEED="${arg#*=}" ;;
        --training_seed)   shift_next=true ;;
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

SEED_OVERRIDES=(
    actor_rollout_ref.actor.fsdp_config.seed="${TRAINING_SEED}"
    actor_rollout_ref.ref.fsdp_config.seed="${TRAINING_SEED}"
    critic.model.fsdp_config.seed="${TRAINING_SEED}"
)
SHARED_OVERRIDES+=("${SEED_OVERRIDES[@]}")

JOB=${POSITIONAL[0]:?Usage: $0 <job> [--local] [--training_seed N]  (see header for job list)}
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
        submit "$job_name" "bash ${SCPO_DIR}/batch_scripts/submit_thought_grpo.sh ${case_key} --local --training_seed=${TRAINING_SEED}" "$n_gpus" "$time_limit"
    fi
}

case $JOB in
    # --- thought_grpo loss ---
    # --- MATH Level 5 + format reward ---
    # --- AIME thought GRPO ---
    # --- GPQA ---
    # --- CSQA ---
    # --- MathQA ---
    # --- NuminaMath Olympiads ---
    qwen14b_numina_oly)       run_or_submit "tgrpo_qwen14b_noly"       cmd_qwen14b_numina_oly      qwen14b_numina_oly      4  48:00:00 ;;
    olmo7b_numina_oly_ep1)    run_or_submit "tgrpo_olmo7b_noly_ep1"    cmd_olmo7b_numina_oly_ep1 ;;
    olmo7b_numina_oly_ep1_kl) run_or_submit "tgrpo_olmo7b_noly_ep1kl"  cmd_olmo7b_numina_oly_ep1_kl ;;
    olmo7b_numina_oly_sk400)  run_or_submit "tgrpo_olmo7b_noly_sk4"   cmd_olmo7b_numina_oly_sk400  olmo7b_numina_oly_sk400  2 ;;
    qwen14b_numina_oly_sk400) run_or_submit "tgrpo_qwen14b_noly_sk4" cmd_qwen14b_numina_oly_sk400 qwen14b_numina_oly_sk400 4 48:00:00 ;;
    # tgrpo_sm: vanilla GRPO + seq-mean-token-mean aggregation (default seed)
    olmo7b_numina_oly_sm)  run_or_submit "tgrpo_sm_olmo7b_noly"  cmd_olmo7b_numina_oly_sm ;;
    qwen14b_numina_oly_sm) run_or_submit "tgrpo_sm_qwen14b_noly" cmd_qwen14b_numina_oly_sm qwen14b_numina_oly_sm 4 48:00:00 ;;
    # --- NuminaMath AoPS ---
    # --- NuminaMath AMC ---
    # --- OLMo 7B ---
    olmo7b_mathlvl5)   run_or_submit "tgrpo_olmo7b_mlvl5" cmd_olmo7b_mathlvl5 ;;
    olmo7b_csqa)       run_or_submit "tgrpo_olmo7b_csqa"  cmd_olmo7b_csqa ;;
    # --- LiveCodeBench ---
    olmo7b_lcb_medium)  run_or_submit "tgrpo_olmo7b_lcbm"  cmd_olmo7b_lcb_medium ;;
    qwen14b_lcb_medium) run_or_submit "tgrpo_qwen14b_lcbm" cmd_qwen14b_lcb_medium qwen14b_lcb_medium  4 48:00:00 ;;
    qwen14b_lcb_hard)   run_or_submit "tgrpo_qwen14b_lcbh" cmd_qwen14b_lcb_hard   qwen14b_lcb_hard    4 48:00:00 ;;
    olmo7b_lcb_hard)    run_or_submit "tgrpo_olmo7b_lcbh"  cmd_olmo7b_lcb_hard ;;

    # olmo7b LCB seed variants (s0, s420)
    olmo7b_lcb_medium_s0)   run_or_submit "tgrpo_olmo7b_lcbm_s0"   cmd_olmo7b_lcb_medium_s0   olmo7b_lcb_medium_s0   2 48:00:00 ;;
    olmo7b_lcb_medium_s420) run_or_submit "tgrpo_olmo7b_lcbm_s420" cmd_olmo7b_lcb_medium_s420 olmo7b_lcb_medium_s420 2 48:00:00 ;;
    olmo7b_lcb_medium_ep1)      run_or_submit "tgrpo_olmo7b_lcbm_ep1"      cmd_olmo7b_lcb_medium_ep1      olmo7b_lcb_medium_ep1      2 24:00:00 ;;
    olmo7b_lcb_medium_ep1_s0)   run_or_submit "tgrpo_olmo7b_lcbm_ep1_s0"   cmd_olmo7b_lcb_medium_ep1_s0   olmo7b_lcb_medium_ep1_s0   2 24:00:00 ;;
    olmo7b_lcb_medium_ep1_s420) run_or_submit "tgrpo_olmo7b_lcbm_ep1_s420" cmd_olmo7b_lcb_medium_ep1_s420 olmo7b_lcb_medium_ep1_s420 2 24:00:00 ;;
    olmo7b_lcb_hard_s0)     run_or_submit "tgrpo_olmo7b_lcbh_s0"   cmd_olmo7b_lcb_hard_s0     olmo7b_lcb_hard_s0     2 48:00:00 ;;
    olmo7b_lcb_hard_s420)   run_or_submit "tgrpo_olmo7b_lcbh_s420" cmd_olmo7b_lcb_hard_s420   olmo7b_lcb_hard_s420   2 48:00:00 ;;
    olmo7b_numina_oly) run_or_submit "tgrpo_olmo7b_noly"  cmd_olmo7b_numina_oly ;;

    # ─── Seed variants (NuminaMath Olympiads) ────────────────────────────────
    olmo7b_numina_oly_s0)    run_or_submit "tgrpo_olmo7b_noly_s0"    cmd_olmo7b_numina_oly_s0 ;;
    olmo7b_numina_oly_s420)  run_or_submit "tgrpo_olmo7b_noly_s420"  cmd_olmo7b_numina_oly_s420 ;;
    qwen14b_numina_oly_s0)   run_or_submit "tgrpo_qwen14b_noly_s0"   cmd_qwen14b_numina_oly_s0   qwen14b_numina_oly_s0   4 48:00:00 ;;
    qwen14b_numina_oly_s420) run_or_submit "tgrpo_qwen14b_noly_s420" cmd_qwen14b_numina_oly_s420 qwen14b_numina_oly_s420 4 48:00:00 ;;
    # --- Polaris d2 ---
    # --- Polaris d3 ---
    # --- Polaris d4 ---
    # --- AceReason-Math ---
    olmo7b_acereason)      run_or_submit "tgrpo_olmo7b_acer"    cmd_olmo7b_acereason ;;
    # --- Omni-MATH ---
    # --- IMO-AnswerBench ---
    # --- AIME baseline ---
    # --- AoPS baseline ---
    # --- OpenMath2 baseline ---
    # --- larger splits ---
    # --- seq-mean-token-mean baseline ---
    # --- submit all ---
    all)
        run_or_submit "tgrpo_olmo7b_mlvl5" cmd_olmo7b_mathlvl5 olmo7b_mathlvl5
        ;;
    *)
        echo "Unknown job: $JOB"
        echo "See script header for available jobs."
        exit 1
        ;;
esac
