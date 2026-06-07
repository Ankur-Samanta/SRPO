#!/bin/bash
# Submit Critique-GRPO baseline training jobs (verl pipeline)
#
# Reference: Zhang et al. "Critique-GRPO: Advancing LLM Reasoning with
#   Natural Language and Numerical Feedback." arXiv:2506.03106.
#
# Usage:
#   ./submit_critique_grpo.sh <job>          # submit to SLURM
#   ./submit_critique_grpo.sh <job> --local   # run locally (set CUDA_VISIBLE_DEVICES first)
#
# Jobs:
#   olmo7b_numina_oly    OLMo 3 7B on NuminaMath Olympiads    (2 GPUs, TP=2)
#   all                  Submit all jobs (2 epochs)
#   *_ep1                1-epoch variant of any job above (eval every 4 steps)
#   all_ep1              Submit all 1-epoch jobs
#
# Environment variables:
#   CRITIQUE_GAMMA=0.1   Shaping function parameter (default 0.1)
#   entropy_coeff        Set via config (critique_grpo_math500.yaml), not an env var

set -e

SCPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH="${SCPO_DIR}/baselines/critique_grpo/scripts/launch_training.sh"
CONDA_PATH="/home/${USER}/miniconda3"
CONDA_ENV="scpo"

mkdir -p "${SCPO_DIR}/batch_scripts/logs"

# ─── Shared overrides for 7B models (TP=2, 2 GPUs) ─────────────────────────
SHARED_OVERRIDES=(
    actor_rollout_ref.model.external_lib=baselines.critique_grpo
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=true
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    trainer.n_gpus_per_node=2
    "trainer.default_local_dir='checkpoints/\${trainer.project_name}/\${trainer.experiment_name}'"
    trainer.total_epochs=2
)

export VERL_LOGGING_LEVEL=INFO

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

# ─── Model shortcuts ──────────────────────────────────────────────────────
QWEN14B="Qwen/Qwen2.5-14B-Instruct"
OLMO7B="allenai/OLMo-3-7B-Instruct"

# 14B models: TP=4, 4 GPUs, lower vLLM memory + micro batches
QWEN14B_OVERRIDES=(
    trainer.n_gpus_per_node=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
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


# ─── Data overrides ───────────────────────────────────────────────────────
NUMINA_OLYMPIADS_DATA=(
    data.train_files=/home/${USER}/data/rlhf/numinamath_olympiads/train.parquet
    data.val_files=/home/${USER}/data/rlhf/numinamath_olympiads/test.parquet
)
# ─── Job definitions ────────────────────────────────────────────────────────


# --- NuminaMath Olympiads ---

# ─── Seed variants ───────────────────────────────────────────────────────────
SEED_0_OVERRIDES=(actor_rollout_ref.actor.fsdp_config.seed=0   actor_rollout_ref.ref.fsdp_config.seed=0   critic.model.fsdp_config.seed=0)
SEED_420_OVERRIDES=(actor_rollout_ref.actor.fsdp_config.seed=420 actor_rollout_ref.ref.fsdp_config.seed=420 critic.model.fsdp_config.seed=420)
cmd_olmo7b_numina_oly_s0()   { _run "$OLMO7B"  numina_oly_olmo7b_cgrpo_s0   "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_olmo7b_numina_oly_s420() { _run "$OLMO7B"  numina_oly_olmo7b_cgrpo_s420 "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_s0()  { _run "$QWEN14B" numina_oly_qwen14b_cgrpo_s0  "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}" trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly_s420(){ _run "$QWEN14B" numina_oly_qwen14b_cgrpo_s420 "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" trainer.test_freq=6 ; }
cmd_qwen14b_numina_oly() { _run "$QWEN14B" numina_oly_qwen14b_cgrpo "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }
cmd_olmo7b_numina_oly()  { _run "$OLMO7B"  numina_oly_olmo7b_cgrpo "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" trainer.resume_mode=disable trainer.test_freq=6 ; }

# ─── 1-epoch variants (eval every 4 steps) ──────────────────────────────────
EP1_OVERRIDES=(
    trainer.total_epochs=1
    trainer.test_freq=4
)


# --- NuminaMath Olympiads ep1 ---
cmd_olmo7b_numina_oly_ep1() { _run "$OLMO7B" numina_oly_olmo7b_cgrpo_ep1 "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${EP1_OVERRIDES[@]}" ; }

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
        submit "$job_name" "bash ${SCPO_DIR}/batch_scripts/submit_critique_grpo.sh ${case_key} --local --training_seed=${TRAINING_SEED}" "$n_gpus" "$time_limit"
    fi
}

case $JOB in
    qwen14b_numina_oly)     run_or_submit "crgrpo_qwen14b_oly"      cmd_qwen14b_numina_oly  qwen14b_numina_oly  4  48:00:00 ;;
    olmo7b_numina_oly)      run_or_submit "crgrpo_olmo7b_oly"       cmd_olmo7b_numina_oly ;;

    # ─── Seed variants (NuminaMath Olympiads) ────────────────────────────────
    olmo7b_numina_oly_s0)    run_or_submit "crgrpo_olmo7b_oly_s0"    cmd_olmo7b_numina_oly_s0 ;;
    olmo7b_numina_oly_s420)  run_or_submit "crgrpo_olmo7b_oly_s420"  cmd_olmo7b_numina_oly_s420 ;;
    qwen14b_numina_oly_s0)   run_or_submit "crgrpo_qwen14b_oly_s0"   cmd_qwen14b_numina_oly_s0   qwen14b_numina_oly_s0   4 48:00:00 ;;
    qwen14b_numina_oly_s420) run_or_submit "crgrpo_qwen14b_oly_s420" cmd_qwen14b_numina_oly_s420 qwen14b_numina_oly_s420 4 48:00:00 ;;
    olmo7b_numina_oly_ep1)  run_or_submit "crgrpo_olmo7b_oly_ep1"  cmd_olmo7b_numina_oly_ep1 ;;
    all)
 for job in olmo7b_numina_oly; do
            run_or_submit "crgrpo_${job}" "cmd_${job}" "${job}"
        done
        ;;
    all_ep1)
 for job in olmo7b_numina_oly_ep1; do
            run_or_submit "crgrpo_${job}" "cmd_${job}" "${job}"
        done
        ;;
    *)
        echo "Unknown job: $JOB"
 echo "Available: olmo7b_numina_oly all"
        echo "         + *_ep1 variants and all_ep1"
        exit 1
        ;;
esac
