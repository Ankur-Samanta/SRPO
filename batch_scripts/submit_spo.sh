#!/bin/bash
# Submit SPO (Segment Policy Optimization) baseline training jobs (verl pipeline)
#
# Reference: SPO (arXiv:2505.23564) https://github.com/AIFrameResearch/SPO
#
# Usage:
#   ./submit_spo.sh <job>          # submit to SLURM
#   ./submit_spo.sh <job> --local   # run locally (set CUDA_VISIBLE_DEVICES first)
#
# Jobs (SPO-Tree, §5):
#   olmo7b_numina_oly_tree, qwen14b_numina_oly_tree  (+ _s0 / _s420 seed variants)
#   all_tree             Submit all tree jobs

set -e

SRPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH_TREE="${SRPO_DIR}/baselines/spo/scripts/launch_tree_training.sh"
CONDA_PATH="/home/${USER}/miniconda3"
CONDA_ENV="srpo"

mkdir -p "${SRPO_DIR}/batch_scripts/logs"

# ─── Shared overrides for 7B models (TP=2, 2 GPUs) ─────────────────────────
SHARED_OVERRIDES=(
    actor_rollout_ref.model.external_lib=baselines.spo
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

# ─── Generic job builder (SPO-Tree launcher / spo_tree_math500 config) ──────
_run_tree() {
    local model_path=$1; shift
    local experiment_name=$1; shift
    local extra_overrides=("$@")

    local cmd_args=(
        "${SHARED_OVERRIDES[@]}"
        actor_rollout_ref.model.path="$model_path"
        trainer.experiment_name="$experiment_name"
        "${extra_overrides[@]}"
    )

    bash "$LAUNCH_TREE" "${cmd_args[@]}"
}

# ─── Model shortcuts ──────────────────────────────────────────────────────
OLMO7B="allenai/OLMo-3-7B-Instruct"
QWEN14B="Qwen/Qwen2.5-14B-Instruct"

QWEN14B_OVERRIDES=(
    trainer.n_gpus_per_node=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
)

SMALL_MODEL_OVERRIDES=(
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    trainer.n_gpus_per_node=1
)


OLMO_OVERRIDES=(
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
)

# ─── Data overrides ───────────────────────────────────────────────────────
NUMINA_OLYMPIADS_DATA=(
    data.train_files=/home/${USER}/data/rlhf/numinamath_olympiads/train.parquet
    data.val_files=/home/${USER}/data/rlhf/numinamath_olympiads/test.parquet
)
# ─── Seed variants (shared by SPO-Tree jobs) ────────────────────────────────
SEED_0_OVERRIDES=(actor_rollout_ref.actor.fsdp_config.seed=0   actor_rollout_ref.ref.fsdp_config.seed=0   critic.model.fsdp_config.seed=0)
SEED_420_OVERRIDES=(actor_rollout_ref.actor.fsdp_config.seed=420 actor_rollout_ref.ref.fsdp_config.seed=420 critic.model.fsdp_config.seed=420)

# ─── Job definitions (SPO-Tree, §5) ────────────────────────────────────────
# Experiment names get _spo_tree suffix so WandB keeps tree runs separate
# from chain runs.


# --- NuminaMath Olympiads (tree) ---
cmd_olmo7b_numina_oly_tree()   { _run_tree "$OLMO7B"  numina_oly_olmo7b_spo_tree  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" ; }

# Seed variants (tree)
cmd_olmo7b_numina_oly_tree_s0()    { _run_tree "$OLMO7B"  numina_oly_olmo7b_spo_tree_s0    "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}" ; }
cmd_olmo7b_numina_oly_tree_s420()  { _run_tree "$OLMO7B"  numina_oly_olmo7b_spo_tree_s420  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" ; }

# qwen14b SPO-tree: default seed (42 via SEED_OVERRIDES) + s0 + s420
cmd_qwen14b_numina_oly_tree()      { _run_tree "$QWEN14B" numina_oly_qwen14b_spo_tree      "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" ; }
cmd_qwen14b_numina_oly_tree_s0()   { _run_tree "$QWEN14B" numina_oly_qwen14b_spo_tree_s0   "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}" ; }
cmd_qwen14b_numina_oly_tree_s420() { _run_tree "$QWEN14B" numina_oly_qwen14b_spo_tree_s420 "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" ; }

# ─── SLURM submission helper ────────────────────────────────────────────────

submit() {
    local job_name=$1
    local cmd=$2
    local n_gpus=${3:-2}
    local time_limit=${4:-24:00:00}

    sbatch --partition="${SLURM_PARTITION:-q1}" \
           --nodes=1 \
           --gpus-per-node=${n_gpus} \
           --cpus-per-gpu=10 \
           --time=${time_limit} \
           --job-name="${job_name}" \
           --output="batch_scripts/logs/${job_name}.out" \
           --error="batch_scripts/logs/${job_name}.err" \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${SRPO_DIR} && ${cmd}'"

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
        submit "$job_name" "bash ${SRPO_DIR}/batch_scripts/submit_spo.sh ${case_key} --local --training_seed=${TRAINING_SEED}" "$n_gpus" "$time_limit"
    fi
}

case $JOB in
    # ─── SPO-Tree ────────────────────────────────────────────────────────────
    olmo7b_numina_oly_tree)  run_or_submit "spotree_olmo7b_oly"  cmd_olmo7b_numina_oly_tree ;;

    # --- Tree seed variants ---
    olmo7b_numina_oly_tree_s0)    run_or_submit "spotree_olmo7b_oly_s0"    cmd_olmo7b_numina_oly_tree_s0 ;;
    olmo7b_numina_oly_tree_s420)  run_or_submit "spotree_olmo7b_oly_s420"  cmd_olmo7b_numina_oly_tree_s420 ;;
    qwen14b_numina_oly_tree)      run_or_submit "spotree_qwen14b_oly"      cmd_qwen14b_numina_oly_tree      qwen14b_numina_oly_tree      4 48:00:00 ;;
    qwen14b_numina_oly_tree_s0)   run_or_submit "spotree_qwen14b_oly_s0"   cmd_qwen14b_numina_oly_tree_s0   qwen14b_numina_oly_tree_s0   4 48:00:00 ;;
    qwen14b_numina_oly_tree_s420) run_or_submit "spotree_qwen14b_oly_s420" cmd_qwen14b_numina_oly_tree_s420 qwen14b_numina_oly_tree_s420 4 48:00:00 ;;

    all_tree)
        for job in olmo7b_numina_oly_tree; do
            run_or_submit "spotree_${job}" "cmd_${job}" "${job}"
        done
        ;;
    *)
        echo "Unknown job: $JOB"
        echo "Available: olmo7b_numina_oly_tree, qwen14b_numina_oly_tree (+ _s0/_s420), all_tree"
        exit 1
        ;;
esac
