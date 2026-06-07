#!/bin/bash
# Full-epoch per-token gradient dump for the LCB-medium OLMo-7B ep1 training.
#
# Unlike submit_grad_dump_l2n_ep1.sh (single-step probe of the *trained* ep1
# policy), this one reproduces the original ep1 training itself with the
# per-token loss dump and branch dump turned on. Every update step's gradient
# lands on disk — so the downstream analysis can see how per-thought credit
# concentration evolves across the actual training arc that produced the
# lcb_medium_olmo7b_srpo_l2new_ep1 checkpoint.
#
# Mirrors cmd_olmo7b_lcb_medium_l2new_ep1 from submit_srpo.sh:
#   SHARED + OLMO_OVERRIDES + LCB_MEDIUM_DATA + EP1_OVERRIDES + resume=disable
# with seed 42 (default). Differences from that command:
#   - test_freq=999, save_freq=999  (skip val + ckpt writes; we only want grad dumps)
#   - dump env vars + lifted caps
#   - distinct project/experiment name so dumps don't collide
#
# Prerequisite: training/srpo_loss.py patched with the env-gated
# tensor dump (writes torch.save when SRPO_PER_TOKEN_DUMP_DIR is set).
#
# Outputs (all under unique paths):
#   logs/srpo_per_token/grad_dump_l2n_ep1_full/loss_pid*_call*.pt
#   logs/srpo_localizations/grad_dump_l2n_ep1_full/branches_pid*.jsonl
#   wandb run under project=srpo_grad_dump
#
# Usage:
#   ./batch_scripts/submit_grad_dump_l2n_ep1_full.sh             # SLURM
#   ./batch_scripts/submit_grad_dump_l2n_ep1_full.sh --local     # interactive

set -e

# ─── Paths (all unique to this run) ─────────────────────────────────────────
SCPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_PATH="/home/${USER}/miniconda3"
CONDA_ENV="scpo"
JOB_NAME="srpo_graddump_l2n_ep1_full"
EXP_NAME="grad_dump_l2n_ep1_full"
PROJECT="srpo_grad_dump"

TRAIN_CKPT_DIR="${SCPO_DIR}/checkpoints/${PROJECT}/${EXP_NAME}"
DUMP_DIR="${SCPO_DIR}/logs/srpo_per_token/${EXP_NAME}"
BRANCH_DIR="${SCPO_DIR}/logs/srpo_localizations/${EXP_NAME}"

mkdir -p "${SCPO_DIR}/batch_scripts/logs"
mkdir -p "${DUMP_DIR}"

# ─── Hydra overrides ────────────────────────────────────────────────────────
OLMO7B="allenai/OLMo-3-7B-Instruct"

OVERRIDES=(
    # Shared SRPO setup (mirror SHARED_OVERRIDES from submit_srpo.sh)
    actor_rollout_ref.model.external_lib=training
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=true
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    trainer.n_gpus_per_node=2
    actor_rollout_ref.rollout.agent.default_agent_loop=srpo_agent
    actor_rollout_ref.actor.policy_loss.loss_mode=srpo
    # Base OLMo (NO LoRA init — we want the original training path from scratch)
    actor_rollout_ref.model.path="${OLMO7B}"
    # OLMo micro-batch sizing (mirror OLMO_OVERRIDES)
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
    # LCB-medium data
    "data.train_files=/home/${USER}/data/rlhf/livecodebench_medium/train.parquet"
    "data.val_files=/home/${USER}/data/rlhf/livecodebench_medium/test.parquet"
    # Seed 42 (matches original ep1)
    actor_rollout_ref.actor.fsdp_config.seed=42
    actor_rollout_ref.ref.fsdp_config.seed=42
    critic.model.fsdp_config.seed=42
    # ── Run isolation ─────────────────────────────────────────────────
    "trainer.project_name=${PROJECT}"
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.default_local_dir='checkpoints/\${trainer.project_name}/\${trainer.experiment_name}'"
    trainer.resume_mode=disable           # start fresh
    trainer.total_epochs=1                # one full LCB-medium epoch (matches ep1 baseline)
    trainer.test_freq=999                 # skip val (we only need grads)
    trainer.save_freq=999                 # skip checkpoint writes
    # train_batch_size left at config default (32 → ~11 update steps for LCB-medium ep1)
)

# ─── Env vars ───────────────────────────────────────────────────────────────
export VERL_LOGGING_LEVEL=INFO
export ICS_L2_PROMPT=l2new
export SRPO_PER_TOKEN_DUMP_DIR="${DUMP_DIR}"
export SRPO_PER_TOKEN_DUMP_MAX=20000      # ~256 microbatches × ~11 steps × ppo_epochs=1 ≈ 2.8k; 20k = generous headroom
export SCGRPO_BRANCH_DUMP_DIR="${BRANCH_DIR}"
export SCGRPO_BRANCH_DUMP_EVERY=1
export SCGRPO_BRANCH_DUMP_MAX=5000        # ~32 prompts × ~11 steps ≈ 352; 5k = generous headroom

# ─── Submit or run locally ──────────────────────────────────────────────────
MODE="${1:-slurm}"

if [[ "$MODE" == "--local" ]]; then
    echo "=== Running locally: ${JOB_NAME} ==="
    echo "    train ckpt:  ${TRAIN_CKPT_DIR} (created fresh)"
    echo "    loss dump:   ${DUMP_DIR}"
    echo "    branch dump: ${BRANCH_DIR}"

    # Inline launch_srpo_training.sh's env — skip the destructive save_experiment.py post-step.
    set -x
    ulimit -n 65535
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    cd "${SCPO_DIR}"

    export VLLM_USE_V1=1
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    export PYTHONPATH="${SCPO_DIR}:${PYTHONPATH}"
    export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-$$} % 1000))

    python3 -c "import training" || { echo "Failed to import training"; exit 1; }

    CONFIG_PATH="${SCPO_DIR}/training/config"
    python3 -m training.scripts.main_ppo_wrapper \
        --config-path="${CONFIG_PATH}" \
        --config-name='srpo_math500' \
        "${OVERRIDES[@]}"
else
    EXPORTS="ALL,VERL_LOGGING_LEVEL=INFO,ICS_L2_PROMPT=l2new"
    EXPORTS="${EXPORTS},SRPO_PER_TOKEN_DUMP_DIR=${DUMP_DIR},SRPO_PER_TOKEN_DUMP_MAX=20000"
    EXPORTS="${EXPORTS},SCGRPO_BRANCH_DUMP_DIR=${BRANCH_DIR},SCGRPO_BRANCH_DUMP_EVERY=1,SCGRPO_BRANCH_DUMP_MAX=5000"

    sbatch --partition=q1 \
           --nodes=1 \
           --gpus-per-node=2 \
           --cpus-per-gpu=10 \
           --time=24:00:00 \
           --job-name="${JOB_NAME}" \
           --output="${SCPO_DIR}/batch_scripts/logs/${JOB_NAME}.out" \
           --error="${SCPO_DIR}/batch_scripts/logs/${JOB_NAME}.err" \
           --export="${EXPORTS}" \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${SCPO_DIR} && bash ${SCPO_DIR}/batch_scripts/submit_grad_dump_l2n_ep1_full.sh --local'"

    echo "Submitted: ${JOB_NAME}"
    echo "  out: batch_scripts/logs/${JOB_NAME}.out"
    echo "  err: batch_scripts/logs/${JOB_NAME}.err"
fi
