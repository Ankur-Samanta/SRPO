#!/usr/bin/env python3
"""Post-training experiment saver.

Collects Hydra config, wandb metrics, per-step training log, and the final
checkpoint into a single experiments/ directory for reproducibility.

Usage (called automatically by launch_training.sh):
    python3 save_experiment.py [hydra CLI overrides...]
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WANDB_DIR = PROJECT_ROOT / "wandb"
def _get_ray_logs_dirs() -> list[Path]:
    """Get all possible Ray log directories (handles RAY_TMPDIR)."""
    candidates = []
    # Check RAY_TMPDIR (may nest as RAY_TMPDIR/ray/session_latest/logs/)
    ray_tmpdir = os.environ.get("RAY_TMPDIR")
    if ray_tmpdir:
        candidates.append(Path(ray_tmpdir) / "ray" / "session_latest" / "logs")
        candidates.append(Path(ray_tmpdir) / "session_latest" / "logs")
    # Default location
    candidates.append(Path("/tmp/ray/session_latest/logs"))
    return [d for d in candidates if d.exists()]


def find_latest_hydra_dir() -> Path | None:
    """Find the most recently modified Hydra output directory."""
    hydra_dirs = list(OUTPUTS_DIR.glob("*/*/.hydra"))
    if not hydra_dirs:
        return None
    return max(hydra_dirs, key=lambda p: p.stat().st_mtime)


def find_latest_wandb_dir() -> Path | None:
    """Find the most recently modified wandb run directory."""
    run_dirs = list(WANDB_DIR.glob("run-*/files"))
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def find_latest_worker_log() -> Path | None:
    """Find the most recently modified Ray worker .out log."""
    logs = []
    for d in _get_ray_logs_dirs():
        logs.extend(d.glob("worker-*-01000000-*.out"))
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def load_resolved_config(hydra_dir: Path) -> dict | None:
    """Load the fully resolved Hydra config as a dict."""
    config_path = hydra_dir / "config.yaml"
    if not config_path.exists():
        return None
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        # Fallback: just read as raw YAML
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)


def find_checkpoint_dir(config: dict) -> Path | None:
    """Find the final checkpoint directory from the resolved config."""
    try:
        local_dir = config["trainer"]["default_local_dir"]
    except (KeyError, TypeError):
        return None

    ckpt_base = Path(local_dir)
    if not ckpt_base.is_absolute():
        ckpt_base = PROJECT_ROOT / ckpt_base

    if not ckpt_base.exists():
        return None

    # Find highest global_step_* subdirectory
    step_dirs = [d for d in ckpt_base.iterdir() if d.is_dir() and d.name.startswith("global_step_")]
    if not step_dirs:
        return None

    def step_number(d: Path) -> int:
        try:
            return int(d.name.split("global_step_")[1])
        except (IndexError, ValueError):
            return -1

    return max(step_dirs, key=step_number)


def parse_worker_log(log_path: Path) -> list[dict]:
    """Parse step: lines from the Ray worker log into a list of dicts."""
    entries = []
    # Key metrics we want to extract
    keys_of_interest = {
        "training/global_step": "step",
        "training/epoch": "epoch",
        "critic/score/mean": "score",
        "actor/pg_loss": "pg_loss",
        "actor/grad_norm": "grad_norm",
        "actor/entropy": "entropy",
        "actor/ppo_kl": "ppo_kl",
        "val-core/math500/acc/mean@1": "val_acc",
        "response_length/mean": "response_length",
        "num_turns/mean": "num_turns",
        "perf/throughput": "throughput",
        "timing_s/step": "step_time_s",
    }

    with open(log_path) as f:
        for line in f:
            if not line.startswith("step:"):
                continue
            # Format: step:N - key:value - key:value ...
            entry = {}
            parts = line.strip().split(" - ")
            for part in parts:
                # Split on first colon only
                idx = part.find(":")
                if idx == -1:
                    continue
                key = part[:idx].strip()
                val = part[idx + 1:].strip()

                if key == "step":
                    entry["log_step"] = int(val)
                elif key in keys_of_interest:
                    field = keys_of_interest[key]
                    try:
                        entry[field] = float(val) if "." in val or "e" in val.lower() else int(val)
                    except ValueError:
                        entry[field] = val

            if entry:
                entries.append(entry)

    return entries


def get_experiment_name(config: dict, cli_overrides: list[str]) -> str:
    """Derive experiment name from config or CLI overrides."""
    # Check CLI overrides first
    for arg in cli_overrides:
        if arg.startswith("trainer.experiment_name="):
            return arg.split("=", 1)[1]

    # Fall back to config
    try:
        return config["trainer"]["experiment_name"]
    except (KeyError, TypeError):
        return "experiment"


def main():
    cli_overrides = sys.argv[1:]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print("Saving experiment artifacts...")
    print(f"{'='*60}")

    # 1. Find and load Hydra config
    hydra_dir = find_latest_hydra_dir()
    config = None
    if hydra_dir:
        config = load_resolved_config(hydra_dir)
        print(f"  Found Hydra config: {hydra_dir}")
    else:
        print("  WARNING: No Hydra output directory found")

    # Determine experiment name
    exp_name = get_experiment_name(config, cli_overrides) if config else "experiment"
    exp_dir = EXPERIMENTS_DIR / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Experiment dir: {exp_dir}")

    # 2. Copy config files
    if hydra_dir:
        config_src = hydra_dir / "config.yaml"
        overrides_src = hydra_dir / "overrides.yaml"
        if config_src.exists():
            shutil.copy2(config_src, exp_dir / "config.yaml")
            print("  Saved config.yaml")
        if overrides_src.exists():
            shutil.copy2(overrides_src, exp_dir / "overrides.yaml")
            print("  Saved overrides.yaml")

    # 3. Copy wandb summary metrics
    wandb_dir = find_latest_wandb_dir()
    if wandb_dir:
        summary_src = wandb_dir / "wandb-summary.json"
        if summary_src.exists():
            shutil.copy2(summary_src, exp_dir / "metrics.json")
            print(f"  Saved metrics.json (from {wandb_dir.parent.name})")
        else:
            print("  WARNING: wandb-summary.json not found")
    else:
        print("  WARNING: No wandb run directory found")

    # 4. Extract per-step training log
    worker_log = find_latest_worker_log()
    if worker_log:
        entries = parse_worker_log(worker_log)
        if entries:
            log_path = exp_dir / "training_log.jsonl"
            with open(log_path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
            print(f"  Saved training_log.jsonl ({len(entries)} steps)")
        else:
            print("  WARNING: No step: lines found in worker log")
    else:
        print("  WARNING: No Ray worker log found")

    # 5. Move final checkpoint
    if config:
        ckpt_dir = find_checkpoint_dir(config)
        if ckpt_dir:
            actor_src = ckpt_dir / "actor"
            if actor_src.exists():
                dest = exp_dir / "checkpoint"
                shutil.move(str(actor_src), str(dest))
                print(f"  Moved checkpoint: {ckpt_dir.name}/actor/ -> checkpoint/")
            else:
                # Move the whole step directory if no actor/ subdirectory
                dest = exp_dir / "checkpoint"
                shutil.move(str(ckpt_dir), str(dest))
                print(f"  Moved checkpoint: {ckpt_dir.name}/ -> checkpoint/")
        else:
            print("  WARNING: No checkpoint found")
    else:
        print("  WARNING: Skipping checkpoint (no config)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Experiment saved to: {exp_dir}")
    saved = [f.name for f in exp_dir.iterdir()]
    print(f"  Contents: {', '.join(sorted(saved))}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
