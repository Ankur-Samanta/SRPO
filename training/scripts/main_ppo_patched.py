"""Drop-in replacement for `python -m verl.trainer.main_ppo` that ensures
training is imported in the Ray TaskRunner actor process.

Why this exists:
  verl's external_lib import runs inside *worker* processes (actor/critic/ref
  workers) via ModelConfig.__post_init__. It does NOT run inside the
  TaskRunner actor, which is the Ray actor where RayPPOTrainer.fit() executes.
  Any monkey-patch on RayPPOTrainer methods or module-level functions in
  verl.trainer.ppo.ray_trainer must therefore run in the TaskRunner process.

How this works:
  TaskRunner.__init__ is overridden to trigger `import training`
  before RayPPOTrainer is instantiated. That import applies all patches
  (compute_data_metrics → ICS metrics).

Env var note:
  SRPO_SUFFIX_MASK toggles suffix masking in the loss (1=on, 0=off /
  GRPO-parity). On single-node setups (n_gpus_per_node=1, nnodes=1) Ray
  workers are subprocesses of the driver and inherit env vars set before
  python starts. For multi-node runs, forward it via runtime_env on the
  actor remote class (ray.remote(...).options(runtime_env={"env_vars": {...}})).

Usage: replace `python -m verl.trainer.main_ppo` with
       `python -m training.scripts.main_ppo_patched`
in launch scripts. All hydra args pass through unchanged.
"""

import hydra
import ray

from verl.trainer.main_ppo import TaskRunner, run_ppo


class _PatchedTaskRunner(TaskRunner):
    """TaskRunner subclass that imports training at actor startup.

    The import runs once per actor process; it triggers the monkey-patches
    in training/__init__.py so they are live before fit() begins.
    """

    def __init__(self, *args, **kwargs):
        import training  # noqa: F401  -- applies monkey-patches
        super().__init__(*args, **kwargs)


# config_path points to this package's config dir. The launcher typically
# overrides with --config-path/--config-name; this default just ensures
# hydra's decorator has a valid searchpath at import time.
@hydra.main(config_path="../config", config_name="srpo_math500", version_base=None)
def main(config):
    from verl.utils.device import auto_set_device
    from verl.experimental.reward_loop import migrate_legacy_reward_impl

    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)

    # Inject our patched TaskRunner as the remote actor class.
    patched_remote = ray.remote(num_cpus=1)(_PatchedTaskRunner)
    run_ppo(config, task_runner_class=patched_remote)


if __name__ == "__main__":
    main()
