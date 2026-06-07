"""SDPO training entrypoint.

Patches verl's TaskRunner.run() to import baselines.sdpo inside
the Ray actor process where update_policy actually executes.
The patches (actor + trainer) are applied at import time via __init__.py.
"""

import baselines.sdpo  # noqa: F401

from verl.trainer.main_ppo import TaskRunner, main

_original_run = TaskRunner.run


def _patched_run(self, config):
    import baselines.sdpo  # noqa: F401
    return _original_run(self, config)


TaskRunner.run = _patched_run

if __name__ == "__main__":
    main()
