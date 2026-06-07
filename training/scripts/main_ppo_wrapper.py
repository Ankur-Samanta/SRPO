"""Wrapper around verl.trainer.main_ppo that imports training first.

Importing training in the driver process applies the
compute_data_metrics monkey-patch (see training/__init__.py),
so ICS aggregations (ics/triggered_rate, iter_accuracy, localization,
training/prompts/*) reach wandb. Without this, only the standard verl
metrics get logged because main_ppo never imports the external lib in
its own process — workers do, but compute_data_metrics runs in the driver.
"""

import training  # noqa: F401 -- triggers monkey-patches in driver
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
