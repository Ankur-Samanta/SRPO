# Vendored TREE modules

These modules are vendored verbatim from the sibling **TREE** project so that
SRPO is self-contained and does not require a `../TREE` checkout to run.

| File | Provides | Used by |
| --- | --- | --- |
| `dataset_loaders.py` | `load_dataset_by_name`, `normalize_answer`, `get_dataset_info` | `evaluation/data_loader.py`, `thought_ics/dataset_utils.py`, `training/scripts/prepare_datasets.py` |
| `third_party_localization.py` | `ThirdPartyModelManager`, `call_3p_error_localization` | `thought_ics/model_init.py`, `thought_ics/correction.py` |

All import sites resolve to this directory; it is the single source of truth for
these modules in SRPO.

Both modules depend only on the standard library plus packages already pinned in
`requirements.txt` (e.g. `datasets`). They do **not** import any other TREE
files. The test-only modules `tree_of_thought.py` / `iterative_self_correction.py`
are intentionally **not** vendored — they pull in the heavier `SIERA.src.models`
package and are not part of the SRPO runtime path.
