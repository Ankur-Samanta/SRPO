"""
Iterative Self-Correction (ICS) module for SCPO.

Ported from TREE's iterative self-correction pipeline. Provides:
- Chain generation and correction loop
- Error localization (batch, incremental, majority vote)
- Solution verification
- Chain caching
- Metrics computation

Submodules:
    thought_ics.correction   - Main ICS pipeline and all correction functions
    thought_ics.chain_cache  - Chain caching (stdlib-only)
    thought_ics.compute_metrics - Experiment metrics (stdlib-only)
    thought_ics.dataset_utils - normalize_answer + load_dataset_by_name
    thought_ics.model_init   - Model initialization (requires SIERA)

Imports are lazy to avoid pulling in heavy dependencies (torch, evaluation)
when only lightweight submodules are needed. Use direct submodule imports:
    from thought_ics.correction import iterative_self_correction
    from thought_ics.chain_cache import save_initial_chains
"""

__all__ = [
    # correction.py
    "extract_boxed_answer",
    "generate_full_chain",
    "verify_single_step",
    "verify_solution_correctness",
    "identify_error_step",
    "identify_error_step_incremental",
    "identify_error_step_with_mv",
    "generate_from_prefix",
    "iterative_self_correction",
    "run_iterative_correction_with_cached_chain",
    # chain_cache.py
    "save_initial_chains",
    "load_initial_chains",
    "list_cached_chains",
    # compute_metrics.py
    "compute_metrics",
    # dataset_utils.py
    "normalize_answer",
    # model_init.py
    "initialize_model",
    "initialize_model_3p",
]


def __getattr__(name):
    """Lazy imports to avoid pulling in torch/evaluation for lightweight usage."""
    if name in (
        "extract_boxed_answer", "generate_full_chain", "verify_single_step",
        "verify_solution_correctness", "identify_error_step",
        "identify_error_step_incremental", "identify_error_step_with_mv",
        "generate_from_prefix", "iterative_self_correction",
        "run_iterative_correction_with_cached_chain",
    ):
        from thought_ics import correction
        return getattr(correction, name)

    if name in ("save_initial_chains", "load_initial_chains", "list_cached_chains"):
        from thought_ics import chain_cache
        return getattr(chain_cache, name)

    if name == "compute_metrics":
        from thought_ics import compute_metrics as _cm
        return _cm.compute_metrics

    if name == "normalize_answer":
        from thought_ics import dataset_utils
        return dataset_utils.normalize_answer

    if name in ("initialize_model", "initialize_model_3p"):
        from thought_ics import model_init
        return getattr(model_init, name)

    raise AttributeError(f"module 'thought_ics' has no attribute {name!r}")
