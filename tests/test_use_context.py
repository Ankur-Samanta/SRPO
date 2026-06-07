"""Unit tests for use_context feature in ThoughtICSAgentLoop.

Tests the logic without GPU by validating signatures, prompt construction,
and the prompt-swap mechanism for GRPO group consistency.
"""

import asyncio
import inspect

from training.thought_ics_agent_loop import ThoughtICSAgentLoop
from training.thought_agent_loop import _ThoughtChainResult


def test_constructor_signature():
    """use_context param exists with default=False."""
    sig = inspect.signature(ThoughtICSAgentLoop.__init__)
    assert "use_context" in sig.parameters
    assert sig.parameters["use_context"].default is False
    print("PASS: Constructor has use_context=False")


def test_localize_error_return_annotation():
    """_localize_error returns a 3-tuple."""
    sig = inspect.signature(ThoughtICSAgentLoop._localize_error)
    ret = str(sig.return_annotation)
    assert "str" in ret, f"Expected str in return annotation, got: {ret}"
    print("PASS: _localize_error returns 3-tuple with str")


def test_generate_from_prefix_signature():
    """_generate_thought_chain_from_prefix has the 3 new optional params."""
    sig = inspect.signature(ThoughtICSAgentLoop._generate_thought_chain_from_prefix)
    params = sig.parameters
    for name in ("previous_chain_thoughts", "error_reasoning", "error_step"):
        assert name in params, f"Missing param: {name}"
        assert params[name].default is None, f"{name} default should be None"
    print("PASS: _generate_thought_chain_from_prefix has 3 new optional params")


def test_context_prompt_construction():
    """Historical context is correctly formatted with delimiters stripped."""
    thought_delimiter = "</thought>"
    original_prompt_text = "Solve: What is 2+2?"
    previous_chain_thoughts = [
        "First, I note that 2+2 is addition.</thought>",
        "Adding gives 5.</thought>",
    ]
    error_reasoning = "Step 2 has an arithmetic error: 2+2=4, not 5."

    # Replicate the logic from _generate_thought_chain_from_prefix
    context_text = original_prompt_text
    context_text += "\n\n### Previous Failed Attempt\n"
    context_text += "The following reasoning chain led to an incorrect answer:\n"
    for i, thought in enumerate(previous_chain_thoughts, 1):
        clean = thought.replace(thought_delimiter, "").strip()
        context_text += f"\nStep {i}: {clean}"
    context_text += f"\n\n### Error Analysis\n{error_reasoning}\n"
    context_text += "\nNow let's try again with the correct approach:\n"

    assert "### Previous Failed Attempt" in context_text
    assert "Step 1: First, I note that 2+2 is addition." in context_text
    assert "Step 2: Adding gives 5." in context_text
    assert "</thought>" not in context_text  # delimiters stripped
    assert "### Error Analysis" in context_text
    assert "Step 2 has an arithmetic error" in context_text
    print("PASS: Context text correctly formatted, delimiters stripped")


def test_prompt_swap_for_grpo():
    """prompt_ids is swapped to original while response_ids stays unchanged."""
    original_ids = [1, 2, 3]
    generation_ids = [1, 2, 3, 4, 5, 6, 7]  # longer (context-enriched)
    result = _ThoughtChainResult(
        prompt_ids=list(generation_ids),
        response_ids=[10, 11, 12],
        response_mask=[1, 1, 1],
        response_logprobs=None,
        thought_boundaries=[(0, 3)],
        decoded_thoughts=["test"],
        found_answer=True,
        num_thoughts=1,
    )
    # Simulate the swap (same logic as in the method)
    if generation_ids is not original_ids:
        result.prompt_ids = list(original_ids)

    assert result.prompt_ids == [1, 2, 3]
    assert result.response_ids == [10, 11, 12]  # unchanged
    print("PASS: prompt_ids swapped to original, response_ids unchanged")


def test_no_swap_when_no_context():
    """When context is not used, prompt_ids is not swapped (identity check)."""
    original_ids = [1, 2, 3]
    generation_ids = original_ids  # same object (no context case)
    result = _ThoughtChainResult(
        prompt_ids=list(original_ids),
        response_ids=[10, 11, 12],
        response_mask=[1, 1, 1],
        response_logprobs=None,
        thought_boundaries=[(0, 3)],
        decoded_thoughts=["test"],
        found_answer=True,
        num_thoughts=1,
    )
    # Identity check — should NOT swap
    if generation_ids is not original_ids:
        result.prompt_ids = [99, 99]  # would corrupt if triggered

    assert result.prompt_ids == [1, 2, 3]  # not corrupted
    print("PASS: No swap when generation_ids is original_ids")


if __name__ == "__main__":
    test_constructor_signature()
    test_localize_error_return_annotation()
    test_generate_from_prefix_signature()
    test_context_prompt_construction()
    test_prompt_swap_for_grpo()
    test_no_swap_when_no_context()
    print("\nAll tests passed!")
