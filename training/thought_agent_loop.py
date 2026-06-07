"""Thought-by-thought agent loop for VERL GRPO.

Generates reasoning chains one thought at a time, delimited by </thought>,
with \\boxed{} as the terminal signal. Each call to the vLLM server generates
a single thought using the stop parameter.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
)
from verl.utils.profiler import simple_timer

from training.prompt_templates import (
    prompt_template_no_examples,
    prompt_template_with_examples,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class _ThoughtChainResult:
    """One trajectory with thought boundary tracking."""

    prompt_ids: list
    response_ids: list
    response_mask: list
    response_logprobs: Optional[list]
    thought_boundaries: list  # list of (start_idx, end_idx) tuples in response_ids
    decoded_thoughts: list  # decoded text of each thought
    found_answer: bool
    num_thoughts: int
    generate_duration: float = 0.0  # wall-clock seconds for the generation loop


class ThoughtAgentLoop(AgentLoopBase):
    """Generates reasoning chains thought-by-thought.

    Each thought is generated as a separate vLLM call with stop=["</thought>"].
    The delimiter tokens are included in output so the model learns to produce them.
    Uses the same request_id across thought steps for prefix caching.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        max_thoughts: int = 10,
        max_tokens_per_thought: int = 512,
        thought_delimiter: str = "</thought>",
        use_examples: bool = True,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, **kwargs)
        self.max_thoughts = max_thoughts
        self.max_tokens_per_thought = max_tokens_per_thought
        self.thought_delimiter = thought_delimiter
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        if use_examples:
            self.template = prompt_template_with_examples()
        else:
            self.template = prompt_template_no_examples()

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Generate a thought-by-thought reasoning chain.

        Args:
            sampling_params: Base sampling params (temperature, top_p, etc.).
            **kwargs: Dataset fields including raw_prompt with the question.

        Returns:
            AgentLoopOutput with accumulated prompt/response ids and logprobs.
        """
        messages = kwargs["raw_prompt"]
        question = self._extract_question(messages)
        result = await self._generate_thought_chain(sampling_params, question)
        return self._chain_result_to_output(result)

    # ------------------------------------------------------------------
    # Thought chain generation
    # ------------------------------------------------------------------

    async def _generate_thought_chain(
        self,
        sampling_params: dict,
        question: str,
    ) -> _ThoughtChainResult:
        """Generate a full thought chain from scratch."""
        prompt_text = self.template.format(question=question)
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(prompt_text, add_special_tokens=True),
        )
        return await self._generate_thoughts_core(list(prompt_ids), sampling_params)

    async def _generate_thoughts_core(
        self,
        initial_prompt_ids: list,
        sampling_params: dict,
        prefix_response_ids: Optional[list] = None,
        prefix_logprobs: Optional[list] = None,
        prefix_boundaries: Optional[list] = None,
        prefix_thoughts: Optional[list] = None,
    ) -> _ThoughtChainResult:
        """Shared thought generation loop, optionally continuing from prefix.

        When prefix data is provided, the loop resumes from the end of the
        prefix.  A new request_id is used so vLLM automatic prefix caching
        handles KV reuse transparently.
        """
        # Initialize accumulators from prefix or empty
        if prefix_response_ids is not None:
            all_response_ids = list(prefix_response_ids)
            all_logprobs = list(prefix_logprobs) if prefix_logprobs else []
            boundaries = list(prefix_boundaries) if prefix_boundaries else []
            thoughts = list(prefix_thoughts) if prefix_thoughts else []
            current_prompt_ids = initial_prompt_ids + all_response_ids
        else:
            all_response_ids = []
            all_logprobs = []
            boundaries = []
            thoughts = []
            current_prompt_ids = list(initial_prompt_ids)

        # Sampling params for thought-by-thought generation
        thought_params = dict(sampling_params)
        thought_params["max_new_tokens"] = self.max_tokens_per_thought
        thought_params.pop("max_tokens", None)
        thought_params["stop"] = [self.thought_delimiter]
        thought_params["include_stop_str_in_output"] = True

        request_id = uuid4().hex
        found_answer = False
        start_step = len(thoughts)
        metrics: dict[str, Any] = {}

        max_model_len = self.config.actor_rollout_ref.rollout.get("max_model_len", 8192)
        with simple_timer("generate_sequences", metrics):
            for step in range(start_step, self.max_thoughts):
                # Stop if the next thought can't fit fully in response_length,
                # or if the running prompt would exceed the model context window
                # (initial_prompt_ids can include long refinement context in critique-grpo).
                if (len(all_response_ids) + self.max_tokens_per_thought > self.response_length
                        or len(current_prompt_ids) + self.max_tokens_per_thought > max_model_len):
                    break

                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=current_prompt_ids,
                    sampling_params=thought_params,
                )

                if not output.token_ids:
                    break

                # Track thought boundary
                start_idx = len(all_response_ids)
                all_response_ids.extend(output.token_ids)
                end_idx = len(all_response_ids)
                boundaries.append((start_idx, end_idx))

                if output.log_probs:
                    all_logprobs.extend(output.log_probs)

                # Decode this thought
                decoded = await self.loop.run_in_executor(
                    None,
                    lambda ids=output.token_ids: self.tokenizer.decode(
                        ids, skip_special_tokens=False
                    ),
                )
                thoughts.append(decoded)

                # Grow prompt for next step (prefix caching reuses KV)
                current_prompt_ids.extend(output.token_ids)

                # Terminal condition
                if "\\boxed{" in decoded:
                    found_answer = True
                    break

                if output.stop_reason == "aborted":
                    break

        # Safety truncation: the pre-generation check above should prevent
        # overshoot, but individual thoughts can vary in length. If the
        # response somehow exceeds response_length, truncate at the last
        # complete thought boundary to avoid training on partial thoughts.
        if len(all_response_ids) > self.response_length:
            logger.warning(
                f"[ThoughtAgent] Response exceeded response_length "
                f"({len(all_response_ids)}/{self.response_length}), "
                f"truncating at last complete thought boundary"
            )
            # Find last boundary that fits entirely
            truncated_boundaries = []
            for start, end in boundaries:
                if end <= self.response_length:
                    truncated_boundaries.append((start, end))
                else:
                    break
            if truncated_boundaries:
                cut = truncated_boundaries[-1][1]
                all_response_ids = all_response_ids[:cut]
                if all_logprobs:
                    all_logprobs = all_logprobs[:cut]
            else:
                all_response_ids = []
                all_logprobs = []
        else:
            truncated_boundaries = list(boundaries)

        return _ThoughtChainResult(
            prompt_ids=initial_prompt_ids,
            response_ids=all_response_ids,
            response_mask=[1] * len(all_response_ids),
            response_logprobs=all_logprobs if all_logprobs else None,
            thought_boundaries=truncated_boundaries,
            decoded_thoughts=thoughts[: len(truncated_boundaries)],
            found_answer=found_answer,
            num_thoughts=len(truncated_boundaries),
            generate_duration=metrics.get("generate_sequences", 0.0),
        )

    # ------------------------------------------------------------------
    # Conversion utilities
    # ------------------------------------------------------------------

    def _chain_result_to_output(
        self, result: _ThoughtChainResult
    ) -> AgentLoopOutput:
        """Convert _ThoughtChainResult to AgentLoopOutput."""
        output = AgentLoopOutput(
            prompt_ids=result.prompt_ids,
            response_ids=result.response_ids,
            response_mask=result.response_mask,
            response_logprobs=result.response_logprobs,
            multi_modal_data={},
            num_turns=result.num_thoughts + 1,
            metrics=AgentLoopMetrics(
                generate_sequences=result.generate_duration,
                tool_calls=0.0,
            ),
        )

        # Build thought_segment_ids from thought_boundaries
        # Each boundary is (start_idx, end_idx) in response_ids
        response_len = len(result.response_ids)
        thought_segment_ids = [0] * self.response_length
        for k, (start, end) in enumerate(result.thought_boundaries):
            for t in range(start, min(end, response_len)):
                thought_segment_ids[t] = k + 1  # 1-indexed

        output.extra_fields.update(
            {
                "turn_scores": [],
                "tool_rewards": [],
                "num_thoughts": result.num_thoughts,
                "found_answer": result.found_answer,
                "thought_segment_ids": thought_segment_ids,
            }
        )
        return output

    def _extract_question(self, messages: list[dict]) -> str:
        """Extract the question text from raw_prompt messages.

        Handles both chat format (list of dicts) and plain string.
        """
        if isinstance(messages, str):
            return messages

        # Look for user message content
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # Handle list-style content (multimodal)
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part["text"]
        # Fallback: use last message content
        if messages:
            content = messages[-1].get("content", "")
            if isinstance(content, str):
                return content
        return ""
