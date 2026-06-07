"""
ThoughtEvaluator: Rubric-based reasoning coherence scorer for Thought-MDP.

Evaluates individual thoughts on coherence dimensions (NOT correctness).
Designed for use as a process reward signal in RL training.

Key Design Principles:
- Self-evaluation: Model evaluates its own thoughts
- Flexible model interface: Pass in a lambda/callable for generation
- Rubric-based: Multiple dimensions contribute to composite score
- Coherence over correctness: We want meaningful reasoning steps, regardless of factual accuracy
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThoughtEvaluationResult:
    """Result of evaluating a single thought."""
    thought: str
    dimension_scores: Dict[str, float]  # dimension_name -> score (0-1)
    composite_score: float              # Weighted sum, normalized to 0-1
    raw_evaluation: str                 # Model's raw evaluation output
    metadata: Dict[str, Any] = field(default_factory=dict)


# Default rubric dimensions and weights (simplified 3-dimension rubric)
DEFAULT_RUBRIC_DIMENSIONS = {
    "forward_progress": {
        "weight": 0.4,
        "description": "Does this thought advance problem-solving? Does it add new information or insight?",
        "criteria": "1=repetition/restating/circular, 3=minor progress, 5=significant advancement"
    },
    "substantiveness": {
        "weight": 0.4,
        "description": "Is this actual reasoning or filler? Real analysis vs meta-commentary?",
        "criteria": "1=pure filler ('let me think'), 3=mixed, 5=concrete substantive reasoning"
    },
    "coherence": {
        "weight": 0.2,
        "description": "Does this thought logically connect to previous reasoning without contradictions?",
        "criteria": "1=contradicts or disconnected, 3=loosely connected, 5=perfectly coherent"
    },
}


def build_evaluation_prompt(
    question: str,
    thought_prefix: List[str],
    new_thought: str,
    rubric_dimensions: Dict[str, Dict],
    thought_delimiter: str = "</thought>",
) -> str:
    """
    Build the self-evaluation prompt.

    Args:
        question: The original problem/question
        thought_prefix: List of previous thoughts in the chain
        new_thought: The thought to evaluate
        rubric_dimensions: Dict of dimension configs
        thought_delimiter: Delimiter between thoughts

    Returns:
        Formatted evaluation prompt
    """
    # Format the thought history
    if thought_prefix:
        history = ""
        for i, t in enumerate(thought_prefix, 1):
            history += f"Thought {i}: {t}{thought_delimiter}\n"
    else:
        history = "(This is the first thought - no previous reasoning)\n"

    # Build rubric section
    rubric_text = ""
    for dim_name, dim_config in rubric_dimensions.items():
        rubric_text += f"- **{dim_name.replace('_', ' ').title()}**: {dim_config['description']}\n"
        rubric_text += f"  Rating scale: {dim_config['criteria']}\n\n"

    prompt = f'''You are a HARSH critic evaluating reasoning quality. Be critical and demanding.
Do NOT solve the problem. Only rate the thought below.

## Problem Context
{question}

## Previous Steps
{history}

## Thought to Evaluate
{new_thought}

## Scoring Rubric (be strict!)

**Forward Progress** (score one to five): Does this ACTUALLY advance the solution?
- Score one: Restating, repeating, going in circles, "let's start over"
- Score two: Barely any new information
- Score three: Some progress
- Score four: Good progress with new insight
- Score five: Major breakthrough

**Substantiveness** (score one to five): Is this REAL reasoning or filler?
- Score one: Pure filler like "Let me think", "I need to reconsider", restating the problem
- Score two: Mostly filler with tiny bit of content
- Score three: Mixed
- Score four: Mostly substantive
- Score five: Dense analysis, specific calculations, concrete reasoning

**Coherence** (score one to five): Logical connection to previous steps?
- Score one: Contradicts or disconnected
- Score three: Loosely connected
- Score five: Perfect continuation

## Your Response
Reply with EXACTLY three lines in this format (replace N with the digit corresponding to your score):
forward_progress: N
substantiveness: N
coherence: N
'''
    return prompt


def parse_evaluation_response(
    response: str,
    rubric_dimensions: Dict[str, Dict],
) -> ThoughtEvaluationResult:
    """
    Parse the model's evaluation response into structured scores.

    Args:
        response: Raw model output
        rubric_dimensions: Dict of dimension configs (for weights)

    Returns:
        ThoughtEvaluationResult with parsed scores
    """
    dimension_scores = {}

    # Parse individual dimension scores - use the LAST match to avoid
    # picking up echoed rubric examples from the prompt. The model's
    # actual scores appear at the end of its response.
    # Handles: "Forward Progress: 4", "Forward Progress: \boxed{4}",
    #          "Forward Progress: [4]", "Forward Progress: **4**"
    for dim_name in rubric_dimensions.keys():
        pattern = rf'{dim_name.replace("_", "[ _]?")}[:\s]+(?:\\boxed\{{|\[|\*\*)?(\d)(?:\}}|\]|\*\*)?'
        matches = list(re.finditer(pattern, response, re.IGNORECASE))
        if matches:
            score = int(matches[-1].group(1))  # last match = model's actual score
            dimension_scores[dim_name] = min(max(score, 1), 5) / 5.0  # Normalize to 0-1
        else:
            dimension_scores[dim_name] = 0.2  # Low default — unparseable means bad

    # Always compute composite from dimension scores (more reliable than parsing boxed)
    # Dimension scores are already normalized to 0-1
    composite_score = sum(
        dimension_scores.get(dim, 0.5) * config["weight"]
        for dim, config in rubric_dimensions.items()
    )

    return ThoughtEvaluationResult(
        thought="",  # Will be filled by caller
        dimension_scores=dimension_scores,
        composite_score=composite_score,
        raw_evaluation=response,
    )


class ThoughtEvaluator:
    """
    Evaluates individual thoughts on reasoning coherence using a rubric.

    Designed for self-evaluation where the same model (or a frozen copy)
    evaluates its own reasoning steps. Provides process reward signals
    for RL training.

    Rubric Dimensions (Simplified 3-dimension):
    - Forward Progress (0.4): Does this thought advance problem-solving? Penalize repetition/filler.
    - Substantiveness (0.4): Is this actual reasoning or meta-commentary/filler?
    - Coherence (0.2): Does it logically connect without contradictions?

    Usage:
        # With a lambda wrapping your model's generate function
        evaluator = ThoughtEvaluator()

        generate_fn = lambda prompt: model_manager.generate([prompt])[0]
        # OR
        generate_fn = lambda prompt: policy_model.generate(tokenizer(prompt))

        result = evaluator.evaluate(
            generate_fn=generate_fn,
            question="What is 2+2?",
            thought_prefix=["I need to add the numbers"],
            new_thought="2 + 2 = 4, so the answer is 4",
        )

        print(f"Composite score: {result.composite_score}")  # 0.0 to 1.0
    """

    def __init__(
        self,
        rubric_dimensions: Optional[Dict[str, Dict]] = None,
        thought_delimiter: str = "</thought>",
        max_eval_tokens: int = 512,
        temperature: float = 0.3,  # Low temp for consistent evaluation
    ):
        """
        Initialize the ThoughtEvaluator.

        Args:
            rubric_dimensions: Custom rubric dimensions and weights.
                               If None, uses DEFAULT_RUBRIC_DIMENSIONS.
            thought_delimiter: Delimiter between thoughts (default: "</thought>")
            max_eval_tokens: Max tokens for evaluation response
            temperature: Sampling temperature for evaluation (lower = more consistent)
        """
        self.rubric_dimensions = rubric_dimensions or DEFAULT_RUBRIC_DIMENSIONS
        self.thought_delimiter = thought_delimiter
        self.max_eval_tokens = max_eval_tokens
        self.temperature = temperature

        # Validate weights sum to ~1
        total_weight = sum(d["weight"] for d in self.rubric_dimensions.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Rubric weights sum to {total_weight}, not 1.0")

    def evaluate(
        self,
        generate_fn: Callable[[str], str],
        question: str,
        thought_prefix: List[str],
        new_thought: str,
    ) -> ThoughtEvaluationResult:
        """
        Evaluate a single thought using the rubric.

        Args:
            generate_fn: Callable that takes a prompt string and returns
                        a completion string. This is your model's generate
                        function wrapped as a lambda.
                        Example: lambda p: model.generate([p])[0]
            question: The original problem/question
            thought_prefix: List of previous thoughts (without delimiters)
            new_thought: The thought to evaluate (without delimiter)

        Returns:
            ThoughtEvaluationResult with dimension scores and composite
        """
        # Build evaluation prompt
        eval_prompt = build_evaluation_prompt(
            question=question,
            thought_prefix=thought_prefix,
            new_thought=new_thought,
            rubric_dimensions=self.rubric_dimensions,
            thought_delimiter=self.thought_delimiter,
        )

        # Get evaluation from model
        try:
            raw_response = generate_fn(eval_prompt)
        except Exception as e:
            logger.error(f"Evaluation generation failed: {e}")
            # Return neutral scores on failure
            return ThoughtEvaluationResult(
                thought=new_thought,
                dimension_scores={d: 0.5 for d in self.rubric_dimensions},
                composite_score=0.5,
                raw_evaluation=f"ERROR: {e}",
                metadata={"error": str(e)},
            )

        # Parse response into scores
        result = parse_evaluation_response(raw_response, self.rubric_dimensions)
        result.thought = new_thought

        return result

    def batch_evaluate(
        self,
        batch_generate_fn: Callable[[List[str]], List[str]],
        questions: List[str],
        thought_prefixes: List[List[str]],
        new_thoughts: List[str],
    ) -> List[ThoughtEvaluationResult]:
        """
        Evaluate multiple thoughts in a single batch call.

        Builds all evaluation prompts, sends them as one batch, and parses
        all responses. This is dramatically faster than sequential evaluate()
        calls when using a vLLM backend that can parallelize internally.

        Args:
            batch_generate_fn: Callable that takes a list of prompts and
                              returns a list of completions.
                              Example: lambda ps: manager.generate(prompts=ps, max_tokens=512)
            questions: List of questions (one per thought to evaluate)
            thought_prefixes: List of thought prefix lists (one per thought)
            new_thoughts: List of thoughts to evaluate

        Returns:
            List of ThoughtEvaluationResult, one per input thought
        """
        assert len(questions) == len(thought_prefixes) == len(new_thoughts), (
            f"Mismatched lengths: {len(questions)}, {len(thought_prefixes)}, {len(new_thoughts)}"
        )

        if not questions:
            return []

        # Build all evaluation prompts
        all_prompts = []
        for question, prefix, thought in zip(questions, thought_prefixes, new_thoughts):
            prompt = build_evaluation_prompt(
                question=question,
                thought_prefix=prefix,
                new_thought=thought,
                rubric_dimensions=self.rubric_dimensions,
                thought_delimiter=self.thought_delimiter,
            )
            all_prompts.append(prompt)

        # Single batch call
        logger.info(f"Batch evaluating {len(all_prompts)} thoughts")
        try:
            all_responses = batch_generate_fn(all_prompts)
        except Exception as e:
            logger.error(f"Batch evaluation generation failed: {e}")
            return [
                ThoughtEvaluationResult(
                    thought=t,
                    dimension_scores={d: 0.5 for d in self.rubric_dimensions},
                    composite_score=0.5,
                    raw_evaluation=f"ERROR: {e}",
                    metadata={"error": str(e)},
                )
                for t in new_thoughts
            ]

        # Parse each response
        results = []
        for thought, response in zip(new_thoughts, all_responses):
            result = parse_evaluation_response(response, self.rubric_dimensions)
            result.thought = thought
            results.append(result)

        return results

    def evaluate_trajectory(
        self,
        generate_fn: Callable[[str], str],
        question: str,
        thoughts: List[str],
    ) -> List[ThoughtEvaluationResult]:
        """
        Evaluate all thoughts in a trajectory.

        Args:
            generate_fn: Model generation callable
            question: The original problem
            thoughts: List of thoughts to evaluate

        Returns:
            List of ThoughtEvaluationResult, one per thought
        """
        results = []

        for i, thought in enumerate(thoughts):
            prefix = thoughts[:i]  # All thoughts before this one
            result = self.evaluate(
                generate_fn=generate_fn,
                question=question,
                thought_prefix=prefix,
                new_thought=thought,
            )
            result.metadata["thought_index"] = i
            result.metadata["is_first"] = (i == 0)
            result.metadata["is_last"] = (i == len(thoughts) - 1)
            results.append(result)

        return results

    def get_trajectory_score(
        self,
        generate_fn: Callable[[str], str],
        question: str,
        thoughts: List[str],
        aggregation: str = "mean",
    ) -> float:
        """
        Get a single score for an entire trajectory.

        Args:
            generate_fn: Model generation callable
            question: The original problem
            thoughts: List of thoughts
            aggregation: How to aggregate scores ("mean", "min", "last")

        Returns:
            Aggregated score for the trajectory (0-1)
        """
        results = self.evaluate_trajectory(generate_fn, question, thoughts)
        scores = [r.composite_score for r in results]

        if not scores:
            return 0.5

        if aggregation == "mean":
            return sum(scores) / len(scores)
        elif aggregation == "min":
            return min(scores)
        elif aggregation == "last":
            return scores[-1]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def get_dimension_weights(self) -> Dict[str, float]:
        """
        Get the current rubric dimension weights.

        Returns:
            Dict mapping dimension names to weights
        """
        return {
            name: config["weight"]
            for name, config in self.rubric_dimensions.items()
        }

    def get_dimension_descriptions(self) -> Dict[str, str]:
        """
        Get the descriptions for each rubric dimension.

        Returns:
            Dict mapping dimension names to descriptions
        """
        return {
            name: config["description"]
            for name, config in self.rubric_dimensions.items()
        }
