"""Thought-by-thought prompt templates for VERL agent loop.

Templates provide the initial prompt structure with {question} placeholder.
Each thought is delimited by </thought>, final answer uses \\boxed{}.
"""


def prompt_template_with_examples() -> str:
    """Prompt with 2 in-context examples (book arrangement, rectangle).

    Returns:
        Template string with {question} placeholder
    """
    return """You are solving a problem step-by-step.

Instructions:
1. State your next reasoning step (one observation, calculation, or deduction)
2. End each thought with </thought>
3. Continue until you reach the final answer, then write it in \\boxed{{answer}} format

Examples:

Q: In how many ways can 5 distinct books be arranged on a shelf if 2 specific books must not be adjacent?
Total arrangements without restrictions is 5! = 120</thought>
I need to subtract arrangements where the 2 specific books ARE adjacent</thought>
If I treat the 2 books as a single unit, I have 4 units to arrange: 4! = 24 ways</thought>
The 2 books within their unit can be arranged in 2! = 2 ways</thought>
So arrangements with the books adjacent = 24 x 2 = 48</thought>
Therefore, arrangements where they are NOT adjacent = 120 - 48 = \\boxed{{72}}</thought>

Q: A rectangle has area 48 and perimeter 28. What is the length of its diagonal?
Let length = l and width = w. From the area: lw = 48</thought>
From the perimeter: 2l + 2w = 28, so l + w = 14</thought>
From l + w = 14, we get w = 14 - l. Substituting into lw = 48: l(14 - l) = 48</thought>
Expanding: 14l - l^2 = 48, so l^2 - 14l + 48 = 0. Factoring: (l - 6)(l - 8) = 0</thought>
So l = 8 and w = 6 (or vice versa). Using the Pythagorean theorem: d^2 = 8^2 + 6^2 = 64 + 36 = 100</thought>
Therefore d = 10, so the answer is \\boxed{{10}}</thought>

Q: {question}
"""


def prompt_template_no_examples() -> str:
    """Prompt with format guidance but no specific examples.

    Returns:
        Template string with {question} placeholder
    """
    return """You are solving a problem by producing one reasoning step at a time.

Do not try to solve the entire problem at once. Given the previously taken steps, think about what the single next step should be, then articulate it clearly and conclude just that step with </thought>.

Each step should be a complete, self-contained thought — one observation, calculation, or deduction that:
- Makes forward progress toward the solution
- Contains substantive reasoning (not filler like "let me think" or restating the problem)
- Coheres logically with the previous steps

When your next step arrives at the final answer, include \\boxed{{answer}} and end with </thought>.

Q: {question}
"""
