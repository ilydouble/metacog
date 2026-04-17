"""PotSandboxWrapper - Lightweight ToT sandbox wrapper

Core Idea
---------
PoT error → inject reflection prompt → continue linear execution

Instead of implementing true tree branching:
1. Monitor code execution result (observation) at each step
2. If error detected → inject reflection prompt before next step
3. Continue linear execution (consumes 1 step budget)

Budget Management
-----------------
- step_limit: total compute budget (action count), including reflection steps
- Reflection steps themselves consume budget, forcing the model to use each step efficiently
"""

from __future__ import annotations

import re
from typing import Any


# Generic fallback reflection
_REFLECTION_DEFAULT = """\
⚠️ Your previous code failed. STOP and reflect:
1. What went wrong? (identify the exact error)
2. Try a DIFFERENT approach, not the same code with minor fixes
3. Consider: mathematical reasoning first, then minimal code

Do NOT repeat the same approach."""

# Reflection prompts differentiated by error type
_REFLECTION_BY_TYPE = {
    "recursion": """\
⚠️ RecursionError: your code has infinite recursion.
1. Add a base case, or switch to an iterative approach (loop instead of recursion)
2. If using sympy/math library, check for circular dependencies
Do NOT call the same function recursively without a termination condition.""",

    "zero_division": """\
⚠️ ZeroDivisionError: your formula divides by zero for some input.
1. Check your constraints — is the denominator always non-zero?
2. Add a guard: `if denominator != 0` or restructure the formula
3. Re-examine your mathematical derivation for boundary cases.""",

    "timeout": """\
⚠️ TimeoutError: your code is too slow or stuck in an infinite loop.
1. Avoid brute-force enumeration over large ranges — use math to reduce the search space
2. Check for infinite loops (while True without a break condition)
3. Use closed-form formulas or modular arithmetic instead of iteration""",

    "syntax": """\
⚠️ SyntaxError: your code has a syntax mistake.
1. Check indentation, missing colons, unmatched parentheses/brackets
2. Fix the syntax error and keep the same mathematical approach
Do NOT change the algorithm, just fix the code structure.""",

    "name": """\
⚠️ NameError: a variable or function is not defined.
1. Check spelling of variable names
2. Make sure all variables are defined before use
3. Import any missing libraries (e.g. `from sympy import *`)""",

    "type": """\
⚠️ TypeError: wrong data type used in an operation.
1. Check if you are mixing integers and floats, or lists and scalars
2. Use `int()`, `float()`, or `list()` conversions where needed
3. Verify function argument types match what the function expects.""",

    "value": """\
⚠️ ValueError: a value is outside the valid range for this operation.
1. Check domain constraints (e.g. sqrt of negative, log of zero)
2. Verify your input values satisfy the mathematical preconditions
3. Add range checks before calling the function.""",

    "wrong_answer": """\
⚠️ Your code ran without errors but produced the WRONG answer.
1. Your mathematical approach or formula is incorrect — do NOT just re-run
2. Re-derive the solution from scratch: what does the problem actually ask?
3. Verify each step of your math manually before coding""",
}

# Error keywords → Error type mapping (arranged by priority)
_ERROR_PATTERNS: list[tuple[str, str]] = [
    (r"RecursionError|maximum recursion depth", "recursion"),
    (r"ZeroDivisionError",                      "zero_division"),
    (r"TimeoutError",                            "timeout"),
    (r"SyntaxError|IndentationError",            "syntax"),
    (r"NameError",                               "name"),
    (r"TypeError",                               "type"),
    (r"ValueError",                              "value"),
    (r"Traceback \(most recent call last\)|Error:|Exception:", "default"),
]


def _classify_error(observation: str) -> str:
    """Return error type key, return empty string if no match"""
    if not observation:
        return ""
    for pattern, error_type in _ERROR_PATTERNS:
        if re.search(pattern, observation, re.IGNORECASE):
            return error_type
    return ""


def _is_error_observation(observation: str) -> bool:
    return bool(_classify_error(observation))


def _get_reflection_prompt(error_type: str, consecutive: int) -> str:
    """Return reflection prompt based on error type and consecutive count"""
    base = _REFLECTION_BY_TYPE.get(error_type, _REFLECTION_DEFAULT)
    if consecutive >= 2:
        base += "\n\n🚨 You have failed multiple times. Completely change your strategy."
    return base


def create_pot_sandbox_wrapper(base_agent: Any) -> Any:
    """Wrap DefaultAgent, add PoT sandbox monitoring and reflection injection
    
    Parameters
    ----------
    base_agent : DefaultAgent
        The original Agent to be wrapped
        
    Returns
    -------
    base_agent : DefaultAgent
        The wrapped Agent (with modified step method)
    """
    original_step = base_agent.step

    # Statistics
    stats = {
        "total_steps": 0,
        "error_steps": 0,       # Number of steps with errors
        "reflection_steps": 0,  # Number of reflections triggered
        "consecutive_errors": 0,
    }

    def wrapped_step() -> list[dict]:
        """The wrapped step method, monitors PoT errors and injects reflection prompts"""
        stats["total_steps"] += 1

        # Execute original step
        result = original_step()

        # Check if observation contains errors
        observation = _get_observation(base_agent.messages)

        error_type = _classify_error(observation)
        if error_type:
            stats["error_steps"] += 1
            stats["consecutive_errors"] += 1

            print(
                f"  [PoT-Sandbox] ⚠️  Step {stats['total_steps']} 检测到错误: {error_type}"
                f"（连续 {stats['consecutive_errors']} 次）",
                flush=True,
            )

            reflection = _get_reflection_prompt(error_type, stats["consecutive_errors"])
            _inject_reflection(base_agent, reflection)
            stats["reflection_steps"] += 1

            print(
                f"  [PoT-Sandbox] 💡 注入反思 prompt [{error_type}]（第 {stats['reflection_steps']} 次）",
                flush=True,
            )
        else:
            # Execution successful, reset consecutive error count
            if stats["consecutive_errors"] > 0:
                print(
                    f"  [PoT-Sandbox] ✓ Step {stats['total_steps']} 执行成功，重置错误计数",
                    flush=True,
                )
            stats["consecutive_errors"] = 0

        return result

    def wrapped_run(*args, **kwargs):
        """The wrapped run method, resets statistics"""
        stats["total_steps"] = 0
        stats["error_steps"] = 0
        stats["reflection_steps"] = 0
        stats["consecutive_errors"] = 0
        result = base_agent._original_run(*args, **kwargs)
        
        # Print statistics
        if stats["reflection_steps"] > 0:
            print(
                f"  [PoT-Sandbox] 📊 共 {stats['total_steps']} 步, "
                f"错误 {stats['error_steps']} 次, "
                f"反思 {stats['reflection_steps']} 次",
                flush=True,
            )
        return result

    # Replace methods
    base_agent._original_run = base_agent.run
    base_agent.step = wrapped_step
    base_agent.run = wrapped_run

    return base_agent


def _get_observation(messages: list[dict]) -> str:
    """Extract observation content from recent messages

    mini-swe-agent 1.1 format:
    - In the messages list, observation message role may be "user" or "tool"
    - The actual execution result is in the message's content, or in extra.outputs
    """
    # Search from back, skip assistant messages, find the nearest observation
    for msg in reversed(messages):
        role = msg.get("role", "")
        extra = msg.get("extra", {})

        # mini-swe-agent 1.1: observation is in extra.outputs
        if "outputs" in extra and extra["outputs"]:
            return "\n".join(str(o) for o in extra["outputs"])

        # Compatible with other formats
        if role in ("tool", "observation"):
            return msg.get("content", "")

    return ""


def _inject_reflection(agent: Any, reflection: str) -> None:
    """Inject reflection prompt into Agent's message history

    Use agent.model.format_message() to format,
    Then insert into conversation history via agent.add_messages().
    """
    reflection_message = agent.model.format_message(
        role="user",
        content=reflection,
    )
    agent.add_messages(reflection_message)
