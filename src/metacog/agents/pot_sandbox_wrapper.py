"""PotSandboxWrapper - 轻量级 ToT 沙箱包装器

核心思路
--------
PoT 报错 → 注入反思 prompt → 继续线性执行

不真正实现树状分支，而是：
1. 监控每一步的代码执行结果（observation）
2. 如果检测到报错 → 在下一步之前注入反思 prompt
3. 继续线性执行（消耗 1 步预算）

预算管理
--------
- step_limit: 总算力预算（action count），包括反思步骤
- 反思步骤本身也消耗预算，强制模型高效利用每一步
"""

from __future__ import annotations

import re
from typing import Any


# 通用兜底反思
_REFLECTION_DEFAULT = """\
⚠️ Your previous code failed. STOP and reflect:
1. What went wrong? (identify the exact error)
2. Try a DIFFERENT approach, not the same code with minor fixes
3. Consider: mathematical reasoning first, then minimal code

Do NOT repeat the same approach."""

# 按错误类型分化的反思 prompt
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

# 错误关键词 → 错误类型映射（按优先级排列）
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
    """返回错误类型 key，未匹配返回空字符串"""
    if not observation:
        return ""
    for pattern, error_type in _ERROR_PATTERNS:
        if re.search(pattern, observation, re.IGNORECASE):
            return error_type
    return ""


def _is_error_observation(observation: str) -> bool:
    return bool(_classify_error(observation))


def _get_reflection_prompt(error_type: str, consecutive: int) -> str:
    """根据错误类型和连续次数返回反思 prompt"""
    base = _REFLECTION_BY_TYPE.get(error_type, _REFLECTION_DEFAULT)
    if consecutive >= 2:
        base += "\n\n🚨 You have failed multiple times. Completely change your strategy."
    return base


def create_pot_sandbox_wrapper(base_agent: Any) -> Any:
    """包装 DefaultAgent，添加 PoT 沙箱监控和反思注入
    
    参数
    ----
    base_agent : DefaultAgent
        被包装的原始 Agent
        
    返回
    ----
    base_agent : DefaultAgent
        包装后的 Agent（修改了 step 方法）
    """
    original_step = base_agent.step

    # 统计信息
    stats = {
        "total_steps": 0,
        "error_steps": 0,       # 发生错误的步数
        "reflection_steps": 0,  # 触发反思的次数
        "consecutive_errors": 0,
    }

    def wrapped_step() -> list[dict]:
        """包装后的 step 方法，监控 PoT 报错并注入反思 prompt"""
        stats["total_steps"] += 1

        # 执行原始 step
        result = original_step()

        # 检查 observation 是否包含错误
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
            # 执行成功，重置连续错误计数
            if stats["consecutive_errors"] > 0:
                print(
                    f"  [PoT-Sandbox] ✓ Step {stats['total_steps']} 执行成功，重置错误计数",
                    flush=True,
                )
            stats["consecutive_errors"] = 0

        return result

    def wrapped_run(*args, **kwargs):
        """包装后的 run 方法，重置统计信息"""
        stats["total_steps"] = 0
        stats["error_steps"] = 0
        stats["reflection_steps"] = 0
        stats["consecutive_errors"] = 0
        result = base_agent._original_run(*args, **kwargs)

        # 打印统计
        if stats["reflection_steps"] > 0:
            print(
                f"  [PoT-Sandbox] 📊 共 {stats['total_steps']} 步, "
                f"错误 {stats['error_steps']} 次, "
                f"反思 {stats['reflection_steps']} 次",
                flush=True,
            )
        return result

    # 替换方法
    base_agent._original_run = base_agent.run
    base_agent.step = wrapped_step
    base_agent.run = wrapped_run

    return base_agent


def _get_observation(messages: list[dict]) -> str:
    """从最近的消息中提取 observation 内容

    mini-swe-agent 1.1 格式：
    - messages 列表中，observation 消息的 role 可能是 "user" 或 "tool"
    - 真正的执行结果在消息的 content 中，或 extra.outputs 中
    """
    # 从后往前找，跳过 assistant 消息，找最近的 observation
    for msg in reversed(messages):
        role = msg.get("role", "")
        extra = msg.get("extra", {})

        # mini-swe-agent 1.1：observation 在 extra.outputs 中
        if "outputs" in extra and extra["outputs"]:
            return "\n".join(str(o) for o in extra["outputs"])

        # 兼容其他格式
        if role in ("tool", "observation"):
            return msg.get("content", "")

    return ""


def _inject_reflection(agent: Any, reflection: str) -> None:
    """向 Agent 的消息历史中注入反思 prompt

    使用 agent.model.format_message() 格式化，
    然后通过 agent.add_messages() 插入到对话历史。
    """
    reflection_message = agent.model.format_message(
        role="user",
        content=reflection,
    )
    agent.add_messages(reflection_message)
