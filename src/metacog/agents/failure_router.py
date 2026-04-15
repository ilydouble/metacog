"""FailureRouter - 操作性失误路由过滤器

在 AnalyzerAgent 分析之前拦截"操作性失误"，阻止其写入 memU。

分类规则
--------
OPERATIONAL（操作性失误，直接丢弃）：
  - SyntaxError: 语法错误，写对就好，无需学习
  - ImportError / ModuleNotFoundError: 库没导入
  - IndentationError: 缩进错误
  - 死循环（TimeoutError / RecursionError）
  - 纯粹的代码格式错误

LOGICAL（逻辑/公式错误，允许进入反思流）：
  - 数学公式错误（计算结果不对）
  - 错误的算法选择
  - 逻辑推理错误
  - 边界条件遗漏

设计原则
--------
只看"最后的错误观察"，快速判断，不调用 LLM。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class FailureType(Enum):
    OPERATIONAL = "operational"  # 操作性失误，丢弃
    LOGICAL = "logical"          # 逻辑错误，允许入库
    UNKNOWN = "unknown"          # 无法判断，允许入库（保守策略）


@dataclass
class RoutingResult:
    failure_type: FailureType
    reason: str
    should_store: bool  # 是否允许写入 memU


# 操作性失误的错误模式（直接丢弃）
_OPERATIONAL_PATTERNS = [
    # 语法错误
    (r"SyntaxError", "语法错误（SyntaxError）"),
    (r"IndentationError", "缩进错误（IndentationError）"),
    (r"TabError", "Tab 错误（TabError）"),
    
    # 导入错误
    (r"ImportError", "缺少导入（ImportError）"),
    (r"ModuleNotFoundError", "模块未安装（ModuleNotFoundError）"),
    (r"No module named", "模块未安装"),
    
    # 运行超时/死循环
    (r"TimeoutError", "超时/死循环（TimeoutError）"),
    (r"RecursionError", "递归深度超限（RecursionError）"),
    (r"maximum recursion depth exceeded", "递归死循环"),
    
    # 变量未定义（通常是代码写错，不是逻辑错误）
    (r"NameError: name '.*' is not defined", "变量未定义（NameError）"),
    
    # 文件/IO 错误（与数学无关）
    (r"FileNotFoundError", "文件不存在（FileNotFoundError）"),
    (r"PermissionError", "权限错误（PermissionError）"),
]

# 逻辑错误的明确标志（确认允许入库）
_LOGICAL_PATTERNS = [
    (r"AssertionError", "断言失败（逻辑错误）"),
    (r"Wrong answer", "答案错误（逻辑错误）"),
    (r"incorrect", "计算结果不正确"),
    (r"ValueError: .*math", "数学值域错误"),
    (r"ZeroDivisionError", "除零错误（逻辑/公式错误）"),
]


def route_failure(
    steps: list[Any],  # 可以是 dict 或其他对象
    loop_detected: bool = False,
    extracted_answer: Optional[str] = None,
    expected_answer: Optional[str] = None,
) -> RoutingResult:
    """判断失败类型，决定是否允许写入 memU
    
    参数
    ----
    steps : list[_Step]
        轨迹步骤列表
    loop_detected : bool
        是否检测到死循环（TrajectoryAnalyzer 的结果）
    extracted_answer : str | None
        提取的答案
    expected_answer : str | None
        期望的答案
        
    返回
    ----
    RoutingResult
        路由结果（是否允许入库）
    """
    # 1. 死循环 → 操作性失误，丢弃
    if loop_detected:
        return RoutingResult(
            failure_type=FailureType.OPERATIONAL,
            reason="检测到死循环，属于操作性失误",
            should_store=False,
        )
    
    # 2. 提取所有 observation 内容（取最后 3 步的输出）
    recent_outputs = []
    for step in steps[-3:]:
        # 兼容 dict 和对象两种格式
        output = step.get("output") if isinstance(step, dict) else getattr(step, "output", None)
        if output:
            recent_outputs.append(output)
    
    combined_output = "\n".join(recent_outputs)
    
    # 3. 检查操作性失误模式
    for pattern, reason in _OPERATIONAL_PATTERNS:
        if re.search(pattern, combined_output, re.IGNORECASE):
            return RoutingResult(
                failure_type=FailureType.OPERATIONAL,
                reason=f"操作性失误: {reason}",
                should_store=False,
            )
    
    # 4. 检查明确的逻辑错误模式
    for pattern, reason in _LOGICAL_PATTERNS:
        if re.search(pattern, combined_output, re.IGNORECASE):
            return RoutingResult(
                failure_type=FailureType.LOGICAL,
                reason=f"逻辑错误: {reason}",
                should_store=True,
            )
    
    # 5. 如果有答案但答案错误 → 逻辑错误
    if extracted_answer and expected_answer:
        if str(extracted_answer).strip() != str(expected_answer).strip():
            return RoutingResult(
                failure_type=FailureType.LOGICAL,
                reason="答案错误（逻辑/公式错误）",
                should_store=True,
            )
    
    # 6. 无法判断 → 保守策略，允许入库
    return RoutingResult(
        failure_type=FailureType.UNKNOWN,
        reason="无法判断失败类型，保守允许入库",
        should_store=True,
    )
