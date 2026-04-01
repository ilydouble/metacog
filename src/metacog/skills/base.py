"""StructuredSkill - 比 raw script 更高层的技能抽象。

一个 Skill 封装了：
- 名称、描述、标签（供 SkillRegistry 和 prompt 注入使用）
- 结构化的输入/输出（dict → SkillResult）
- 统一的错误处理

示例::

    class SympySolverSkill(StructuredSkill):
        name = "sympy_solver"
        description = "Solve algebraic equations using SymPy."
        tags = ["algebra", "sympy"]

        def _run(self, inputs: dict) -> SkillResult:
            expr = inputs.get("expression", "")
            # ... 调用 sympy ...
            return SkillResult(success=True, output={"solution": sol})
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillResult:
    """Skill 执行结果。"""
    success: bool
    output: dict = field(default_factory=dict)
    error: str = ""
    metadata: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success


class StructuredSkill(ABC):
    """结构化技能基类。

    子类只需实现 `_run(inputs: dict) -> SkillResult`，
    基类负责输入校验和错误捕获。
    """

    # 子类声明这些类属性
    name: str = ""
    description: str = ""
    tags: list[str] = []

    # 可选：声明必须字段，基类自动校验
    required_inputs: list[str] = []

    def run(self, inputs: dict | None = None) -> SkillResult:
        """执行技能（对外接口）。"""
        inputs = inputs or {}

        # 输入校验
        missing = [k for k in self.required_inputs if k not in inputs]
        if missing:
            return SkillResult(
                success=False,
                error=f"Missing required inputs: {missing}",
            )

        try:
            return self._run(inputs)
        except Exception as exc:
            return SkillResult(success=False, error=str(exc))

    @abstractmethod
    def _run(self, inputs: dict) -> SkillResult:
        """子类实现具体逻辑。"""

    def to_prompt_line(self) -> str:
        """生成注入 prompt 的单行描述。"""
        tag_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"- **{self.name}**{tag_str}: {self.description}"

    def __repr__(self) -> str:
        return f"<Skill {self.name!r}>"


@dataclass
class FileSkill:
    """文件型技能的元数据容器（无需继承 StructuredSkill）。

    技能逻辑存放在独立的 Python 文件里，agent 通过 import 直接调用。
    本类只持有元数据，供 SkillRegistry 生成 prompt 描述和管理使用。

    对应文件约定格式（SKILL_META 字典）::

        SKILL_META = {
            "name": "modular_inverse",
            "description": "Compute modular inverse pow(a, -1, m).",
            "tags": ["number_theory"],
            "module": "skill_modular_inverse",
            "usage": "from skill_modular_inverse import modular_inverse\\nresult = modular_inverse(3, 7)",
        }
    """
    name: str
    description: str
    tags: list[str]
    module: str          # import 时使用的模块名（即文件名去掉 .py）
    usage: str = ""      # 单行或多行使用示例
    file_path: str = ""  # 绝对路径，调试用

    def to_prompt_lines(self) -> str:
        """生成注入 prompt 的多行描述。"""
        tag_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        lines = [f"- **{self.name}**{tag_str}: {self.description}"]
        if self.usage:
            for line in self.usage.strip().splitlines():
                lines.append(f"  {line}")
        return "\n".join(lines)

    # 兼容 SkillRegistry 的统一接口
    def to_prompt_line(self) -> str:
        return self.to_prompt_lines()

