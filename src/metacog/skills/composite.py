"""CompositionalSkill - 将多个 StructuredSkill 组合成流水线。

示例::

    skill = CompositionalSkill(
        name="geometry_pipeline",
        description="Parse geometry problem then solve with sympy.",
        skills=[ParseGeometrySkill(), SympySolverSkill()],
    )
    result = skill.run({"problem": "..."})
    # ParseGeometrySkill 的 output 自动合并到下一个 skill 的 inputs

支持两种组合模式：
- pipeline（默认）：顺序执行，前一个 output merge 到下一个 inputs，任一失败即停止
- parallel：并行执行所有 skill，汇总所有 output（inputs 相同）
"""

from __future__ import annotations

from typing import Literal

from .base import SkillResult, StructuredSkill


class CompositionalSkill(StructuredSkill):
    """流水线 / 并行组合技能。"""

    def __init__(
        self,
        name: str,
        description: str,
        skills: list[StructuredSkill],
        mode: Literal["pipeline", "parallel"] = "pipeline",
        tags: list[str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.skills = skills
        self.mode = mode
        self.tags = tags or []
        self.required_inputs = []

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _run(self, inputs: dict) -> SkillResult:
        if self.mode == "pipeline":
            return self._run_pipeline(inputs)
        return self._run_parallel(inputs)

    def _run_pipeline(self, inputs: dict) -> SkillResult:
        """顺序执行：前一个 output 合并进 inputs。"""
        current_inputs = dict(inputs)
        all_outputs: dict = {}
        steps: list[dict] = []

        for skill in self.skills:
            result = skill.run(current_inputs)
            steps.append({"skill": skill.name, "success": result.success, "error": result.error})

            if not result.success:
                return SkillResult(
                    success=False,
                    error=f"Pipeline failed at '{skill.name}': {result.error}",
                    output=all_outputs,
                    metadata={"steps": steps},
                )
            # 合并 output 到下一步的 inputs
            all_outputs.update(result.output)
            current_inputs = {**current_inputs, **result.output}

        return SkillResult(success=True, output=all_outputs, metadata={"steps": steps})

    def _run_parallel(self, inputs: dict) -> SkillResult:
        """并行执行：所有 skill 用相同 inputs，output 合并（key 冲突取后者）。"""
        all_outputs: dict = {}
        steps: list[dict] = []
        any_failed = False

        for skill in self.skills:
            result = skill.run(inputs)
            steps.append({"skill": skill.name, "success": result.success, "error": result.error})
            if result.success:
                all_outputs.update(result.output)
            else:
                any_failed = True

        return SkillResult(
            success=not any_failed,
            output=all_outputs,
            error="" if not any_failed else "One or more parallel skills failed",
            metadata={"steps": steps},
        )

    def __repr__(self) -> str:
        names = [s.name for s in self.skills]
        return f"<CompositionalSkill {self.name!r} mode={self.mode} skills={names}>"

