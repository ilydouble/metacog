"""SkillAgent - 订阅 SUCCESS_ANALYSIS，积累成功模式，达到阈值后生成新 skill 文件。

流程
----
1. 收到 SUCCESS_ANALYSIS 事件，按 tags 归组写入 pattern_buffer
2. 同组累计达到 threshold（默认 3）次
3. 调 LLM 生成完整 Python skill 文件（含 SKILL_META + 函数）
4. ast.parse() 语法验证 + 检查 SKILL_META 存在
5. 写入 skills_dir/skill_<name>.py
6. 注册进 SkillRegistry
7. 发布 SKILL_CREATED 事件，清空该组 buffer
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from minisweagent.models.litellm_model import LitellmModel

from ..bus import Event, EventBus, EventType
from ..skills.registry import SkillRegistry
from .base import BaseAgent

_GENERATE_SYSTEM = """\
You are a Python expert generating a reusable math skill module.
Given several successful math problem summaries that share the same technique,
write a complete Python file that implements the technique as reusable functions.

The file MUST contain:
1. A SKILL_META dictionary at the top level with keys: name, description, tags, module, usage
2. One or more reusable Python functions implementing the technique
3. A brief docstring per function

Output ONLY the Python code, no markdown fences, no explanation."""

_GENERATE_USER = """\
These successful solutions all used the same technique: {technique}
Tags: {tags}

Success summaries:
{summaries}

Generate a Python skill file named skill_{safe_name}.py.
The SKILL_META "module" field must be "skill_{safe_name}".
Make the functions general and reusable, not tied to specific problem numbers."""


class SkillAgent(BaseAgent):
    """技能生成智能体。

    参数
    ----
    skill_registry : SkillRegistry
        注册表，生成 skill 后动态注册进去。
    skills_dir : Path
        生成的 skill 文件写入此目录（与 ExecutorAgent 共享 PYTHONPATH）。
    threshold : int
        同组 SUCCESS_ANALYSIS 积累多少次后触发生成（默认 3）。
    """

    name = "skill_agent"

    def __init__(
        self,
        model: LitellmModel,
        bus: EventBus,
        skill_registry: SkillRegistry,
        skills_dir: Path,
        threshold: int = 3,
    ) -> None:
        super().__init__(model, bus)
        self.registry = skill_registry
        self.skills_dir = Path(skills_dir)
        self.threshold = threshold
        # buffer: tag_key -> list of {"technique", "summary", "problem_id"}
        self.pattern_buffer: dict[str, list[dict]] = {}

    def _register_handlers(self) -> None:
        self.bus.subscribe(EventType.SUCCESS_ANALYSIS, self._on_success)

    # ------------------------------------------------------------------ #
    # 事件处理
    # ------------------------------------------------------------------ #

    def _on_success(self, event: Event) -> None:
        data = event.data
        tags = sorted(data.get("tags", []))
        tag_key = ",".join(tags) or "general"
        technique = data.get("technique", "unknown")

        bucket = self.pattern_buffer.setdefault(tag_key, [])
        bucket.append({
            "technique": technique,
            "summary": data.get("summary", ""),
            "problem_id": data.get("problem_id", ""),
        })

        if len(bucket) >= self.threshold:
            self._try_generate_skill(tag_key, technique, tags, bucket.copy())
            del self.pattern_buffer[tag_key]

    # ------------------------------------------------------------------ #
    # 生成 skill
    # ------------------------------------------------------------------ #

    def _try_generate_skill(
        self,
        tag_key: str,
        technique: str,
        tags: list[str],
        summaries: list[dict],
    ) -> None:
        safe_name = re.sub(r"[^a-z0-9]+", "_", technique.lower()).strip("_") or "generated"

        # 避免覆盖已有 skill
        target_path = self.skills_dir / f"skill_{safe_name}.py"
        if target_path.exists() or self.registry.get(safe_name):
            return

        # 调 LLM 生成代码
        summaries_text = "\n".join(
            f"{i+1}. [{s['problem_id']}] {s['summary']}"
            for i, s in enumerate(summaries)
        )
        user_msg = _GENERATE_USER.format(
            technique=technique,
            tags=", ".join(tags),
            summaries=summaries_text,
            safe_name=safe_name,
        )
        try:
            code = self._llm_call(
                system=_GENERATE_SYSTEM,
                user=user_msg,
                temperature=0.2,
                max_tokens=1024,
            ).strip()
        except Exception:
            return

        # 去掉 LLM 可能输出的 markdown 围栏
        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"\s*```$", "", code)

        if not self._validate(code):
            return

        # 写文件
        target_path.write_text(code, encoding="utf-8")

        # 注册
        try:
            self.registry.register_from_file(target_path)
        except Exception:
            target_path.unlink(missing_ok=True)
            return

        self._publish(Event(
            type=EventType.SKILL_CREATED,
            data={"name": safe_name, "path": str(target_path), "tags": tags},
        ))

    # ------------------------------------------------------------------ #
    # 验证
    # ------------------------------------------------------------------ #

    def _validate(self, code: str) -> bool:
        """语法检查 + 确认含 SKILL_META。"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        # 检查顶层有 SKILL_META 赋值
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "SKILL_META":
                        return True
        return False

