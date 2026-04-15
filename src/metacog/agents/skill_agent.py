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
Given several successful math problem summaries OR failed lessons that share the same technique,
write a complete Python file that implements the technique as reusable functions.

The file MUST contain:
1. A SKILL_METADATA dictionary at the module level (not SKILL_META) with keys:
   - name: str (function name, snake_case)
   - description: str (what this skill does, max 50 words)
   - when_to_use: str (when to use this skill, max 80 words, be specific)
   - tags: list[str] (e.g. ["number_theory", "modular_arithmetic"])

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
        procedural_memory=None,  # ProceduralMemory instance
    ) -> None:
        super().__init__(model, bus)
        self.registry = skill_registry
        self.skills_dir = Path(skills_dir)
        self.threshold = threshold
        self.procedural_memory = procedural_memory
        # buffer: tag_key -> list of {"technique", "summary", "problem_id"}
        self.pattern_buffer: dict[str, list[dict]] = {}

    def _register_handlers(self) -> None:
        self.bus.subscribe(EventType.SUCCESS_ANALYSIS, self._on_success)
        self.bus.subscribe(EventType.MEMORY_UPDATED, self._on_lesson_learned)

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

    def _on_lesson_learned(self, event: Event) -> None:
        """处理失败教训，尝试生成预防性技能"""
        data = event.data

        # 检查是否包含代码建议（PoT 增强的）
        advice = data.get("actionable_advice", "")
        if "CODE:" not in advice:
            return  # 没有代码建议，跳过

        tags = data.get("tags", [])
        problem_id = data.get("problem_id", "")

        # 提取代码片段
        code_part = advice.split("CODE:", 1)[1].strip() if "CODE:" in advice else ""

        # 生成一个基于教训的技能
        technique = f"avoid_{tags[0]}_error" if tags else "error_prevention"
        self._try_generate_skill_from_lesson(technique, tags, code_part, problem_id)

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
        print(f"  [SkillAgent] 调用 LLM 生成技能代码...", flush=True)
        try:
            code = self._llm_call(
                system=_GENERATE_SYSTEM,
                user=user_msg,
                temperature=0.2,
                max_tokens=1024,
            ).strip()
            print(f"  [SkillAgent] LLM 返回代码长度: {len(code)} 字符", flush=True)
        except Exception as e:
            print(f"  [SkillAgent] ✗ LLM 调用失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return

        # 去掉 LLM 可能输出的 markdown 围栏
        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"\s*```$", "", code)

        print(f"  [SkillAgent] 验证代码语法...", flush=True)
        if not self._validate(code):
            print(f"  [SkillAgent] ✗ 代码验证失败（缺少 SKILL_METADATA 或语法错误）", flush=True)
            print(f"  [SkillAgent] 代码前 200 字符: {code[:200]}...", flush=True)
            return

        print(f"  [SkillAgent] ✓ 代码验证通过", flush=True)

        # 写文件
        target_path.write_text(code, encoding="utf-8")

        # 注册到 SkillRegistry
        try:
            self.registry.register_from_file(target_path)
        except Exception:
            target_path.unlink(missing_ok=True)
            return

        # 🔥 注册到程序记忆（ProceduralMemory）
        if self.procedural_memory:
            try:
                metadata = self._extract_metadata(code)
                if metadata:
                    self.procedural_memory.add_skill(
                        name=metadata.get("name", safe_name),
                        description=metadata.get("description", ""),
                        when_to_use=metadata.get("when_to_use", ""),
                        tags=metadata.get("tags", tags),
                        file_path=str(target_path),
                    )
            except Exception as e:
                print(f"  [SkillAgent] ⚠️  无法注册到程序记忆: {e}", flush=True)

        self._publish(Event(
            type=EventType.SKILL_CREATED,
            data={"name": safe_name, "path": str(target_path), "tags": tags},
        ))

    # ------------------------------------------------------------------ #
    # 验证
    # ------------------------------------------------------------------ #

    def _validate(self, code: str) -> bool:
        """语法检查 + 确认含 SKILL_METADATA。"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        # 检查顶层有 SKILL_METADATA 赋值
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "SKILL_METADATA":
                        return True
        return False

    def _extract_metadata(self, code: str) -> dict | None:
        """从代码中提取 SKILL_METADATA"""
        try:
            # 创建一个空的命名空间
            namespace = {}
            # 执行代码（只执行顶层赋值）
            exec(code, namespace)
            # 提取 SKILL_METADATA
            return namespace.get("SKILL_METADATA")
        except Exception:
            return None

    def _try_generate_skill_from_lesson(
        self,
        technique: str,
        tags: list[str],
        code_snippet: str,
        problem_id: str,
    ) -> None:
        """从失败教训生成预防性技能"""
        safe_name = f"skill_{technique}_{problem_id}"
        target_path = self.skills_dir / f"{safe_name}.py"

        if target_path.exists():
            return

        user_msg = f"""Generate a skill based on this lesson learned:
Technique: {technique}
Tags: {', '.join(tags)}
Code hint: {code_snippet}
Problem ID: {problem_id}

Create a reusable Python module with SKILL_METADATA that prevents this type of error."""

        try:
            code = self._llm_call(
                system=_GENERATE_SYSTEM,
                user=user_msg,
                temperature=0.2,
                max_tokens=1024,
            ).strip()
        except Exception:
            return

        # 清理代码
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

        # 注册到程序记忆
        if self.procedural_memory:
            try:
                metadata = self._extract_metadata(code)
                if metadata:
                    self.procedural_memory.add_skill(
                        name=metadata.get("name", safe_name),
                        description=metadata.get("description", ""),
                        when_to_use=metadata.get("when_to_use", ""),
                        tags=metadata.get("tags", tags),
                        file_path=str(target_path),
                    )
            except Exception:
                pass

