"""SkillRegistry - 运行时注册、查找、生成 prompt 描述。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional, Union

from .base import FileSkill, StructuredSkill

_AnySkill = Union[StructuredSkill, FileSkill]


class SkillRegistry:
    """技能注册表。

    支持两种技能：
    - StructuredSkill 子类实例（Python 代码内定义）
    - FileSkill（从独立 .py 文件加载，agent 通过 import 调用）

    用法::

        registry = SkillRegistry()

        # 注册代码内 skill
        registry.register(SympySolverSkill())

        # 从文件加载 skill（文件须含 SKILL_META 字典）
        registry.register_from_file(Path("outputs/exp/skills/skill_modular_inverse.py"))

        # 生成注入 prompt 的文本
        print(registry.as_prompt_text())
    """

    def __init__(self) -> None:
        self._skills: dict[str, _AnySkill] = {}

    # ------------------------------------------------------------------ #
    # 注册
    # ------------------------------------------------------------------ #

    def register(self, skill: _AnySkill) -> None:
        """注册技能（name 必须唯一）。"""
        if not skill.name:
            raise ValueError(f"Skill {skill!r} has no name.")
        self._skills[skill.name] = skill

    def register_from_file(self, path: Path) -> FileSkill:
        """从 .py 文件读取 SKILL_META，注册为 FileSkill。

        文件必须包含顶层 SKILL_META 字典，格式::

            SKILL_META = {
                "name": "modular_inverse",
                "description": "...",
                "tags": ["number_theory"],
                "module": "skill_modular_inverse",
                "usage": "from skill_modular_inverse import modular_inverse\\n...",
            }

        返回注册好的 FileSkill 对象。
        """
        path = Path(path).resolve()
        # 动态加载模块以读取 SKILL_META
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load skill file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module  # 避免重复加载
        spec.loader.exec_module(module)  # type: ignore

        meta: dict = getattr(module, "SKILL_META", None)
        if not meta:
            raise ValueError(f"Skill file {path} has no SKILL_META dict.")

        skill = FileSkill(
            name=meta["name"],
            description=meta.get("description", ""),
            tags=meta.get("tags", []),
            module=meta.get("module", path.stem),
            usage=meta.get("usage", ""),
            file_path=str(path),
        )
        self.register(skill)
        return skill

    def register_from_dir(self, skills_dir: Path) -> list[FileSkill]:
        """扫描目录，注册所有含 SKILL_META 的 skill_*.py 文件。"""
        registered: list[FileSkill] = []
        for f in sorted(Path(skills_dir).glob("skill_*.py")):
            try:
                registered.append(self.register_from_file(f))
            except Exception as exc:
                pass  # 跳过格式不对的文件
        return registered

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #

    def unregister(self, name: str) -> None:
        self._skills.pop(name, None)

    def get(self, name: str) -> Optional[_AnySkill]:
        return self._skills.get(name)

    def by_tag(self, tag: str) -> list[_AnySkill]:
        return [s for s in self._skills.values() if tag in s.tags]

    def all(self) -> list[_AnySkill]:
        return list(self._skills.values())

    # ------------------------------------------------------------------ #
    # Prompt 生成
    # ------------------------------------------------------------------ #

    def as_prompt_text(self, tag: Optional[str] = None) -> str:
        """生成注入 system_template 的技能列表文本。"""
        skills = self.by_tag(tag) if tag else self.all()
        if not skills:
            return ""
        lines = ["## Available Skills", "You can import these directly in your Python code:\n"]
        for s in skills:
            lines.append(s.to_prompt_line())
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"<SkillRegistry skills={list(self._skills.keys())}>"

