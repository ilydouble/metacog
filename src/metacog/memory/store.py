"""记忆存储 - 读写 memories.yaml，支持 append/update/query_by_tag。

YAML 格式::

    version: 1
    memories:
      - id: mem_001
        title: "整除问题：避免 sympy.solve"
        content: "For modular arithmetic, prefer pow(a, -1, m) over sympy.solve()."
        tags: [modular_arithmetic, sympy]
        created_at: "2024-01-01T00:00:00"
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class MemoryEntry:
    """单条记忆。"""
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:8]}")
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "tags": self.tags,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            id=d.get("id", f"mem_{uuid.uuid4().hex[:8]}"),
            title=d.get("title", ""),
            content=d.get("content", ""),
            tags=d.get("tags", []),
            created_at=d.get("created_at", datetime.now().isoformat(timespec="seconds")),
        )


class MemoryStore:
    """YAML 记忆存储，线程不安全（单进程单线程使用）。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: list[MemoryEntry] = []
        self._load()

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if self.path.exists():
            data = yaml.safe_load(self.path.read_text()) or {}
            self._entries = [MemoryEntry.from_dict(d) for d in data.get("memories", [])]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "memories": [e.to_dict() for e in self._entries]}
        self.path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    def append(self, entry: MemoryEntry) -> None:
        """追加一条记忆并保存。"""
        self._entries.append(entry)
        self.save()

    def update(self, entry_id: str, **kwargs) -> bool:
        """按 id 更新字段，返回是否找到。"""
        for e in self._entries:
            if e.id == entry_id:
                for k, v in kwargs.items():
                    if hasattr(e, k):
                        setattr(e, k, v)
                self.save()
                return True
        return False

    def query_by_tag(self, tag: str) -> list[MemoryEntry]:
        """按 tag 过滤记忆。"""
        return [e for e in self._entries if tag in e.tags]

    def all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def as_prompt_text(self, tag: Optional[str] = None) -> str:
        """将记忆格式化为注入 prompt 的文本。"""
        entries = self.query_by_tag(tag) if tag else self._entries
        if not entries:
            return ""
        lines = ["## Learned Strategies\n"]
        for i, e in enumerate(entries, 1):
            lines.append(f"{i}. **{e.title}** — {e.content}")
        return "\n".join(lines) + "\n"

    def __len__(self) -> int:
        return len(self._entries)

