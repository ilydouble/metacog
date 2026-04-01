"""MemoryManagerAgent - 订阅 AnalysisEvent，将 lesson 写入 MemoryStore。"""

from __future__ import annotations

from minisweagent.models.litellm_model import LitellmModel

from ..bus import Event, EventBus, EventType
from ..memory.store import MemoryEntry, MemoryStore
from .base import BaseAgent


class MemoryManagerAgent(BaseAgent):
    """记忆管理智能体。

    订阅 AnalysisEvent → 决定是追加新记忆还是合并到已有记忆 → 更新 MemoryStore。

    合并策略（简单版）：
    - 如果 MemoryStore 里已有相同 tags 的记忆超过阈值 `merge_threshold`，
      则调用 LLM 将新 lesson 与已有记忆合并成一条；
    - 否则直接追加。
    """

    name = "memory_manager"

    _MERGE_SYSTEM = """\
You are managing a memory store for a math problem-solving agent.
Given an existing memory entry and a new lesson, merge them into a single, improved entry.
Output JSON only:
{
  "title": "merged title (max 10 words)",
  "content": "merged content (2-4 sentences, actionable, in English)",
  "tags": ["tag1", "tag2"]
}
"""

    def __init__(
        self,
        model: LitellmModel,
        bus: EventBus,
        memory_store: MemoryStore,
        merge_threshold: int = 3,
    ) -> None:
        super().__init__(model, bus)
        self.store = memory_store
        self.merge_threshold = merge_threshold

    def _register_handlers(self) -> None:
        self.bus.subscribe(EventType.ANALYSIS, self._on_analysis)

    def _on_analysis(self, event: Event) -> None:
        analysis = event.data.get("analysis", {})
        problem_id = event.data.get("problem_id", "unknown")

        if "error" in analysis and "lesson_title" not in analysis:
            return  # 分析失败，跳过

        title = analysis.get("lesson_title", "Untitled lesson")
        content = analysis.get("lesson_content", "")
        tags = analysis.get("tags", [])

        if not content:
            return

        # 检查是否需要合并
        existing = []
        for tag in tags:
            existing.extend(self.store.query_by_tag(tag))
        # 去重
        seen_ids: set[str] = set()
        candidates = [e for e in existing if not (e.id in seen_ids or seen_ids.add(e.id))]  # type: ignore

        if candidates and len(candidates) >= self.merge_threshold:
            # 取最相关的一条做合并
            target = candidates[0]
            merged = self._merge(target, title, content, tags)
            if merged:
                self.store.update(
                    target.id,
                    title=merged.get("title", target.title),
                    content=merged.get("content", target.content),
                    tags=merged.get("tags", target.tags),
                )
                self._publish(Event(
                    type=EventType.MEMORY_UPDATED,
                    data={"action": "merged", "entry_id": target.id, "problem_id": problem_id},
                ))
                return

        # 追加新记忆
        entry = MemoryEntry(title=title, content=content, tags=tags)
        self.store.append(entry)
        self._publish(Event(
            type=EventType.MEMORY_UPDATED,
            data={"action": "appended", "entry_id": entry.id, "problem_id": problem_id},
        ))

    def _merge(self, existing: MemoryEntry, new_title: str, new_content: str, new_tags: list[str]) -> dict | None:
        """调用 LLM 合并两条记忆，返回合并后的 dict。"""
        import json, re
        user_msg = (
            f"Existing entry:\nTitle: {existing.title}\nContent: {existing.content}\n\n"
            f"New lesson:\nTitle: {new_title}\nContent: {new_content}"
        )
        try:
            response = self._llm_call(self._MERGE_SYSTEM, user_msg, temperature=0.0, max_tokens=256)
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return None

