"""MemoryManagerAgent - Subscribes to AnalysisEvent, writes structured lessons to memU vector database.

Key Points
----------
1. Receives structured JSON output from AnalyzerAgent
2. Concatenates root_cause + actionable_advice as vectorized text
3. Stores in memU semantic memory layer
4. Keeps YAML as human debug backup (optional)
"""

from __future__ import annotations

from pathlib import Path

from minisweagent.models.litellm_model import LitellmModel

from ..bus import Event, EventBus, EventType
from ..memory.memu_client import MemUClient
from ..memory.store import MemoryEntry, MemoryStore
from .base import BaseAgent


class MemoryManagerAgent(BaseAgent):
    """Memory manager agent (memU version).

    Subscribes to AnalysisEvent → extracts structured JSON → writes to memU vector database.

    Core design:
    - Vectorized content: activation_condition + root_cause_analysis + teacher_suggested_approach
    - Metadata: all semantic_memory fields for future retrieval and injection
    - Optional YAML backup for human debugging
    """

    name = "memory_manager"

    def __init__(
        self,
        model: LitellmModel,
        bus: EventBus,
        memory_store: MemoryStore | None = None,  # 保留为可选的 debug 备份
        memu_persist_dir: Path | str | None = None,  # memU 向量库目录
        collection_name: str = "math_lessons",
    ) -> None:
        super().__init__(model, bus)
        self.store = memory_store  # Optional YAML backup

        # Initialize memU client (math_lessons collection — stores both distilled lessons and hints)
        self.memu = MemUClient(
            collection_name=collection_name,
            persist_dir=memu_persist_dir,
        )

    def _register_handlers(self) -> None:
        self.bus.subscribe(EventType.ANALYSIS, self._on_analysis)

    def _on_analysis(self, event: Event) -> None:
        """Handle AnalysisEvent, write semantic lesson to memU vector database."""
        analysis     = event.data.get("analysis", {})
        problem_id   = event.data.get("problem_id", "unknown")
        problem_text = event.data.get("problem_text", "")

        # Skip completely invalid analysis
        if "error" in analysis or analysis.get("skip"):
            return

        skip_error_analysis = analysis.get("skip_error_analysis", False)
        solution_hint: dict = analysis.get("solution_hint") or {}

        # ── Phase 1 blocked, only solution hint available ──────────────────
        if skip_error_analysis:
            if not solution_hint:
                print(f"  [MemoryManager] ⚠️  Phase 1 failed and no solution hint, skipping", flush=True)
                return

            print(f"  [MemoryManager] ℹ️  Storing hint-only entry in math_lessons (Phase 1 failed)", flush=True)
            key_insight    = solution_hint.get("key_insight", "")
            approach       = solution_hint.get("approach", [])
            common_pitfall = solution_hint.get("common_pitfall", "")

            # approach is a list — join for content text, store as list in metadata
            if isinstance(approach, list):
                approach_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(approach))
            else:
                approach_text = str(approach)

            # Vectorized content: use problem text as retrieval anchor (no problem_type available)
            # Strip insight and approach to avoid diluting the vector space
            content = f"Problem snippet: {problem_text[:300]}\n"

            metadata = {
                "tags":            ["hint_only"],
                "key_insight":     key_insight,
                "approach":        approach,       # list stored directly
                "common_pitfall":  common_pitfall,
                "problem_id":      problem_id,
                "source_problem":  problem_text[:200],
            }
            mem_id = self.memu.add_memory(content=content.strip(), metadata=metadata)
            print(f"  [MemoryManager] ✓ Stored (hint-only) in math_lessons: {mem_id[:12]}", flush=True)
            self._publish_updated(mem_id, problem_id, ["hint_only"])
            return

        # ── Normal flow: flat JSON fields ──────────────────────────────────
        problem_type   = analysis.get("problem_type", "")
        error          = analysis.get("error", "")
        fix            = analysis.get("fix", "")
        reasoning_path = analysis.get("reasoning_path", "")
        solution_steps = analysis.get("solution_steps", "")
        common_traps   = analysis.get("common_traps", "")

        if not fix:
            print(f"  [MemoryManager] ⚠️  Missing 'fix' field, skipping", flush=True)
            return

        # Vectorized content: USE ONLY problem_type AND problem text snippet for retrieval matching
        # Do NOT include error, fix, or solution steps here, as they dilute the vector space
        # since the query will only be a raw problem text.
        content = (
            f"Problem type: {problem_type}\n"
            f"Problem snippet: {problem_text[:300]}"
        )

        metadata = {
            "tags":           ["semantic_lesson"],
            "problem_type":   problem_type,
            "error":          error,
            "fix":            fix,
            "reasoning_path": reasoning_path,
            "solution_steps": solution_steps,
            "common_traps":   common_traps,
            "problem_id":     problem_id,
            "source_problem": problem_text[:200],
        }

        try:
            memory_id = self.memu.add_memory(content=content, metadata=metadata)
            print(f"  [MemoryManager] ✓ Semantic lesson stored: {memory_id[:12]}", flush=True)
        except Exception as exc:
            print(f"  [MemoryManager] ✗ Storage failed: {exc}", flush=True)
            return

        # Optional YAML backup for human debugging
        if self.store:
            try:
                entry = MemoryEntry(
                    title=f"[semantic] {activation_condition[:60]}",
                    content=content,
                    tags=["semantic_lesson"],
                )
                self.store.append(entry)
            except Exception:
                pass

        self._publish_updated(memory_id, problem_id, ["semantic_lesson"])

    def _publish_updated(self, memory_id: str, problem_id: str, tags: list) -> None:
        self._publish(Event(
            type=EventType.MEMORY_UPDATED,
            data={
                "action":     "added_to_memu",
                "memory_id":  memory_id,
                "problem_id": problem_id,
                "tags":       tags,
            },
        ))

