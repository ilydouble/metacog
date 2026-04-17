"""MemoryManagerAgent - 订阅 AnalysisEvent，将结构化 lesson 写入 memU 向量库。

改造要点
--------
1. 接收 AnalyzerAgent 输出的结构化 JSON
2. 将 root_cause + actionable_advice 拼接为向量化文本
3. 存入 memU 的语义记忆层
4. 保留 YAML 作为人类 debug 备份（可选）
"""

from __future__ import annotations

from pathlib import Path

from minisweagent.models.litellm_model import LitellmModel

from ..bus import Event, EventBus, EventType
from ..memory.memu_client import MemUClient
from ..memory.store import MemoryEntry, MemoryStore
from .base import BaseAgent


class MemoryManagerAgent(BaseAgent):
    """记忆管理智能体（memU 版本）。

    订阅 AnalysisEvent → 提取结构化 JSON → 写入 memU 向量库。

    核心改造
    --------
    - 不再使用 YAML 作为主存储（保留为人类 debug 备份）
    - 将 root_cause + actionable_advice 作为向量化内容
    - 将 problem_tags, error_symptom 作为元数据
    - 使用 memU 的语义检索能力
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
        self.store = memory_store  # 可选的 YAML 备份

        # 初始化 memU 客户端（核心存储）
        self.memu = MemUClient(
            collection_name=collection_name,
            persist_dir=memu_persist_dir,
        )

    def _register_handlers(self) -> None:
        self.bus.subscribe(EventType.ANALYSIS, self._on_analysis)

    def _on_analysis(self, event: Event) -> None:
        """处理失败分析结果，存入 memU 向量库"""
        analysis = event.data.get("analysis", {})
        problem_id = event.data.get("problem_id", "unknown")
        problem_text = event.data.get("problem_text", "")

        # 跳过完全无效的分析
        if "error" in analysis or analysis.get("skip"):
            return

        # ========================================
        # 🔥 支持"只有解题思路，没有错误分析"的情况
        # ========================================
        skip_error_analysis = analysis.get("skip_error_analysis", False)
        solution_hint: dict = analysis.get("solution_hint") or {}

        if skip_error_analysis:
            # Phase 1 失败，但 Phase 2 成功
            if not solution_hint:
                print(f"  [MemoryManager] ⚠️  Phase 1 失败且无解题思路，跳过存储", flush=True)
                return

            print(f"  [MemoryManager] ℹ️  仅存储解题思路（Phase 1 失败）", flush=True)

            # 构造一个"仅包含解题思路"的特殊记忆条目
            key_insight = solution_hint.get("key_insight", "")
            approach = solution_hint.get("approach", "")
            common_pitfall = solution_hint.get("common_pitfall", "")

            document_content = f"Problem type: (analysis failed, hint only)\n"
            if key_insight:
                document_content += f"Correct approach: {key_insight}\n"
            if approach:
                document_content += f"Key steps: {approach}\n"
            if common_pitfall:
                document_content += f"Common pitfall: {common_pitfall}\n"

            metadata = {
                "tags": ["hint_only"],  # 特殊标记
                "symptom": "phase1_failed",
                "problem_id": problem_id,
                "source_problem": problem_text[:200],
            }

            # 存入语义记忆（作为"参考解法"而非"错误教训"）
            mem_id = self.memu.add_memory(
                content=document_content.strip(),
                metadata=metadata
            )
            print(f"  [MemoryManager] ✓ 写入 memU (hint-only): {mem_id} | tags={metadata['tags']}", flush=True)
            return

        # 正常流程：提取结构化字段
        raw_tags = analysis.get("problem_tags", "")
        problem_tags = [t.strip() for t in raw_tags.split(",") if t.strip()] if isinstance(raw_tags, str) else raw_tags

        # 🔥 ChromaDB 不允许空 list
        if not problem_tags:
            problem_tags = ["general"]

        problem_type = analysis.get("problem_type", "")
        error_symptom = analysis.get("error_symptom", "")
        root_cause = analysis.get("root_cause", "")
        actionable_advice = analysis.get("actionable_advice", "")

        if not root_cause or not actionable_advice:
            print(f"  [MemoryManager] 警告: 缺少核心内容，跳过存储", flush=True)
            return

        # ========================================
        # 核心改造：构造 memU 文档
        # ========================================
        # 1. 拼接向量化内容
        # Problem_Type 放最前面作为 embedding 锚点，与 query（题目原文）共享数学词汇，
        # 使得余弦相似度能真正反映题型相关性，而非被中文分析文字主导。
        if problem_type:
            document_content = (
                f"Problem type: {problem_type}\n"
                f"Error: {root_cause}\n"
                f"Fix: {actionable_advice}"
            )
        else:
            document_content = f"错误原因：{root_cause}\n解决策略：{actionable_advice}"

        # 重新获取 solution_hint（之前已经在前面声明了，这里需要更新）
        solution_hint = analysis.get("solution_hint") or {}
        if solution_hint:
            key_insight = solution_hint.get("key_insight", "")
            approach = solution_hint.get("approach", "")
            common_pitfall = solution_hint.get("common_pitfall", "")
            hint_text = f"\n解题思路：{key_insight}"
            if approach:
                hint_text += f"\n解题步骤：{approach}"
            if common_pitfall:
                hint_text += f"\n常见陷阱：{common_pitfall}"
            document_content += hint_text
            print(f"  [MemoryManager] ✓ 附加解题思路 hint", flush=True)

        # 2. 构造元数据（用于过滤和混合检索）
        metadata = {
            "tags": problem_tags,  # list[str]
            "symptom": error_symptom,
            "problem_id": problem_id,
            "source_problem": problem_text[:200],  # 截取前200字符作为线索
        }

        # 3. 写入 memU 向量库（不做去重，交给 MemoryEvaluator 延迟合并）
        try:
            memory_id = self.memu.add_memory(
                content=document_content,
                metadata=metadata
            )
            print(f"  [MemoryManager] ✓ 写入 memU: {memory_id} | tags={problem_tags}", flush=True)
        except Exception as exc:
            print(f"  [MemoryManager] ✗ memU 写入失败: {exc}", flush=True)
            return

        # ========================================
        # 可选：同时写入 YAML 作为人类 debug 备份
        # ========================================
        if self.store:
            try:
                entry = MemoryEntry(
                    title=f"[{'/'.join(problem_tags[:2])}] {error_symptom[:50]}",
                    content=document_content,
                    tags=problem_tags,
                )
                self.store.append(entry)
            except Exception:
                pass  # YAML 备份失败不影响核心流程

        # 4. 发布更新完成事件
        self._publish(Event(
            type=EventType.MEMORY_UPDATED,
            data={
                "action": "added_to_memu",
                "memory_id": memory_id,
                "problem_id": problem_id,
                "tags": problem_tags,
            },
        ))

