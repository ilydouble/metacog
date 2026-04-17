"""EpisodicMemory - 情景记忆（成功案例检索）

负责存储和检索成功解题的完整推理链，用于类比学习。

设计思路
--------
1. 存储成功案例的：题目 + 关键推理步骤 + 解法洞察
2. 根据当前题目检索相似的成功案例
3. 注入完整的推理链，供 Agent 类比学习

与其他记忆的区别
----------------
- 语义记忆（Semantic）: 抽象的教训/知识
- 程序记忆（Procedural）: 可复用的技能
- 情景记忆（Episodic）: 具体的成功案例
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .memu_client import MemUClient, MemorySearchResult


@dataclass
class SuccessCase:
    """成功案例结构"""
    problem_id: str
    problem_text: str
    solution_steps: list[str]  # 关键推理步骤
    key_insight: str  # 核心洞察
    tags: list[str]
    answer: str


class EpisodicMemory:
    """情景记忆管理器
    
    负责：
    1. 存储成功案例到向量库
    2. 根据题目检索相似的成功案例
    3. 格式化案例为 prompt（供类比学习）
    """
    
    def __init__(
        self,
        collection_name: str = "episodic_memory",
        persist_dir: Path | str | None = None,
    ) -> None:
        self.memu = MemUClient(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )
    
    def add_success_case(
        self,
        problem_id: str,
        problem_text: str,
        solution_steps: list[str],
        key_insight: str,
        tags: list[str],
        answer: str,
        problem_type: str = "",
    ) -> str:
        """Add a success case to episodic memory (episodic fields only).

        Parameters
        ----------
        problem_id : str
            Problem ID
        problem_text : str
            Full problem description
        solution_steps : list[str]
            verified_cot_trajectory as a single-item list
        key_insight : str
            state_evaluation_metric (value-network heuristic)
        tags : list[str]
            Inferred from problem_type keywords
        answer : str
            Final answer (not stored in vector content)
        problem_type : str
            Abstract problem type description

        Returns
        -------
        memory_id : str
            ID in the vector store
        """
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(solution_steps))
        tags_text  = ", ".join(tags)

        if problem_type:
            content = (
                f"Problem type: {problem_type}\n"
                f"Key insight: {key_insight}\n"
                f"Tags: {tags_text}\n"
                f"Approach:\n{steps_text}"
            )
        else:
            content = (
                f"Problem: {problem_text[:200]}\n\n"
                f"Key insight: {key_insight}\n\n"
                f"Approach:\n{steps_text}"
            )

        metadata = {
            "problem_id":   problem_id,
            "problem_text": problem_text[:500],
            "problem_type": problem_type,
            "key_insight":  key_insight,
            "tags":         tags,
            "answer":       answer,
            "type":         "success_case",
            "num_steps":    len(solution_steps),
        }

        memory_id = self.memu.add_memory(content=content, metadata=metadata)
        return memory_id
    
    def search_similar_cases(
        self,
        query: str,
        top_k: int = 2,
        tag_filter: Optional[list[str]] = None,
    ) -> list[MemorySearchResult]:
        """根据题目检索相似的成功案例
        
        参数
        ----
        query : str
            当前题目文本
        top_k : int
            返回最相似的 K 个案例
        tag_filter : list[str] | None
            按标签过滤（可选）
            
        返回
        ----
        cases : list[MemorySearchResult]
            检索到的成功案例列表
        """
        results = self.memu.search(query=query, top_k=top_k)
        return results
    
    def count(self) -> int:
        """返回情景记忆中的成功案例数量"""
        return self.memu.count()
    
    def format_cases_for_prompt(
        self,
        cases: list[MemorySearchResult],
        max_cases: int = 2,
    ) -> str:
        """Format success cases as prompt text.

        Notes:
        - Never exposes the answer (avoid anchoring bias)
        - Clearly states this is for inspiration only, not a guaranteed match
        - Renders new fields: problem_type, key_insight, tags, approach
        """
        if not cases:
            return ""

        cases = cases[:max_cases]
        lines = [
            "## 💡 Similar Success Case (For Reference Only)",
            "The following case(s) may share structural or technical similarities with your current problem.",
            "**Important**: Similarity does NOT guarantee the same solution approach applies.",
            "Use the reasoning pattern as **inspiration**, and adapt it to your specific problem.\n",
        ]

        for i, case in enumerate(cases, 1):
            meta        = case.metadata or {}
            similarity  = 1 - case.distance
            problem_type = meta.get("problem_type", "")
            key_insight  = meta.get("key_insight", "")
            tags         = meta.get("tags", [])
            tags_str     = ", ".join(tags) if isinstance(tags, list) else str(tags)

            # approach is stored as a list in metadata (or in content as numbered lines)
            approach = meta.get("approach", [])

            lines.append(f"### Case {i} (Similarity: {similarity:.0%} | Tags: {tags_str})")
            if problem_type:
                lines.append(f"**Problem type**: {problem_type}")
            if key_insight:
                lines.append(f"**Key insight**: {key_insight}")
            if approach:
                if isinstance(approach, list):
                    steps = "\n".join(f"  {step}" for step in approach)
                else:
                    steps = str(approach)
                lines.append(f"**Approach**:\n{steps}")
            lines.append("\n⚠️  **Adapt, don't copy**: Only use this approach if the problem structure truly matches.\n")

        return "\n".join(lines)
