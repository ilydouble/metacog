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
        """添加一个成功案例到情景记忆
        
        参数
        ----
        problem_id : str
            题目 ID
        problem_text : str
            题目完整描述
        solution_steps : list[str]
            关键推理步骤（3-5 步）
        key_insight : str
            核心洞察（为什么这个方法有效）
        tags : list[str]
            题目标签
        answer : str
            答案
            
        返回
        ----
        memory_id : str
            向量库中的 ID
        """
        # 拼接向量化内容
        # 以抽象问题类型为核心，避免被题目里具体数字/符号主导
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(solution_steps))
        tags_text = ", ".join(tags)
        if problem_type:
            # 建议一：用抽象类型 + 洞察 + 标签作为 embedding，泛化性更强
            content = (
                f"Problem type: {problem_type}\n"
                f"Key insight: {key_insight}\n"
                f"Tags: {tags_text}\n"
                f"Approach:\n{steps_text}"
            )
        else:
            # fallback：沿用旧格式（GLM 没有生成 problem_type 时）
            content = (
                f"题目：{problem_text[:200]}\n\n"
                f"关键推理步骤：\n{steps_text}\n\n"
                f"核心洞察：{key_insight}\n\n"
                f"答案：{answer}"
            )

        # 构造元数据（完整信息保留在 metadata，不影响 embedding）
        metadata = {
            "problem_id": problem_id,
            "problem_text": problem_text[:500],
            "problem_type": problem_type,
            "key_insight": key_insight,
            "tags": tags,
            "answer": answer,
            "type": "success_case",
            "num_steps": len(solution_steps),
        }
        
        # 存入向量库
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
        """将成功案例格式化为 prompt 文本

        注意：
        - 不暴露答案（避免锚定偏差）
        - 明确告知模型这只是"可能相似"的例子，不保证完全匹配
        - 只提供推理方法，不提供结论
        """
        if not cases:
            return ""

        cases = cases[:max_cases]
        lines = [
            "## 💡 Possibly Similar Problem (For Reference Only)",
            "The following is a problem that may share a similar structure or technique.",
            "It is NOT guaranteed to be the same type. Use the reasoning approach as inspiration,",
            "but solve the current problem independently.\n",
        ]

        for i, case in enumerate(cases, 1):
            meta = case.metadata
            similarity = 1 - case.distance
            tags = ", ".join(meta.get("tags", []))

            # 从 content 中提取关键步骤和洞察（不含答案）
            content = case.content
            steps_section = ""
            insight_section = ""
            for line in content.split("\n"):
                if "关键推理步骤" in line or "Key Steps" in line:
                    continue
                elif "核心洞察" in line or "Core Insight" in line:
                    insight_section = line.replace("核心洞察：", "Insight: ").replace("Core Insight:", "Insight:")
                elif "答案" in line or "Answer" in line:
                    continue  # 跳过答案行
                elif line.strip().startswith(("1.", "2.", "3.", "4.", "5.")):
                    steps_section += line.strip() + "\n"

            lines.append(f"### Reference Case {i} (similarity: {similarity:.0%}, tags: {tags})")
            if steps_section:
                lines.append(f"Key reasoning steps:\n{steps_section.strip()}")
            if insight_section:
                lines.append(insight_section.strip())
            lines.append("\n⚠️  Apply the method above only if it fits the current problem.\n")

        return "\n".join(lines)
