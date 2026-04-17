"""ProceduralMemory - 程序记忆（技能向量检索）

负责管理和检索技能（skills），使用向量数据库实现语义匹配。

设计思路
--------
1. 每个 skill 包含：代码 + 元数据（description, when_to_use, tags）
2. 将 description + when_to_use 向量化存储
3. 根据题目语义检索最相关的 Top-K 个 skill
4. 只注入相关 skill，避免上下文膨胀
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .memu_client import MemUClient, MemorySearchResult


@dataclass
class SkillMetadata:
    """技能元数据"""
    name: str
    description: str
    when_to_use: str
    tags: list[str]
    file_path: str


class ProceduralMemory:
    """程序记忆管理器
    
    负责：
    1. 存储 skill 的元数据到向量库
    2. 根据题目检索最相关的 skill
    3. 返回 skill 的完整信息（元数据 + 代码）
    """
    
    def __init__(
        self,
        collection_name: str = "procedural_memory",
        persist_dir: Path | str | None = None,
    ) -> None:
        self.memu = MemUClient(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )
    
    def add_skill(
        self,
        name: str,
        description: str,
        when_to_use: str,
        tags: list[str],
        file_path: str,
    ) -> str:
        """添加一个技能到程序记忆
        
        参数
        ----
        name : str
            技能名称（函数名）
        description : str
            技能描述（做什么）
        when_to_use : str
            使用指南（何时用）
        tags : list[str]
            标签
        file_path : str
            技能文件路径
            
        返回
        ----
        memory_id : str
            向量库中的 ID
        """
        # 拼接向量化内容
        content = f"{description}\n使用场景：{when_to_use}"

        # 🔥 去重检查
        try:
            similar_skills = self.memu.search(query=content, top_k=1)
            if similar_skills:
                similarity = 1 - similar_skills[0].distance
                if similarity > self.dedup_threshold:
                    # 相似技能已存在，跳过
                    return similar_skills[0].id
        except Exception:
            pass  # 去重失败不影响存储

        # 构造元数据
        metadata = {
            "name": name,
            "description": description,
            "when_to_use": when_to_use,
            "tags": tags,
            "file_path": file_path,
            "type": "skill",
        }

        # 存入向量库
        memory_id = self.memu.add_memory(content=content, metadata=metadata)
        return memory_id
    
    def search_skills(
        self,
        query: str,
        top_k: int = 3,
        tag_filter: Optional[list[str]] = None,
    ) -> list[MemorySearchResult]:
        """根据题目检索最相关的技能
        
        参数
        ----
        query : str
            题目文本
        top_k : int
            返回最相关的 K 个技能
        tag_filter : list[str] | None
            按标签过滤（可选）
            
        返回
        ----
        skills : list[MemorySearchResult]
            检索到的技能列表
        """
        # TODO: 未来可以添加 tag_filter 支持
        results = self.memu.search(query=query, top_k=top_k)
        return results
    
    def get_skill_by_name(self, name: str) -> Optional[MemorySearchResult]:
        """根据名称获取技能（精确匹配）
        
        参数
        ----
        name : str
            技能名称
            
        返回
        ----
        skill : MemorySearchResult | None
        """
        # 使用名称作为查询，取第一个结果
        results = self.memu.search(query=name, top_k=1)
        if results and results[0].metadata.get("name") == name:
            return results[0]
        return None
    
    def count(self) -> int:
        """返回程序记忆中的技能数量"""
        return self.memu.count()
    
    def format_skills_for_prompt(
        self,
        skills: list[MemorySearchResult],
        include_code: bool = False,
    ) -> str:
        """将技能列表格式化为 prompt 文本
        
        参数
        ----
        skills : list[MemorySearchResult]
            技能列表
        include_code : bool
            是否包含代码（默认只包含元数据）
            
        返回
        ----
        prompt : str
            格式化的 prompt 文本
        """
        if not skills:
            return ""
        
        lines = ["## 🛠️ Available Skills (Retrieved for Current Problem)\n"]

        for skill in skills:
            meta = skill.metadata
            name = meta.get("name", "unknown")
            desc = meta.get("description", "")
            when = meta.get("when_to_use", "")
            tags = meta.get("tags", [])

            lines.append(f"### {name}")
            lines.append(f"**Description**: {desc}")
            lines.append(f"**When to use**: {when}")
            lines.append(f"**Tags**: {', '.join(tags)}")

            if include_code:
                file_path = meta.get("file_path", "")
                if file_path and Path(file_path).exists():
                    try:
                        code = Path(file_path).read_text()
                        lines.append(f"**Code**:\n```python\n{code}\n```")
                    except Exception:
                        pass
            
            lines.append("")  # 空行
        
        return "\n".join(lines)
