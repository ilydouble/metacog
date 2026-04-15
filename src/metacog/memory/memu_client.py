"""memU - 轻量级向量记忆微服务客户端

基于 ChromaDB 实现语义记忆的存储和检索。
核心理念：存入时结构化，提取时向量化（Top-K）。

用法::

    client = MemUClient(collection_name="math_lessons", persist_dir="./memu_db")
    
    # 写入记忆
    client.add_memory(
        content="错误原因：忘记了费马小定理的使用前提是模数为素数。解决策略：计算逆元前先检查模数是否为素数。",
        metadata={"tags": ["数论", "模运算"], "symptom": "逆元计算错误"}
    )
    
    # 检索记忆
    results = client.search(query="如何计算模逆元", top_k=2)
    for mem in results:
        print(mem.content)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError(
        "memU 需要 chromadb。请安装: pip install chromadb"
    )


@dataclass
class MemoryResult:
    """检索结果"""
    id: str
    content: str
    metadata: dict[str, Any]
    distance: float  # 越小越相似


# 为了兼容性，保留旧的别名
MemorySearchResult = MemoryResult


class MemUClient:
    """memU 向量记忆客户端
    
    参数
    ----
    collection_name : str
        记忆集合名称（不同实验可以用不同 collection）
    persist_dir : Path | str
        向量数据库持久化目录
    embedding_model : str
        embedding 模型名称（默认使用 ChromaDB 内置的 all-MiniLM-L6-v2）
    """
    
    def __init__(
        self,
        collection_name: str = "math_lessons",
        persist_dir: Path | str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.collection_name = collection_name
        
        # 初始化 ChromaDB 客户端
        if persist_dir:
            persist_dir = Path(persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            # 内存模式（测试用）
            self.client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
        
        # 获取或创建 collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦距离
        )
    
    def add_memory(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> str:
        """添加一条记忆
        
        参数
        ----
        content : str
            记忆内容（会被向量化）
        metadata : dict
            元数据（tags, symptom 等），用于混合检索和过滤
        memory_id : str
            可选的记忆 ID，不提供则自动生成
            
        返回
        ----
        memory_id : str
            记忆的唯一 ID
        """
        if not memory_id:
            memory_id = f"mem_{uuid.uuid4().hex[:12]}"

        metadata = metadata or {}

        # 🔥 添加统计字段
        import time
        metadata["created_at"] = metadata.get("created_at", time.time())
        metadata["used_count"] = metadata.get("used_count", 0)
        metadata["last_used_at"] = metadata.get("last_used_at", 0)
        metadata["success_count"] = metadata.get("success_count", 0)  # 使用后成功次数
        metadata["total_uses"] = metadata.get("total_uses", 0)  # 总使用次数

        # 存入 ChromaDB
        self.collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata]
        )

        return memory_id
    
    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[MemoryResult]:
        """语义检索记忆
        
        参数
        ----
        query : str
            查询文本（通常是当前题目描述）
        top_k : int
            返回最相关的 K 条记忆
        filter_metadata : dict
            元数据过滤条件（例如只检索特定 tag 的记忆）
            
        返回
        ----
        results : list[MemoryResult]
            按相似度排序的记忆列表
        """
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": top_k,
        }
        
        if filter_metadata:
            kwargs["where"] = filter_metadata
        
        results = self.collection.query(**kwargs)
        
        # 解析结果
        memories: list[MemoryResult] = []
        if results["ids"] and results["ids"][0]:
            for i, mem_id in enumerate(results["ids"][0]):
                memories.append(MemoryResult(
                    id=mem_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] or {},
                    distance=results["distances"][0][i] if results["distances"] else 0.0,
                ))
        
        return memories
    
    def clear(self) -> None:
        """清空所有记忆（慎用）"""
        # 删除并重建 collection
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def count(self) -> int:
        """返回记忆总数"""
        return self.collection.count()

    def update_usage_stats(
        self,
        memory_id: str,
        success: bool = False
    ) -> None:
        """更新记忆的使用统计

        参数
        ----
        memory_id : str
            记忆 ID
        success : bool
            本次使用后是否成功
        """
        import time

        # 获取当前元数据
        result = self.collection.get(ids=[memory_id], include=["metadatas"])
        if not result["ids"]:
            return

        metadata = result["metadatas"][0] or {}

        # 更新统计
        metadata["used_count"] = metadata.get("used_count", 0) + 1
        metadata["last_used_at"] = time.time()
        metadata["total_uses"] = metadata.get("total_uses", 0) + 1

        if success:
            metadata["success_count"] = metadata.get("success_count", 0) + 1

        # 更新到数据库
        self.collection.update(
            ids=[memory_id],
            metadatas=[metadata]
        )

    def get_all_memories(self) -> list[MemoryResult]:
        """获取所有记忆（用于评估）"""
        result = self.collection.get(include=["documents", "metadatas"])

        memories = []
        if result["ids"]:
            for i, mem_id in enumerate(result["ids"]):
                memories.append(MemoryResult(
                    id=mem_id,
                    content=result["documents"][i],
                    metadata=result["metadatas"][i] or {},
                    distance=0.0
                ))

        return memories

    def delete_memory(self, memory_id: str) -> None:
        """删除一条记忆"""
        self.collection.delete(ids=[memory_id])
