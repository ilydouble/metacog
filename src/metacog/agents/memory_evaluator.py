"""MemoryEvaluator - 记忆评估和清理智能体

负责定期评估记忆质量、清理低质量记忆、生成质量报告。

设计原则
--------
1. 基于使用频率、成功率、时效性计算质量分数
2. 标记低质量记忆（quality_score < threshold）
3. 清理从未使用且质量低的记忆
4. 生成记忆质量分析报告

质量评分算法
-----------
quality_score = (
    0.4 * usage_score +      # 使用频率
    0.4 * success_rate +     # 成功率
    0.2 * recency_score      # 时效性
)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..bus import Event, EventBus, EventType
from ..memory.memu_client import MemUClient
from .base import BaseAgent


@dataclass
class MemoryQualityReport:
    """记忆质量报告"""
    total_memories: int
    high_quality: int  # score >= 0.7
    medium_quality: int  # 0.4 <= score < 0.7
    low_quality: int  # score < 0.4
    unused_memories: int  # used_count == 0
    deleted_count: int
    avg_quality_score: float


class MemoryEvaluatorAgent(BaseAgent):
    """记忆评估智能体
    
    定期评估三层记忆的质量，清理低质量记忆
    """
    
    def __init__(
        self,
        bus: EventBus,
        semantic_memory: MemUClient,
        procedural_memory: Any,  # ProceduralMemory
        episodic_memory: Any,  # EpisodicMemory
        eval_interval: int = 10,  # 每 N 道题评估一次
        quality_threshold: float = 0.3,  # 低于此分数的记忆会被标记
        cleanup_threshold: float = 0.2,  # 低于此分数且未使用的记忆会被删除
    ) -> None:
        # MemoryEvaluator 不需要 model
        super().__init__(model=None, bus=bus)
        self.semantic_memory = semantic_memory
        self.procedural_memory = procedural_memory
        self.episodic_memory = episodic_memory
        self.eval_interval = eval_interval
        self.quality_threshold = quality_threshold
        self.cleanup_threshold = cleanup_threshold
        
        self.problems_solved = 0
    
    def _register_handlers(self) -> None:
        self.bus.subscribe(EventType.TRAJECTORY, self._on_trajectory)
    
    def _on_trajectory(self, event: Event) -> None:
        """每道题结束后检查是否需要评估"""
        self.problems_solved += 1
        
        if self.problems_solved % self.eval_interval == 0:
            print(f"\n{'='*70}", flush=True)
            print(f"[MemoryEvaluator] 🔍 开始记忆质量评估（已解 {self.problems_solved} 题）", flush=True)
            print(f"{'='*70}\n", flush=True)
            
            # 评估三层记忆
            self._evaluate_all_memories()
    
    def _evaluate_all_memories(self) -> None:
        """评估所有记忆层"""
        reports = {}
        
        # 1. 语义记忆（教训）
        if self.semantic_memory:
            report = self._evaluate_memory_layer(
                self.semantic_memory,
                layer_name="语义记忆（教训）"
            )
            reports["semantic"] = report
        
        # 2. 程序记忆（技能）
        if self.procedural_memory:
            report = self._evaluate_memory_layer(
                self.procedural_memory.memu,
                layer_name="程序记忆（技能）"
            )
            reports["procedural"] = report
        
        # 3. 情景记忆（成功案例）
        if self.episodic_memory:
            report = self._evaluate_memory_layer(
                self.episodic_memory.memu,
                layer_name="情景记忆（成功案例）"
            )
            reports["episodic"] = report
        
        # 🔥 检测并合并重复记忆
        self._merge_duplicates()

        # 生成总结报告
        self._print_summary_report(reports)
    
    def _evaluate_memory_layer(
        self,
        memu: MemUClient,
        layer_name: str
    ) -> MemoryQualityReport:
        """评估单个记忆层"""
        print(f"  [MemoryEvaluator] 评估 {layer_name}...", flush=True)
        
        # 获取所有记忆
        memories = memu.get_all_memories()
        
        if not memories:
            print(f"    → 无记忆，跳过", flush=True)
            return MemoryQualityReport(0, 0, 0, 0, 0, 0, 0.0)
        
        high_quality = 0
        medium_quality = 0
        low_quality = 0
        unused = 0
        deleted = 0
        total_score = 0.0
        
        for memory in memories:
            # 计算质量分数
            score = self._calculate_quality_score(memory.metadata)
            total_score += score
            
            # 分类
            if score >= 0.7:
                high_quality += 1
            elif score >= 0.4:
                medium_quality += 1
            else:
                low_quality += 1
            
            # 检查是否未使用
            if memory.metadata.get("used_count", 0) == 0:
                unused += 1
            
            # 清理逻辑
            if score < self.cleanup_threshold and memory.metadata.get("used_count", 0) == 0:
                # 从未使用且质量极低 → 删除
                memu.delete_memory(memory.id)
                deleted += 1
                print(f"    ✗ 删除低质量记忆: {memory.id} (score={score:.2f}, unused)", flush=True)
        
        avg_score = total_score / len(memories) if memories else 0.0
        
        return MemoryQualityReport(
            total_memories=len(memories),
            high_quality=high_quality,
            medium_quality=medium_quality,
            low_quality=low_quality,
            unused_memories=unused,
            deleted_count=deleted,
            avg_quality_score=avg_score
        )
    
    def _calculate_quality_score(self, metadata: dict) -> float:
        """计算记忆质量分数（0-1）"""
        # 1. 使用频率分数（归一化到 0-1）
        used_count = metadata.get("used_count", 0)
        usage_score = min(used_count / 5.0, 1.0)  # 使用 5 次视为满分
        
        # 2. 成功率分数
        total_uses = metadata.get("total_uses", 0)
        success_count = metadata.get("success_count", 0)
        success_rate = success_count / total_uses if total_uses > 0 else 0.5  # 未使用默认 0.5
        
        # 3. 时效性分数（新记忆分数更高）
        created_at = metadata.get("created_at", time.time())
        age_days = (time.time() - created_at) / 86400  # 转换为天数
        recency_score = max(0, 1.0 - age_days / 30.0)  # 30 天后衰减到 0
        
        # 加权平均
        quality_score = (
            0.4 * usage_score +
            0.4 * success_rate +
            0.2 * recency_score
        )
        
        return quality_score
    
    def _merge_duplicates(self, similarity_threshold: float = 0.85) -> None:
        """检测并合并相似记忆

        参数
        ----
        similarity_threshold : float
            相似度阈值（默认 0.95）
        """
        print(f"  [MemoryEvaluator] 🔍 检测重复记忆 (阈值={similarity_threshold})...", flush=True)

        merged_count = 0

        # 对三层记忆分别处理
        for layer_name, memu in [
            ("语义记忆", self.semantic_memory),
            ("程序记忆", self.procedural_memory.memu if self.procedural_memory else None),
            ("情景记忆", self.episodic_memory.memu if self.episodic_memory else None),
        ]:
            if not memu:
                continue

            memories = memu.get_all_memories()
            if len(memories) < 2:
                continue

            # 检测重复
            to_delete = set()

            for i, mem1 in enumerate(memories):
                if mem1.id in to_delete:
                    continue

                for mem2 in memories[i+1:]:
                    if mem2.id in to_delete:
                        continue

                    # 计算相似度（使用向量检索）
                    try:
                        similar = memu.search(query=mem1.content, top_k=5)

                        for s in similar:
                            if s.id == mem2.id:
                                similarity = 1 - s.distance

                                if similarity > similarity_threshold:
                                    # 合并：保留高质量的，删除低质量的
                                    score1 = self._calculate_quality_score(mem1.metadata)
                                    score2 = self._calculate_quality_score(mem2.metadata)

                                    if score1 >= score2:
                                        to_delete.add(mem2.id)
                                        print(f"    ✗ 合并重复: 删除 {mem2.id} (相似度={similarity:.3f}, 保留更高质量的 {mem1.id})", flush=True)
                                    else:
                                        to_delete.add(mem1.id)
                                        print(f"    ✗ 合并重复: 删除 {mem1.id} (相似度={similarity:.3f}, 保留更高质量的 {mem2.id})", flush=True)

                                    merged_count += 1
                                    break
                    except Exception:
                        continue

            # 执行删除
            for mem_id in to_delete:
                try:
                    memu.delete_memory(mem_id)
                except Exception:
                    pass

        if merged_count > 0:
            print(f"  [MemoryEvaluator] ✓ 合并了 {merged_count} 对重复记忆", flush=True)
        else:
            print(f"  [MemoryEvaluator] ✓ 未发现重复记忆", flush=True)

    def _print_summary_report(self, reports: dict[str, MemoryQualityReport]) -> None:
        """打印汇总报告"""
        print(f"\n{'='*70}", flush=True)
        print(f"[MemoryEvaluator] 📊 记忆质量评估报告", flush=True)
        print(f"{'='*70}\n", flush=True)

        for layer_name, report in reports.items():
            print(f"  {layer_name}:", flush=True)
            print(f"    总计: {report.total_memories} 条", flush=True)
            print(f"    高质量 (≥0.7): {report.high_quality} 条", flush=True)
            print(f"    中等质量 (0.4-0.7): {report.medium_quality} 条", flush=True)
            print(f"    低质量 (<0.4): {report.low_quality} 条", flush=True)
            print(f"    未使用: {report.unused_memories} 条", flush=True)
            print(f"    已删除: {report.deleted_count} 条", flush=True)
            print(f"    平均质量分: {report.avg_quality_score:.3f}", flush=True)
            print()

        print(f"{'='*70}\n", flush=True)
