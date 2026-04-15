"""ExecutionMonitor - 执行监控和智能刹车机制

核心功能
--------
1. **Early Stopping**: 检测死循环和无效重复
2. **Mid-Task Reflection**: 强制中途反思
3. **Timeout Protection**: 限制无效暴力破解

设计原则
--------
- 不让模型在错误的树上吊死
- 早发现，早中断，早反思
- 节省算力，提高效率
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecutionState:
    """单步执行状态"""
    step_num: int
    command: str
    output: str
    timestamp: float
    state_hash: str  # 用于检测重复


@dataclass
class MonitoringStats:
    """监控统计"""
    total_steps: int = 0
    repeated_actions: int = 0
    error_count: int = 0
    timeout_count: int = 0
    forced_stops: int = 0
    
    # 状态哈希频率统计
    state_hashes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # 连续错误计数
    consecutive_errors: int = 0


class ExecutionMonitor:
    """执行监控器
    
    负责：
    1. 检测死循环（状态哈希重复）
    2. 检测无效重复（连续相同操作）
    3. 触发中途反思
    4. 超时保护
    """
    
    def __init__(
        self,
        max_repeated_states: int = 3,  # 最多允许 3 次相同状态
        max_consecutive_errors: int = 3,  # 最多允许 3 次连续错误
        mid_task_reflection_threshold: float = 0.5,  # 在 50% 步数时触发反思
        code_timeout: int = 10,  # 代码执行超时（秒）
    ) -> None:
        self.max_repeated_states = max_repeated_states
        self.max_consecutive_errors = max_consecutive_errors
        self.mid_task_reflection_threshold = mid_task_reflection_threshold
        self.code_timeout = code_timeout
        
        # 执行历史
        self.execution_history: list[ExecutionState] = []
        self.stats = MonitoringStats()
    
    def reset(self) -> None:
        """重置监控状态（新题目开始时调用）"""
        self.execution_history.clear()
        self.stats = MonitoringStats()
    
    def record_step(
        self,
        step_num: int,
        command: str,
        output: str,
        is_error: bool = False,
    ) -> None:
        """记录一步执行"""
        # 计算状态哈希
        state_hash = self._compute_state_hash(command, output)
        
        # 记录执行状态
        state = ExecutionState(
            step_num=step_num,
            command=command,
            output=output,
            timestamp=time.time(),
            state_hash=state_hash
        )
        self.execution_history.append(state)
        
        # 更新统计
        self.stats.total_steps += 1
        self.stats.state_hashes[state_hash] += 1
        
        if is_error:
            self.stats.error_count += 1
            self.stats.consecutive_errors += 1
        else:
            self.stats.consecutive_errors = 0  # 成功则重置
    
    def should_early_stop(self) -> tuple[bool, str]:
        """检查是否应该提前终止
        
        返回
        ----
        (should_stop, reason)
        """
        # 检查 1: 状态重复过多（死循环）
        for state_hash, count in self.stats.state_hashes.items():
            if count >= self.max_repeated_states:
                return True, f"检测到死循环：相同状态重复 {count} 次"
        
        # 检查 2: 连续错误过多
        if self.stats.consecutive_errors >= self.max_consecutive_errors:
            return True, f"连续 {self.stats.consecutive_errors} 次错误，当前路径无效"
        
        # 检查 3: 检测"无意义的重复操作"
        if self._detect_meaningless_repetition():
            return True, "检测到无意义的重复操作"
        
        return False, ""
    
    def should_trigger_mid_reflection(
        self,
        current_step: int,
        step_limit: int
    ) -> bool:
        """检查是否应该触发中途反思
        
        在达到步数限制的 50% 时触发
        """
        threshold_step = int(step_limit * self.mid_task_reflection_threshold)
        
        # 只在恰好达到阈值时触发一次
        if current_step == threshold_step:
            return True
        
        return False
    
    def get_mid_reflection_prompt(self, current_step: int) -> str:
        """生成中途反思的 prompt"""
        recent_errors = self._get_recent_errors(n=3)
        
        prompt = f"""⚠️ CRITICAL CHECKPOINT (Step {current_step}) ⚠️

You have been working on this problem for {current_step} steps with limited progress.
STOP your current approach immediately and conduct a deep reflection:

1. Review the problem statement again - are you solving the RIGHT question?
2. Analyze your recent attempts:
{self._format_recent_errors(recent_errors)}

3. Identify the FUNDAMENTAL issue with your current path
4. Propose a COMPLETELY DIFFERENT approach

If you continue the same failing strategy, this attempt will be terminated.
Choose a new method NOW."""
        
        return prompt
    
    def get_execution_summary(self) -> dict:
        """获取执行摘要（供 AnalyzerAgent 分析）"""
        return {
            "total_steps": self.stats.total_steps,
            "repeated_actions": self.stats.repeated_actions,
            "error_count": self.stats.error_count,
            "consecutive_errors": self.stats.consecutive_errors,
            "forced_stops": self.stats.forced_stops,
            "unique_states": len(self.stats.state_hashes),
            "most_repeated_state_count": max(self.stats.state_hashes.values()) if self.stats.state_hashes else 0,
        }
    
    # ------------------------------------------------------------------ #
    # 内部工具方法
    # ------------------------------------------------------------------ #
    
    def _compute_state_hash(self, command: str, output: str) -> str:
        """计算状态哈希（用于检测重复）
        
        只取命令和输出的核心部分，忽略时间戳等动态信息
        """
        # 简化命令（去除空格、换行）
        cmd_normalized = command.strip().lower()
        
        # 简化输出（只取前 200 字符，忽略具体数值）
        out_normalized = output[:200].strip().lower()
        
        # 计算哈希
        content = f"{cmd_normalized}||{out_normalized}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _detect_meaningless_repetition(self) -> bool:
        """检测无意义的重复操作
        
        例如：连续 3 次执行相同的代码但都失败
        """
        if len(self.execution_history) < 3:
            return False
        
        # 检查最近 3 步
        recent = self.execution_history[-3:]
        
        # 如果最近 3 步的命令完全相同
        commands = [s.command.strip() for s in recent]
        if len(set(commands)) == 1:
            # 并且输出都包含错误信息
            outputs = [s.output.lower() for s in recent]
            error_keywords = ["error", "traceback", "exception", "failed"]
            
            if all(any(kw in out for kw in error_keywords) for out in outputs):
                return True
        
        return False
    
    def _get_recent_errors(self, n: int = 3) -> list[ExecutionState]:
        """获取最近的 n 个错误步骤"""
        errors = [
            state for state in self.execution_history
            if any(kw in state.output.lower() for kw in ["error", "traceback", "exception"])
        ]
        return errors[-n:] if errors else []
    
    def _format_recent_errors(self, errors: list[ExecutionState]) -> str:
        """格式化最近的错误"""
        if not errors:
            return "   No recent errors (but no progress either)"
        
        lines = []
        for err in errors:
            lines.append(f"   Step {err.step_num}: {err.command[:60]}...")
            lines.append(f"   → Error: {err.output[:100]}...")
        
        return "\n".join(lines)
