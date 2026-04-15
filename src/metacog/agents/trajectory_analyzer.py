"""TrajectoryAnalyzer - 轨迹后验分析器

事后分析轨迹，识别死循环和无效重复模式。
虽然不能实时阻止（那需要修改 mini-swe-agent 内部），但可以：
1. 在 AnalyzerAgent 中标记这些模式
2. 将"避免死循环"的教训存入 memU
3. 下次遇到类似情况时提前警告
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional


@dataclass
class LoopPattern:
    """检测到的循环模式"""
    start_step: int
    end_step: int
    loop_count: int
    action_signature: str
    description: str


class TrajectoryAnalyzer:
    """轨迹后验分析器
    
    分析已完成的轨迹，识别死循环和反模式
    """
    
    def __init__(self) -> None:
        pass
    
    def analyze_trajectory_for_loops(
        self,
        steps: list
    ) -> Optional[LoopPattern]:
        """分析轨迹中的死循环模式

        参数
        ----
        steps : list
            轨迹步骤列表，可能是 dict 或 _Step 对象

        返回
        ----
        LoopPattern | None
            如果检测到明显的循环，返回模式描述
        """
        if len(steps) < 3:
            return None  # 至少需要 3 步才能检测连续重复

        # 计算每步的签名（command + output 的哈希）
        signatures = []
        for step in steps:
            sig = self._compute_step_signature(step)
            signatures.append(sig)

        # 检测连续重复
        consecutive_repeats = self._find_consecutive_repeats(signatures)
        if consecutive_repeats:
            return consecutive_repeats

        # 检测交替模式（A-B-A-B-A-B）
        alternating_pattern = self._find_alternating_pattern(signatures, steps)
        if alternating_pattern:
            return alternating_pattern

        return None
    
    def detect_inefficient_approach(
        self,
        steps: list,
        step_limit: int
    ) -> Optional[str]:
        """检测低效的解题方法

        判断标准：检测 Python 代码内容本身是否重复
        - 提取每步实际执行的 Python 代码
        - 归一化（去注释、去数字、去空白）后计算哈希
        - 若 60% 以上的代码块哈希相同 → 逻辑高度重复，没有实质改进
        """
        if len(steps) < step_limit * 0.7:
            return None  # 步数不够多，不判断

        # 提取每步的 Python 代码签名
        code_sigs = []
        for step in steps:
            cmd = getattr(step, 'command', '') if hasattr(step, 'command') else step.get('command', '')
            code = self._extract_python_code(cmd)
            if code:
                sig = self._normalize_code_signature(code)
                code_sigs.append(sig)

        if len(code_sigs) < 3:
            return None  # 代码块太少，无法判断

        # 检查最常见的代码签名占比
        counts = Counter(code_sigs)
        most_common_sig, most_common_count = counts.most_common(1)[0]

        if most_common_count / len(code_sigs) > 0.6:
            return (f"低效方法：{most_common_count}/{len(code_sigs)} 个代码块"
                    f"结构高度相似（归一化后相同），未实质性改进解题逻辑")

        return None
    
    # ------------------------------------------------------------------ #
    # 内部工具方法
    # ------------------------------------------------------------------ #
    
    def _compute_step_signature(self, step) -> str:
        """计算步骤签名（用于检测重复）

        参数可以是：
        - dict: {'action': ..., 'observation': ...}
        - _Step: dataclass with .command and .output
        """
        # 兼容处理不同的步骤格式
        if hasattr(step, 'command'):  # _Step 对象
            command = getattr(step, 'command', '').strip().lower()
            output = getattr(step, 'output', '')[:200].strip().lower()
        elif isinstance(step, dict):  # 字典
            command = step.get("action", step.get("command", "")).strip().lower()
            output = step.get("observation", step.get("output", ""))[:200].strip().lower()
        else:
            # 其他类型，尝试转字符串
            command = str(step).lower()
            output = ""

        # 简化：去除数字、空格
        import re
        command = re.sub(r'\d+', 'N', command)
        output = re.sub(r'\d+', 'N', output)

        content = f"{command}::{output}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _find_consecutive_repeats(
        self,
        signatures: list[str]
    ) -> Optional[LoopPattern]:
        """查找连续重复的模式"""
        if len(signatures) < 3:
            return None
        
        # 检查最近 3-5 步是否重复
        for window_size in [3, 4, 5]:
            if len(signatures) < window_size:
                continue
            
            recent = signatures[-window_size:]
            if len(set(recent)) == 1:  # 全部相同
                return LoopPattern(
                    start_step=len(signatures) - window_size + 1,
                    end_step=len(signatures),
                    loop_count=window_size,
                    action_signature=recent[0],
                    description=f"连续 {window_size} 步执行相同操作"
                )
        
        return None
    
    def _find_alternating_pattern(
        self,
        signatures: list[str],
        steps: list[dict]
    ) -> Optional[LoopPattern]:
        """查找交替模式（A-B-A-B-A-B）"""
        if len(signatures) < 4:
            return None
        
        # 检查最近 4-6 步是否呈现 A-B 交替
        recent_sigs = signatures[-6:]
        
        if len(recent_sigs) >= 4:
            # 检查是否是 A-B-A-B 模式
            if (len(set(recent_sigs[::2])) == 1 and  # 偶数位相同
                len(set(recent_sigs[1::2])) == 1 and  # 奇数位相同
                recent_sigs[0] != recent_sigs[1]):    # 两者不同
                
                return LoopPattern(
                    start_step=len(signatures) - len(recent_sigs) + 1,
                    end_step=len(signatures),
                    loop_count=len(recent_sigs) // 2,
                    action_signature=f"{recent_sigs[0]}<->{recent_sigs[1]}",
                    description=f"检测到 A-B 交替模式，重复 {len(recent_sigs)//2} 次"
                )
        
        return None
    
    def _extract_python_code(self, cmd: str) -> str:
        """从 shell 命令中提取实际的 Python 代码内容

        兼容两种格式：
        1. heredoc: python3 << 'EOF'\n...code...\nEOF（有或无结束符）
        2. python3 -c '...'
        """
        # heredoc 格式（有无 EOF 结尾都兼容）
        m = re.search(
            r"python3\s*<<\s*['\"]?EOF['\"]?\s*\n(.*?)(?:\nEOF\s*$|\Z)",
            cmd, re.DOTALL
        )
        if m:
            return m.group(1).strip()
        # python3 -c 格式
        m = re.search(r"python3\s+-c\s+['\"](.+?)['\"]", cmd, re.DOTALL)
        if m:
            return m.group(1).strip()
        return ""

    def _normalize_code_signature(self, code: str) -> str:
        """归一化 Python 代码并计算哈希，用于判断逻辑是否重复

        归一化规则：
        1. 去掉注释（# 行）
        2. 把所有数字替换为 N（不同参数视为同一逻辑）
        3. 把字符串字面量替换为 STR
        4. 合并连续空白
        5. 转小写后取 MD5 前 8 位
        """
        code = re.sub(r'#.*', '', code)                     # 去注释
        code = re.sub(r'["\'].*?["\']', 'STR', code)       # 去字符串
        code = re.sub(r'\b\d+\.?\d*\b', 'N', code)         # 数字 → N
        code = re.sub(r'\s+', ' ', code).strip().lower()    # 压缩空白 + 小写
        return hashlib.md5(code.encode()).hexdigest()[:8]
