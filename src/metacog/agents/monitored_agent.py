"""MonitoredAgent - 带监控的 Agent 包装器

核心思路
--------
不修改 mini-swe-agent 的代码，而是通过包装器在每一步执行后插入监控逻辑：
1. Early Stopping: 检测死循环
2. Mid-Task Reflection: 强制中途反思
3. Timeout Protection: 超时保护

使用方法
--------
from .monitored_agent import create_monitored_agent

agent = create_monitored_agent(
    base_agent_class=DefaultAgent,
    monitor=execution_monitor,
    step_limit=10
)
"""

from __future__ import annotations

from typing import Any, Type

from minisweagent.agents.default import DefaultAgent


def create_monitored_agent(
    base_agent: DefaultAgent,
    monitor: Any,  # ExecutionMonitor
    step_limit: int = 10,
) -> DefaultAgent:
    """创建带监控的 Agent
    
    参数
    ----
    base_agent : DefaultAgent
        原始的 agent 实例
    monitor : ExecutionMonitor
        监控器实例
    step_limit : int
        步数限制
        
    返回
    ----
    monitored_agent : DefaultAgent
        包装后的 agent（接口不变）
    """
    if monitor is None:
        return base_agent  # 如果没有监控器，直接返回原始 agent

    # 保存原始的 step 和 run 方法
    original_step = base_agent.step
    original_run = base_agent.run

    # 当前步数计数器
    step_counter = {"count": 0, "mid_reflection_triggered": False}

    def monitored_run(*args, **kwargs):
        """包装后的 run 方法 - 在每次运行开始时重置计数器"""
        step_counter["count"] = 0
        step_counter["mid_reflection_triggered"] = False
        monitor.reset()  # 重置监控器状态
        return original_run(*args, **kwargs)

    def monitored_step(*args, **kwargs):
        """包装后的 step 方法"""
        step_counter["count"] += 1
        current_step = step_counter["count"]
        
        # ==========================================
        # 干预点 1: Early Stopping（提前检测）
        # ==========================================
        should_stop, reason = monitor.should_early_stop()
        if should_stop:
            print(f"\n🛑 [Monitor] Early Stop triggered at step {current_step}", flush=True)
            print(f"   Reason: {reason}", flush=True)
            monitor.stats.forced_stops += 1
            
            # 返回一个终止信号
            return {
                "observation": f"[FORCED STOP] {reason}",
                "done": True,
                "info": {"early_stop": True, "reason": reason}
            }
        
        # ==========================================
        # 干预点 2: Mid-Task Reflection（中途反思）
        # ==========================================
        if (not step_counter["mid_reflection_triggered"] and 
            monitor.should_trigger_mid_reflection(current_step, step_limit)):
            
            print(f"\n⚠️  [Monitor] Mid-Task Reflection triggered at step {current_step}/{step_limit}", flush=True)
            
            # 生成反思 prompt
            reflection_prompt = monitor.get_mid_reflection_prompt(current_step)
            
            # 将反思 prompt 注入到下一次对话中
            # 这里我们通过修改 observation 来实现
            step_counter["mid_reflection_triggered"] = True
            
            # 先执行一次正常 step，然后添加反思提示
            result = original_step(*args, **kwargs)
            
            # 在 observation 中注入反思 prompt
            if isinstance(result, dict) and "observation" in result:
                result["observation"] = f"{reflection_prompt}\n\n{result['observation']}"
            
            return result
        
        # ==========================================
        # 正常执行
        # ==========================================
        result = original_step(*args, **kwargs)
        
        # ==========================================
        # 记录执行状态
        # ==========================================
        # 提取命令和输出（根据 mini-swe-agent 的实际结构）
        command = kwargs.get("action", "") or args[0] if args else ""
        output = result.get("observation", "") if isinstance(result, dict) else str(result)
        
        # 检测是否是错误
        is_error = False
        if isinstance(output, str):
            error_keywords = ["error", "traceback", "exception", "failed", "syntax error"]
            is_error = any(kw in output.lower() for kw in error_keywords)
        
        # 记录到监控器
        monitor.record_step(
            step_num=current_step,
            command=str(command)[:200],  # 限制长度
            output=str(output)[:500],  # 限制长度
            is_error=is_error
        )
        
        return result

    # 替换 step 和 run 方法
    base_agent.step = monitored_step
    base_agent.run = monitored_run

    return base_agent


def add_code_timeout_protection(
    code_str: str,
    timeout_seconds: int = 10
) -> str:
    """为 Python 代码添加超时保护
    
    参数
    ----
    code_str : str
        原始 Python 代码
    timeout_seconds : int
        超时时间（秒）
        
    返回
    ----
    protected_code : str
        包装了超时保护的代码
    """
    protected = f"""
import signal
import sys

def timeout_handler(signum, frame):
    print(f"⏱️ TIMEOUT: Code execution exceeded {timeout_seconds} seconds", file=sys.stderr)
    print("This usually indicates an infinite loop or inefficient algorithm.", file=sys.stderr)
    print("Consider using more efficient mathematical methods.", file=sys.stderr)
    sys.exit(124)  # Timeout exit code

# Set timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout_seconds})

try:
    # Original code
{_indent_code(code_str, 4)}
finally:
    signal.alarm(0)  # Cancel the alarm
"""
    return protected


def _indent_code(code: str, spaces: int) -> str:
    """缩进代码"""
    indent = " " * spaces
    return "\n".join(indent + line if line.strip() else line 
                     for line in code.split("\n"))
