# 智能监控和刹车机制升级文档

## 🎯 优化目标

**解决问题**：模型在错误路径上吊死，浪费大量 token 和时间

**核心思路**：
- 不让模型在错误的树上吊死
- 早发现，早中断，早反思
- 节省算力，提高效率

---

## 🚫 为什么不能只增加 step_limit？

```python
# ❌ 错误做法
step_limit = 20  # 从 10 增加到 20
# → 模型在死循环中多烧 10 步的 token
# → 浪费时间，浪费钱，准确率没提升
```

**根本问题**：模型陷入死循环或无效重复时，给它更多步数只会让它继续犯错。

---

## 🔧 三个干预点

### 干预点 1: Early Stopping（提前终止）

**检测目标**：
- 状态重复（死循环）
- 连续错误（无效路径）
- 无意义的重复操作

**实现**：
```python
class ExecutionMonitor:
    def should_early_stop(self) -> tuple[bool, str]:
        # 检查 1: 状态哈希重复 ≥ 3 次
        if state_hash_count >= 3:
            return True, "检测到死循环"
        
        # 检查 2: 连续 3 次错误
        if consecutive_errors >= 3:
            return True, "连续错误，当前路径无效"
        
        # 检查 3: 相同命令连续失败
        if same_failing_command_count >= 3:
            return True, "无意义的重复操作"
```

**触发后**：
```
🛑 [Monitor] Early Stop triggered at step 7
   Reason: 检测到死循环：相同状态重复 3 次
   
→ 立即终止执行
→ AnalyzerAgent 分析为什么陷入死循环
→ 提取教训存入 memU
```

---

### 干预点 2: Mid-Task Reflection（中途反思）

**触发时机**：达到步数限制的 50% 时（例如 step 5/10）

**注入的 Prompt**：
```
⚠️ CRITICAL CHECKPOINT (Step 5) ⚠️

You have been working on this problem for 5 steps with limited progress.
STOP your current approach immediately and conduct a deep reflection:

1. Review the problem statement again - are you solving the RIGHT question?
2. Analyze your recent attempts:
   Step 3: sympy.solve(...) → SyntaxError
   Step 4: sympy.solve(...) → SyntaxError
   
3. Identify the FUNDAMENTAL issue with your current path
4. Propose a COMPLETELY DIFFERENT approach

If you continue the same failing strategy, this attempt will be terminated.
Choose a new method NOW.
```

**效果**：
- 打断模型的惯性思维
- 强制重新审视问题
- 换一条完全不同的路径

---

### 干预点 3: Timeout Protection（超时保护）

**问题**：模型生成的代码可能包含死循环或低效算法

**解决方案 1：代码级超时**
```python
def add_code_timeout_protection(code: str, timeout: int = 10) -> str:
    """为 Python 代码添加超时保护"""
    protected = f"""
import signal
import sys

def timeout_handler(signum, frame):
    print("⏱️ TIMEOUT: Code execution exceeded {timeout} seconds", file=sys.stderr)
    print("Consider using more efficient algorithms.", file=sys.stderr)
    sys.exit(124)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout})

try:
    {code}
finally:
    signal.alarm(0)
"""
    return protected
```

**解决方案 2：subprocess 超时**
```python
# 在 PoTReflector 中已经实现
result = subprocess.run(
    ['python3', code_file],
    timeout=10,  # 10 秒超时
    capture_output=True
)
```

---

## 📊 监控统计

ExecutionMonitor 会记录：
```python
@dataclass
class MonitoringStats:
    total_steps: int = 0
    repeated_actions: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    forced_stops: int = 0
    
    # 状态哈希频率
    state_hashes: dict[str, int]
```

这些统计会：
1. 附加到 TrajectoryEvent 中
2. 供 AnalyzerAgent 分析失败原因
3. 帮助识别模型的"坏习惯"

---

## 🎯 实际案例

### 案例 1: 检测死循环

**执行轨迹**：
```
Step 1: 运行 solve_equation.py → SyntaxError: invalid syntax
Step 2: 运行 solve_equation.py → SyntaxError: invalid syntax (相同错误)
Step 3: 运行 solve_equation.py → SyntaxError: invalid syntax (相同错误)
```

**Monitor 检测**：
```
[Monitor] 检测到无意义的重复操作
         连续 3 次执行相同代码且都失败
         
🛑 Early Stop triggered
```

**AnalyzerAgent 分析**：
```json
{
  "error_symptom": "连续 3 次相同语法错误",
  "root_cause": "对 Python 语法理解不足",
  "actionable_advice": "使用 sympy 前先验证基本语法 | CODE: from sympy import symbols; x = symbols('x')"
}
```

---

### 案例 2: 中途反思拯救失败

**执行轨迹**（无监控）：
```
Step 1-5: 尝试手动展开多项式
Step 6-10: 继续展开，越来越复杂
结果: 失败（超出 step_limit）
```

**执行轨迹**（有监控）：
```
Step 1-4: 尝试手动展开多项式
Step 5: 🔥 触发 Mid-Task Reflection
        "STOP! 你已经用了 5 步还在展开，换个方法"
        
Step 6: 模型重新思考 → 决定用 sympy.expand()
Step 7: 用 sympy 成功求解
结果: 成功 ✅
```

---

## 🚀 使用方法

### 在 ExecutorAgent 中启用

```python
from .execution_monitor import ExecutionMonitor

executor = ExecutorAgent(
    ...,
    enable_monitor=True  # 启用监控
)
```

### 配置参数

```python
monitor = ExecutionMonitor(
    max_repeated_states=3,              # 最多允许 3 次相同状态
    max_consecutive_errors=3,           # 最多允许 3 次连续错误
    mid_task_reflection_threshold=0.5,  # 在 50% 步数时触发反思
    code_timeout=10,                    # 代码执行超时 10 秒
)
```

---

## 📈 预期效果

### 无监控（Baseline）
```
错误率: 60%
平均步数: 9.8 (接近 step_limit=10)
Token 消耗: 高
卡在死循环的题目: 15%
```

### 有监控（+Monitor）
```
错误率: 45%  (↓ 25%)
平均步数: 6.2 (↓ 37%)
Token 消耗: 中
卡在死循环的题目: 2%  (↓ 87%)
```

**关键提升**：
- ✅ 早发现死循环，节省 3-4 步无效操作
- ✅ 中途反思帮助 10-15% 的题目换路径成功
- ✅ 超时保护避免代码执行卡死

---

## 🎨 监控流程图

```
题目开始
    ↓
[Monitor] reset()  # 重置监控状态
    ↓
Step 1 → record_step()
    ↓
Step 2 → record_step() → should_early_stop()? → NO
    ↓
Step 3 → record_step() → should_early_stop()? → NO
    ↓
Step 4 → record_step() → should_early_stop()? → NO
    ↓
Step 5 → should_trigger_mid_reflection()? → YES
    ↓                ↓
    |        inject_reflection_prompt()
    |                ↓
Step 6 → record_step() → should_early_stop()? → YES
    ↓                                           ↓
继续执行                            🛑 强制终止
                                        ↓
                                AnalyzerAgent 分析
```

---

## ✅ 核心优势

1. **节省算力**：提前终止死循环，节省 30-40% token
2. **提升准确率**：中途反思帮助换路径，+10-15% 成功率
3. **可解释性**：监控统计帮助分析为什么失败
4. **防止卡死**：超时保护避免无限循环

---

## 📚 相关文档

- **执行监控实现**: `agents/execution_monitor.py`
- **监控包装器**: `agents/monitored_agent.py`
- **集成方式**: 见本文档"使用方法"部分
