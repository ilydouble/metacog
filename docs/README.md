# MetaCog — 数学解题智能体进化框架

> 一个研究性框架，探索如何让 LLM 智能体在批量解数学题的过程中**自我进化**：从静态 prompt 到动态记忆，再到多智能体协同生成可复用技能。

## 目录

- [项目概述](#项目概述)
- [方法一：基线（Baseline）](#方法一基线-baseline)
- [方法二：Evolve — 元智能体 Prompt 进化](#方法二evolve--元智能体-prompt-进化)
- [方法三：ReCreate — 题目级 ReCreate-Agent](#方法三recreate--题目级-recreate-agent)
- [方法四：Metacog — 多智能体事件驱动系统](#方法四metacog--多智能体事件驱动系统)
- [数据集](#数据集)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [方法对比总结](#方法对比总结)

---

## 项目概述

本项目在 **AIME 24 / AIME 25 / AMC 23** 三个竞赛数学数据集上，实验了四种不同程度的「元认知」策略：

| 方法 | 脚本 | 核心思路 | 学习时机 |
|------|------|----------|----------|
| **Baseline** | `run_math_test.py` | 固定 prompt，无学习 | 无 |
| **Evolve** | `run_math_test_evolve.py` | 元智能体每轮后优化 scaffold | 每轮结束后 |
| **ReCreate** | `run_math_test_recreate.py` | ReCreate-Agent 每题后分析 + 创建工具/记忆 + 批次合成 | 每题后 + 每批合成 |
| **Metacog** | `run_math_test_metacog.py` | 多智能体事件驱动，三层 memU 向量记忆 + 技能双通道 + 双模型架构 | 每题后实时 |

所有方法共用同一套底层 Solver（[mini-swe-agent](src/mini-swe-agent)），通过 bash tool call 调用 Python 进行计算。

---

## 方法一：基线（Baseline）

**脚本**：`scripts/run_math_test.py`

### 核心设计

最简单的实现：固定 system prompt + instance template，对每道题独立运行一个 `DefaultAgent`，题目之间没有任何信息传递。

```
数据集
  └── 题目 1 → DefaultAgent(固定 prompt) → 答案 → 记录结果
  └── 题目 2 → DefaultAgent(固定 prompt) → 答案 → 记录结果
  └── ...
```

### 解题流程

1. Agent 读取题目，进入 bash tool call 循环
2. 每步调用 LLM → 生成 bash 命令 → 执行 → 观察输出
3. 当 stdout 出现 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 时提交答案
4. 与标准答案比对，记录 pass/fail

### 关键参数

```bash
python scripts/run_math_test.py \
  --config scripts/configs/math_test_config.yaml \  # 模型配置（默认值已内置）
  --data-source aime24 \                            # 数据集：aime24 / aime25 / amc23
  --max-instances 30 \                              # 最多跑几道题
  --output outputs/baseline_aime24
```

### 输出结构

```
outputs/baseline/
  results.jsonl        # 每题的 passed/answer/expected 等
  trajectories/        # 每题的完整对话轨迹 .traj.json
```

---

## 方法二：Evolve — 元智能体 Prompt 进化

**脚本**：`scripts/run_math_test_evolve.py`

### 核心设计

引入**元智能体**（Meta-Agent）：每轮（Round）跑完一批题后，把失败案例喂给同一个 LLM，让它分析失败模式并输出改进后的 `system_template + instance_template`，作为下一轮的 scaffold。

```
Round 0: scaffold_v000 → 跑 N 题 → 收集失败案例
           ↓
      Meta-Agent 分析失败 → 输出改进后的 YAML
           ↓
Round 1: scaffold_v001 → 跑 N 题 → 收集失败案例
           ↓
      Meta-Agent 分析失败 → 输出改进后的 YAML
           ↓
Round 2: scaffold_v002 → ...
```

### 详细流程

1. **初始化**：将 `INITIAL_SYSTEM_TEMPLATE` + `INITIAL_INSTANCE_TEMPLATE` 写入 `evolve_workspace/global_v000/scaffold.yaml`
2. **解题阶段**：用当前 scaffold 对每道题运行 `DefaultAgent`，收集轨迹和结果
3. **进化阶段**：
   - 提取所有失败题目的解题步骤摘要
   - 将摘要 + 当前 scaffold 一起喂给 Meta-Agent（同一个 LLM）
   - Meta-Agent 输出包含 `system_template` 和 `instance_template` 字段的 YAML
4. **版本管理**：新 scaffold 存入 `global_v001/`，更新 `current` 软链接
5. **多轮迭代**：重复步骤 2-4，记录每轮 pass rate 变化

### 关键参数

```bash
python scripts/run_math_test_evolve.py \
  --config scripts/configs/math_test_config.yaml \  # 学生模型配置
  --data-source aime24 \
  --max-instances 10 \
  --max-rounds 3 \                                  # 进化轮数
  --output outputs/math_evolve \
  --teacher-model zai/glm-4.7 \                    # 教师模型（元智能体，可选）
  --teacher-api-key YOUR_ZAI_API_KEY
```

### 输出结构

```
outputs/math_test_evolve_aime24/
  evolve_workspace/
    global_v000/scaffold.yaml    # 初始版本
    global_v001/scaffold.yaml    # 第 1 轮优化后
    current -> global_vXXX       # 软链接，指向最新版本
  round_000/                     # 第 0 轮（使用 global_v000）
    results.jsonl
    trajectories/
  round_001/                     # 第 1 轮（使用 global_v001）
    ...
  evolution_log.json             # 每轮 pass rate 对比
```

### 优势与局限

| 优势 | 局限 |
|------|------|
| 实现简单，只需一个 LLM | 每轮结束才学习，实时性差 |
| 可改进任意 prompt 字段 | 改进对象仅为文字 prompt |
| 无需额外组件 | 依赖 Meta-Agent 能写出好 prompt |

---

## 方法三：ReCreate — 题目级 ReCreate-Agent

**脚本**：`scripts/run_math_test_recreate.py`

### 核心设计

在 Evolve 的基础上引入**题目级 ReCreate-Agent**：每道题跑完后，立即启动一个独立的 ReCreate-Agent 来分析该题的轨迹；除了修改 scaffold 文字外，还能**创建新工具**和**写入结构化记忆**。一批结束后由合成元智能体（Synthesis Agent）整合所有建议，生成新的全局 scaffold + 工具库 + 记忆库。

```
Batch 0（使用 global_v000）:
  题目 1 → Solver → 轨迹 → ReCreate-Agent → scaffold_diff.txt + 工具/记忆更新
  题目 2 → Solver → 轨迹 → ReCreate-Agent → scaffold_diff.txt + 工具/记忆更新
  题目 3 → Solver → 轨迹 → ReCreate-Agent → scaffold_diff.txt + 工具/记忆更新
           ↓
      Synthesis Agent 合并所有 diff、工具、记忆 → global_v001
           ↓
Batch 1（使用 global_v001，Solver 已可使用新工具和新记忆）:
  ...
```

### 详细流程

1. **初始化工作区**：`recreate_workspace/global_v000/` 包含 `scaffold.yaml`、`agent_tools/`、`agent_memory/`
2. **题目级分析**：每道题完成后，ReCreate-Agent 在独立沙箱中：
   - 读取 Solver 轨迹（`.traj.json`），生成摘要
   - 分析失败/成功原因
   - 调用工具脚本向 `agent_memory/` 写入结构化记忆（供下一批 Solver 检索）
   - 可在 `agent_tools/` 创建新的可复用工具（供下一批 Solver 调用）
   - 调用 `scaffold_editor.py` 修改 scaffold 字段，输出改动建议
3. **批次合成**：一批题全部完成后：
   - Synthesis Agent 汇总所有题目的 scaffold diff
   - 合并各题新增的工具和记忆
   - 生成统一的新全局版本（`global_v001/scaffold.yaml` + `agent_tools/` + `agent_memory/`）
   - 更新 `current` 软链接
4. **隔离执行**：每道题的 ReCreate-Agent 在独立目录中运行，互不干扰；下一批 Solver 统一使用合并后的全局版本

### 记忆系统（agent_memory）

每个全局版本（`global_vXXX/`）都包含一个 `agent_memory/` 目录，内含三个文件：

| 文件 | 说明 |
|------|------|
| `memories.yaml` | 结构化记忆库，每条记忆含 `title / content / tags / created` 字段 |
| `search_memory.py` | Solver 在解题时调用，按关键词检索相关记忆 |
| `write_memory.py` | Solver 在解题后调用，将本题所学写入记忆库 |

记忆系统有**两个控制层次**，均由 ReCreate-Agent 来管理：

- **Level 1 — 静态记忆内容**：ReCreate-Agent 分析轨迹后，通过 `tools/memory_manager.py add` 向 `agent_memory/memories.yaml` 直接写入教训，Solver 下次解题时可检索到。
- **Level 2 — 记忆使用策略**：ReCreate-Agent 通过 `tools/scaffold_editor.py` 修改 scaffold 的 `memory_template` 字段，定义 Solver **何时**读写记忆（如"遇到任何错误都先搜索记忆"）。

### 工具库（agent_tools）

每个全局版本包含一个 `agent_tools/` 目录，ReCreate-Agent 可在此创建**可复用的 Python 脚本工具**，供后续批次的 Solver 直接调用：

```
agent_tools/
  math/                          # 数学类工具（按领域分类）
    modular_arithmetic/
      main.py                    # 模运算辅助函数
    combinatorics/
      main.py                    # 组合数学计算工具
  debugging/
    step_checker/
      main.py                    # 分步验证中间结果
```

创建工具后，ReCreate-Agent 还需更新 scaffold 的 `system_template`，告知 Solver 这些工具的存在及调用方式。工具在批次合成时会被合并进新全局版本，持续积累。

### ReCreate-Agent 的工具集

ReCreate-Agent 本身是一个 bash Agent，通过命令行调用以下**元操作工具**：

| 工具脚本 | 功能 |
|---------|------|
| `tools/read_trajectory.py summary` | 生成 Solver 轨迹摘要（含失败步骤分析） |
| `tools/scaffold_editor.py str_replace` | 精确修改 scaffold 的任意字段 |
| `tools/scaffold_editor.py view` | 查看当前 scaffold 内容 |
| `tools/memory_manager.py add` | 向 `agent_memory/memories.yaml` 添加结构化记忆 |
| `tools/memory_manager.py list/search` | 查看/检索已有记忆 |

### 与 Evolve 的对比

| 维度 | Evolve | ReCreate |
|------|--------|---------|
| **分析粒度** | 批次级（一轮所有题失败后统一分析） | 题目级（每道题跑完立即分析） |
| **进化对象** | 仅 prompt 文字（`system_template` / `instance_template`） | Prompt + 工具库 + 记忆库（三者同步进化） |
| **工具创建** | ✗ 不支持 | ✓ 可生成新的可执行 Python 工具 |
| **记忆系统** | ✗ 无 | ✓ 双层记忆（静态内容 + 使用策略均可调整） |
| **改进时机** | 每轮结束后（延迟较大） | 每题结束后（实时性更高） |
| **架构复杂度** | 低（单 Meta-Agent 分析批次） | 中（每题一个 ReCreate-Agent + 批次 Synthesis Agent） |
| **合成方式** | Meta-Agent 直接输出新 YAML | Synthesis Agent 整合多个 diff + 工具 + 记忆 |

**核心区别**：Evolve 只是在每轮后"重写 prompt"，ReCreate 则让每题的 ReCreate-Agent 真正"扩展 Solver 的能力边界"——写入可检索的知识（记忆）、创建可复用的代码（工具），并改进 Solver 使用这些能力的策略（scaffold）。

### 关键参数

```bash
python scripts/run_math_test_recreate.py \
  --config scripts/configs/math_test_config.yaml \  # 学生模型配置
  --data-source aime24 \
  --max-instances 10 \
  --max-rounds 2 \                                  # 进化批次数
  --output outputs/math_recreate \
  --teacher-model zai/glm-4.7 \                    # ReCreate-Agent + Synthesis 使用（可选）
  --teacher-api-key YOUR_ZAI_API_KEY
```

### 输出结构

```
outputs/math_test_recreate_aime24/
  recreate_workspace/
    global_v000/
      scaffold.yaml              # 初始版本
      agent_tools/               # 初始工具库（空）
      agent_memory/
        memories.yaml            # 记忆库（初始为空）
        search_memory.py         # Solver 用于检索记忆
        write_memory.py          # Solver 用于写入记忆
    global_v001/
      scaffold.yaml              # 第 0 批合成后的 prompt
      agent_tools/               # 积累的可复用工具
      agent_memory/memories.yaml # 积累的结构化记忆
    current -> global_vXXX       # 软链接，指向最新版本
  batch_000/
    aime24_0001/
      scaffold.yaml              # 拷贝自 global_v000
      agent_tools/               # 题目级工具（ReCreate-Agent 新增）
      agent_memory/memories.yaml # 题目级记忆（ReCreate-Agent 写入）
      aime24_0001.traj.json      # Solver 解题轨迹
      scaffold_diff.txt          # ReCreate-Agent 的 scaffold 改动建议
  runs_recreate/                 # ReCreate-Agent 自身的对话轨迹
  evolution_log.json
```

### 优势与局限

| 优势 | 局限 |
|------|------|
| 题目级实时分析，改进时机更早 | 架构复杂，每题多一个 Agent 执行过程 |
| Prompt + 工具 + 记忆三维度同时进化 | 工具创建质量依赖 LLM 的代码能力 |
| 积累的工具可被 Solver 直接执行 | 本地小模型（如 Qwen-9B）格式稳定性需调优 |
| 记忆结构化存储，可跨批次复用 | 合成阶段需协调多个来源的 diff |

每道题结束
  └── ReCreate-Agent 分析轨迹
        ├── (几乎必做) 创建新工具 → agent_tools/<category>/<name>/main.py
        ├── (按需) 写入记忆     → agent_memory/memories.yaml
        ├── (按需) 改记忆策略   → scaffold 的 memory_template 字段
        └── (最后) 改提示词     → system_template / instance_template
              ↓
      以上改动暂存在该题目录，等批次结束后由 Synthesis Agent 合并进 global_vXXX
---

## 方法四：Metacog — 多智能体事件驱动系统

**脚本**：`scripts/run_math_test_metacog.py`

### 核心设计

最复杂的方案：用**事件总线（EventBus）**连接多个专职智能体，形成**两条并行的学习通道**，并引入**三层记忆体系**和**双模型架构**：

- **失败通道**：失败 → `FailureRouter` 过滤操作性失误 → `AnalyzerAgent` 蒸馏教训 → `MemoryManagerAgent` 写入 memU 向量库 → 下题语义检索注入
- **成功通道**：成功 → `AnalyzerAgent` 提取技术模式 → `SkillAgent` 积累生成 Python skill → 下题可直接 `import`；同时 → `SuccessAnalyzer` 存入情景记忆供类比学习

```
每道题结束
  ├── passed=False
  │     → FailureRouter（阻断操作性失误：SyntaxError/ImportError 等）
  │     → AnalyzerAgent（教师模型复盘）
  │           → 死循环检测 → PoT 验证 → 蒸馏结构化教训
  │           → ANALYSIS 事件
  │                 → MemoryManagerAgent → 写入 memU 语义记忆层
  │                       → MEMORY_UPDATED 事件
  │
  └── passed=True
        → AnalyzerAgent（成功分析）
        │     → 提取 technique + tags
        │     → SUCCESS_ANALYSIS 事件
        │           → SkillAgent
        │                 → 按 tags 归组，写入 pattern_buffer
        │                 → 同组 ≥ threshold 次 → 调 LLM 生成 skill_xxx.py
        │                 → ast.parse() 语法验证
        │                 → 写入 <output>/skills/
        │                 → 注册进 SkillRegistry + ProceduralMemory
        │                 → SKILL_CREATED 事件
        │
        └── SuccessAnalyzer（独立订阅同一 TRAJECTORY 事件）
              → 提取关键推理步骤 + 核心洞察
              → 存入 EpisodicMemory（情景记忆，向量库）

每 10 题：MemoryEvaluatorAgent 评估三层记忆质量，清理无效记忆
```

### 三层记忆体系

Metacog 使用**基于 ChromaDB 的 memU 向量库**替代原有的单一 YAML 文件，支持语义 Top-K 检索：

| 记忆层 | 存储内容 | 检索方式 | 注入方式 |
|--------|----------|----------|----------|
| **语义记忆**（Semantic）| 失败教训（错误原因 + 解决策略）| 用当前题目做 Top-K 向量检索 | 注入 system prompt，默认 `rag_top_k=2` |
| **程序记忆**（Procedural）| Skill 元数据（名称/描述/适用场景）| 用当前题目做 Top-K 向量检索 | 注入 system prompt，默认 `skill_top_k=3` |
| **情景记忆**（Episodic）| 成功案例（关键推理步骤 + 核心洞察）| 用当前题目做 Top-K 向量检索（相似度≥60%）| 注入 system prompt，默认 `case_top_k=1` |

所有三层记忆都持久化到 `<output>/memu_db/`（ChromaDB），支持跨会话续跑。原有 `memories.yaml` 保留为人类 debug 的可读备份。

### 双模型架构

```
学生模型（--model）         教师模型（--teacher-model）
Qwen 9B / LM Studio        GLM-4.7 / 智谱 API
    ↓                           ↓
解题（ExecutorAgent）       复盘（AnalyzerAgent
                            MemoryManagerAgent
                            SkillAgent
                            SuccessAnalyzer）
```

- **学生模型**：负责实际解题，从 `math_test_config.yaml` 的 `model` 段读取，默认为 vLLM 服务器上的 `Qwen3.5-9B`
- **教师模型**（可选）：负责轨迹复盘、蒸馏教训、生成技能，默认退化为学生模型。推荐使用 `zai/glm-4.7`（支持 200K 上下文，无需分块蒸馏）

### 七个核心组件

#### ExecutorAgent（`src/metacog/agents/executor.py`）

每道题的实际解题者：

1. 用当前题目文本去三层记忆做 Top-K 语义检索
2. 将检索结果拼入 system prompt（语义记忆 + 技能描述 + 成功案例）
3. 设置 `PYTHONPATH` 包含 `<output>/skills/`，让 agent 可以 `import` 已生成的 skill
4. 通过 `PotSandboxWrapper` 包装 `DefaultAgent`，执行报错时自动注入反思 prompt
5. 解题结束后记录各层记忆的使用统计（成功/失败），用于记忆质量评分
6. 发布 `TRAJECTORY` 事件

#### AnalyzerAgent（`src/metacog/agents/analyzer.py`）

订阅 `TRAJECTORY` 事件，根据 `passed` 字段走不同分支：

**失败路径**：
1. `TrajectoryAnalyzer` 检测死循环模式
2. `FailureRouter` 过滤操作性失误（SyntaxError / ImportError 等），阻止其写入记忆
3. 调用教师模型直接蒸馏完整轨迹（GLM-4.7 支持 200K 上下文，无需分块）为结构化记忆条目（`problem_tags / error_symptom / root_cause / actionable_advice`）
4. 若标记 `needs_code_verification=true`，则调用 `PoTReflector` 生成并执行验证代码，用代码片段增强 `actionable_advice`
5. 发布 `ANALYSIS` 事件

**成功路径**：
- 调用教师模型蒸馏完整成功轨迹，提取 `technique_name + tags + can_be_skill`
- 发布 `SUCCESS_ANALYSIS` 事件

#### SuccessAnalyzer（`src/metacog/agents/success_analyzer.py`）

独立订阅 `TRAJECTORY` 事件（仅处理 `passed=True`）：
- 调用教师模型提取 3-5 个关键推理步骤和核心洞察
- 将完整成功案例存入 `EpisodicMemory`（情景记忆向量库）
- 供后续题目做类比学习

#### MemoryManagerAgent（`src/metacog/agents/memory_manager.py`）

订阅 `ANALYSIS` 事件：
- 将 `root_cause + actionable_advice` 拼接为向量化文本，存入 **memU 语义记忆层**
- 将 `problem_tags / error_symptom` 作为元数据
- 同步写入 `memories.yaml`（YAML 人类可读备份）
- 发布 `MEMORY_UPDATED` 事件

#### SkillAgent（`src/metacog/agents/skill_agent.py`）

订阅 `SUCCESS_ANALYSIS` 事件：
- 按 `tags` 将成功技术归组，维护 `pattern_buffer`
- 当某组积累 ≥ `threshold`（默认 3）次后，调 LLM 生成独立 Python skill 文件
- 文件须包含 `SKILL_METADATA` dict（name / description / when_to_use / tags）
- 用 `ast.parse()` 验证语法
- 写入 `<output>/skills/skill_xxx.py`，同步注册到 `SkillRegistry` 和 `ProceduralMemory`
- 发布 `SKILL_CREATED` 事件

#### MemoryEvaluatorAgent（`src/metacog/agents/memory_evaluator.py`）

每 `eval_interval`（默认 10）道题运行一次：
- 对三层记忆计算质量分数：`0.4 × 使用频率 + 0.4 × 成功率 + 0.2 × 时效性`
- 标记质量低于 `quality_threshold`（默认 0.3）的记忆
- 删除低于 `cleanup_threshold`（默认 0.2）且从未使用的记忆

#### PotSandboxWrapper（`src/metacog/agents/pot_sandbox_wrapper.py`）

对 `DefaultAgent` 的轻量包装，无需修改底层代码：
- 监控每步代码执行的 observation
- 检测到报错时，根据错误类型（递归/除零/超时/类型错误）注入针对性反思 prompt
- 继续线性执行（反思步骤消耗正常步数预算，不引入树状分支）

### Skill 文件格式

```python
# skills/skill_modular_inverse.py
SKILL_METADATA = {
    "name": "modular_inverse",
    "description": "Compute modular inverse using pow(a, -1, m).",
    "when_to_use": "Use when computing the modular inverse of a in Z/mZ, where m is prime.",
    "tags": ["number_theory", "modular_arithmetic"],
}

def modular_inverse(a: int, m: int) -> int:
    """Compute modular inverse of a modulo m."""
    return pow(a, -1, m)
```

Agent 在解题时可以直接 `from skill_modular_inverse import modular_inverse` 调用，无需重复实现。

### 事件总线（`src/metacog/bus.py`）

```python
class EventType:
    TRAJECTORY      = "trajectory"       # ExecutorAgent 完成一题
    ANALYSIS        = "analysis"         # 失败分析结果
    MEMORY_UPDATED  = "memory_updated"   # 记忆写入完成
    SUCCESS_ANALYSIS= "success_analysis" # 成功技术提取结果
    SKILL_CREATED   = "skill_created"    # 新 skill 文件生成
```

轻量同步总线，所有智能体通过 `bus.subscribe()` / `bus.publish()` 通信，无需直接互相引用。

### 关键参数

```bash
python scripts/run_math_test_metacog.py \
  --config scripts/configs/math_test_config.yaml \ # 学生模型配置（默认已内置）
  --data-source aime24 \
  --max-instances 30 \
  --output outputs/math_metacog \
  --fresh \                                        # 清除旧记忆，重新开始
  --teacher-model zai/glm-4.7 \                   # 教师模型（可选）
  --teacher-api-key YOUR_ZAI_API_KEY \             # 智谱 API Key（使用教师模型时必填）
  --start 0 \                                      # 从第几题开始（0-based，续跑用）
  --no-semantic \                                  # 消融实验：禁用语义记忆
  --no-episodic                                    # 消融实验：禁用情景记忆
```

在脚本内可调整 `SkillAgent(threshold=3)` 降低阈值（如改为 1），让技能更快生成。

### 输出结构

```
outputs/metacog/
  memu_db/                         # ChromaDB 向量数据库（核心存储）
    chroma.sqlite3                 # 向量索引
    ...
  memory/
    memories.yaml                  # YAML 备份（人类可读 debug 用）
  skills/
    skill_modular_inverse.py       # 自动生成的 Python skill 文件
    skill_crt.py
    ...
  trajectories/
    aime24_0001.traj.json          # 每题解题轨迹
  results.jsonl
```

### 优势与局限

| 优势 | 局限 |
|------|------|
| 三层记忆体系：语义 / 程序 / 情景各司其职 | 实现最复杂，依赖 ChromaDB |
| memU Top-K 检索，只注入最相关记忆，不污染上下文 | SkillAgent 需足够多同类题才触发 |
| FailureRouter 过滤操作性失误，只学逻辑错误 | 教师模型需要云端 API（智谱 GLM），本地模型降级 |
| PoT 验证：用代码增强记忆的 actionable_advice | 情景记忆类比效果依赖相似题目的积累量 |
| MemoryEvaluator 定期清理，防止记忆噪声积累 | |
| 消融实验参数支持，便于对比各记忆层贡献 | |

---

## 数据集

| 数据集 | 题目数 | 类型 | 答案范围 |
|--------|--------|------|----------|
| **AIME 2024** | 30 | 竞赛数论/组合/几何 | 整数 0-999 |
| **AIME 2025** | 30 | 竞赛数论/组合/几何 | 整数 0-999 |
| **AMC 2023** | 40 | 竞赛选择题 | A-E 选项 |

数据文件位于 `datasets/math/data/`，JSON 格式，每条记录包含 `problem`、`answer`、`instance_id` 等字段。

---

## 项目结构

```
metacog/
├── datasets/math/data/          # AIME/AMC 数据集
├── scripts/
│   ├── run_math_test.py         # 方法一：Baseline
│   ├── run_math_test_evolve.py  # 方法二：Evolve
│   ├── run_math_test_recreate.py# 方法三：ReCreate
│   ├── run_math_test_metacog.py # 方法四：Metacog
│   ├── analyze_failures.py      # 结果分析工具
│   ├── configs/
│   │   └── math_test_config.yaml# 配置文件示例
│   ├── evolve_utils/            # Evolve/ReCreate 共用工具
│   │   ├── evolution.py         # 核心进化逻辑
│   │   ├── scaffold_ops.py      # Scaffold 版本管理
│   │   ├── stats.py             # 统计工具
│   │   ├── trajectory.py        # 轨迹解析工具
│   │   └── utils.py             # 通用工具
│   └── utils/
│       ├── answer_extraction.py # 从输出中提取最终答案
│       └── evaluation.py        # 答案比对与正确率统计
├── src/
│   ├── metacog/                 # 方法四核心库
│   │   ├── bus.py               # 事件总线
│   │   ├── agents/
│   │   │   ├── base.py          # BaseAgent 基类
│   │   │   ├── executor.py      # 解题 Agent（三层记忆 Top-K 检索）
│   │   │   ├── analyzer.py      # 轨迹分析 Agent（PoT + 死循环检测）
│   │   │   ├── memory_manager.py# 记忆管理 Agent（写入 memU 语义层）
│   │   │   ├── skill_agent.py   # 技能生成 Agent
│   │   │   ├── success_analyzer.py  # 成功案例分析（写入情景记忆）
│   │   │   ├── memory_evaluator.py  # 记忆质量评估与清理
│   │   │   ├── failure_router.py    # 操作性失误过滤（阻止 SyntaxError 等入库）
│   │   │   ├── pot_sandbox_wrapper.py # PoT 报错反思包装器
│   │   │   ├── pot_reflector.py     # PoT 代码验证生成器
│   │   │   ├── trajectory_analyzer.py# 死循环检测
│   │   │   ├── execution_monitor.py # 执行监控器
│   │   │   └── monitored_agent.py   # 带监控的 Agent 包装器
│   │   ├── memory/
│   │   │   ├── store.py         # YAML memories.yaml 读写（人类可读备份）
│   │   │   ├── memu_client.py   # memU 向量库客户端（基于 ChromaDB）
│   │   │   ├── episodic_memory.py   # 情景记忆（成功案例向量存储）
│   │   │   └── procedural_memory.py # 程序记忆（Skill 元数据向量存储）
│   │   └── skills/
│   │       ├── registry.py      # Skill 注册表
│   │       ├── base.py          # StructuredSkill 基类
│   │       ├── composite.py     # 复合技能
│   │       └── math/            # 预置数学 skill 文件
│   ├── mini-swe-agent/          # 底层 Agent 框架
│   └── recreate_agent/          # 方法三 ReCreate 核心库
│       ├── recreate_agent.py    # ReCreate-Agent 实现
│       ├── agent_runner.py      # 批量运行管理
│       ├── scaffold.py          # Scaffold 管理
│       ├── result_collector.py  # 结果收集
│       ├── stats_collector.py   # 统计收集
│       ├── setup_workspace.py   # 工作区初始化
│       ├── adapters/            # 数据集适配器
│       ├── evaluators/          # 各数据集评测器
│       ├── prompts/             # Jinja2 提示词模板
│       └── tools/               # ReCreate-Agent 元操作工具
│           ├── memory_manager.py
│           ├── read_trajectory.py
│           ├── scaffold_editor.py
│           ├── search_memory.py
│           └── write_memory.py
├── tests/                       # 单元测试
└── outputs/                     # 实验结果（git-ignored）
```

---

## 环境配置

### 依赖安装

```bash
conda activate medte
pip install -e src/mini-swe-agent
pip install chromadb          # 方法四 Metacog 必须：memU 向量库
pip install typer rich pyyaml # 方法四 Metacog 必须：CLI 和配置
```

### 学生模型（vLLM 服务器）

所有四种方法的学生模型配置统一写在 `scripts/configs/math_test_config.yaml` 的 `model` 段：

```yaml
# scripts/configs/math_test_config.yaml（model 段）
model:
  model: "openai//root/autodl-tmp/huggingface/Qwen3.5-9B"
  api_base: "https://<YOUR_SERVER>:8443/v1"
  api_key: "sk_123456"
  temperature: 0.0
  max_tokens: 8192
  extra_body:
    chat_template_kwargs:
      enable_thinking: false    # vLLM 关闭 thinking 模式的正确方式
```

- `extra_body.chat_template_kwargs.enable_thinking: false` 是 vLLM 的参数传递方式，与 LM Studio 的 `think: false` 不同
- 所有脚本默认读取此配置文件，CLI 参数（`--model`、`--api-base`）可覆盖

### 教师模型（智谱 GLM，可选）

方法二（Evolve）、方法三（ReCreate）、方法四（Metacog）均支持双模型架构：
学生模型（Qwen）负责解题，教师模型（GLM-4.7）负责分析/进化。

```bash
# 方式一：命令行传入
python scripts/run_math_test_metacog.py \
  --teacher-model zai/glm-4.7 \
  --teacher-api-key YOUR_ZAI_API_KEY ...

# 方式二：环境变量
export ZAI_API_KEY="YOUR_ZAI_API_KEY"
python scripts/run_math_test_metacog.py --teacher-model zai/glm-4.7 ...
```

不指定 `--teacher-model` 时，分析/进化/记忆写入退化为使用学生模型。

---

## 快速开始

> **前提**：`conda activate medte`，且 `scripts/configs/math_test_config.yaml` 中的 `api_base` 已指向正确的 vLLM 服务器地址。

---

### 方法一：Baseline（基线）

```bash
# AIME 2025，30 道题
python scripts/run_math_test.py \
  --data-source aime25 \
  --max-instances 30 \
  --output outputs/baseline_aime25

# AIME 2024，完整 30 道题
python scripts/run_math_test.py \
  --data-source aime24 \
  --max-instances 30 \
  --output outputs/baseline_aime24
```

---

### 方法二：Evolve（元智能体 Prompt 进化）

```bash
# 推荐：GLM-4.7 作为教师模型进行 Scaffold 进化，Qwen 求解
python scripts/run_math_test_evolve.py \
  --data-source aime25 \
  --max-instances 30 \
  --max-rounds 3 \
  --output outputs/evolve_aime25 \
  --teacher-model zai/glm-4.7 \
  --teacher-api-key YOUR_ZAI_API_KEY

# 单模型版本（不使用教师模型，退化为 Qwen 自进化）
python scripts/run_math_test_evolve.py \
  --data-source aime25 \
  --max-instances 30 \
  --max-rounds 3 \
  --output outputs/evolve_aime25_no_teacher
```

> **注意**：重新开始实验前，清除旧 workspace：`rm -rf outputs/evolve_aime25/evolve_workspace`

---

### 方法三：ReCreate（题目级 ReCreate-Agent）

```bash
# 推荐：GLM-4.7 驱动 ReCreate-Agent 和 Batch Synthesis
python scripts/run_math_test_recreate.py \
  --data-source aime25 \
  --max-instances 30 \
  --max-rounds 2 \
  --output outputs/recreate_aime25 \
  --teacher-model zai/glm-4.7 \
  --teacher-api-key YOUR_ZAI_API_KEY

# 单模型版本
python scripts/run_math_test_recreate.py \
  --data-source aime25 \
  --max-instances 30 \
  --max-rounds 2 \
  --output outputs/recreate_aime25_no_teacher
```

> **注意**：重新开始实验前，清除旧 workspace：`rm -rf outputs/recreate_aime25/recreate_workspace`

---

### 方法四：Metacog（多智能体事件驱动 + 三层记忆）

```bash
# 推荐：全功能 + 双模型（全新开始）
python scripts/run_math_test_metacog.py \
  --data-source aime25 \
  --max-instances 30 \
  --output outputs/metacog_aime25 \
  --teacher-model zai/glm-4.7 \
  --teacher-api-key YOUR_ZAI_API_KEY \
  --fresh

# 续跑（自动加载已有三层记忆继续积累）
python scripts/run_math_test_metacog.py \
  --data-source aime25 \
  --max-instances 30 \
  --output outputs/metacog_aime25 \
  --teacher-model zai/glm-4.7 \
  --teacher-api-key YOUR_ZAI_API_KEY

# 从第 N 题开始续跑（0-based，例如从第 21 题开始）
python scripts/run_math_test_metacog.py \
  --data-source aime25 \
  --max-instances 30 \
  --output outputs/metacog_aime25 \
  --teacher-model zai/glm-4.7 \
  --teacher-api-key YOUR_ZAI_API_KEY \
  --start 20

# 消融实验：禁用语义记忆
python scripts/run_math_test_metacog.py \
  --data-source aime25 --max-instances 30 \
  --output outputs/ablation_no_semantic --no-semantic --fresh \
  --teacher-model zai/glm-4.7 --teacher-api-key YOUR_ZAI_API_KEY

# 消融实验：禁用情景记忆
python scripts/run_math_test_metacog.py \
  --data-source aime25 --max-instances 30 \
  --output outputs/ablation_no_episodic --no-episodic --fresh \
  --teacher-model zai/glm-4.7 --teacher-api-key YOUR_ZAI_API_KEY
```

---

### 消融实验对照表

为确保公平对比，四种方法共用同一套初始提示词（`INITIAL_SYSTEM_TEMPLATE` / `INITIAL_INSTANCE_TEMPLATE`），由 `math_test_config.yaml` 统一管理模型参数。

| 实验组 | 脚本 | 教师模型 | 输出目录 |
|--------|------|----------|---------|
| Baseline | `run_math_test.py` | — | `outputs/baseline_aime25` |
| Evolve | `run_math_test_evolve.py` | GLM-4.7 | `outputs/evolve_aime25` |
| ReCreate | `run_math_test_recreate.py` | GLM-4.7 | `outputs/recreate_aime25` |
| Metacog（全功能）| `run_math_test_metacog.py` | GLM-4.7 | `outputs/metacog_aime25` |
| Metacog（禁语义）| `run_math_test_metacog.py --no-semantic` | GLM-4.7 | `outputs/ablation_no_semantic` |
| Metacog（禁情景）| `run_math_test_metacog.py --no-episodic` | GLM-4.7 | `outputs/ablation_no_episodic` |

---

### 结果分析

```bash
python scripts/analyze_failures.py --output outputs/baseline_aime25
python scripts/analyze_failures.py --output outputs/metacog_aime25
```

---

## 方法对比总结

```
学习粒度      粗 ←──────────────────────────────────────────────→ 细
              Baseline    Evolve       ReCreate          Metacog
学习时机         无        批次后        题目后              实时
改进对象         无        Prompt    Prompt+工具+记忆   Prompt+三层记忆+代码
工具可执行性     无          无        ✓ 批次间复用       ✓ Python import
记忆系统         无          无        ✓ YAML 持久化      ✓ ChromaDB 向量库
记忆检索         无          无        线性检索           语义 Top-K 检索
记忆层数         无          无        1 层（YAML）       3 层（语义/程序/情景）
失误过滤         无          无           无              ✓ FailureRouter
记忆质量管理     无          无           无              ✓ MemoryEvaluator
PoT 验证         无          无           无              ✓ 代码增强记忆
模型架构         单模型      单模型       单模型            双模型（学生+教师）
实现复杂度       低          低           中                高
```
