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

---

## 项目概述

本项目在 **AIME 24 / AIME 25 / AMC 23** 三个竞赛数学数据集上，实验了四种不同程度的「元认知」策略：

| 方法 | 脚本 | 核心思路 | 学习时机 |
|------|------|----------|----------|
| **Baseline** | `run_math_test.py` | 固定 prompt，无学习 | 无 |
| **Evolve** | `run_math_test_evolve.py` | 元智能体每轮后优化 scaffold | 每轮结束后 |
| **ReCreate** | `run_math_test_recreate.py` | ReCreate-Agent 每题后分析 + 创建工具/记忆 + 批次合成 | 每题后 + 每批合成 |
| **Metacog** | `run_math_test_metacog.py` | 多智能体事件驱动，记忆 + 技能双通道 | 每题后实时 |

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
  --data-source aime24 \       # 数据集：aime24 / aime25 / amc23
  --max-instances 10 \         # 最多跑几道题
  --step-limit 15 \            # 每题最多几步
  --output outputs/baseline
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
  --data-source aime24 \
  --max-instances 10 \
  --rounds 3 \              # 进化轮数
  --output outputs/evolve
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
  --data-source aime24 \
  --max-instances 10 \
  --max-rounds 2 \
  --output outputs/recreate
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

最复杂的方案：用**事件总线（EventBus）**连接多个专职智能体，形成两条并行的学习通道：

- **失败通道**：失败 → 提取教训 → 写入结构化记忆 → 下题注入文字提示
- **成功通道**：成功 → 提取技术模式 → 生成 Python skill 文件 → 下题可直接 `import`

```
每道题结束
  ├── passed=False → AnalyzerAgent（失败分析）
  │     → 分块摘要 → 蒸馏成一句 lesson
  │     → ANALYSIS 事件
  │           → MemoryManagerAgent → 写/合并 memories.yaml
  │                 → 下题 system prompt 注入记忆文字
  │
  └── passed=True  → AnalyzerAgent（成功分析）
        → 分块摘要 → 提取 technique + tags
        → SUCCESS_ANALYSIS 事件
              → SkillAgent
                    → 按 tags 归组，写入 pattern_buffer
                    → 同组 ≥ threshold 次 → 调 LLM 生成 skill_xxx.py
                    → ast.parse() 语法验证
                    → 写入 <output>/skills/
                    → 注册进 SkillRegistry
                    → SKILL_CREATED 事件
                    → 下题 PYTHONPATH 可直接 import
```

### 四个核心智能体

#### ExecutorAgent（`src/metacog/agents/executor.py`）

每道题的实际解题者：

1. 从 `MemoryStore` 加载最新记忆，拼入 system prompt
2. 从 `SkillRegistry` 生成技能描述，拼入 system prompt
3. 设置 `PYTHONPATH` 包含 `<output>/skills/`，让 agent 可以 `import` 已生成的 skill
4. 运行 `DefaultAgent` 解题
5. 发布 `TRAJECTORY` 事件

#### AnalyzerAgent（`src/metacog/agents/analyzer.py`）

订阅 `TRAJECTORY` 事件，根据 `passed` 字段走不同分支：

**失败路径**：
- 将轨迹按 `chunk_size` 步分块，每块调 LLM 得一句摘要
- 把所有摘要 + 题目元数据再次喂给 LLM，蒸馏成 1-2 句可执行建议
- 发布 `ANALYSIS` 事件（含 `lesson` 字段）

**成功路径**：
- 同样分块摘要
- 从成功轨迹中提取"用了什么方法"（`technique` + `tags`）
- 发布 `SUCCESS_ANALYSIS` 事件

#### MemoryManagerAgent（`src/metacog/agents/memory_manager.py`）

订阅 `ANALYSIS` 事件：

- 将 `lesson` 存入 `MemoryEntry`（含 `title / content / tags / created_at`）
- 写入 `memories.yaml`（YAML 格式，支持增量追加）
- 发布 `MEMORY_UPDATED` 事件
- 下一道题的 ExecutorAgent 读取时自动注入最新记忆

#### SkillAgent（`src/metacog/agents/skill_agent.py`）

订阅 `SUCCESS_ANALYSIS` 事件：

- 按 `tags` 将成功技术归组，维护 `pattern_buffer`
- 当某组积累 ≥ `threshold`（默认 3）次后，调 LLM 生成独立 Python skill 文件
- 文件须包含 `SKILL_META` dict（name / description / tags / usage）
- 用 `ast.parse()` 验证语法
- 写入 `<output>/skills/skill_xxx.py`
- 注册进 `SkillRegistry`，发布 `SKILL_CREATED` 事件

### Skill 文件格式

```python
# skills/skill_modular_inverse.py
SKILL_META = {
    "name": "modular_inverse",
    "description": "Compute modular inverse using pow(a, -1, m).",
    "tags": ["number_theory", "modular_arithmetic"],
    "module": "skill_modular_inverse",
    "usage": "from skill_modular_inverse import modular_inverse\nx = modular_inverse(3, 7)  # → 5",
}

def modular_inverse(a: int, m: int) -> int:
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
  --data-source aime24 \
  --max-instances 10 \
  --output outputs/metacog \
  --fresh                    # 清除旧记忆，重新开始
```

在脚本内可调整 `SkillAgent(threshold=3)` 降低阈值（如改为 1），让技能更快生成。

### 输出结构

```
outputs/metacog/
  memory/
    memories.yaml              # 累积的失败教训（结构化 YAML）
  skills/
    skill_modular_inverse.py   # 自动生成的 Python skill 文件
    skill_crt.py
    ...
  trajectories/
    aime24_0001.traj.json      # 每题解题轨迹
  results.jsonl
```

### 优势与局限

| 优势 | 局限 |
|------|------|
| 每题后实时学习，响应最快 | 实现最复杂 |
| 双通道：失败学教训 + 成功生代码 | SkillAgent 需足够多同类题才触发 |
| Skill 可直接执行，不依赖 prompt 理解 | 9B 小模型对注入记忆的利用率不稳定 |
| 记忆结构化存储，可跨实验复用 | |

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
│   ├── evolve_utils/            # Evolve/ReCreate 共用工具
│   │   ├── evolution.py         # 核心进化逻辑
│   │   ├── scaffold_ops.py      # Scaffold 版本管理
│   │   └── trajectory.py        # 轨迹解析工具
│   └── utils/
│       ├── answer_extraction.py # 从输出中提取最终答案
│       └── evaluation.py        # 答案比对与正确率统计
├── src/
│   ├── metacog/                 # 方法四核心库
│   │   ├── bus.py               # 事件总线
│   │   ├── agents/
│   │   │   ├── executor.py      # 解题 Agent
│   │   │   ├── analyzer.py      # 轨迹分析 Agent
│   │   │   ├── memory_manager.py# 记忆管理 Agent
│   │   │   └── skill_agent.py   # 技能生成 Agent
│   │   ├── memory/
│   │   │   └── store.py         # memories.yaml 读写
│   │   └── skills/
│   │       ├── registry.py      # Skill 注册表
│   │       ├── base.py          # StructuredSkill 基类
│   │       └── math/            # 预置数学 skill 文件
│   ├── mini-swe-agent/          # 底层 Agent 框架
│   └── recreate_agent/          # 方法三 ReCreate 核心库
│       ├── recreate_agent.py    # ReCreate-Agent 实现
│       ├── agent_runner.py      # 批量运行管理
│       └── adapters/            # 数据集适配器
└── outputs/                     # 实验结果（git-ignored）
```

---

## 环境配置

### 依赖安装

```bash
conda activate medte
pip install -e src/mini-swe-agent
```

### 本地模型（LM Studio）

```bash
# 启动 LM Studio，加载 Qwen3.5-9b 模型
# Base URL: http://0.0.0.0:1234/v1
# Model ID: lm_studio/qwen/qwen3.5-9b

# 绕过本地代理（必须）
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
```

在脚本中已内置以下环境变量设置：

```python
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"
```

---

## 快速开始

### 1. 运行基线实验

```bash
conda activate medte
python scripts/run_math_test.py \
  --data-source aime24 --max-instances 5 \
  --output outputs/baseline_test
```

### 2. 运行 Evolve 实验

```bash
python scripts/run_math_test_evolve.py \
  --data-source aime24 --max-instances 10 --rounds 3 \
  --output outputs/evolve_test
```

### 3. 运行 ReCreate 实验

```bash
python scripts/run_math_test_recreate.py \
  --data-source aime24 --max-instances 10 --rounds 2 \
  --output outputs/recreate_test
```

### 4. 运行 Metacog 实验

```bash
# 全新开始（清除旧记忆）
python scripts/run_math_test_metacog.py \
  --data-source aime24 --max-instances 10 \
  --output outputs/metacog_test --fresh

# 续跑（自动加载已有记忆继续积累）
python scripts/run_math_test_metacog.py \
  --data-source aime24 --max-instances 10 \
  --output outputs/metacog_test
```

### 5. 结果对比

```bash
# 查看各实验的 pass rate
python scripts/analyze_failures.py --output outputs/baseline_test
python scripts/analyze_failures.py --output outputs/metacog_test
```

---

## 方法对比总结

```
学习粒度      粗 ←─────────────────────────────────────→ 细
              Baseline    Evolve      ReCreate       Metacog
学习时机         无        批次后       题目后          实时
改进对象         无        Prompt   Prompt+工具+记忆  Prompt+记忆+代码
工具可执行性     无          无       ✓ 批次间复用     ✓ Python import
记忆系统         无          无       ✓ YAML 持久化    ✓ YAML 持久化
实现复杂度       低          低          中              高
```
