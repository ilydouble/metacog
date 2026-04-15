# memU 升级改造文档

## 🎯 改造目标

**拯救 Qwen3.5 9B 的上下文注意力**，将 memU 作为独立的记忆微服务无缝嵌入到 EventBus 架构中。

**核心思路**：**存入时结构化，提取时向量化（Top-K）**

---

## ⚡ 改造前 vs 改造后

### 改造前（YAML 全量加载）

```python
# MemoryStore 包含 100 条记忆
system_prompt = base_prompt + "\n\n" + mem_store.as_prompt_text()
# → 注入所有 100 条记忆，几千 tokens，9B 模型崩溃
```

**问题**：
- ❌ 9B 模型看到几千 tokens 的记忆列表，注意力崩溃
- ❌ 90% 的记忆与当前题目完全无关
- ❌ 模型无法从海量信息中提取关键教训

### 改造后（memU RAG Top-K）

```python
# 用当前题目做语义检索
retrieved = memu.search(query=current_problem, top_k=2)
system_prompt = base_prompt + "\n\n" + format_memories(retrieved)
# → 只注入最相关的 2 条记忆，100-200 tokens，精准打击
```

**优势**：
- ✅ 9B 模型只看到 1-2 条高度相关的教训
- ✅ 信息密度极高，指令遵循能力飙升
- ✅ 向量检索自动找到与当前题目最相似的历史错误

---

## 🔪 三刀改造详解

### 第一刀：`analyzer.py` - 强制输出结构化 JSON

**改造点**：锁死 LLM 输出为严格的 JSON 格式

**新的 Prompt 模板**：
```python
{
  "problem_tags": ["number_theory", "modular_arithmetic"],
  "error_symptom": "在计算逆元时使用了错误的公式",
  "root_cause": "忘记了费马小定理的使用前提是模数为素数",
  "actionable_advice": "计算 a 的逆元前，先检查模数 m 是否为素数..."
}
```

**验证机制**：
- 检查必需字段：`problem_tags`, `error_symptom`, `root_cause`, `actionable_advice`
- 缺少任何字段立即跳过，避免污染向量库

**为什么这样做**：
- 保证存入 memU 的数据极其干净
- 规避 9B 模型的"幻觉致死"问题
- 结构化数据更易于向量化和检索

---

### 第二刀：`memory_manager.py` - 拦截 YAML，写入 memU

**改造点**：将 YAML 存储替换为向量数据库

**新的存储逻辑**：
```python
# 1. 拼接向量化内容（核心教训）
document_content = f"错误原因：{root_cause}\n解决策略：{actionable_advice}"

# 2. 构造元数据（用于过滤和混合检索）
metadata = {
    "tags": problem_tags,
    "symptom": error_symptom,
    "problem_id": problem_id
}

# 3. 写入 memU 向量库
memory_id = memu.add_memory(content=document_content, metadata=metadata)
```

**保留 YAML 作为备份**：
- YAML 文件仍然生成，但 Agent 不再读取
- 仅用于人类 debug 和审计

**为什么这样做**：
- 向量化存储支持语义检索
- 元数据支持按 tag 过滤
- 彻底解耦存储和检索逻辑

---

### 第三刀：`executor.py` - RAG 动态检索注入

**改造点**：用当前题目做 Top-K 检索，只注入最相关记忆

**新的 Prompt 组装逻辑**：
```python
# 基于当前题目进行 Top-K RAG 检索
retrieved_memories = memu.search(
    query=current_problem,  # 用题目文本做查询
    top_k=2  # 🔥 关键参数：只取最相关的 2 条
)

# 组装极度精简的记忆注入
memory_injection = "## 📚 历史避坑指南\n"
for idx, mem in enumerate(retrieved_memories, 1):
    memory_injection += f"{idx}. {mem.content}\n"

system_prompt = base_prompt + "\n\n" + memory_injection
```

**关键参数**：
- `rag_top_k=2`：救命参数！控制注入多少条记忆
- 可在运行脚本时调整（1-3 条为最佳实践）

**为什么这样做**：
- 9B 模型再也不会看到几千 tokens 的记忆列表
- 每次只看到 1-2 句"直击灵魂的教条"
- 向量检索自动处理相似度匹配

---

## 📦 依赖安装

需要安装 ChromaDB（轻量级向量数据库）：

```bash
pip install chromadb
```

---

## 🚀 使用方法

### 运行示例

```bash
# 在 AIME 2025 数据集上运行 10 道题
python scripts/run_math_test_metacog.py \
  --data-source aime25 \
  --max-instances 10 \
  --output outputs/metacog_memu

# 清除旧记忆，重新开始
python scripts/run_math_test_metacog.py \
  --data-source aime25 \
  --max-instances 10 \
  --output outputs/metacog_memu \
  --fresh
```

### 输出结构

```
outputs/metacog_memu/
├── memu_db/                     # memU 向量数据库（核心存储）
│   └── chroma.sqlite3           # ChromaDB 持久化文件
├── memory/
│   └── memories.yaml            # YAML 备份（人类 debug 用）
├── skills/
│   ├── skill_modular_inverse.py
│   └── ...
├── trajectories/
│   └── ...
└── results.jsonl
```

---

## ⚙️ 调优参数

### ExecutorAgent
- **`rag_top_k`**（默认 2）：🔥 **最关键参数**
  - `1`：极度精简，只注入最相关的 1 条记忆
  - `2`：推荐值，平衡精准度和覆盖面
  - `3`：稍宽松，适合复杂题目
  - `≥4`：不推荐，9B 模型可能又会注意力分散

### MemUClient
- **`collection_name`**：记忆集合名称（不同实验可以用不同 collection）
- **`persist_dir`**：向量数据库持久化目录

---

## 🎨 核心优势

### 1. 信息密度极高
9B 模型再也不会看到长达几千 tokens 的 `memories.yaml`。它每次只看到一两句类似：

> "计算 a 的逆元前，先检查模数 m 是否为素数。如果是，用 pow(a, m-2, m)；如果不是，必须用扩展欧几里得算法。"

这种直击灵魂的教条，它的指令遵循能力会直线飙升。

### 2. 规避"幻觉致死"
因为我们把总结任务锁死在了 JSON 格式，9B 模型在提取教训时不容易瞎编乱造。这保证了 memU 向量库里存储的都是高质量资产。

### 3. 彻底解耦
EventBus 优势彻底发挥：
- ExecutorAgent 根本不知道外面发生了什么进化
- 它只觉得自己每做一道新题，系统就会神机妙算地给它一个"锦囊妙计"
- MemoryManagerAgent 和 AnalyzerAgent 完全独立工作

---

## 📊 预期效果

### 改造前（YAML 全量加载）
- 记忆数量：50 条
- 注入 tokens：~3000
- 9B 模型表现：注意力分散，教训遵循率 < 20%

### 改造后（memU RAG Top-2）
- 记忆数量：50 条（向量库）
- 注入 tokens：~150
- 9B 模型表现：注意力集中，教训遵循率 > 70%

---

## 🔍 技术细节

### 向量化模型
- 使用 ChromaDB 内置的 `all-MiniLM-L6-v2`
- 轻量级（80MB），适合本地部署
- 支持中英文混合文本

### 检索算法
- 使用余弦相似度（cosine similarity）
- HNSW 索引，查询速度 < 10ms

### 持久化
- SQLite 后端，单文件存储
- 支持增量写入，无需重建索引
