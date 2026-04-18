#!/usr/bin/env python3
"""Math 测试 - metacog 多智能体版本

每道题结束后立即触发 AnalyzerAgent 分析轨迹，
MemoryManagerAgent 更新记忆，下一道题动态加载最新记忆。

用法::

    # 全新实验（记忆住在 outputs/exp_001/memory/memories.yaml）
    python scripts/run_math_test_metacog.py \\
        --data-source aime25 -n 10 \\
        --output outputs/exp_001

    # 续跑同一实验（自动加载已有记忆继续累积）
    python scripts/run_math_test_metacog.py \\
        --data-source aime25 -n 10 \\
        --output outputs/exp_001

    # 重置记忆重新跑（删除旧 memories.yaml 后再开始）
    python scripts/run_math_test_metacog.py \\
        --data-source aime25 -n 10 \\
        --output outputs/exp_001 --fresh
"""

import os
os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

# 路径设置
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src" / "mini-swe-agent" / "src"))
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # scripts/

from minisweagent.models.litellm_model import LitellmModel

from metacog.bus import EventBus, EventType, Event
from metacog.memory.store import MemoryStore
from metacog.memory.memu_client import MemUClient
from metacog.skills.registry import SkillRegistry
from metacog.agents.executor import ExecutorAgent
from metacog.agents.analyzer import AnalyzerAgent
from metacog.agents.memory_manager import MemoryManagerAgent
from metacog.agents.skill_agent import SkillAgent

from utils.evaluation import compute_accuracy

app = typer.Typer()
console = Console()

# ------------------------------------------------------------------ #
# 默认 Scaffold
# ------------------------------------------------------------------ #

DEFAULT_SYSTEM = """You are a helpful assistant that solves math problems."""

DEFAULT_INSTANCE = """Please solve the following math problem:

{{task}}

## Instructions

1. Read and analyze the problem carefully.
2. Use the bash tool to run Python code for calculations if needed.
3. Show your step-by-step reasoning.
4. Give your final answer in the format: \\boxed{your_answer}

## Command Execution Rules

- Your response MUST include AT LEAST ONE bash tool call every turn.
- Use the bash tool to run Python scripts for any calculations.
- **IMPORTANT**: Once you have determined the final answer, you MUST submit it in the **same bash
  block** as your final calculation — do NOT split computation and submission into separate steps.
  Append the following lines at the end of your final Python script (replace ANSWER with your number):

  ```python
  # ... your calculation code above ...
  print(f'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')
  print(ANSWER)
  ```

  Or use printf in the same bash block after the python call:
  ```bash
  python3 << 'EOF'
  # ... your calculation ...
  EOF
  printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n%s\\n' ANSWER
  ```

## Important Notes

- For AIME problems, answers are integers from 0 to 999.
- The submission MUST print COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT on its own line,
  followed by your numeric answer on the next line.
- Combining calculation and submission in ONE bash call saves a step — always do this.

Example: if your answer is 73, your final bash call should be:
  python3 << 'EOF'
  result = 73
  print('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')
  print(result)
  EOF

Now please solve the problem."""


# ------------------------------------------------------------------ #
# 数据加载（复用 run_math_test_evolve 的逻辑）
# ------------------------------------------------------------------ #

def load_dataset(data_source: str, base_path: str, max_instances: int, start: int = 0) -> list[dict]:
    data_file = Path(base_path) / f"{data_source}.json"
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    problems = []
    with open(data_file) as f:
        for line in f:
            try:
                problems.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    # 如果按行解析为空，尝试整体解析（JSON 数组格式）
    if not problems:
        content = data_file.read_text()
        data = json.loads(content)
        if isinstance(data, list):
            problems = data
        elif isinstance(data, dict):
            problems = [data]
    # 应用 start offset 和 max_instances
    problems = problems[start:]
    if max_instances:
        problems = problems[:max_instances]
    return problems


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

@app.command()
def main(
    data_source: Annotated[str, typer.Option("--data-source", "-d")] = "aime25",
    max_instances: Annotated[int, typer.Option("--max-instances", "-n")] = 5,
    scaffold: Annotated[Optional[Path], typer.Option("--scaffold", "-s")] = None,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/math_metacog"),
    fresh: Annotated[bool, typer.Option("--fresh", help="删除该实验已有的记忆，从零开始")] = False,
    model: Annotated[Optional[str], typer.Option("--model", "-m")] = None,
    api_base: Annotated[Optional[str], typer.Option("--api-base")] = None,
    api_key: Annotated[Optional[str], typer.Option("--api-key")] = None,
    config: Annotated[Optional[Path], typer.Option("--config", "-c")] = Path("scripts/configs/math_test_config.yaml"),
    base_path: Annotated[str, typer.Option("--base-path")] = "datasets/math/data",
    teacher_model: Annotated[Optional[str], typer.Option("--teacher-model")] = None,  # 🔥 教师模型（如 zai/glm-4.7）
    teacher_api_key: Annotated[Optional[str], typer.Option("--teacher-api-key")] = None,  # 🔥 智谱 API Key
    start: Annotated[int, typer.Option("--start", help="从第几题开始（0-based，如 --start 21 表示从第22题）")] = 0,
    no_semantic: Annotated[bool, typer.Option("--no-semantic", help="消融实验：禁用语义记忆（教训），只用情景记忆")] = False,
    no_episodic: Annotated[bool, typer.Option("--no-episodic", help="消融实验：禁用情景记忆（成功案例），只用语义记忆")] = False,
) -> None:
    """Math 多智能体测试（每题结束立即分析并更新记忆）。

    记忆默认存储在 <output>/memory/memories.yaml。
    不同 --output 实验天然隔离；--fresh 可重置当前实验的记忆。
    """

    # ── 加载配置文件（model + agent 两段）────────────────────────────────────────
    config_data: dict = {}
    if config and config.exists():
        config_data = yaml.safe_load(config.read_text()) or {}
        console.print(f"[dim]✓ 已加载配置: {config}[/dim]")

    # scaffold（agent 段 → --scaffold 文件 → 内置默认值，优先级从低到高）
    scaffold_data: dict = {"system_template": DEFAULT_SYSTEM, "instance_template": DEFAULT_INSTANCE,
                           "step_limit": 10, "cost_limit": 2.0}
    scaffold_data.update(config_data.get("agent", {}))
    if scaffold and scaffold.exists():
        sc = yaml.safe_load(scaffold.read_text()) or {}
        for k in ("system_template", "instance_template", "step_limit", "cost_limit"):
            if k in sc:
                scaffold_data[k] = sc[k]
        console.print(f"[green]✓ scaffold: {scaffold}[/green]")

    # ==========================================
    # 模型初始化（双模型架构）—— 与基线 run_math_test.py 保持一致
    # ==========================================

    # 从配置文件读取 model 段，CLI 参数可覆盖
    model_config = config_data.get("model", {})
    model_name   = model    or model_config.get("model",       "openai//root/autodl-tmp/huggingface/Qwen3.5-9B")
    _api_base    = api_base or model_config.get("api_base",    None)
    _api_key     = api_key  or model_config.get("api_key",     "sk_123456")
    temperature  = model_config.get("temperature", 0.0)
    max_tokens   = model_config.get("max_tokens",  8192)
    think        = model_config.get("think",        None)
    extra_body   = model_config.get("extra_body",   None)

    # 1. 学生模型（Qwen vLLM 服务器）：负责解题
    model_kwargs: dict = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "drop_params": True,
        "tool_choice": "required",
    }
    if _api_base:
        model_kwargs["api_base"] = _api_base
        model_kwargs["api_key"]  = _api_key
    if think is not None:
        model_kwargs["think"] = think
    if extra_body is not None:
        model_kwargs["extra_body"] = extra_body

    litellm_model = LitellmModel(model_name=model_name, model_kwargs=model_kwargs,
                                 cost_tracking="ignore_errors")
    console.print(f"[cyan]🎓 学生模型（解题）: {model_name}[/cyan]")
    console.print(f"[dim]   model_kwargs: {model_kwargs}[/dim]")

    # 2. 教师模型（GLM-4.7 / 强推理模型）：负责复盘和写 memU 经验
    if teacher_model:
        teacher_kwargs = {
            "temperature": 0.0,
            "max_tokens": 4096,
            "drop_params": True,
        }

        # 🔥 智谱 API 配置
        if teacher_api_key:
            teacher_kwargs["api_key"] = teacher_api_key
        else:
            # 如果未提供 API Key，尝试从环境变量读取
            import os
            env_key = os.getenv("ZAI_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
            if env_key:
                teacher_kwargs["api_key"] = env_key
            else:
                console.print("[red]错误：使用智谱模型需要提供 --teacher-api-key 或设置 ZAI_API_KEY 环境变量[/red]")
                raise typer.Exit(1)

        # 🔥 设置智谱 API 地址（开放平台）
        teacher_kwargs["api_base"] = "https://open.bigmodel.cn/api/paas/v4"

        analyzer_model = LitellmModel(
            model_name=teacher_model,
            model_kwargs=teacher_kwargs,
            cost_tracking="ignore_errors",
        )
        console.print(f"[magenta]🧑‍🏫 教师模型（复盘）: {teacher_model}[/magenta]")
    else:
        # 未指定教师模型，退化为单模型模式
        analyzer_model = litellm_model
        console.print(f"[yellow]⚠️  未指定教师模型，复盘使用学生模型[/yellow]")

    # ==========================================
    # memU 向量记忆库（核心存储）
    # ==========================================
    memu_dir = output / "memu_db"
    memu_dir.mkdir(parents=True, exist_ok=True)

    if fresh and memu_dir.exists():
        import shutil
        shutil.rmtree(memu_dir)
        memu_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[yellow]✓ --fresh: 已清空 memU 向量库 {memu_dir}[/yellow]")

    # 1. 语义记忆（Semantic Memory）- 存储教训
    memu_client = MemUClient(
        collection_name="math_lessons",
        persist_dir=memu_dir
    )

    memu_count = memu_client.count()
    status = "新建" if memu_count == 0 else f"续跑，已有 {memu_count} 条"
    if no_semantic:
        console.print(f"[yellow]语义记忆（教训）: 已禁用（消融实验 --no-semantic）[/yellow]")
    else:
        console.print(f"[cyan]语义记忆（教训）: {memu_dir} ({status})[/cyan]")

    # 2. 程序记忆（Procedural Memory）- 存储技能
    from metacog.memory.procedural_memory import ProceduralMemory
    procedural_memory = ProceduralMemory(
        collection_name="procedural_memory",
        persist_dir=memu_dir
    )
    skill_count = procedural_memory.count()
    skill_status = "新建" if skill_count == 0 else f"已有 {skill_count} 个"
    console.print(f"[cyan]程序记忆（技能）: {memu_dir} ({skill_status})[/cyan]")

    # 3. 情景记忆（Episodic Memory）- 存储成功案例
    from metacog.memory.episodic_memory import EpisodicMemory
    episodic_memory = EpisodicMemory(
        collection_name="episodic_memory",
        persist_dir=memu_dir
    )
    case_count = episodic_memory.count()
    case_status = "新建" if case_count == 0 else f"已有 {case_count} 个"
    console.print(f"[cyan]情景记忆（成功案例）: {memu_dir} ({case_status})[/cyan]")

    # ==========================================
    # YAML 记忆（保留为人类 debug 备份）
    # ==========================================
    memory_dir = output / "memory"
    memory_file = memory_dir / "memories.yaml"
    memory_dir.mkdir(parents=True, exist_ok=True)
    mem_store = MemoryStore(memory_file)  # 可选备份
    console.print(f"[dim]YAML 备份: {memory_file} ({len(mem_store)} 条)[/dim]")

    # 技能目录：seed skills 从 src 复制过来，动态生成的 skill 也放这里
    skills_dir = output / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    _seed_src = _root / "src" / "metacog" / "skills" / "math"
    if _seed_src.exists():
        import shutil
        for f in _seed_src.glob("skill_*.py"):
            dst = skills_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)

    # 技能注册：扫描 skills_dir 里所有 skill_*.py
    registry = SkillRegistry()
    loaded = registry.register_from_dir(skills_dir)
    if loaded:
        console.print(f"[cyan]Skills: {[s.name for s in loaded]}[/cyan]")

    # 输出目录
    output.mkdir(parents=True, exist_ok=True)
    traj_dir = output / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # 事件总线 + 四个 Agent
    # 注意：bus 按注册顺序同步调用，日志 handler 在 Agent handler 之前注册，
    # 确保打印先于实际处理，让用户能看到当前在哪个阶段。
    bus = EventBus()

    # ExecutorAgent：传入三层记忆 + RAG Top-K 参数
    executor = ExecutorAgent(
        litellm_model,
        bus,
        scaffold_data,
        mem_store,  # 保留兼容性
        registry,
        traj_dir,
        skills_dir=skills_dir,
        memu_client=None if no_semantic else memu_client,          # 🔥 语义记忆（消融可禁用）
        rag_top_k=2,
        procedural_memory=procedural_memory,                        # 🔥 程序记忆（技能）
        skill_top_k=3,
        episodic_memory=None if no_episodic else episodic_memory,  # 🔥 情景记忆（消融可禁用）
        case_top_k=1
    )

    # ── 日志：Solver 完成，Analyzer 即将开始（注册在 AnalyzerAgent 之前）
    @bus.on(EventType.TRAJECTORY)
    def log_trajectory(event: Event) -> None:
        d = event.data
        outcome = "[green]✓ 解出[/green]" if d.get("passed") else "[red]✗ 失败[/red]"
        console.print(
            f"  {outcome} | 步数={d.get('n_steps', 0)} | "
            f"耗时={d.get('time', 0):.1f}s | 答案={d.get('extracted_answer') or '—'}"
        )
        label = "成功" if d.get("passed") else "失败"
        console.print(f"  [dim]→ [Analyzer] 正在分析{label}轨迹...[/dim]")

    # AnalyzerAgent: 使用教师模型复盘，更精准提取数学痛点
    _analyzer = AnalyzerAgent(
        analyzer_model,  # 🔥 教师模型（GLM-5 或退化为学生模型）
        bus,
        chunk_size=3,
        enable_pot=True,
        enable_loop_detection=True
    )

    # 🔥 SuccessAnalyzer: 使用教师模型提取成功案例关键步骤
    from metacog.agents.success_analyzer import SuccessAnalyzer
    _success_analyzer = SuccessAnalyzer(
        analyzer_model,  # 🔥 教师模型
        bus,
        episodic_memory=episodic_memory
    )

    # 🔥 MemoryEvaluator: 定期评估记忆质量，清理低质量记忆
    from metacog.agents.memory_evaluator import MemoryEvaluatorAgent
    _memory_evaluator = MemoryEvaluatorAgent(
        bus,
        semantic_memory=memu_client,
        procedural_memory=procedural_memory,
        episodic_memory=episodic_memory,
        eval_interval=10,  # 每 10 道题评估一次
        quality_threshold=0.3,  # 低于 0.3 的记忆会被标记
        cleanup_threshold=0.2  # 低于 0.2 且未使用的记忆会被删除
    )
    console.print("[green]✓ 记忆评估智能体已启动（每 10 题评估一次）[/green]")

    # ── 日志：Analyzer 完成分析，MemoryManager 即将写入（注册在 MemoryManagerAgent 之前）
    @bus.on(EventType.ANALYSIS)
    def log_analysis(event: Event) -> None:
        a = event.data.get("analysis", {})
        tags = a.get("problem_tags", [])
        symptom = a.get("error_symptom", "?")
        console.print(f"  [dim]→ [MemoryManager] 写入 memU: [{'/'.join(tags[:2])}] {symptom}[/dim]")

    # MemoryManagerAgent：使用教师模型（实际不调用 LLM，但保持接口统一）
    # --no-semantic 时不写入语义记忆（消融实验）
    _mem_manager = MemoryManagerAgent(
        analyzer_model,
        bus,
        memory_store=mem_store,  # YAML 备份
        memu_persist_dir=None if no_semantic else memu_dir,  # 消融时禁止写入 memU
        collection_name="math_lessons"
    )

    # ── 日志：成功技术提取完成，SkillAgent 即将处理（注册在 SkillAgent 之前）
    @bus.on(EventType.SUCCESS_ANALYSIS)
    def log_success(event: Event) -> None:
        technique = event.data.get("technique", "?")
        tags = event.data.get("tags", [])
        console.print(f"  [dim]→ [SkillAgent] 成功模式: {technique} {tags}[/dim]")

    # SkillAgent：使用教师模型生成更高质量的 skill
    _skill_agent = SkillAgent(
        analyzer_model,  # 🔥 教师模型
        bus,
        registry,
        skills_dir,
        threshold=3,
        procedural_memory=procedural_memory
    )

    # ── 日志：记忆实际写入完成
    @bus.on(EventType.MEMORY_UPDATED)
    def log_memory(event: Event) -> None:
        action = event.data.get("action", "?")
        mem_id = event.data.get("memory_id", event.data.get("entry_id", "?"))
        pid = event.data.get("problem_id", "?")
        tags = event.data.get("tags", [])
        console.print(f"  [magenta]  ↑ memU {action}: {mem_id} | {tags} (题 {pid})[/magenta]")

    # ── 日志：Skill 文件生成
    @bus.on(EventType.SKILL_CREATED)
    def log_skill(event: Event) -> None:
        console.print(f"  [bold green]  ★ 新 Skill 生成: {event.data['name']} "
                      f"tags={event.data['tags']}[/bold green]")

    # 加载数据集
    console.print(f"\n[bold]加载数据集: {data_source}[/bold]")
    problems = load_dataset(data_source, base_path, max_instances, start=start)
    console.print(f"共 {len(problems)} 道题 | 语义记忆: {memu_client.count()} 条 | 情景记忆: {episodic_memory.count()} 条 | Skills: {len(registry)}\n")

    # 逐题运行
    results = []
    for i, prob in enumerate(problems, 1):
        pid = str(prob.get("id", f"prob_{i}"))
        console.rule(f"[bold cyan][{i}/{len(problems)}] {pid}[/bold cyan]")
        console.print(f"  [dim]→ [Solver] 解题中... (语义: {memu_client.count()} 条 | 情景: {episodic_memory.count()} 条 | RAG Top-K={executor.rag_top_k} | limit={scaffold_data.get('step_limit',10)} 步)[/dim]")
        record = executor.run_problem(prob)
        results.append(record)
        # ← log_trajectory handler 已经在 run_problem 内部打印了结果行
        console.print(f"  [dim]────────────────────────────── 累计: 语义记忆 {memu_client.count()} 条 | 情景记忆 {episodic_memory.count()} 条[/dim]")

    # 保存结果
    results_file = output / "results.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 摘要
    acc = compute_accuracy(results)
    console.print(f"\n[bold]结果: {acc['passed']}/{acc['total']} ({acc['pass_rate']:.1%})[/bold]")
    console.print(f"语义记忆: {memu_client.count()} 条 | 情景记忆: {episodic_memory.count()} 条")
    console.print(f"YAML 备份: {len(mem_store)} 条")
    console.print(f"Skill 数: {len(registry)}")
    console.print(f"输出: {output}")
    console.print(f"[dim]memU 数据库: {memu_dir}[/dim]")


if __name__ == "__main__":
    app()

