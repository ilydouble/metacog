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

DEFAULT_SYSTEM = "You are a helpful assistant that solves math problems."
DEFAULT_INSTANCE = """\
Please solve the following math problem:

{{task}}

## Instructions
1. Analyze the problem carefully.
2. Use bash to run Python code for calculations.
3. Submit your answer with:
   printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n%s\\n' ANSWER

For AIME problems, the answer is an integer from 0 to 999.
"""


# ------------------------------------------------------------------ #
# 数据加载（复用 run_math_test_evolve 的逻辑）
# ------------------------------------------------------------------ #

def load_dataset(data_source: str, base_path: str, max_instances: int) -> list[dict]:
    data_file = Path(base_path) / f"{data_source}.json"
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    problems = []
    with open(data_file) as f:
        for i, line in enumerate(f):
            if max_instances and i >= max_instances:
                break
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
    model: Annotated[str, typer.Option("--model", "-m")] = "lm_studio/qwen/qwen3.5-9b",
    api_base: Annotated[Optional[str], typer.Option("--api-base")] = "http://0.0.0.0:1234/v1",
    config: Annotated[Optional[Path], typer.Option("--config", "-c")] = None,
    base_path: Annotated[str, typer.Option("--base-path")] = "datasets/math/data",
) -> None:
    """Math 多智能体测试（每题结束立即分析并更新记忆）。

    记忆默认存储在 <output>/memory/memories.yaml。
    不同 --output 实验天然隔离；--fresh 可重置当前实验的记忆。
    """

    # 加载 scaffold
    scaffold_data: dict = {"system_template": DEFAULT_SYSTEM, "instance_template": DEFAULT_INSTANCE,
                           "step_limit": 10, "cost_limit": 2.0}
    if config:
        cfg = yaml.safe_load(config.read_text()) or {}
        scaffold_data.update(cfg.get("agent", {}))
    if scaffold and scaffold.exists():
        sc = yaml.safe_load(scaffold.read_text()) or {}
        for k in ("system_template", "instance_template", "step_limit", "cost_limit"):
            if k in sc:
                scaffold_data[k] = sc[k]
        console.print(f"[green]✓ scaffold: {scaffold}[/green]")

    # 模型
    model_kwargs = {"temperature": 0.0, "max_tokens": 8192, "drop_params": True, "tool_choice": "required"}
    if api_base:
        model_kwargs["api_base"] = api_base
        model_kwargs["api_key"] = "lm-studio"
    litellm_model = LitellmModel(model_name=model, model_kwargs=model_kwargs,
                                 cost_tracking="ignore_errors")

    # 记忆存储：记忆跟着实验走，住在 <output>/memory/
    memory_dir = output / "memory"
    memory_file = memory_dir / "memories.yaml"
    memory_dir.mkdir(parents=True, exist_ok=True)

    if fresh and memory_file.exists():
        memory_file.unlink()
        console.print(f"[yellow]✓ --fresh: 已清空记忆 {memory_file}[/yellow]")

    mem_store = MemoryStore(memory_file)
    status = "新建" if len(mem_store) == 0 else f"续跑，已有 {len(mem_store)} 条"
    console.print(f"[cyan]记忆: {memory_file} ({status})[/cyan]")

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
    executor = ExecutorAgent(litellm_model, bus, scaffold_data, mem_store, registry, traj_dir,
                             skills_dir=skills_dir)

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

    _analyzer = AnalyzerAgent(litellm_model, bus)

    # ── 日志：Analyzer 完成分析，MemoryManager 即将写入（注册在 MemoryManagerAgent 之前）
    @bus.on(EventType.ANALYSIS)
    def log_analysis(event: Event) -> None:
        a = event.data.get("analysis", {})
        title = a.get("lesson_title", "?")
        ftype = a.get("failure_type", "?")
        console.print(f"  [dim]→ [MemoryManager] 写入记忆: [{ftype}] {title}[/dim]")

    _mem_manager = MemoryManagerAgent(litellm_model, bus, mem_store)

    # ── 日志：成功技术提取完成，SkillAgent 即将处理（注册在 SkillAgent 之前）
    @bus.on(EventType.SUCCESS_ANALYSIS)
    def log_success(event: Event) -> None:
        technique = event.data.get("technique", "?")
        tags = event.data.get("tags", [])
        console.print(f"  [dim]→ [SkillAgent] 成功模式: {technique} {tags}[/dim]")

    _skill_agent = SkillAgent(litellm_model, bus, registry, skills_dir, threshold=3)

    # ── 日志：记忆实际写入完成
    @bus.on(EventType.MEMORY_UPDATED)
    def log_memory(event: Event) -> None:
        action = event.data.get("action", "?")
        eid = event.data.get("entry_id", "?")
        pid = event.data.get("problem_id", "?")
        console.print(f"  [magenta]  ↑ 记忆 {action}: {eid} (题 {pid})[/magenta]")

    # ── 日志：Skill 文件生成
    @bus.on(EventType.SKILL_CREATED)
    def log_skill(event: Event) -> None:
        console.print(f"  [bold green]  ★ 新 Skill 生成: {event.data['name']} "
                      f"tags={event.data['tags']}[/bold green]")

    # 加载数据集
    console.print(f"\n[bold]加载数据集: {data_source}[/bold]")
    problems = load_dataset(data_source, base_path, max_instances)
    console.print(f"共 {len(problems)} 道题 | 记忆: {len(mem_store)} 条 | Skills: {len(registry)}\n")

    # 逐题运行
    results = []
    for i, prob in enumerate(problems, 1):
        pid = str(prob.get("id", f"prob_{i}"))
        console.rule(f"[bold cyan][{i}/{len(problems)}] {pid}[/bold cyan]")
        console.print(f"  [dim]→ [Solver] 解题中... (记忆 {len(mem_store)} 条 | limit={scaffold_data.get('step_limit',10)} 步)[/dim]")
        record = executor.run_problem(prob)
        results.append(record)
        # ← log_trajectory handler 已经在 run_problem 内部打印了结果行
        console.print(f"  [dim]────────────────────────────── 累计: 记忆 {len(mem_store)} 条[/dim]")

    # 保存结果
    results_file = output / "results.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 摘要
    acc = compute_accuracy(results)
    console.print(f"\n[bold]结果: {acc['passed']}/{acc['total']} ({acc['pass_rate']:.1%})[/bold]")
    console.print(f"记忆条数: {len(mem_store)}")
    console.print(f"Skill 数: {len(registry)}")
    console.print(f"输出: {output}")


if __name__ == "__main__":
    app()

