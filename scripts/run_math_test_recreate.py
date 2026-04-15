#!/usr/bin/env python3
"""Math 数据集 ReCreate 进化测试脚本

ReCreate 策略：题目级别优化 + 批次级别合成
1. 初始化 scaffold 工作区（global_v000，含 agent_tools / agent_memory）
2. 每一批（round）用当前 scaffold 跑一批数学题
3. 每道题跑完后立刻触发 ReCreate-Agent 分析轨迹，提出局部改动
4. 一批全部跑完后，由合成元智能体（run_batch_synthesis）统一整合所有改动
5. 生成新的 global_vXXX，更新 current 软链接，继续下一轮

目录结构：
  outputs/math_recreate/
    recreate_workspace/
      global_v000/scaffold.yaml   ← 初始版本
      global_v001/scaffold.yaml   ← 第 0 批合成后
      current -> global_vXXX      ← 软链接
    batch_000/                    ← 第 0 批
      aime24_0000/
        scaffold.yaml             ← 拷贝自当前全局
        agent_tools/
        agent_memory/
        aime24_0000.traj.json     ← Solver 轨迹
        eval_result.json
        scaffold_diff.txt         ← ReCreate-Agent 产生的改动（如有）
    runs_recreate/                ← ReCreate-Agent 自身轨迹
    evolution_log.json
"""

import os

os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

# ── 路径设置 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src" / "mini-swe-agent" / "src"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

from recreate_agent.adapters.base import UnifiedInstance, UnifiedResult
from recreate_agent.adapters.math_adapter import MathAdapter

from utils.answer_extraction import extract_final_answer, normalize_answer
from utils.evaluation import compare_answers, compute_accuracy
from evolve_utils import safe_load_yaml
from evolve_utils.scaffold_ops import init_agent_memory, load_scaffold
from evolve_utils.utils import BatchResult
from evolve_utils.evolution import run_recreate_evolution_isolated, run_batch_synthesis

app = typer.Typer()
console = Console()

# ── 初始 Scaffold（与 evolve 版本相同，作为起点）───────────────────────────────

INITIAL_SYSTEM_TEMPLATE = "You are a helpful assistant that solves math problems."

INITIAL_INSTANCE_TEMPLATE = """Please solve the following math problem:

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


# ── VerboseAgent（与 evolve 版本相同）────────────────────────────────────────

class VerboseAgent(DefaultAgent):
    """实时打印每一步思考和执行过程的 Agent。"""

    def query(self) -> dict:
        print(f"\n=== Step {self.n_calls + 1} · 正在调用模型... ===")
        message = super().query()
        content = message.get("content", "") or ""
        if content:
            preview = content[:600] if len(content) <= 600 else content[:600] + "\n...(截断)"
            print(f"--- 模型回复 ---\n{preview}\n--- 回复结束 ---")
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        for action in actions:
            cmd = action.get("command", "")
            print(f">>> 执行命令: {cmd[:200]}")
        observations = super().execute_actions(message)
        for obs in observations:
            output = obs.get("extra", {}).get("raw_output", "") or obs.get("content", "")
            if output:
                preview = output[:300] if len(output) <= 300 else output[:300] + "\n...(截断)"
                print(f"<<< 执行结果:\n{preview}")
        return observations


# ── Workspace & Scaffold 管理 ────────────────────────────────────────────────

def init_recreate_workspace(workspace: Path) -> Path:
    """初始化 ReCreate workspace，创建 global_v000（含 agent_tools / agent_memory）。"""
    workspace.mkdir(parents=True, exist_ok=True)
    v000 = workspace / "global_v000"
    v000.mkdir(exist_ok=True)

    scaffold_file = v000 / "scaffold.yaml"
    if not scaffold_file.exists():
        scaffold = {
            "system_template": INITIAL_SYSTEM_TEMPLATE,
            "instance_template": INITIAL_INSTANCE_TEMPLATE,
            "step_limit": 10,
            "cost_limit": 2.0,
        }
        scaffold_file.write_text(yaml.dump(scaffold, allow_unicode=True, default_style=None))
        console.print(f"[green]✓ 初始化 scaffold: {v000}[/green]")

    # 初始化 agent_tools 和 agent_memory
    tools_src = ROOT / "src" / "recreate_agent" / "tools"
    agent_tools_dst = v000 / "agent_tools"
    if not agent_tools_dst.exists() and tools_src.exists():
        shutil.copytree(tools_src, agent_tools_dst)

    init_agent_memory(v000)

    current = workspace / "current"
    if not current.exists() and not current.is_symlink():
        current.symlink_to("global_v000")

    return v000


def load_current_scaffold(workspace: Path) -> dict:
    """从 current 软链接加载当前 scaffold。"""
    current = workspace / "current"
    scaffold_file = current / "scaffold.yaml"
    data = safe_load_yaml(scaffold_file)
    if not data:
        raise RuntimeError(f"无法加载 scaffold: {scaffold_file}")
    return data


def update_current_symlink(workspace: Path, version_name: str):
    """更新 current 软链接指向新版本。"""
    current = workspace / "current"
    if current.exists() or current.is_symlink():
        current.unlink()
    current.symlink_to(version_name)


# ── 数据加载 ─────────────────────────────────────────────────────────────────

def load_dataset(data_source: str, base_path: str, max_instances: int | None = None) -> list[dict]:
    """加载数学题目数据集，返回原始 dict 列表。"""
    data_file = Path(base_path) / f"{data_source}.json"
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    problems = []
    with open(data_file) as f:
        for i, line in enumerate(f):
            if max_instances is not None and i >= max_instances:
                break
            try:
                problems.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return problems


def problem_to_unified_instance(problem: dict, data_source: str, orig_idx: int) -> UnifiedInstance:
    """把原始 dict 转为 UnifiedInstance。"""
    problem_text = problem.get("problem") or problem.get("question", "")
    answer = str(problem.get("expected_answer") or problem.get("answer", "")).strip()
    instance_id = problem.get("id", f"{data_source}_{orig_idx:04d}")
    subject = problem.get("source", data_source)
    return UnifiedInstance(
        instance_id=str(instance_id),
        problem_statement=problem_text,
        difficulty="",
        category=subject,
        domain_data={"answer": answer, "source": data_source, "orig_idx": orig_idx},
    )


# ── 单题运行（ReCreate 版）──────────────────────────────────────────────────

def run_single_problem_recreate(
    instance: UnifiedInstance,
    model: LitellmModel,
    scaffold: dict,
    instance_dir: Path,
    current_global_dir: Path,
) -> tuple[dict, UnifiedResult, str]:
    """
    运行单个题目，保存轨迹到 instance_dir，返回 (result_dict, UnifiedResult, exit_status)。
    """
    instance_dir.mkdir(parents=True, exist_ok=True)

    # 拷贝当前全局 scaffold 到 instance 目录（供 ReCreate-Agent 修改）
    shutil.copy(current_global_dir / "scaffold.yaml", instance_dir / "scaffold.yaml")

    tools_src = current_global_dir / "agent_tools"
    tools_dst = instance_dir / "agent_tools"
    if tools_src.exists() and not tools_dst.exists():
        shutil.copytree(tools_src, tools_dst)

    memory_src = current_global_dir / "agent_memory"
    memory_dst = instance_dir / "agent_memory"
    if memory_src.exists() and not memory_dst.exists():
        shutil.copytree(memory_src, memory_dst)
    else:
        init_agent_memory(instance_dir)

    traj_path = instance_dir / f"{instance.instance_id}.traj.json"
    problem = instance.problem_statement
    expected_answer = instance.domain_data.get("answer", "")

    start_time = time.time()
    error = None
    extracted_answer = None
    passed = False
    cost = 0.0
    n_steps = 0
    submission = ""
    exit_status = "unknown"

    try:
        env = LocalEnvironment()
        agent = VerboseAgent(
            model=model,
            env=env,
            system_template=scaffold.get("system_template", INITIAL_SYSTEM_TEMPLATE),
            instance_template=scaffold.get("instance_template", INITIAL_INSTANCE_TEMPLATE),
            step_limit=scaffold.get("step_limit", 10),
            cost_limit=scaffold.get("cost_limit", 2.0),
            output_path=traj_path,
        )
        result_info = agent.run(task=problem)
        exit_status = result_info.get("exit_status", "unknown")
        submission = result_info.get("submission", "")

        extracted_answer = extract_final_answer(submission)
        if extracted_answer:
            extracted_answer = normalize_answer(extracted_answer)

        if not extracted_answer:
            for msg in reversed(agent.messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "") or ""
                    fallback = extract_final_answer(content)
                    if fallback:
                        extracted_answer = normalize_answer(fallback)
                        break

        if extracted_answer:
            passed = compare_answers(extracted_answer, expected_answer)

        cost = agent.cost
        n_steps = agent.n_calls

    except Exception as e:
        error = str(e)
        exit_status = type(e).__name__
        console.print(f"[red]错误: {instance.instance_id} - {e}[/red]")

    elapsed = round(time.time() - start_time, 2)

    eval_result_data = {
        "instance_id": instance.instance_id,
        "success": passed,
        "score": 1.0 if passed else 0.0,
        "expected_answer": expected_answer,
        "extracted_answer": extracted_answer or "",
        "exit_status": exit_status,
        "n_steps": n_steps,
        "cost": cost,
        "time": elapsed,
        "error": error or "",
    }
    (instance_dir / "eval_result.json").write_text(
        json.dumps(eval_result_data, indent=2, ensure_ascii=False)
    )

    unified_result = UnifiedResult(
        instance_id=instance.instance_id,
        success=passed,
        score=1.0 if passed else 0.0,
        error=error or "",
        details=eval_result_data,
        formatted_output=submission[:500] if submission else "",
    )

    result_dict = {
        "id": instance.instance_id,
        "source": instance.domain_data.get("source", ""),
        "problem": problem[:200] + "..." if len(problem) > 200 else problem,
        "expected_answer": expected_answer,
        "extracted_answer": extracted_answer,
        "submission": submission[:500] if submission else "",
        "passed": passed,
        "cost": cost,
        "n_steps": n_steps,
        "time": elapsed,
        "error": error,
        "exit_status": exit_status,
    }
    return result_dict, unified_result, exit_status


# ── 批次结果保存 ──────────────────────────────────────────────────────────────

def save_batch_results(results: list[dict], batch_dir: Path, batch_idx: int, scaffold_version: int) -> dict:
    batch_dir.mkdir(parents=True, exist_ok=True)
    results_file = batch_dir / "results.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    accuracy = compute_accuracy(results)
    summary = {
        "batch": batch_idx,
        "scaffold_version": scaffold_version,
        "timestamp": datetime.now().isoformat(),
        **accuracy,
        "total_cost": round(sum(r["cost"] for r in results), 4),
        "total_time": round(sum(r["time"] for r in results), 2),
    }
    (batch_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def print_batch_summary(summary: dict):
    console.print(
        f"  批次 {summary['batch']} (scaffold v{summary['scaffold_version']:03d}): "
        f"通过率 [cyan]{summary['pass_rate']:.1%}[/cyan] "
        f"({summary['passed']}/{summary['total']}) | "
        f"成本 ${summary['total_cost']:.4f} | "
        f"耗时 {summary['total_time']:.1f}s"
    )


def print_final_comparison(all_summaries: list[dict]):
    table = Table(title="ReCreate 进化结果对比")
    table.add_column("批次", style="cyan")
    table.add_column("Scaffold 版本", style="dim")
    table.add_column("通过率", style="green")
    table.add_column("通过/总数", style="white")
    table.add_column("成本", style="yellow")
    table.add_column("耗时", style="white")
    for s in all_summaries:
        table.add_row(
            str(s["batch"]),
            f"v{s['scaffold_version']:03d}",
            f"{s['pass_rate']:.1%}",
            f"{s['passed']}/{s['total']}",
            f"${s['total_cost']:.4f}",
            f"{s['total_time']:.1f}s",
        )
    console.print(table)


# ── 主命令 ────────────────────────────────────────────────────────────────────

@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", "-c", help="配置文件路径")] = Path(
        "scripts/configs/math_test_config.yaml"
    ),
    data_source: Annotated[str, typer.Option("--data-source", "-d", help="数据源")] = "aime24",
    max_instances: Annotated[int, typer.Option("--max-instances", "-n", help="每批最多测试题数")] = 5,
    max_rounds: Annotated[int, typer.Option("--max-rounds", "-r", help="最多进化批次数")] = 3,
    output: Annotated[Path, typer.Option("--output", "-o", help="输出根目录")] = Path(
        "outputs/math_recreate"
    ),
    recreate_model: Annotated[str | None, typer.Option("--recreate-model", help="ReCreate-Agent 模型（默认与 solver 相同，优先级低于 --teacher-model）")] = None,
    recreate_temp: Annotated[float, typer.Option("--recreate-temp", help="ReCreate-Agent / 教师模型温度")] = 0.2,
    teacher_model: Annotated[str | None, typer.Option("--teacher-model", help="ReCreate-Agent 和 Synthesis 使用的教师模型，如 zai/glm-4.7（优先级高于 --recreate-model）")] = None,
    teacher_api_key: Annotated[str | None, typer.Option("--teacher-api-key", help="教师模型的 API Key（智谱 ZAI_API_KEY）")] = None,
):
    """运行 Math 数据集 ReCreate 进化测试（题目级别优化 + 批次级别合成）"""
    with open(config) as f:
        config_data = yaml.safe_load(f)

    model_config = config_data.get("model", {})
    model_name = model_config.get("model", "gpt-4o-mini")
    temperature = model_config.get("temperature", 0.0)
    max_tokens = model_config.get("max_tokens", 4096)
    api_base = model_config.get("api_base", None)
    api_key = model_config.get("api_key", None)
    think = model_config.get("think", None)
    extra_body = model_config.get("extra_body", None)

    # ── 学生模型（求解）──────────────────────────────────────────────────────────
    model_kwargs: dict = {"temperature": temperature, "max_tokens": max_tokens, "drop_params": True}
    if api_base:
        model_kwargs["api_base"] = api_base
        model_kwargs["api_key"] = api_key or "lm-studio"
        # 供 evolution.py 里的 ReCreate-Agent 使用（仅学生模型 base，teacher 另行覆盖）
        os.environ["LLM_API_BASE"] = api_base
        os.environ["LLM_API_KEY"] = api_key or "lm-studio"

    # 关闭 thinking 模式（二选一，按部署方式选择）：
    # - LM Studio：think: false
    # - vLLM：extra_body.chat_template_kwargs.enable_thinking: false
    if think is not None:
        model_kwargs["think"] = think
        os.environ["LLM_THINK"] = str(think).lower()  # 供 evolution.py 里的 ReCreate-Agent 使用
    if extra_body is not None:
        model_kwargs["extra_body"] = extra_body
        # 序列化后传给 evolution.py，让 ReCreate-Agent（LitellmTextbasedModel）也能透传
        os.environ["LLM_EXTRA_BODY"] = json.dumps(extra_body)
    else:
        os.environ.pop("LLM_EXTRA_BODY", None)

    # ── 教师模型（ReCreate-Agent + Batch Synthesis）──────────────────────────────
    if teacher_model:
        _teacher_key = (
            teacher_api_key
            or os.getenv("ZAI_API_KEY")
            or os.getenv("ZHIPUAI_API_KEY")
        )
        if not _teacher_key:
            console.print("[red]错误：使用教师模型需要提供 --teacher-api-key 或设置 ZAI_API_KEY 环境变量[/red]")
            raise typer.Exit(1)
        recreate_model_name = teacher_model
        # 写入专用 env vars，供 evolution.py 优先读取（覆盖 LLM_API_BASE）
        os.environ["RECREATE_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"
        os.environ["RECREATE_API_KEY"] = _teacher_key
    else:
        # 未指定教师模型，退化为 --recreate-model 或学生模型
        recreate_model_name = recreate_model or model_name
        # 确保 RECREATE_* 不残留（防止环境变量污染）
        os.environ.pop("RECREATE_API_BASE", None)
        os.environ.pop("RECREATE_API_KEY", None)

    console.print(f"\n[bold blue]Math ReCreate 进化测试[/bold blue]")
    console.print(f"数据源: [cyan]{data_source}[/cyan]  每批题数: [cyan]{max_instances}[/cyan]  批次: [cyan]{max_rounds}[/cyan]")
    console.print(f"[cyan]🎓 学生模型（求解）: {model_name}[/cyan]")
    if teacher_model:
        console.print(f"[magenta]🧑‍🏫 教师模型（ReCreate + Synthesis）: {recreate_model_name}[/magenta]")
    else:
        console.print(f"[yellow]⚠️  未指定教师模型，ReCreate/Synthesis 使用: {recreate_model_name}[/yellow]")
    console.print(f"输出目录: [cyan]{output}[/cyan]\n")

    base_path = config_data.get("dataset", {}).get("base_path", "datasets/math/data")
    try:
        raw_problems = load_dataset(data_source, base_path, max_instances)
        console.print(f"[green]已加载 {len(raw_problems)} 个问题[/green]\n")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # 转换为 UnifiedInstance
    instances = [
        problem_to_unified_instance(p, data_source, i) for i, p in enumerate(raw_problems)
    ]

    # 初始化 ReCreate workspace
    workspace = output / "recreate_workspace"
    init_recreate_workspace(workspace)

    runs_recreate_dir = output / "runs_recreate"
    runs_recreate_dir.mkdir(parents=True, exist_ok=True)

    # 获取 domain config（通过 MathAdapter）
    adapter = MathAdapter(data_source=data_source)
    domain_config = adapter.get_recreate_agent_config()
    domain = domain_config.get("domain", "math")

    # 创建 solver model（LocalEnvironment 每题单独创建，防止状态污染）
    litellm_model = LitellmModel(
        model_name=model_name,
        model_kwargs=model_kwargs,
        cost_tracking="ignore_errors",
    )

    all_summaries = []
    current_version = 0  # 对应 global_v000

    for batch_idx in range(max_rounds):
        console.print(f"\n[bold]{'='*60}[/bold]")
        console.print(f"[bold cyan]批次 {batch_idx} (Scaffold v{current_version:03d})[/bold cyan]")
        console.print(f"[bold]{'='*60}[/bold]")

        current_global_dir = workspace / f"global_v{current_version:03d}"
        scaffold = load_current_scaffold(workspace)

        batch_dir = output / f"batch_{batch_idx:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict] = []
        batch_results: dict[str, BatchResult] = {}

        for i, instance in enumerate(instances):
            console.print(f"  [{i+1}/{len(instances)}] {instance.instance_id}")
            instance_dir = batch_dir / instance.instance_id

            r, unified_result, exit_status = run_single_problem_recreate(
                instance=instance,
                model=litellm_model,
                scaffold=scaffold,
                instance_dir=instance_dir,
                current_global_dir=current_global_dir,
            )
            results.append(r)
            traj_path = instance_dir / f"{instance.instance_id}.traj.json"

            status = "[green]✓[/green]" if r["passed"] else "[red]✗[/red]"
            console.print(
                f"  {status} 期望={r['expected_answer']}  提取={r['extracted_answer'] or '(无)'}  步数={r['n_steps']}"
            )

            # 最后一批跳过进化（只评测）
            if batch_idx < max_rounds - 1:
                console.print(f"    [dim]>>> 触发 ReCreate-Agent 分析轨迹...[/dim]")
                try:
                    scaffold_changed = run_recreate_evolution_isolated(
                        instance=instance,
                        eval_result=unified_result,
                        agent_exit_status=exit_status,
                        working_dir=instance_dir,
                        base_scaffold_dir=current_global_dir,
                        traj_path=traj_path,
                        container_id="",
                        recreate_model=recreate_model_name,
                        recreate_temp=recreate_temp,
                        domain=domain,
                        domain_config=domain_config,
                        runs_recreate_dir=runs_recreate_dir,
                    )
                except Exception as e:
                    scaffold_changed = False
                    console.print(f"    [yellow]ReCreate-Agent 出错: {e}[/yellow]")

                batch_results[instance.instance_id] = BatchResult(
                    instance_id=instance.instance_id,
                    success=r["passed"],
                    score=1.0 if r["passed"] else 0.0,
                    scaffold_changed=scaffold_changed,
                    has_new_tools=False,
                    has_new_memories=False,
                    error=r.get("error") or "",
                    exit_status=exit_status,
                    duration=r["time"],
                )

        # 保存批次结果
        summary = save_batch_results(results, batch_dir, batch_idx, current_version)
        all_summaries.append(summary)
        console.print()
        print_batch_summary(summary)

        # 最后一批不合成
        if batch_idx == max_rounds - 1:
            console.print("[dim]最后一批，不执行 batch synthesis。[/dim]")
            break

        # Batch Synthesis：合并所有实例改动成新版本
        new_version = current_version + 1
        new_global_dir = workspace / f"global_v{new_version:03d}"

        console.print(f"\n[yellow]>>> Batch Synthesis: 合成新版 scaffold v{new_version:03d}...[/yellow]")
        try:
            synthesis_ok = run_batch_synthesis(
                workspace=workspace,
                batch_dir=batch_dir,
                batch_results=batch_results,
                current_global_dir=current_global_dir,
                new_global_dir=new_global_dir,
                batch_idx=batch_idx,
                recreate_model=recreate_model_name,
                recreate_temp=recreate_temp,
                domain=domain,
                domain_config=domain_config,
                runs_recreate_dir=runs_recreate_dir,
            )
        except Exception as e:
            synthesis_ok = False
            console.print(f"[red]Batch Synthesis 失败: {e}[/red]")
            if not new_global_dir.exists():
                shutil.copytree(current_global_dir, new_global_dir)

        if synthesis_ok or new_global_dir.exists():
            current_version = new_version
            update_current_symlink(workspace, f"global_v{current_version:03d}")
            console.print(f"[green]✓ 已切换至新版本 global_v{current_version:03d}[/green]")
        else:
            console.print("[dim]合成未产生新版本，继续使用当前版本。[/dim]")

    # 保存进化日志
    log_file = output / "evolution_log.json"
    log_file.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False))

    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print("[bold]ReCreate 进化测试完成[/bold]")
    console.print(f"[bold]{'='*60}[/bold]\n")
    print_final_comparison(all_summaries)
    console.print(f"\n[green]所有结果已保存到: {output}[/green]")


if __name__ == "__main__":
    app()

