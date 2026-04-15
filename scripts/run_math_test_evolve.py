#!/usr/bin/env python3
"""Math 数据集进化测试脚本

基于 evolve_utils 的简化进化方法：
1. 初始化 scaffold（系统提示 + 实例模板）存入 evolve_workspace/global_v000/
2. 每轮用当前 scaffold 运行一批数学题
3. 收集失败案例，用元智能体（同一 LLM）分析并优化 scaffold
4. 保存改进版本到 global_v001, global_v002, ...，并更新 current 软链接
5. 多轮迭代，最终对比各轮通过率

目录结构：
  outputs/math_evolve/
    evolve_workspace/
      global_v000/scaffold.yaml   # 初始版本
      global_v001/scaffold.yaml   # 第 1 轮优化后
      current -> global_vXXX      # 软链接指向最新版本
    round_000/                    # 第 0 轮结果（使用 global_v000）
      results.jsonl
      run_summary.json
      trajectories/
    round_001/                    # 第 1 轮结果（使用 global_v001）
      ...
    evolution_log.json            # 每轮通过率对比
"""

import os

os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Optional

import litellm
import typer
import yaml
from rich.console import Console
from rich.table import Table

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "mini-swe-agent" / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

from utils.answer_extraction import extract_final_answer, normalize_answer
from utils.evaluation import compare_answers, compute_accuracy
from evolve_utils import safe_load_yaml

app = typer.Typer()
console = Console()

# ============================================================
# 初始 Scaffold 模板（与基线相同，作为起点）
# ============================================================

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

# ============================================================
# 元智能体 Prompt（用于分析失败并优化 scaffold）
# ============================================================

META_AGENT_SYSTEM = """You are a meta-agent that analyzes AI assistant failures on math problems \
and improves the assistant's prompt templates to help it perform better.

You will be given:
1. The current system template and instance template
2. A summary of failed problems: the problem statement, expected answer, what the agent submitted, and error patterns

Your task: analyze the failure patterns and propose improved templates that help the agent:
- Better structure its mathematical reasoning
- Avoid the common mistakes you observe
- More reliably submit its final answer in the correct format
- Handle difficult computation within the 30-second bash timeout limit

IMPORTANT: Respond with ONLY a YAML code block. No explanation outside the YAML.

Format:
```yaml
system_template: |
  <improved system template>

instance_template: |
  <improved instance template>
```

Rules:
- Keep the {{task}} placeholder in instance_template (exactly as-is)
- Keep the submission format: printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n%s\\n' ANSWER
- Do NOT add {{memory}} or other undefined placeholders
- Your YAML must be valid and parseable"""

META_AGENT_INSTANCE_TPL = """## Current Templates

### System Template
{system_template}

### Instance Template
{instance_template}

## Failure Summary ({n_failures}/{n_total} problems failed in this round)

{failure_summaries}

## Task
Analyze the failure patterns above. What went wrong? How can the templates be improved?
Respond with ONLY the improved YAML block (system_template + instance_template)."""


# ============================================================
# VerboseAgent（同基线）
# ============================================================

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


# ============================================================
# Scaffold 管理
# ============================================================

def init_scaffold_workspace(workspace: Path) -> Path:
    """初始化 scaffold 工作区，创建 global_v000 和 current 软链接。"""
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


def save_scaffold_version(workspace: Path, scaffold: dict, version: int) -> Path:
    """保存新版本 scaffold 并更新 current 软链接。"""
    version_name = f"global_v{version:03d}"
    version_dir = workspace / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    scaffold_file = version_dir / "scaffold.yaml"
    scaffold_file.write_text(yaml.dump(scaffold, allow_unicode=True, default_style=None))

    # 更新 current 软链接
    current = workspace / "current"
    if current.exists() or current.is_symlink():
        current.unlink()
    current.symlink_to(version_name)

    console.print(f"[green]✓ 保存新版本: {version_name}[/green]")
    return version_dir


# ============================================================
# 单题运行
# ============================================================

def run_single_problem(
    problem_data: dict,
    model: LitellmModel,
    env: LocalEnvironment,
    scaffold: dict,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """运行单个问题，返回结果字典。"""
    problem_id = str(problem_data.get("id", "unknown"))
    # 兼容 amc23 ("question") 和 aime24/25 ("problem")
    problem = problem_data.get("problem") or problem_data.get("question", "")
    # 兼容 aime24 ("expected_answer") 和 aime25/amc23 ("answer"，可能是整数)
    expected_answer = str(
        problem_data.get("expected_answer") or problem_data.get("answer", "")
    ).strip()

    start_time = time.time()
    error = None
    extracted_answer = None
    passed = False
    cost = 0.0
    n_steps = 0
    submission = ""

    try:
        agent = VerboseAgent(
            model=model,
            env=env,
            system_template=scaffold.get("system_template", INITIAL_SYSTEM_TEMPLATE),
            instance_template=scaffold.get("instance_template", INITIAL_INSTANCE_TEMPLATE),
            step_limit=scaffold.get("step_limit", 10),
            cost_limit=scaffold.get("cost_limit", 2.0),
            output_path=output_dir / f"{problem_id}.traj.json" if output_dir else None,
        )

        result = agent.run(task=problem)
        submission = result.get("submission", "")
        print(f"[DEBUG] submission = {repr(submission[:200]) if submission else repr(submission)}")

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
        console.print(f"[red]错误: {problem_id} - {e}[/red]")

    return {
        "id": problem_id,
        "source": problem_data.get("source", ""),
        "problem": problem[:200] + "..." if len(problem) > 200 else problem,
        "expected_answer": expected_answer,
        "extracted_answer": extracted_answer,
        "submission": submission[:500] if submission else "",
        "passed": passed,
        "cost": cost,
        "n_steps": n_steps,
        "time": round(time.time() - start_time, 2),
        "error": error,
    }


# ============================================================
# 元智能体：分析失败，优化 scaffold
# ============================================================

def build_failure_summaries(results: list[dict]) -> str:
    """构建失败案例摘要，供元智能体分析。"""
    failures = [r for r in results if not r["passed"]]
    lines = []
    for i, r in enumerate(failures, 1):
        lines.append(f"### Failure {i}: {r['id']}")
        lines.append(f"- Problem (truncated): {r['problem'][:300]}")
        lines.append(f"- Expected: {r['expected_answer']}")
        lines.append(f"- Extracted: {r['extracted_answer'] or '(none)'}")
        lines.append(f"- Submission: {repr(r['submission'][:200]) if r['submission'] else '(empty)'}")
        lines.append(f"- Steps taken: {r['n_steps']}")
        if r["error"]:
            lines.append(f"- Error: {r['error'][:200]}")
        lines.append("")
    return "\n".join(lines) if lines else "No failures."


def parse_yaml_from_response(response_text: str) -> dict | None:
    """从元智能体响应中提取 YAML 代码块并解析。"""
    # 尝试找 ```yaml ... ``` 代码块
    match = re.search(r"```(?:yaml)?\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        yaml_text = match.group(1)
    else:
        # 尝试整个响应作为 YAML
        yaml_text = response_text

    try:
        data = yaml.safe_load(yaml_text)
        if isinstance(data, dict) and "system_template" in data and "instance_template" in data:
            return data
    except yaml.YAMLError:
        pass
    return None


def evolve_scaffold(
    scaffold: dict,
    results: list[dict],
    model_name: str,
    model_kwargs: dict,
) -> dict | None:
    """
    调用元智能体（LLM）分析失败案例，返回优化后的 scaffold。
    如果解析失败返回 None（保持原版本不变）。
    """
    n_total = len(results)
    n_failures = sum(1 for r in results if not r["passed"])

    if n_failures == 0:
        console.print("[green]所有题目通过，无需优化 scaffold！[/green]")
        return None

    failure_summaries = build_failure_summaries(results)

    user_message = META_AGENT_INSTANCE_TPL.format(
        system_template=scaffold.get("system_template", ""),
        instance_template=scaffold.get("instance_template", ""),
        n_failures=n_failures,
        n_total=n_total,
        failure_summaries=failure_summaries,
    )

    console.print(f"[yellow]元智能体分析中（{n_failures}/{n_total} 题失败）...[/yellow]")

    try:
        response = litellm.completion(
            model=model_name,
            messages=[
                {"role": "system", "content": META_AGENT_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            **{k: v for k, v in model_kwargs.items() if k not in ("drop_params",)},
        )
        response_text = response.choices[0].message.content or ""
        console.print(f"[dim]元智能体响应长度: {len(response_text)} 字符[/dim]")
    except Exception as e:
        console.print(f"[red]元智能体调用失败: {e}[/red]")
        return None

    improved = parse_yaml_from_response(response_text)
    if improved:
        # 保留 step_limit / cost_limit 不变
        improved["step_limit"] = scaffold.get("step_limit", 10)
        improved["cost_limit"] = scaffold.get("cost_limit", 2.0)
        console.print("[green]✓ 成功解析改进后的 scaffold[/green]")
        return improved
    else:
        console.print("[yellow]警告: 无法从元智能体响应中解析有效 YAML，保持原 scaffold[/yellow]")
        return None


# ============================================================
# 数据加载 & 结果保存（复用基线逻辑）
# ============================================================

def load_dataset(data_source: str, base_path: str, max_instances: int | None = None) -> list[dict]:
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


def save_round_results(results: list[dict], round_dir: Path, round_idx: int, scaffold_version: int):
    """保存单轮结果。"""
    round_dir.mkdir(parents=True, exist_ok=True)
    results_file = round_dir / "results.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    accuracy = compute_accuracy(results)
    summary = {
        "round": round_idx,
        "scaffold_version": scaffold_version,
        "timestamp": datetime.now().isoformat(),
        **accuracy,
        "total_cost": round(sum(r["cost"] for r in results), 4),
        "total_time": round(sum(r["time"] for r in results), 2),
    }
    (round_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def print_round_summary(summary: dict):
    console.print(
        f"  轮次 {summary['round']} (scaffold v{summary['scaffold_version']:03d}): "
        f"通过率 [cyan]{summary['pass_rate']:.1%}[/cyan] "
        f"({summary['passed']}/{summary['total']}) | "
        f"成本 ${summary['total_cost']:.4f} | "
        f"耗时 {summary['total_time']:.1f}s"
    )


def print_final_comparison(all_summaries: list[dict]):
    """打印各轮通过率对比表。"""
    table = Table(title="进化结果对比")
    table.add_column("轮次", style="cyan")
    table.add_column("Scaffold 版本", style="dim")
    table.add_column("通过率", style="green")
    table.add_column("通过/总数", style="white")
    table.add_column("成本", style="yellow")
    table.add_column("耗时", style="white")

    for s in all_summaries:
        table.add_row(
            str(s["round"]),
            f"v{s['scaffold_version']:03d}",
            f"{s['pass_rate']:.1%}",
            f"{s['passed']}/{s['total']}",
            f"${s['total_cost']:.4f}",
            f"{s['total_time']:.1f}s",
        )
    console.print(table)


# ============================================================
# 主命令
# ============================================================

@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", "-c", help="配置文件路径")] = Path(
        "scripts/configs/math_test_config.yaml"
    ),
    data_source: Annotated[str, typer.Option("--data-source", "-d", help="数据源")] = "aime24",
    max_instances: Annotated[int, typer.Option("--max-instances", "-n", help="每轮最多测试问题数")] = 5,
    max_rounds: Annotated[int, typer.Option("--max-rounds", "-r", help="最多进化轮数")] = 3,
    output: Annotated[Path, typer.Option("--output", "-o", help="输出根目录")] = Path(
        "outputs/math_evolve"
    ),
    save_trajectories: Annotated[bool, typer.Option(help="是否保存 agent 轨迹")] = True,
    teacher_model: Annotated[Optional[str], typer.Option("--teacher-model", help="元智能体（进化）使用的教师模型，如 zai/glm-4.7")] = None,
    teacher_api_key: Annotated[Optional[str], typer.Option("--teacher-api-key", help="教师模型的 API Key（智谱 ZAI_API_KEY）")] = None,
):
    """运行 Math 数据集进化测试（多轮 Scaffold 优化）"""
    # 加载配置
    with open(config) as f:
        config_data = yaml.safe_load(f)
    config_data.setdefault("run", {})["max_instances"] = max_instances

    model_config = config_data.get("model", {})
    model_name = model_config.get("model", "gpt-4o-mini")
    temperature = model_config.get("temperature", 0.0)
    max_tokens = model_config.get("max_tokens", 4096)
    api_base = model_config.get("api_base", None)
    api_key = model_config.get("api_key", None)
    think = model_config.get("think", None)
    extra_body = model_config.get("extra_body", None)

    # ── 学生模型（求解）──────────────────────────────────────────────────────────
    model_kwargs: dict = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "drop_params": True,
        "tool_choice": "required",
    }
    if api_base:
        model_kwargs["api_base"] = api_base
        model_kwargs["api_key"] = api_key or "lm-studio"
    # 关闭 thinking：LM Studio 用 think，vLLM 用 extra_body
    if think is not None:
        model_kwargs["think"] = think
    if extra_body is not None:
        model_kwargs["extra_body"] = extra_body

    # ── 教师模型（元智能体进化 scaffold）────────────────────────────────────────
    if teacher_model:
        # 优先使用命令行传入的 key，其次读环境变量
        _teacher_key = (
            teacher_api_key
            or os.getenv("ZAI_API_KEY")
            or os.getenv("ZHIPUAI_API_KEY")
        )
        if not _teacher_key:
            console.print("[red]错误：使用教师模型需要提供 --teacher-api-key 或设置 ZAI_API_KEY 环境变量[/red]")
            raise typer.Exit(1)
        teacher_model_name = teacher_model
        teacher_model_kwargs: dict = {
            "temperature": 0.0,
            "max_tokens": 4096,
            "drop_params": True,
            "api_base": "https://open.bigmodel.cn/api/paas/v4",
            "api_key": _teacher_key,
        }
    else:
        # 未指定教师模型，退化为与学生模型相同
        teacher_model_name = model_name
        teacher_model_kwargs = {k: v for k, v in model_kwargs.items() if k != "tool_choice"}

    console.print(f"\n[bold blue]Math 进化测试[/bold blue]")
    console.print(f"数据源: [cyan]{data_source}[/cyan]  最大问题数: [cyan]{max_instances}[/cyan]  轮数: [cyan]{max_rounds}[/cyan]")
    console.print(f"[cyan]🎓 学生模型（求解）: {model_name}[/cyan]")
    if teacher_model:
        console.print(f"[magenta]🧑‍🏫 教师模型（进化）: {teacher_model_name}[/magenta]")
    else:
        console.print(f"[yellow]⚠️  未指定教师模型，进化使用学生模型[/yellow]")
    console.print(f"输出目录: [cyan]{output}[/cyan]\n")

    # 加载数据集
    base_path = config_data.get("dataset", {}).get("base_path", "datasets/math/data")
    try:
        problems = load_dataset(data_source, base_path, max_instances)
        console.print(f"[green]已加载 {len(problems)} 个问题[/green]\n")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # 初始化 scaffold 工作区
    workspace = output / "evolve_workspace"
    init_scaffold_workspace(workspace)

    # 创建 LitellmModel & LocalEnvironment
    litellm_model = LitellmModel(
        model_name=model_name,
        model_kwargs=model_kwargs,
        cost_tracking="ignore_errors",
    )
    local_env = LocalEnvironment()

    all_summaries = []
    current_version = 0  # 对应 global_v000

    for round_idx in range(max_rounds):
        console.print(f"\n[bold]{'='*60}[/bold]")
        console.print(f"[bold cyan]第 {round_idx} 轮 (Scaffold v{current_version:03d})[/bold cyan]")
        console.print(f"[bold]{'='*60}[/bold]")

        # 加载当前 scaffold
        scaffold = load_current_scaffold(workspace)

        # 输出目录
        round_dir = output / f"round_{round_idx:03d}"
        traj_dir = round_dir / "trajectories" if save_trajectories else None
        if traj_dir:
            traj_dir.mkdir(parents=True, exist_ok=True)

        # 运行所有题目
        results = []
        for i, problem in enumerate(problems):
            console.print(f"  [{i+1}/{len(problems)}] {problem.get('id', 'unknown')}")
            r = run_single_problem(
                problem_data=problem,
                model=litellm_model,
                env=local_env,
                scaffold=scaffold,
                output_dir=traj_dir,
            )
            results.append(r)
            status = "[green]✓[/green]" if r["passed"] else "[red]✗[/red]"
            console.print(
                f"  {status} 期望={r['expected_answer']}  提取={r['extracted_answer'] or '(无)'}  步数={r['n_steps']}"
            )

        # 保存本轮结果
        summary = save_round_results(results, round_dir, round_idx, current_version)
        all_summaries.append(summary)
        console.print()
        print_round_summary(summary)

        # 最后一轮不再进化
        if round_idx == max_rounds - 1:
            console.print("[dim]最后一轮，不执行 scaffold 优化。[/dim]")
            break

        # 元智能体进化 scaffold（使用教师模型）
        console.print(f"\n[yellow]>>> 元智能体优化 Scaffold（{teacher_model_name}）...[/yellow]")
        improved_scaffold = evolve_scaffold(
            scaffold=scaffold,
            results=results,
            model_name=teacher_model_name,
            model_kwargs=teacher_model_kwargs,
        )

        if improved_scaffold:
            current_version += 1
            save_scaffold_version(workspace, improved_scaffold, current_version)
        else:
            console.print("[dim]scaffold 未改变，下轮继续使用当前版本。[/dim]")

    # 保存进化日志
    log_file = output / "evolution_log.json"
    log_file.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False))

    # 最终对比
    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print("[bold]进化测试完成[/bold]")
    console.print(f"[bold]{'='*60}[/bold]\n")
    print_final_comparison(all_summaries)
    console.print(f"\n[green]所有结果已保存到: {output}[/green]")


if __name__ == "__main__":
    app()

