#!/usr/bin/env python3
"""Math 数据集测试脚本

使用 mini-swe-agent 测试 Math 数据集（aime24, aime25, amc23）。
"""

import os

# 设置环境变量，忽略本地模型的成本跟踪错误
os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"

# 让 localhost 请求绕过代理，避免 502 Bad Gateway
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "mini-swe-agent" / "src"))

from minisweagent import Agent, Environment, Model
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

from utils.answer_extraction import extract_final_answer, normalize_answer
from utils.evaluation import compare_answers, compute_accuracy

app = typer.Typer()
console = Console()


class VerboseAgent(DefaultAgent):
    """实时打印每一步思考和执行过程的 Agent。"""

    def query(self) -> dict:
        print(f"\n=== Step {self.n_calls + 1} · 正在调用模型... ===")
        message = super().query()
        content = message.get("content", "") or ""
        print(f"[DEBUG] message keys: {list(message.keys())}, content len: {len(content)}")
        if content:
            preview = content[:800] if len(content) <= 800 else content[:800] + "\n...(截断)"
            print("--- 模型回复 ---")
            print(preview)
            print("--- 回复结束 ---")
        else:
            print("!!! 模型回复为空 (thinking模式未关闭？或模型未响应) !!!")
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        for action in actions:
            cmd = action.get("command", "")
            print(f">>> 执行命令: {cmd[:300]}")
        observations = super().execute_actions(message)
        for obs in observations:
            output = obs.get("extra", {}).get("raw_output", "") or obs.get("content", "")
            if output:
                preview = output[:400] if len(output) <= 400 else output[:400] + "\n...(截断)"
                print(f"<<< 执行结果:\n{preview}")
        return observations


# Math 问题求解的 Prompt 模板
MATH_SYSTEM_TEMPLATE = """You are a helpful assistant that solves math problems."""

MATH_INSTANCE_TEMPLATE = """Please solve the following math problem:

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


def scan_tools(tools_dir: Path) -> str:
    """扫描 tools 目录，返回可用工具列表的文本说明。

    工具目录结构：
      tools_dir/<category>/<tool_name>/main.py  (或 main.sh)
      tools_dir/<category>/<tool_name>/README.md  (可选，含描述)

    Returns:
        工具列表说明字符串，若无工具则返回空字符串。
    """
    if not tools_dir or not tools_dir.exists():
        return ""

    lines = []
    for category_dir in sorted(tools_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for tool_dir in sorted(category_dir.iterdir()):
            if not tool_dir.is_dir():
                continue
            # 确认有可执行入口
            has_py = (tool_dir / "main.py").exists()
            has_sh = (tool_dir / "main.sh").exists()
            if not (has_py or has_sh):
                continue

            # 尝试从 README.md frontmatter 读取描述
            desc = ""
            readme = tool_dir / "README.md"
            if readme.exists():
                import re
                text = readme.read_text()
                m = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
                if m:
                    desc = m.group(1).strip()

            ext = "main.py" if has_py else "main.sh"
            cmd = f"python3 $TOOLS_DIR/{category_dir.name}/{tool_dir.name}/{ext}" if has_py \
                  else f"bash $TOOLS_DIR/{category_dir.name}/{tool_dir.name}/{ext}"
            entry = f"- `{category_dir.name}/{tool_dir.name}`: {desc or '(no description)'}\n  Call: `{cmd}`"
            lines.append(entry)

    if not lines:
        return ""
    return "## Available Tools\n\n" + "\n".join(lines) + "\n"


def load_config(config_path: Path) -> dict:
    """加载配置文件"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset(data_source: str, base_path: str, max_instances: int | None = None) -> list[dict]:
    """加载数据集
    
    Args:
        data_source: 数据源名称（aime24, aime25, amc23）
        base_path: 数据集基础路径
        max_instances: 最大加载数量
        
    Returns:
        问题列表
    """
    data_file = Path(base_path) / f"{data_source}.json"
    
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    
    problems = []
    with open(data_file) as f:
        for i, line in enumerate(f):
            if max_instances is not None and i >= max_instances:
                break
            try:
                problem = json.loads(line)
                problems.append(problem)
            except json.JSONDecodeError as e:
                console.print(f"[yellow]警告: 跳过无效行 {i+1}: {e}[/yellow]")
    
    return problems


def create_agent(model: Model, env: Environment, config: dict) -> Agent:
    """创建 Agent"""
    agent_config = config.get("agent", {})
    
    agent = DefaultAgent(
        model=model,
        env=env,
        system_template=MATH_SYSTEM_TEMPLATE,
        instance_template=MATH_INSTANCE_TEMPLATE,
        step_limit=agent_config.get("step_limit", 50),
        cost_limit=agent_config.get("cost_limit", 2.0),
    )
    
    return agent


def extract_problem_fields(problem_data: dict) -> tuple[str, str, str]:
    """从不同数据集格式中统一提取 (problem_id, problem_text, expected_answer)。

    各数据集字段差异：
      - aime24: id, problem, expected_answer (str)
      - aime25: id, problem, answer (str)
      - amc23:  id, question, answer (int)
    """
    problem_id = str(problem_data.get("id", "unknown"))
    # 题目文本：amc23 用 "question"，其余用 "problem"
    problem = problem_data.get("problem") or problem_data.get("question", "")
    # 答案：优先 "expected_answer"，其次 "answer"，统一转字符串
    expected_answer = str(
        problem_data.get("expected_answer") or problem_data.get("answer", "")
    ).strip()
    return problem_id, problem, expected_answer


def run_single_problem(
    problem_data: dict,
    model: Model,
    env: Environment,
    config: dict,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """运行单个问题

    Args:
        problem_data: 问题数据
        model: 模型实例
        env: 环境实例
        config: 配置
        output_dir: 输出目录（用于保存轨迹）

    Returns:
        结果字典
    """
    problem_id, problem, expected_answer = extract_problem_fields(problem_data)
    
    start_time = time.time()
    error = None
    extracted_answer = None
    passed = False
    cost = 0.0
    n_steps = 0
    submission = ""
    
    try:
        # 创建新的 agent
        agent_cfg = config.get("agent", {})
        agent = VerboseAgent(
            model=model,
            env=env,
            system_template=agent_cfg.get("system_template", MATH_SYSTEM_TEMPLATE),
            instance_template=agent_cfg.get("instance_template", MATH_INSTANCE_TEMPLATE),
            step_limit=agent_cfg.get("step_limit", 50),
            cost_limit=agent_cfg.get("cost_limit", 2.0),
            output_path=output_dir / f"{problem_id}.traj.json" if output_dir else None,
        )
        
        # 运行 agent
        result = agent.run(task=problem)

        # 获取提交的内容
        submission = result.get("submission", "")
        print(f"[DEBUG] submission = {repr(submission[:200]) if submission else repr(submission)}")

        # 提取答案（优先从 submission 提取）
        extracted_answer = extract_final_answer(submission)
        if extracted_answer:
            extracted_answer = normalize_answer(extracted_answer)

        # 回退一：submission 为空时，从 assistant 消息内容里搜索
        if not extracted_answer:
            print("[DEBUG] submission 中未找到答案，回退到对话历史搜索...")
            for msg in reversed(agent.messages):
                role = msg.get("role", "")
                content = msg.get("content", "") or ""
                if role == "assistant" and content:
                    fallback = extract_final_answer(content)
                    if fallback:
                        extracted_answer = normalize_answer(fallback)
                        print(f"[DEBUG] 从 assistant 消息中找到答案: {extracted_answer}")
                        break

        # 回退二：扫描 tool（bash 执行结果）消息，找 COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT 信号
        # 触发场景：_check_finished 未能捕获（如 returncode != 0），或 submission 被后续 LimitsExceeded 覆盖
        if not extracted_answer:
            print("[DEBUG] 回退到 bash 输出扫描...")
            for msg in reversed(agent.messages):
                if msg.get("role") != "tool":
                    continue
                raw = msg.get("extra", {}).get("raw_output", "") or msg.get("content", "") or ""
                lines = raw.splitlines()
                for i, line in enumerate(lines):
                    if line.strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and i + 1 < len(lines):
                        candidate = normalize_answer(lines[i + 1].strip())
                        if candidate:
                            extracted_answer = candidate
                            print(f"[DEBUG] 从 bash 输出中找到提交答案: {extracted_answer}")
                            break
                if extracted_answer:
                    break

        # 评估
        if extracted_answer:
            passed = compare_answers(extracted_answer, expected_answer)
        
        cost = agent.cost
        n_steps = agent.n_calls
        
    except Exception as e:
        error = str(e)
        console.print(f"[red]错误: {problem_id} - {e}[/red]")
    
    elapsed_time = time.time() - start_time
    
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
        "time": round(elapsed_time, 2),
        "error": error,
    }


def save_results(results: list[dict], output_dir: Path, config: dict, data_source: str):
    """保存结果
    
    Args:
        results: 结果列表
        output_dir: 输出目录
        config: 配置
        data_source: 数据源
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存详细结果
    results_file = output_dir / "results.jsonl"
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # 计算统计
    accuracy = compute_accuracy(results)
    total_cost = sum(r["cost"] for r in results)
    total_time = sum(r["time"] for r in results)
    
    # 保存摘要
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.get("model", {}).get("model"),
            "data_source": data_source,
            "max_instances": config.get("run", {}).get("max_instances"),
        },
        **accuracy,
        "total_cost": round(total_cost, 4),
        "total_time": round(total_time, 2),
    }
    
    summary_file = output_dir / "run_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary


def print_summary(summary: dict):
    """打印结果摘要"""
    table = Table(title="测试结果摘要")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    
    table.add_row("总问题数", str(summary["total"]))
    table.add_row("通过数", str(summary["passed"]))
    table.add_row("失败数", str(summary["failed"]))
    table.add_row("通过率", f"{summary['pass_rate']:.2%}")
    table.add_row("总成本", f"${summary['total_cost']:.4f}")
    table.add_row("总时间", f"{summary['total_time']:.2f}s")
    
    console.print(table)


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", "-c", help="配置文件路径")] = Path("scripts/configs/math_test_config.yaml"),
    scaffold: Annotated[Optional[Path], typer.Option("--scaffold", "-s", help="scaffold.yaml 路径（覆盖配置文件中的模板）")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="模型名称（覆盖配置文件）")] = None,
    data_source: Annotated[str, typer.Option("--data-source", "-d", help="数据源（aime24, aime25, amc23）")] = "aime24",
    max_instances: Annotated[int, typer.Option("--max-instances", "-n", help="最大测试问题数量")] = 2,
    output: Annotated[Path, typer.Option("--output", "-o", help="输出目录")] = Path("outputs/math_test"),
    save_trajectories: Annotated[bool, typer.Option(help="是否保存 agent 轨迹")] = True,
    tools_dir: Annotated[Optional[Path], typer.Option("--tools-dir", help="agent_tools 目录路径（ReCreate 生成的工具）")] = None,
):
    """运行 Math 数据集测试"""
    # 加载配置
    config_data = load_config(config)

    # 如果指定了 scaffold，将其模板注入 agent 配置（优先级高于配置文件）
    if scaffold is not None:
        if not scaffold.exists():
            console.print(f"[red]scaffold 文件不存在: {scaffold}[/red]")
            raise typer.Exit(1)
        scaffold_data = yaml.safe_load(scaffold.read_text())
        agent_cfg = config_data.setdefault("agent", {})
        for key in ("system_template", "instance_template", "step_limit", "cost_limit"):
            if key in scaffold_data:
                agent_cfg[key] = scaffold_data[key]
        console.print(f"[green]✓ 已加载 scaffold: {scaffold}[/green]")

    # 如果指定了 tools_dir，将工具列表附加到 system_template
    if tools_dir is not None:
        if not tools_dir.exists():
            console.print(f"[red]tools_dir 不存在: {tools_dir}[/red]")
            raise typer.Exit(1)
        tools_listing = scan_tools(tools_dir)
        if tools_listing:
            n_tools = tools_listing.count("- `")
            agent_cfg = config_data.setdefault("agent", {})
            existing_sys = agent_cfg.get("system_template", MATH_SYSTEM_TEMPLATE)
            agent_cfg["system_template"] = existing_sys.rstrip() + "\n\n" + tools_listing
            console.print(f"[green]✓ 已加载工具目录: {tools_dir} ({n_tools} 个工具)[/green]")
        else:
            console.print(f"[yellow]tools_dir 中未找到可用工具: {tools_dir}[/yellow]")
        # 记录 tools_dir 路径，供 LocalEnvironment 使用
        config_data["_tools_dir"] = str(tools_dir.resolve())

    # 命令行参数覆盖配置文件
    if model:
        config_data.setdefault("model", {})["model"] = model
    config_data.setdefault("run", {})["max_instances"] = max_instances
    
    # 显示配置
    console.print(f"\n[bold blue]Math 数据集测试[/bold blue]")
    console.print(f"数据源: [cyan]{data_source}[/cyan]")
    console.print(f"模型: [cyan]{config_data.get('model', {}).get('model', 'unknown')}[/cyan]")
    console.print(f"最大问题数: [cyan]{max_instances}[/cyan]")
    console.print(f"输出目录: [cyan]{output}[/cyan]\n")
    
    # 加载数据集
    base_path = config_data.get("dataset", {}).get("base_path", "datasets/math/data")
    console.print(f"[yellow]加载数据集: {data_source}...[/yellow]")
    
    try:
        problems = load_dataset(data_source, base_path, max_instances)
        console.print(f"[green]已加载 {len(problems)} 个问题[/green]\n")
    except FileNotFoundError as e:
        console.print(f"[red]错误: {e}[/red]")
        raise typer.Exit(1)
    
    # 创建模型和环境
    model_config = config_data.get("model", {})
    model_name = model_config.get("model", "gpt-4o-mini")
    temperature = model_config.get("temperature", 0.0)
    max_tokens = model_config.get("max_tokens", 4096)
    api_base = model_config.get("api_base", None)
    api_key = model_config.get("api_key", None)
    num_ctx = model_config.get("num_ctx", None)
    think = model_config.get("think", None)
    extra_body = model_config.get("extra_body", None)

    # 构建 model_kwargs
    model_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "drop_params": True,  # 丢弃不支持的参数
    }

    # 本地模型需要设置 api_base 和 api_key
    if api_base:
        model_kwargs["api_base"] = api_base
        model_kwargs["api_key"] = api_key or "lm-studio"

    # Ollama 上下文窗口大小（减少 KV Cache 内存占用）
    if num_ctx:
        model_kwargs["num_ctx"] = num_ctx

    # 关闭 thinking 模式（二选一，按部署方式选择）：
    # - LM Studio：think: false（在系统 prompt 注入 /no_think）
    # - vLLM：extra_body.chat_template_kwargs.enable_thinking: false（透传到请求 body）
    if think is not None:
        model_kwargs["think"] = think
    if extra_body is not None:
        model_kwargs["extra_body"] = extra_body
    
    console.print(f"[yellow]创建模型: {model_name}[/yellow]")
    console.print(f"[yellow]model_kwargs: {model_kwargs}[/yellow]")
    
    litellm_model = LitellmModel(
        model_name=model_name,
        model_kwargs=model_kwargs,
        cost_tracking="ignore_errors",  # 本地模型忽略成本跟踪错误
    )
    
    # 创建环境：如果有 tools_dir，通过 TOOLS_DIR 环境变量暴露给 agent
    _tools_dir_str = config_data.pop("_tools_dir", None)
    env_vars = {}
    if _tools_dir_str:
        env_vars["TOOLS_DIR"] = _tools_dir_str
    local_env = LocalEnvironment(env=env_vars) if env_vars else LocalEnvironment()
    
    # 创建输出目录
    output.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output / "trajectories" if save_trajectories else None
    if trajectories_dir:
        trajectories_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行测试
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for i, problem in enumerate(problems):
            task_id = progress.add_task(
                f"[{i+1}/{len(problems)}] {problem.get('id', 'unknown')}",
                total=None
            )
            
            result = run_single_problem(
                problem_data=problem,
                model=litellm_model,
                env=local_env,
                config=config_data,
                output_dir=trajectories_dir,
            )
            results.append(result)
            
            # 显示结果
            status = "[green]✓[/green]" if result["passed"] else "[red]✗[/red]"
            progress.update(
                task_id,
                description=f"[{i+1}/{len(problems)}] {problem.get('id', 'unknown')} {status} (cost: ${result['cost']:.4f}, time: {result['time']:.1f}s)"
            )
            progress.remove_task(task_id)
            
            # 实时显示结果
            console.print(
                f"  [{i+1}/{len(problems)}] {problem.get('id', 'unknown')} {status} "
                f"(cost: ${result['cost']:.4f}, time: {result['time']:.1f}s)"
            )
    
    # 保存结果
    console.print(f"\n[yellow]保存结果...[/yellow]")
    summary = save_results(results, output, config_data, data_source)
    
    # 显示摘要
    console.print()
    print_summary(summary)
    
    console.print(f"\n[green]结果已保存到: {output}[/green]")


if __name__ == "__main__":
    app()
