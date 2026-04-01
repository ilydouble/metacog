#!/usr/bin/env python3
"""Analyze failure trajectories from math_test_aime25."""
import json
from pathlib import Path

TRAJ_DIR = Path("outputs/math_test_aime25/trajectories")
RESULTS  = Path("outputs/math_test_aime25/results.jsonl")

results = []
with open(RESULTS) as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

failures = [r for r in results if not r["passed"]]

# ── 分类 ────────────────────────────────────────────────────────────────────
wrong_answer = [r for r in failures if r["extracted_answer"]]
no_submission = [r for r in failures if not r["extracted_answer"]]

print(f"总题数: {len(results)}  通过: {len(results)-len(failures)}  失败: {len(failures)}")
print(f"  ├─ 答案错误:   {len(wrong_answer)} 道")
print(f"  └─ 未提交答案: {len(no_submission)} 道")
print()

# ── 答案错误 ────────────────────────────────────────────────────────────────
if wrong_answer:
    print("=== 类型1：答案错误（算出了答案但算错了）===")
    for r in wrong_answer:
        print(f"  {r['id']:6s}  期望={r['expected_answer']:>4s}  提交={r['extracted_answer']:>4s}  步数={r['n_steps']}")
    print()

# ── 未提交（步数耗尽） ──────────────────────────────────────────────────────
print("=== 类型2：未提交答案（步数耗尽 / LimitsExceeded）===")
for r in no_submission:
    traj_path = TRAJ_DIR / f"{r['id']}.traj.json"
    if not traj_path.exists():
        print(f"  {r['id']:6s}  轨迹文件不存在")
        continue

    data = json.loads(traj_path.read_text())
    msgs = data if isinstance(data, list) else data.get("messages", [])

    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]

    # 统计 tool 报错次数
    error_count = sum(
        1 for m in tool_msgs
        if "returncode>1" in (m.get("content") or "")
        or "Traceback" in (m.get("content") or "")
    )

    # 最后一步 assistant 的 action
    last_cmds = []
    for m in reversed(assistant_msgs):
        actions = m.get("extra", {}).get("actions", [])
        if actions:
            last_cmds = [a.get("command", "")[:80] for a in actions]
            break

    # 最后一个 tool 输出摘要
    last_tool_out = ""
    if tool_msgs:
        raw = (tool_msgs[-1].get("content") or "")
        last_tool_out = raw.replace("\n", " ")[:120]

    # 每步都在做什么（计算型 vs 推理型）
    step_types = []
    for m in assistant_msgs:
        actions = m.get("extra", {}).get("actions", [])
        for a in actions:
            cmd = a.get("command", "")
            if "python" in cmd.lower() or "sympy" in cmd.lower():
                step_types.append("python")
            elif "printf" in cmd or "COMPLETE" in cmd:
                step_types.append("submit")
            else:
                step_types.append("other")

    print(f"  {r['id']:6s}  steps={r['n_steps']}  tool_errors={error_count}")
    print(f"    step_types : {step_types}")
    print(f"    last_cmd   : {last_cmds[0] if last_cmds else '(none)'}")
    print(f"    last_output: {last_tool_out[:100]}")
    print()

