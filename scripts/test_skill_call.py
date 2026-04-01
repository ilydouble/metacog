#!/usr/bin/env python3
"""最简测试：验证模型能通过 PYTHONPATH 调用 skill。

让模型解一道需要模逆元的简单题，system prompt 里注入 skill 描述，
观察模型是否会 import skill 并得到正确答案。

用法::

    conda run -n medte python scripts/test_skill_call.py
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src" / "mini-swe-agent" / "src"))
sys.path.insert(0, str(_root / "src"))

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

from metacog.skills.registry import SkillRegistry

# ------------------------------------------------------------------ #
# 配置
# ------------------------------------------------------------------ #
MODEL   = "lm_studio/qwen/qwen3.5-9b"
API_BASE = "http://0.0.0.0:1234/v1"
SEED_SKILLS_DIR = _root / "src" / "metacog" / "skills" / "math"

# ------------------------------------------------------------------ #
# 注册 skill，生成描述
# ------------------------------------------------------------------ #
registry = SkillRegistry()
registry.register_from_dir(SEED_SKILLS_DIR)
skill_text = registry.as_prompt_text()
print("=== Skill prompt ===")
print(skill_text)

# ------------------------------------------------------------------ #
# System prompt
# ------------------------------------------------------------------ #
SYSTEM = f"""\
You are a math problem solver. Use bash (python3) to compute answers.
Submit your final answer with:
  printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n%s\\n' ANSWER

{skill_text}
"""

TASK = """\
Find x such that 17 * x ≡ 1 (mod 100).
(i.e., the modular inverse of 17 modulo 100)
Submit the answer as a single integer.
"""

# ------------------------------------------------------------------ #
# 用 PYTHONPATH 指向 seed skills 目录
# ------------------------------------------------------------------ #
existing_pypath = os.environ.get("PYTHONPATH", "")
pythonpath = f"{SEED_SKILLS_DIR}:{existing_pypath}" if existing_pypath else str(SEED_SKILLS_DIR)

with tempfile.TemporaryDirectory() as tmpdir:
    traj_path = Path(tmpdir) / "test.traj.json"
    model = LitellmModel(
        model_name=MODEL,
        model_kwargs={"api_base": API_BASE, "api_key": "lm-studio", "temperature": 0.0},
        cost_tracking="ignore_errors",
    )
    env = LocalEnvironment(env={"PYTHONPATH": pythonpath})
    agent = DefaultAgent(
        model=model,
        env=env,
        system_template=SYSTEM,
        instance_template="{{task}}",
        step_limit=6,
        cost_limit=2.0,
        output_path=traj_path,
    )

    print("=== Running agent ===")
    result = agent.run(task=TASK)
    submission = result.get("submission", "").strip()
    print(f"\n=== Submission: {submission!r} ===")
    print(f"Expected: 41  (17*41=697=6*100+97... let's check: 17*53=901=9*100+1 → 53)")
    # 真正答案：pow(17, -1, 100) = 53
    expected = str(pow(17, -1, 100))
    print(f"Correct answer: {expected}")
    print(f"PASS: {submission == expected}")

