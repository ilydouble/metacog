"""ExecutorAgent - 跑 Solver，每题结束后发 TrajectoryEvent。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

from ..bus import Event, EventBus, EventType
from ..memory.store import MemoryStore
from ..skills.registry import SkillRegistry
from .base import BaseAgent


class ExecutorAgent(BaseAgent):
    """Solver 执行智能体。

    每道题：
    1. 从 MemoryStore 动态加载最新记忆，注入 system_template
    2. 从 SkillRegistry 生成技能描述，注入 system_template
    3. 跑 DefaultAgent
    4. 发布 TrajectoryEvent（供 AnalyzerAgent 消费）
    """

    name = "executor"

    def __init__(
        self,
        model: LitellmModel,
        bus: EventBus,
        scaffold: dict,
        memory_store: MemoryStore | None = None,
        skill_registry: SkillRegistry | None = None,
        output_dir: Path | None = None,
        skills_dir: Path | None = None,
    ) -> None:
        super().__init__(model, bus)
        self.scaffold = scaffold
        self.memory_store = memory_store
        self.skill_registry = skill_registry
        self.output_dir = output_dir
        self.skills_dir = skills_dir

    def run_problem(self, problem_data: dict) -> dict:
        """运行单道题，返回结果 dict，并发布 TrajectoryEvent。"""
        from utils.answer_extraction import extract_final_answer, normalize_answer
        from utils.evaluation import compare_answers

        problem_id = str(problem_data.get("id", "unknown"))
        problem = problem_data.get("problem") or problem_data.get("question", "")
        expected_answer = str(
            problem_data.get("expected_answer") or problem_data.get("answer", "")
        ).strip()

        # 动态拼接 system_template
        system_template = self.scaffold.get("system_template", "")
        if self.memory_store:
            mem_text = self.memory_store.as_prompt_text()
            if mem_text:
                system_template = system_template.rstrip() + "\n\n" + mem_text
        if self.skill_registry:
            skill_text = self.skill_registry.as_prompt_text()
            if skill_text:
                system_template = system_template.rstrip() + "\n\n" + skill_text

        start_time = time.time()
        traj_path = (self.output_dir / f"{problem_id}.traj.json") if self.output_dir else None

        # 构造 PYTHONPATH：把 skills_dir 追加到当前环境的 PYTHONPATH 前面
        import os as _os
        env_extra: dict[str, str] = {}
        if self.skills_dir and self.skills_dir.exists():
            existing = _os.environ.get("PYTHONPATH", "")
            env_extra["PYTHONPATH"] = f"{self.skills_dir}:{existing}" if existing else str(self.skills_dir)

        local_env = LocalEnvironment(env=env_extra) if env_extra else LocalEnvironment()
        agent = DefaultAgent(
            model=self.model,
            env=local_env,
            system_template=system_template,
            instance_template=self.scaffold.get("instance_template", "{{task}}"),
            step_limit=self.scaffold.get("step_limit", 10),
            cost_limit=self.scaffold.get("cost_limit", 2.0),
            output_path=traj_path,
        )

        error = None
        extracted_answer = None
        passed = False
        submission = ""

        try:
            result = agent.run(task=problem)
            submission = result.get("submission", "")
            extracted_answer = extract_final_answer(submission)
            if extracted_answer:
                extracted_answer = normalize_answer(extracted_answer)
            if extracted_answer:
                passed = compare_answers(extracted_answer, expected_answer)
        except Exception as exc:
            error = str(exc)

        record = {
            "id": problem_id,
            "problem": problem[:300],
            "expected_answer": expected_answer,
            "extracted_answer": extracted_answer,
            "submission": submission[:500],
            "passed": passed,
            "n_steps": agent.n_calls,
            "cost": agent.cost,
            "time": round(time.time() - start_time, 2),
            "error": error,
            "traj_path": str(traj_path) if traj_path else None,
        }

        # 发布轨迹事件
        self._publish(Event(type=EventType.TRAJECTORY, data=record))
        return record

