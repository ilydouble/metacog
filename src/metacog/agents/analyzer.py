"""AnalyzerAgent - 订阅 TrajectoryEvent，分段分析轨迹，蒸馏成精简记忆条目。

设计原则
--------
- 上下文受限：每次 LLM 调用只喂一小块（chunk），不把整条轨迹塞进去。
- 高度精简：每块只产出一句话摘要；最终蒸馏只产出 1-2 句可执行建议。
- 分段流程：
    轨迹文件 → parse_steps → 按 chunk_size 分块
    → 每块调 LLM 得一句摘要
    → 所有摘要 + 题目元数据 → 调 LLM 蒸馏 → AnalysisEvent
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from minisweagent.models.litellm_model import LitellmModel

from ..bus import Event, EventBus, EventType
from .base import BaseAgent

# ------------------------------------------------------------------ #
# 每步截断长度（字符数）
# ------------------------------------------------------------------ #
_THOUGHT_LIMIT = 300   # assistant 思考内容
_CMD_LIMIT     = 400   # bash 命令
_OUTPUT_LIMIT  = 400   # 执行输出


@dataclass
class _Step:
    """轨迹中的一个 (思考, 命令, 输出) 三元组。"""
    thought: str
    command: str
    output: str

    def render(self) -> str:
        parts = []
        if self.thought:
            parts.append(f"[Thought] {self.thought}")
        if self.command:
            parts.append(f"[Cmd] {self.command}")
        if self.output:
            parts.append(f"[Out] {self.output}")
        return "\n".join(parts)


# ------------------------------------------------------------------ #
# Prompt 模板
# ------------------------------------------------------------------ #
_CHUNK_SYSTEM = """\
You are reviewing a segment of an AI math-solving trajectory.
Summarize what the agent tried and what happened in ONE sentence (max 25 words).
Be factual. No advice yet."""

_DISTILL_SYSTEM = """\
You are distilling trajectory summaries into a single actionable lesson for a math AI agent.
Output JSON only—no other text:
{
  "failure_type": "wrong_approach|arithmetic_error|step_limit|missing_constraint|other",
  "lesson_title": "≤8 words",
  "lesson_content": "1-2 sentences, concrete and actionable, written for the agent to read next time",
  "tags": ["tag1", "tag2"]
}
Keep lesson_content under 40 words. Focus on what to do differently, not what went wrong."""

_SUCCESS_CHUNK_SYSTEM = """\
You are reviewing a segment of a SUCCESSFUL AI math-solving trajectory.
Summarize what technique the agent used in ONE sentence (max 25 words).
Be specific about the mathematical method."""

_SUCCESS_DISTILL_SYSTEM = """\
You are extracting a reusable mathematical technique from successful trajectory summaries.
Output JSON only—no other text:
{
  "technique_name": "snake_case name, ≤5 words",
  "summary": "1-2 sentences describing the reusable technique concretely",
  "tags": ["number_theory|combinatorics|algebra|geometry|arithmetic|other"],
  "can_be_skill": true
}
Focus on techniques that could be reused for similar problems."""

_SUCCESS_DISTILL_USER = """\
Problem: {problem}
Expected: {expected_answer}  Steps used: {n_steps}

Trajectory summaries (one per chunk):
{summaries}"""

_DISTILL_USER = """\
Problem: {problem}
Expected: {expected_answer}  Got: {extracted_answer}  Steps: {n_steps}

Trajectory summaries (one per chunk):
{summaries}"""


class AnalyzerAgent(BaseAgent):
    """轨迹分析智能体（分段读取 + 蒸馏）。

    参数
    ----
    chunk_size : int
        每个分析块包含的步数（默认 3）。
    """

    name = "analyzer"

    def __init__(
        self,
        model: LitellmModel,
        bus: EventBus,
        chunk_size: int = 3,
    ) -> None:
        super().__init__(model, bus)
        self.chunk_size = chunk_size

    def _register_handlers(self) -> None:
        self.bus.subscribe(EventType.TRAJECTORY, self._on_trajectory)

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #

    def _on_trajectory(self, event: Event) -> None:
        data = event.data
        if data.get("passed", False):
            self._on_success(data)
            return

        steps = self._parse_steps(data.get("traj_path"))
        if not steps:
            return

        # 分块摘要
        chunks = [steps[i: i + self.chunk_size]
                  for i in range(0, len(steps), self.chunk_size)]
        summaries: list[str] = []
        for chunk in chunks:
            s = self._summarize_chunk(chunk)
            if s:
                summaries.append(s)

        if not summaries:
            return

        # 蒸馏成最终记忆条目
        try:
            analysis = self._distill(
                summaries=summaries,
                problem=data.get("problem", "")[:300],
                expected_answer=data.get("expected_answer", ""),
                extracted_answer=data.get("extracted_answer") or "(none)",
                n_steps=data.get("n_steps", 0),
            )
        except Exception as exc:
            analysis = {"error": str(exc)}

        if not analysis or analysis.get("skip"):
            return

        self._publish(Event(
            type=EventType.ANALYSIS,
            data={"problem_id": data.get("id"), "analysis": analysis},
        ))

    # ------------------------------------------------------------------ #
    # 轨迹解析
    # ------------------------------------------------------------------ #

    def _parse_steps(self, traj_path: Optional[str]) -> list[_Step]:
        """把轨迹文件解析为 _Step 列表（每步 = 思考 + 命令 + 输出）。"""
        if not traj_path:
            return []
        p = Path(traj_path)
        if not p.exists():
            return []
        try:
            msgs = json.loads(p.read_text()).get("messages", [])
        except Exception:
            return []

        steps: list[_Step] = []
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            role = msg.get("role", "")

            if role == "assistant":
                thought = (msg.get("content") or "")[:_THOUGHT_LIMIT]
                extra = msg.get("extra") or {}
                actions = extra.get("actions") or []
                cmd = (actions[0].get("command", "") if actions else "")[:_CMD_LIMIT]

                # 下一条是 tool 输出
                output = ""
                if i + 1 < len(msgs) and msgs[i + 1].get("role") == "tool":
                    tool_extra = msgs[i + 1].get("extra") or {}
                    output = (tool_extra.get("raw_output") or
                              msgs[i + 1].get("content") or "")[:_OUTPUT_LIMIT]
                    i += 1  # 消费掉 tool 消息

                steps.append(_Step(thought=thought, command=cmd, output=output))

            i += 1

        return steps

    # ------------------------------------------------------------------ #
    # LLM 调用
    # ------------------------------------------------------------------ #

    def _summarize_chunk(self, chunk: list[_Step]) -> str:
        """用一句话摘要一个 chunk。失败时返回空字符串。"""
        body = "\n\n".join(f"Step {j+1}:\n{s.render()}"
                           for j, s in enumerate(chunk))
        try:
            return self._llm_call(
                system=_CHUNK_SYSTEM,
                user=body,
                temperature=0.0,
                max_tokens=60,
            ).strip()
        except Exception:
            return ""

    def _distill(
        self,
        summaries: list[str],
        problem: str,
        expected_answer: str,
        extracted_answer: str,
        n_steps: int,
    ) -> dict:
        """把所有块摘要蒸馏成最终结构化记忆。"""
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))
        user_msg = _DISTILL_USER.format(
            problem=problem,
            expected_answer=expected_answer,
            extracted_answer=extracted_answer,
            n_steps=n_steps,
            summaries=numbered,
        )
        response = self._llm_call(
            system=_DISTILL_SYSTEM,
            user=user_msg,
            temperature=0.0,
            max_tokens=200,
        )
        return self._parse_json(response)

    # ------------------------------------------------------------------ #
    # 成功路径
    # ------------------------------------------------------------------ #

    def _on_success(self, data: dict) -> None:
        """分析成功轨迹，提取可复用技术，发布 SUCCESS_ANALYSIS 事件。"""
        steps = self._parse_steps(data.get("traj_path"))
        if not steps:
            return

        chunks = [steps[i: i + self.chunk_size]
                  for i in range(0, len(steps), self.chunk_size)]
        summaries: list[str] = []
        for chunk in chunks:
            s = self._summarize_chunk_success(chunk)
            if s:
                summaries.append(s)

        if not summaries:
            return

        try:
            analysis = self._distill_success(
                summaries=summaries,
                problem=data.get("problem", "")[:300],
                expected_answer=data.get("expected_answer", ""),
                n_steps=data.get("n_steps", 0),
            )
        except Exception as exc:
            return

        if not analysis or not analysis.get("can_be_skill", False):
            return

        self._publish(Event(
            type=EventType.SUCCESS_ANALYSIS,
            data={
                "problem_id": data.get("id"),
                "technique": analysis.get("technique_name", "unknown"),
                "summary": analysis.get("summary", ""),
                "tags": analysis.get("tags", []),
            },
        ))

    def _summarize_chunk_success(self, chunk: list[_Step]) -> str:
        body = "\n\n".join(f"Step {j+1}:\n{s.render()}"
                           for j, s in enumerate(chunk))
        try:
            return self._llm_call(
                system=_SUCCESS_CHUNK_SYSTEM,
                user=body,
                temperature=0.0,
                max_tokens=60,
            ).strip()
        except Exception:
            return ""

    def _distill_success(
        self,
        summaries: list[str],
        problem: str,
        expected_answer: str,
        n_steps: int,
    ) -> dict:
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))
        user_msg = _SUCCESS_DISTILL_USER.format(
            problem=problem,
            expected_answer=expected_answer,
            n_steps=n_steps,
            summaries=numbered,
        )
        response = self._llm_call(
            system=_SUCCESS_DISTILL_SYSTEM,
            user=user_msg,
            temperature=0.0,
            max_tokens=200,
        )
        return self._parse_json(response)

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #

    def _parse_json(self, text: str) -> dict:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {"error": "failed to parse", "raw": text[:100]}

