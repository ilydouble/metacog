"""AnalyzerAgent - Subscribes to TrajectoryEvent, analyzes trajectory in segments, distills into concise memory entries.

Design Principles
-----------------
- Context-limited: Each LLM call is fed only a small chunk, not the entire trajectory.
- Highly concise: Each chunk produces one-sentence summary; final distillation produces 1-2 actionable suggestions.
- **Program-of-Thoughts (PoT)**: Generate verification code for math errors, extract reusable logic
- Segmented workflow:
    Trajectory file → parse_steps → split by chunk_size
    → Each chunk calls LLM to get one-sentence summary
    → All summaries + problem metadata → call LLM to distill → AnalysisEvent
    → (Optional) PoT verification → generate code → execute → extract reusable patterns
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from minisweagent.models.litellm_model import LitellmModel

from ..bus import Event, EventBus, EventType
from .base import BaseAgent
from .pot_reflector import PoTReflector
from .trajectory_analyzer import TrajectoryAnalyzer

# ------------------------------------------------------------------ #
# Truncation length for each step (characters)
# ------------------------------------------------------------------ #
_THOUGHT_LIMIT = 300   # assistant thought content
_CMD_LIMIT     = 400   # bash command
_OUTPUT_LIMIT  = 400   # execution output


@dataclass
class _Step:
    """A (thought, command, output) triple in the trajectory."""
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
# Prompt Templates
# ------------------------------------------------------------------ #
_CHUNK_SYSTEM = """\
You are reviewing a segment of an AI math-solving trajectory.
Summarize what the agent tried and what happened in ONE sentence (max 25 words).
Be factual. No advice yet."""

_DISTILL_SYSTEM = """\
# Role
You are an AI Core Evaluator specializing in "Meta-cognitive Distillation of Failures". Your task is to analyze an agent's failed execution trajectory and distill it EXCLUSIVELY into abstract, structured Semantic Memory (Historical Lessons). You act as the "Teacher Model" to extract universal rules and cognitive traps.

# Objective 1: The Filtering Gate
- IGNORE trivial errors: Pure syntax errors, API timeouts, or unhandled formatting exceptions without logical exploration should be discarded. Output: {"status": "discard"}
- CAPTURE cognitive errors: Strategic failures, mathematical traps, or intractable routing (e.g., naive numerical brute-force on exact algebraic systems).

# Objective 2: Absolute De-parameterization
Replace all specific numerical values (e.g., 38, 14, 196) and precise coordinates with generic mathematical variables (e.g., Length L, Polygon P).

# Output Format (JSON)
If it is a cognitive failure, output a standard JSON block inside ```json tags. The keys MUST exactly match the schema below. ALL generated content MUST be entirely in English.

```json
{
  "problem_type": "<Abstract mathematical or algorithmic description of the problem category>",
  "error": "<Diagnose the root strategic or algorithmic failure (e.g., 'Over-reliance on coordinate brute-force')>",
  "fix": "<The imperative 'Teacher' instruction on what to do instead. Must be highly actionable>",
  "reasoning_path": "<Abstract reasoning pathway for the correct approach>",
  "solution_steps": "<Provide 3 abstract steps for the correct approach, e.g., '1. Identify shape 2. Apply theorem 3. Solve system'>",
  "common_traps": "<Document the deceptive algorithmic 'trap' the agent fell into, serving as a warning>"
}
```

# Execution Constraints
- Do NOT leak the specific final answer into the output.
- Focus strictly on strategy and routing, not basic arithmetic mistakes."""

_SUCCESS_CHUNK_SYSTEM = """\
You are reviewing a segment of a SUCCESSFUL AI math-solving trajectory.
Summarize what technique the agent used in ONE sentence (max 25 words).
Be specific about the mathematical method."""

_SUCCESS_DISTILL_SYSTEM = """\
Extract a reusable technique from successful solution.

CRITICAL OUTPUT FORMAT - After your analysis, you MUST end your response with EXACTLY this:

[Technique_Name]: snake_case_name
[Summary]: technique description
[Tags]: tag1, tag2
[Can_Be_Skill]: true

EXAMPLE:
[Technique_Name]: modular_inverse_computation
[Summary]: use built-in pow(a,-1,m) for modular inverse instead of manual extended euclidean
[Tags]: number_theory, modular_arithmetic
[Can_Be_Skill]: true

You can think first, but your response MUST END with the format above."""

_SUCCESS_DISTILL_USER = """\
Problem: {problem}
Expected: {expected_answer}  Steps used: {n_steps}

Full trajectory:
{trajectory}"""

_DISTILL_USER = """\
Problem: {problem}
Expected: {expected_answer}  Got: {extracted_answer}  Steps: {n_steps}

Full trajectory:
{trajectory}"""

# ------------------------------------------------------------------ #
# Phase 2: Solution hint prompt (only looks at problem, not contaminated by failed trajectory)
# ------------------------------------------------------------------ #
_HINT_SYSTEM = """\
You are a math olympiad coach. A student just failed to solve a problem.
Given ONLY the problem statement, provide a concise solution hint.

OUTPUT FORMAT (your ENTIRE response must be ONLY this):

[Key_Insight]: the core mathematical idea, theorem, or technique needed
[Approach]: brief 2-3 step strategy (numbered, e.g. 1. do X  2. do Y)
[Common_Pitfall]: the most likely mistake to avoid

EXAMPLE:
[Key_Insight]: Use perpendicular bisector — |z-a|=|z-b| defines a line, then check tangency with circle
[Approach]: 1. Simplify second equation to a linear constraint  2. Apply point-to-line distance formula  3. Solve |distance| = radius for both ± cases
[Common_Pitfall]: Missing the second tangent case (distance = −radius)

Do NOT give the full solution. Output ONLY the three lines above."""

_HINT_USER = """\
Problem: {problem}

This is a competition math problem. The answer is an integer from 0-999.
Provide a concise hint focusing on the KEY mathematical insight."""


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
        enable_pot: bool = True,  # 🔥 是否启用 PoT 反思
        enable_loop_detection: bool = True,  # 🔥 是否启用死循环检测
    ) -> None:
        super().__init__(model, bus)
        self.chunk_size = chunk_size
        self.enable_pot = enable_pot
        self.pot_reflector = PoTReflector() if enable_pot else None
        self.enable_loop_detection = enable_loop_detection
        self.traj_analyzer = TrajectoryAnalyzer() if enable_loop_detection else None

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
            print("  [Analyzer] 无法读取轨迹文件，跳过分析", flush=True)
            return

        # ========================================
        # 🔥 Dead loop detection (post-hoc analysis)
        # ========================================
        loop_pattern = None
        inefficient_approach = None

        if self.enable_loop_detection and self.traj_analyzer:
            loop_pattern = self.traj_analyzer.analyze_trajectory_for_loops(steps)
            if loop_pattern:
                print(f"  [Analyzer] 🔁 检测到死循环: {loop_pattern.description}", flush=True)
                print(f"             步骤 {loop_pattern.start_step}-{loop_pattern.end_step}", flush=True)

            step_limit = data.get("step_limit", 10)
            inefficient_approach = self.traj_analyzer.detect_inefficient_approach(steps, step_limit)
            if inefficient_approach:
                print(f"  [Analyzer] ⚠️  {inefficient_approach}", flush=True)

        # ========================================
        # 🔥 Failure routing (block operational mistakes from being stored)
        # ========================================
        from .failure_router import route_failure
        routing = route_failure(
            steps=steps,
            loop_detected=bool(loop_pattern),
            extracted_answer=data.get("extracted_answer"),
            expected_answer=data.get("expected_answer"),
        )

        # Phase 1 is blocked for operational mistakes, but Phase 2 (hint) always runs
        # because hint generation only depends on the problem text, not the trajectory
        skip_phase1 = not routing.should_store
        if skip_phase1:
            print(f"  [Router] 🚫 Phase 1 blocked: {routing.reason}", flush=True)
        else:
            print(f"  [Router] ✅ Phase 1 allowed: {routing.reason}", flush=True)

        print(f"  [Analyzer] Running: distill trajectory (Phase 1) + generate hint (Phase 2)...", flush=True)

        # Build extra context (dead loop / inefficiency detection results)
        extra_context = ""
        if loop_pattern:
            extra_context += f"\n[LOOP DETECTED] {loop_pattern.description}"
        if inefficient_approach:
            extra_context += f"\n[INEFFICIENT] {inefficient_approach}"

        problem_text = data.get("problem", "")

        # ========================================
        # 🔥 Phase 1 + Phase 2 parallel: error analysis & solution hint
        # ========================================
        analysis = {"skip": True, "error": "not_started"}
        hint_result: Optional[dict] = None

        with ThreadPoolExecutor(max_workers=2) as pool:
            # Phase 1: only submit if Router allows
            if not skip_phase1:
                future_distill = pool.submit(
                    self._distill,
                    steps=steps,
                    problem=problem_text[:300],
                    expected_answer=data.get("expected_answer", ""),
                    extracted_answer=data.get("extracted_answer") or "(none)",
                    n_steps=data.get("n_steps", 0),
                    extra_context=extra_context,
                )
            else:
                future_distill = None

            # Phase 2: always submit regardless of Router decision
            future_hint = pool.submit(
                self._generate_hint,
                problem=problem_text[:500],
            )

            # Phase 1: only call .result() if it was submitted
            if future_distill is not None:
                try:
                    analysis = future_distill.result()
                except Exception as exc:
                    print(f"  [Analyzer] ❌ Distillation exception occurred: {exc}", flush=True)
                    import traceback
                    traceback.print_exc()
                    analysis = {"skip": True, "error": str(exc)}
            else:
                # Phase 1 was blocked by Router, mark as skipped
                analysis = {"skip": True, "error": "blocked_by_router"}

            try:
                hint_result = future_hint.result()
                if hint_result:
                    print(f"  [Analyzer] ✅ Solution hint generation completed", flush=True)
            except Exception as exc:
                print(f"  [Analyzer] ⚠️ Solution hint generation failed: {exc}", flush=True)

        # ========================================
        # 🔥 Phase 1 and Phase 2 decoupling
        # ========================================
        # Even if Phase 1 (error analysis) fails, Phase 2 (solution hint) is still valuable,
        # should publish event to let MemoryManager store solution hint for future reference
        phase1_failed = "error" in analysis or analysis.get("skip")
        phase2_available = hint_result is not None

        if phase1_failed and not phase2_available:
            # Both phases failed, return directly
            print(f"  [Analyzer] ⚠️  Both Phase 1 and Phase 2 failed, skipping storage", flush=True)
            return

        if phase1_failed:
            # Phase 1 failed but Phase 2 succeeded: keep only solution hint
            print(f"  [Analyzer] ⚠️  Phase 1 failed, but Phase 2 succeeded, storing only solution hint", flush=True)
            analysis = {
                "skip_error_analysis": True,  # Mark skip error analysis
                "solution_hint": hint_result,
            }
        else:
            # Phase 1 succeeded: merge Phase 2 normally
            if phase2_available:
                analysis["solution_hint"] = hint_result

        # ========================================
        # 🔥 PoT Enhancement: Program verification for math calculation errors
        # ========================================
        needs_verification = str(analysis.get("needs_code_verification", "false")).lower().strip() == "true"
        if self.enable_pot and needs_verification:
            print(f"  [Analyzer] 🧪 Enabling PoT verification...", flush=True)
            pot_result = self._apply_pot_verification(
                problem=data.get("problem", ""),
                steps=steps,
                analysis=analysis
            )
            if pot_result:
                # Enhance actionable_advice with code logic
                analysis["actionable_advice"] = pot_result
                print(f"  [Analyzer] ✅ PoT enhancement completed", flush=True)

        # Publish ANALYSIS event (contains structured JSON and problem text)
        self._publish(Event(
            type=EventType.ANALYSIS,
            data={
                "problem_id": data.get("id"),
                "analysis": analysis,  # Structured JSON: {problem_tags, error_symptom, root_cause, actionable_advice}
                "problem_text": data.get("problem", ""),  # Original problem text (for memU storage)
            },
        ))

    # ------------------------------------------------------------------ #
    # Chunking Strategy
    # ------------------------------------------------------------------ #

    def _adaptive_chunking(self, steps: list[_Step]) -> list[list[_Step]]:
        """🔥 Adaptive chunking strategy: dynamically adjust chunk_size based on trajectory length

        Strategy:
        - <= 5 steps: no chunking (1 GLM call, direct distillation)
        - 6-10 steps: chunk_size=4 (2-3 GLM calls)
        - > 10 steps: chunk_size=5 (controlled to 3-4 GLM calls)

        Goals:
        - Short trajectories: save cost (reduce 1-2 GLM calls)
        - Long trajectories: maintain quality (avoid excessive chunking)
        """
        n_steps = len(steps)

        if n_steps <= 5:
            # Short trajectory: no chunking, distill as whole
            print(f"    [Chunking] Short trajectory ({n_steps} steps), no chunking", flush=True)
            return [steps]
        elif n_steps <= 10:
            # 中等轨迹：chunk_size=4
            chunk_size = 4
            print(f"    [Chunking] 中等轨迹（{n_steps} 步），chunk_size={chunk_size}", flush=True)
        else:
            # 长轨迹：chunk_size=5
            chunk_size = 5
            print(f"    [Chunking] 长轨迹（{n_steps} 步），chunk_size={chunk_size}", flush=True)

        # 分块
        chunks = [steps[i: i + chunk_size] for i in range(0, n_steps, chunk_size)]
        return chunks

    # ------------------------------------------------------------------ #
    # 轨迹解析
    # ------------------------------------------------------------------ #

    def _parse_steps(self, traj_path: Optional[str]) -> list[_Step]:
        """把轨迹文件解析为 _Step 列表（每步 = 思考 + 命令 + 输出）。

        🔥 修复：mini-swe-agent 使用 OpenAI tool_calls 格式
        命令在 tool_calls[0]["function"]["arguments"]["command"] 里
        而不是在 extra.actions 里
        """
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
                # 🔥 新格式：从 tool_calls 提取命令
                cmd = ""
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    try:
                        func = tool_calls[0].get("function", {})
                        if func.get("name") == "bash":
                            args_str = func.get("arguments", "{}")
                            args = json.loads(args_str)
                            cmd = args.get("command", "")[:_CMD_LIMIT]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

                # 兼容旧格式：从 extra.actions 提取
                if not cmd:
                    extra = msg.get("extra") or {}
                    actions = extra.get("actions") or []
                    cmd = (actions[0].get("command", "") if actions else "")[:_CMD_LIMIT]

                # 思考内容（content 可能为空，从命令中提取摘要作为 fallback）
                thought = (msg.get("content") or "")[:_THOUGHT_LIMIT]
                if not thought and cmd:
                    # 如果 content 为空，用命令的前 100 字符作为 thought
                    thought = f"Execute: {cmd[:100]}"

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
        """Summarize a chunk in one sentence. Returns empty string on failure."""
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
        steps: list[_Step],
        problem: str,
        expected_answer: str,
        extracted_answer: str,
        n_steps: int,
        extra_context: str = "",
    ) -> dict:
        """Distill the full trajectory into structured semantic memory (JSON format)."""
        import json as _json

        trajectory = "\n\n".join(f"Step {i+1}:\n{s.render()}" for i, s in enumerate(steps))
        if extra_context:
            trajectory += extra_context
        user_msg = _DISTILL_USER.format(
            problem=problem,
            expected_answer=expected_answer,
            extracted_answer=extracted_answer,
            n_steps=n_steps,
            trajectory=trajectory,
        )
        response = self._llm_call(
            system=_DISTILL_SYSTEM,
            user=user_msg,
            temperature=0.0,
            max_tokens=600,
            extra_body={"thinking": {"type": "disabled"}},
        )

        print(f"  [Analyzer] LLM response length: {len(response)} chars", flush=True)
        print(f"  [Analyzer] LLM raw response (first 500 chars):\n{response[:500]}", flush=True)

        # Parse JSON output
        clean = response.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean.strip())

        try:
            outer = _json.loads(clean)
        except _json.JSONDecodeError as exc:
            print(f"  [Analyzer] ✗ JSON parse failed: {exc}", flush=True)
            return {"skip": True, "error": "parse_failed"}

        # Handle discard signal
        if outer.get("status") == "discard":
            print(f"  [Analyzer] ⏭  Discarded by filtering gate", flush=True)
            return {"skip": True, "error": "trivial_failure_discarded"}

        # Flat fields — no nesting
        required_fields = ["problem_type", "error", "fix",
                           "reasoning_path", "solution_steps", "common_traps"]
        missing = [f for f in required_fields if not outer.get(f)]
        if missing:
            print(f"  [Analyzer] ✗ Missing required fields: {missing}", flush=True)
            return {"skip": True, "error": "parse_failed"}

        return {
            "problem_type":    outer["problem_type"],
            "error":           outer["error"],
            "fix":             outer["fix"],
            "reasoning_path":  outer["reasoning_path"],
            "solution_steps":  outer["solution_steps"],
            "common_traps":    outer["common_traps"],
        }

    def _generate_hint(self, problem: str) -> Optional[dict]:
        """Phase 2：GLM 只看题目，独立生成解题思路提示（不受失败轨迹影响）。

        返回包含 key_insight / approach / common_pitfall 的 dict，或 None（失败时）。
        """
        user_msg = _HINT_USER.format(problem=problem)
        try:
            response = self._llm_call(
                system=_HINT_SYSTEM,
                user=user_msg,
                temperature=0.0,
                max_tokens=200,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as exc:
            print(f"  [Analyzer] ⚠️ Phase 2 LLM 调用失败: {exc}", flush=True)
            return None

        parsed = self._parse_markdown(response)
        required = ["key_insight", "approach"]
        if "error" in parsed or not all(f in parsed for f in required):
            print(f"  [Analyzer] ⚠️ Phase 2 解析失败，跳过 hint", flush=True)
            return None

        return {
            "key_insight": parsed.get("key_insight", ""),
            "approach": parsed.get("approach", ""),
            "common_pitfall": parsed.get("common_pitfall", ""),
        }

    # ------------------------------------------------------------------ #
    # 成功路径
    # ------------------------------------------------------------------ #

    def _on_success(self, data: dict) -> None:
        """分析成功轨迹，提取可复用技术，发布 SUCCESS_ANALYSIS 事件。"""
        steps = self._parse_steps(data.get("traj_path"))
        if not steps:
            return

        # GLM-4.7 支持 200K 上下文，直接蒸馏完整轨迹，无需分块
        print(f"  [Analyzer] 成功轨迹 {len(steps)} 步 → 直接蒸馏（无分块）...", flush=True)

        print(f"  [Analyzer] 蒸馏成功技术...", flush=True)
        try:
            analysis = self._distill_success(
                steps=steps,
                problem=data.get("problem", "")[:300],
                expected_answer=data.get("expected_answer", ""),
                n_steps=data.get("n_steps", 0),
            )
        except Exception as exc:
            return

        can_be_skill = str(analysis.get("can_be_skill", "false")).lower().strip() == "true"
        if not analysis or "error" in analysis or not can_be_skill:
            return

        # 解析 tags（逗号分隔的字符串 → list）
        raw_tags = analysis.get("tags", "")
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()] if isinstance(raw_tags, str) else raw_tags

        self._publish(Event(
            type=EventType.SUCCESS_ANALYSIS,
            data={
                "problem_id": data.get("id"),
                "technique": analysis.get("technique_name", "unknown"),
                "summary": analysis.get("summary", ""),
                "tags": tags,
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
        steps: list[_Step],
        problem: str,
        expected_answer: str,
        n_steps: int,
    ) -> dict:
        """把完整成功轨迹直接蒸馏成可复用技术（GLM-4.7 大上下文，无需分块）。"""
        trajectory = "\n\n".join(f"Step {i+1}:\n{s.render()}" for i, s in enumerate(steps))
        user_msg = _SUCCESS_DISTILL_USER.format(
            problem=problem,
            expected_answer=expected_answer,
            n_steps=n_steps,
            trajectory=trajectory,
        )
        response = self._llm_call(
            system=_SUCCESS_DISTILL_SYSTEM,
            user=user_msg,
            temperature=0.0,
            max_tokens=200,
            extra_body={"thinking": {"type": "disabled"}},
        )

        return self._parse_json(response)

    # ------------------------------------------------------------------ #
    # PoT 程序辅助反思
    # ------------------------------------------------------------------ #

    def _apply_pot_verification(
        self,
        problem: str,
        steps: list[_Step],
        analysis: dict,
    ) -> Optional[str]:
        """应用 PoT 验证，返回增强的 actionable_advice

        流程：
        1. 生成验证代码的 prompt
        2. 调用 LLM 生成 Python/SymPy 代码
        3. 执行代码获取正确结果
        4. 提取可复用的代码模式
        5. 将代码逻辑融入 actionable_advice
        """
        if not self.pot_reflector:
            return None

        try:
            # 将 steps 渲染为文字摘要供 PoT 参考
            step_summaries = [s.render() for s in steps[:5]]  # 最多前5步

            # 1. 生成验证代码 prompt
            code_gen_prompt = self.pot_reflector.generate_verification_code(
                problem=problem,
                failed_output="",
                summaries=step_summaries
            )

            # 2. 调用 LLM 生成代码
            print(f"  [PoT] 生成验证代码...", flush=True)
            code = self._llm_call(
                system="You are a Python code generator. Output ONLY code, no explanation.",
                user=code_gen_prompt,
                temperature=0.0,
                max_tokens=400,
                extra_body={"thinking": {"type": "disabled"}},
            )

            # 3. 执行验证代码
            print(f"  [PoT] 执行验证代码...", flush=True)
            verification = self.pot_reflector.execute_verification_code(code)

            if not verification.success:
                print(f"  [PoT] ⚠️ 代码执行失败: {verification.output[:100]}", flush=True)
                return None

            # 4. 构造增强的 actionable_advice
            original_advice = analysis.get("actionable_advice", "")
            enhanced_advice = self._build_enhanced_advice(
                original_advice=original_advice,
                verification=verification,
                problem_tags=analysis.get("problem_tags", [])
            )

            return enhanced_advice

        except Exception as exc:
            print(f"  [PoT] ✗ PoT 验证失败: {exc}", flush=True)
            return None

    def _build_enhanced_advice(
        self,
        original_advice: str,
        verification,
        problem_tags: list[str],
    ) -> str:
        """Construct enhanced actionable_advice"""
        code_snippet = self._extract_key_code_snippet(
            verification.code,
            problem_tags
        )

        if verification.reusable_pattern:
            return f"{original_advice} | CODE: {verification.reusable_pattern}"
        elif code_snippet:
            return f"{original_advice} | CODE: {code_snippet}"
        else:
            return original_advice

    def _extract_key_code_snippet(
        self,
        code: str,
        problem_tags: list[str]
    ) -> Optional[str]:
        """Extract key reusable fragments from complete code"""
        lines = code.split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('import') or line.startswith('from'):
                continue

            # 检测关键函数调用
            if 'solve(' in line or 'pow(' in line or 'factorial' in line or 'binomial' in line:
                return self._clean_snippet(line)

        return None

    def _clean_snippet(self, line: str) -> str:
        """清理代码片段"""
        if '=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2:
                line = parts[1].strip()

        line = line.rstrip(';').strip()
        if len(line) > 80:
            line = line[:77] + "..."

        return line

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #

    def _parse_markdown(self, text: str) -> dict:
        """Parse Markdown key-value pair format, for example:
        [Root_Cause]: xxx
        [Actionable_Advice]: xxx

        兼容智谱 GLM 的推理模式：忽略前面的思考，只提取 [Key]: Value。
        """
        result = {}

        # 🔥 找到第一个 [Key]:，从这里开始提取（丢弃前面的思考）
        first_bracket = text.find('[')
        if first_bracket > 0:
            text = text[first_bracket:]

        # 🔥 改进的正则表达式：
        # 1. 支持 Key_Steps 或 Key-Steps 或 KeySteps
        # 2. 值部分匹配到下一个 [Key] 或字符串结尾（用 \Z 而非 $）
        # 3. 移除值末尾的空白字符
        pattern = re.compile(r"\[([A-Za-z_-]+)\]:\s*(.+?)(?=\n\[|\Z)", re.DOTALL | re.MULTILINE)
        for match in pattern.finditer(text):
            key_raw = match.group(1)
            # 统一转换：Key_Steps -> key_steps, Key-Steps -> key_steps
            key = key_raw.lower().replace('-', '_')
            value = match.group(2).strip()
            result[key] = value

        return result if result else {"error": "failed to parse", "raw": text[:300]}

    def _parse_json(self, text: str) -> dict:
        """兼容旧代码，实际调用 _parse_markdown"""
        return self._parse_markdown(text)



