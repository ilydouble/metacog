"""AnalyzerAgent - 订阅 TrajectoryEvent，分段分析轨迹，蒸馏成精简记忆条目。

设计原则
--------
- 上下文受限：每次 LLM 调用只喂一小块（chunk），不把整条轨迹塞进去。
- 高度精简：每块只产出一句话摘要；最终蒸馏只产出 1-2 句可执行建议。
- **Program-of-Thoughts (PoT)**: 对数学错误生成验证代码，提取可复用逻辑
- 分段流程：
    轨迹文件 → parse_steps → 按 chunk_size 分块
    → 每块调 LLM 得一句摘要
    → 所有摘要 + 题目元数据 → 调 LLM 蒸馏 → AnalysisEvent
    → (可选) PoT 验证 → 生成代码 → 执行 → 提取可复用模式
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
from .pot_reflector import PoTReflector
from .trajectory_analyzer import TrajectoryAnalyzer

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
You are a math tutor analyzing student errors. Extract a reusable lesson.

OUTPUT FORMAT (MANDATORY - your ENTIRE response must be ONLY this):

[Problem_Tags]: tag1, tag2
[Error_Symptom]: what went wrong
[Root_Cause]: why it happened
[Actionable_Advice]: how to fix it
[Needs_Code_Verification]: true

EXAMPLE:
[Problem_Tags]: number_theory, modular_arithmetic
[Error_Symptom]: forgot modular reduction after multiplication
[Root_Cause]: missed the requirement to apply mod at each step
[Actionable_Advice]: always apply % m after each arithmetic operation in modular context
[Needs_Code_Verification]: true

Do NOT include any explanation. Output ONLY the five lines above."""

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
        # 🔥 死循环检测（事后分析）
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
        # 🔥 失败路由（阻断操作性失误入库）
        # ========================================
        from .failure_router import route_failure
        routing = route_failure(
            steps=steps,
            loop_detected=bool(loop_pattern),
            extracted_answer=data.get("extracted_answer"),
            expected_answer=data.get("expected_answer"),
        )

        if not routing.should_store:
            print(f"  [Router] 🚫 阻断入库: {routing.reason}", flush=True)
            return

        print(f"  [Router] ✅ 允许入库: {routing.reason}", flush=True)

        # GLM-4.7 支持 200K 上下文，直接蒸馏完整轨迹，无需分块
        print(f"  [Analyzer] 失败轨迹 {len(steps)} 步 → 直接蒸馏（无分块）...", flush=True)

        # 构建附加上下文（死循环/低效检测结果）
        extra_context = ""
        if loop_pattern:
            extra_context += f"\n[LOOP DETECTED] {loop_pattern.description}"
        if inefficient_approach:
            extra_context += f"\n[INEFFICIENT] {inefficient_approach}"

        # 蒸馏成最终记忆条目
        print(f"  [Analyzer] 蒸馏完整轨迹 → 记忆条目...", flush=True)
        try:
            analysis = self._distill(
                steps=steps,
                problem=data.get("problem", "")[:300],
                expected_answer=data.get("expected_answer", ""),
                extracted_answer=data.get("extracted_answer") or "(none)",
                n_steps=data.get("n_steps", 0),
                extra_context=extra_context,
            )
        except Exception as exc:
            print(f"  [Analyzer] ❌ 蒸馏发生异常: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            analysis = {"skip": True, "error": str(exc)}

        if "error" in analysis or analysis.get("skip"):
            return

        # ========================================
        # 🔥 PoT 增强：对数学计算错误进行程序验证
        # ========================================
        needs_verification = str(analysis.get("needs_code_verification", "false")).lower().strip() == "true"
        if self.enable_pot and needs_verification:
            print(f"  [Analyzer] 🧪 启用 PoT 验证...", flush=True)
            pot_result = self._apply_pot_verification(
                problem=data.get("problem", ""),
                steps=steps,
                analysis=analysis
            )
            if pot_result:
                # 用代码逻辑增强 actionable_advice
                analysis["actionable_advice"] = pot_result
                print(f"  [Analyzer] ✅ PoT 增强完成", flush=True)

        # 发布 ANALYSIS 事件（包含结构化 JSON 和题目原文）
        self._publish(Event(
            type=EventType.ANALYSIS,
            data={
                "problem_id": data.get("id"),
                "analysis": analysis,  # 结构化 JSON: {problem_tags, error_symptom, root_cause, actionable_advice}
                "problem_text": data.get("problem", ""),  # 题目原文（供 memU 存储）
            },
        ))

    # ------------------------------------------------------------------ #
    # 分块策略
    # ------------------------------------------------------------------ #

    def _adaptive_chunking(self, steps: list[_Step]) -> list[list[_Step]]:
        """🔥 自适应分块策略：根据轨迹长度动态调整 chunk_size

        策略：
        - <= 5 步：不分块（1 次 GLM 调用，直接蒸馏）
        - 6-10 步：chunk_size=4（2-3 次 GLM 调用）
        - > 10 步：chunk_size=5（控制在 3-4 次 GLM 调用）

        目标：
        - 短轨迹：节省成本（减少 1-2 次 GLM 调用）
        - 长轨迹：保持质量（不会过度分块）
        """
        n_steps = len(steps)

        if n_steps <= 5:
            # 短轨迹：不分块，整体蒸馏
            print(f"    [Chunking] 短轨迹（{n_steps} 步），不分块", flush=True)
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
        steps: list[_Step],
        problem: str,
        expected_answer: str,
        extracted_answer: str,
        n_steps: int,
        extra_context: str = "",
    ) -> dict:
        """把完整轨迹直接蒸馏成结构化记忆（GLM-4.7 大上下文，无需分块）。"""
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
            max_tokens=300,
            extra_body={"thinking": {"type": "disabled"}},
        )

        print(f"  [Analyzer] LLM 返回长度: {len(response)} 字符", flush=True)
        print(f"  [Analyzer] LLM 原始响应（前 500 字符）:\n{response[:500]}", flush=True)

        # 严格 Markdown 解析和验证
        parsed = self._parse_json(response)

        # 验证必需字段
        required_fields = ["problem_tags", "error_symptom", "root_cause", "actionable_advice"]
        if "error" in parsed or not all(f in parsed for f in required_fields):
            print(f"  [Analyzer] ✗ 解析失败或缺少必需字段", flush=True)
            print(f"  [Analyzer] 解析结果: {parsed}", flush=True)
            print(f"  [Analyzer] 缺少的字段: {[f for f in required_fields if f not in parsed]}", flush=True)
            return {"skip": True, "error": "parse_failed"}

        # 确保 problem_tags 是列表
        if not isinstance(parsed["problem_tags"], list):
            parsed["problem_tags"] = [str(parsed["problem_tags"])]

        return parsed

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
        """构造增强的 actionable_advice"""
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
        """从完整代码中提取关键的可复用片段"""
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
        """解析 Markdown 键值对格式，例如：
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



