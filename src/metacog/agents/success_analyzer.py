"""SuccessAnalyzer - 成功案例分析器

订阅 TrajectoryEvent（成功的），提取关键推理步骤和核心洞察，存入情景记忆。

设计原则
--------
1. 只处理成功的轨迹（passed=True）
2. 提取 3-5 个关键推理步骤（不是全部步骤）
3. 蒸馏出核心洞察（为什么这个方法有效）
4. 存入情景记忆供后续检索
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from minisweagent.models.litellm_model import LitellmModel

from ..bus import Event, EventBus, EventType
from ..memory.episodic_memory import EpisodicMemory
from .base import BaseAgent


class SuccessAnalyzer(BaseAgent):
    """成功案例分析器
    
    订阅成功的 TrajectoryEvent，提取关键步骤和洞察，存入情景记忆
    """
    
    def __init__(
        self,
        model: LitellmModel,
        bus: EventBus,
        episodic_memory: EpisodicMemory,
    ) -> None:
        super().__init__(model, bus)
        self.episodic_memory = episodic_memory
    
    def _register_handlers(self) -> None:
        self.bus.subscribe(EventType.TRAJECTORY, self._on_trajectory)
    
    def _on_trajectory(self, event: Event) -> None:
        """处理轨迹事件（只处理成功的）"""
        data = event.data

        print(f"  [SuccessAnalyzer] 收到轨迹事件: id={data.get('id')}, passed={data.get('passed')}", flush=True)

        # 只处理成功的轨迹
        if not data.get("passed", False):
            print(f"  [SuccessAnalyzer] 跳过（未成功）", flush=True)
            return

        problem_id = data.get("id", "unknown")
        problem_text = data.get("problem", "")
        answer = data.get("extracted_answer", "")
        traj_path = data.get("traj_path")

        print(f"  [SuccessAnalyzer] passed=True, traj_path={traj_path}", flush=True)

        if not traj_path or not Path(traj_path).exists():
            print(f"  [SuccessAnalyzer] ✗ 轨迹文件不存在: {traj_path}", flush=True)
            return

        print(f"  [SuccessAnalyzer] 分析成功案例: {problem_id}", flush=True)
        
        # 读取轨迹
        print(f"  [SuccessAnalyzer] 读取轨迹文件...", flush=True)
        try:
            traj_data = json.loads(Path(traj_path).read_text())
            print(f"  [SuccessAnalyzer] 轨迹文件解析成功", flush=True)
        except Exception as e:
            print(f"  [SuccessAnalyzer] ✗ 轨迹文件读取失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return

        # mini-swe-agent 1.1 的轨迹在 "messages" 字段中
        trajectory = traj_data.get("messages") or traj_data.get("trajectory") or traj_data.get("history", [])
        print(f"  [SuccessAnalyzer] 原始消息数: {len(trajectory)} 条（含 system/user/assistant/tool/exit）", flush=True)

        if not trajectory:
            print(f"  [SuccessAnalyzer] ✗ 轨迹为空", flush=True)
            return

        # 过滤：步数 < 2 说明模型完全没有写代码（纯文字作答），不值得存
        # 使用 event 里 Executor 传入的 n_steps，与日志显示口径一致
        # （避免重新统计 assistant tool_call 消息数导致的口径不一致：
        #   PoT 注入的 user 反思消息会被日志计入步数，但不会产生 assistant 消息）
        n_steps = data.get("n_steps", 0)
        if n_steps < 2:
            print(f"  [SuccessAnalyzer] ⚠️  步数太少（{n_steps} 步），模型未写代码，跳过情景记忆存储", flush=True)
            return
        
        # 提取关键步骤
        try:
            analysis = self._extract_key_steps(
                problem=problem_text,
                trajectory=trajectory,
                answer=answer,
            )
        except Exception as exc:
            print(f"  [SuccessAnalyzer] ✗ 提取失败: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            return
        
        if not analysis or "error" in analysis:
            return
        
        # 存入情景记忆（Markdown 格式：key_steps 用 | 分隔，tags 用逗号分隔）
        raw_steps = analysis.get("key_steps", "")
        key_steps = [s.strip() for s in raw_steps.split("|") if s.strip()] if isinstance(raw_steps, str) else raw_steps
        core_insight = analysis.get("core_insight", "")
        raw_tags = analysis.get("tags", "")
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()] if isinstance(raw_tags, str) else raw_tags
        problem_type = analysis.get("problem_type", "")

        # 🔥 ChromaDB 不允许空 list，如果 tags 为空，给默认值
        if not tags:
            tags = ["general"]

        # 检查必需字段
        missing_fields = []
        if not key_steps:
            missing_fields.append("key_steps")
        if not core_insight:
            missing_fields.append("core_insight")

        if missing_fields:
            print(f"  [SuccessAnalyzer] ⚠️  缺少关键字段: {missing_fields}，跳过存储", flush=True)
            print(f"  [SuccessAnalyzer] 解析到的字段: {list(analysis.keys())}", flush=True)
            print(f"  [SuccessAnalyzer] raw_steps='{raw_steps[:100]}', core_insight='{core_insight[:100]}'", flush=True)
            return
        
        try:
            memory_id = self.episodic_memory.add_success_case(
                problem_id=problem_id,
                problem_text=problem_text,
                solution_steps=key_steps,
                key_insight=core_insight,
                tags=tags,
                answer=answer,
                problem_type=problem_type,
            )
            print(f"  [SuccessAnalyzer] ✓ 存入情景记忆: {memory_id[:12]} | tags={tags}", flush=True)
        except Exception as exc:
            print(f"  [SuccessAnalyzer] ✗ 存储失败: {exc}", flush=True)
            import traceback
            traceback.print_exc()
    
    def _extract_key_steps(
        self,
        problem: str,
        trajectory: list[dict],
        answer: str,
    ) -> dict:
        """🔥 改进：使用 GLM 分析成功案例，提取深层洞察

        策略：
        1. 提取前 5 步关键操作（包含思考和代码）
        2. 调用 GLM 分析"为什么这个方法有效"
        3. 解析结构化输出（与失败分析保持一致的格式）
        4. 如果 GLM 失败，fallback 到规则提取
        """
        print(f"    [SuccessAnalyzer] 🔥 使用 GLM 分析成功案例...", flush=True)
        print(f"    [SuccessAnalyzer] 原始消息数: {len(trajectory)} 条", flush=True)

        # 🔥 第一步：提取前 5 步的关键内容（思考 + 代码）
        trajectory_summary = self._extract_trajectory_summary(trajectory[:5])

        # 🔥 第二步：提取代码块（作为辅助信息）
        code_blocks = self._extract_code_blocks(trajectory)
        all_code = "\n".join(code_blocks[:3]) if code_blocks else "无代码"  # 最多取前 3 个代码块

        print(f"    [SuccessAnalyzer] 轨迹摘要长度: {len(trajectory_summary)} 字符", flush=True)
        print(f"    [SuccessAnalyzer] 代码块数量: {len(code_blocks)}", flush=True)

        # 🔥 第三步：调用 GLM 分析
        try:
            analysis = self._llm_analyze_success(
                problem=problem,
                trajectory_summary=trajectory_summary,
                code_blocks=all_code,
                answer=answer
            )

            if analysis and "error" not in analysis:
                print(f"    [SuccessAnalyzer] ✓ GLM 分析成功", flush=True)
                print(f"      - key_steps: {analysis.get('key_steps', '')[:80]}...", flush=True)
                print(f"      - core_insight: {analysis.get('core_insight', '')[:80]}...", flush=True)
                return analysis

            print(f"    [SuccessAnalyzer] ⚠️  GLM 分析失败，fallback 到规则提取", flush=True)
        except Exception as exc:
            print(f"    [SuccessAnalyzer] ⚠️  GLM 调用异常: {exc}，fallback 到规则提取", flush=True)
            import traceback
            traceback.print_exc()

        # 🔥 第四步：fallback 到规则提取
        return self._fallback_rule_based_extraction(problem, trajectory, answer)
    
    def _extract_trajectory_summary(self, first_steps: list[dict]) -> str:
        """提取前几步的关键操作（思考 + 代码）

        🔥 修复：mini-swe-agent 的 assistant 消息 content 为空
        实际思考和命令在 tool_calls 里，需要从那里提取摘要
        """
        import json as json_lib
        summaries = []

        for i, step in enumerate(first_steps, 1):
            role = step.get("role", "")

            if role == "assistant":
                # 🔥 新格式：从 tool_calls 提取命令摘要
                tool_calls = step.get("tool_calls", [])
                cmd_summary = ""

                for tc in tool_calls:
                    func = tc.get("function", {})
                    if func.get("name") == "bash":
                        try:
                            args_str = func.get("arguments", "{}")
                            args = json_lib.loads(args_str)
                            cmd = args.get("command", "")

                            # 提取关键信息：如果是 Python 代码，取注释；否则取命令前 100 字符
                            if "python3" in cmd and "<<" in cmd:
                                # 提取 heredoc 中的前几行注释
                                lines = cmd.split('\n')
                                comments = [l.strip() for l in lines if l.strip().startswith('#')]
                                cmd_summary = ' '.join(comments[:3])[:200]  # 前 3 行注释
                            else:
                                cmd_summary = cmd[:100]  # 直接命令
                        except (json_lib.JSONDecodeError, KeyError):
                            pass

                # 兼容旧格式：如果 content 有内容，也提取
                content = step.get("content", "")
                if content:
                    cmd_summary = content[:200]

                if cmd_summary:
                    summaries.append(f"Step {i}: {cmd_summary}")

            elif role == "tool":
                # 提取工具输出的关键部分（前 100 字符）
                output = step.get("content", "")
                if output:
                    output_snippet = output[:100].replace('\n', ' ')
                    summaries.append(f"  Output: {output_snippet}")

        return "\n\n".join(summaries[:10])  # 最多 10 项（5 步 + 5 输出）

    def _extract_code_blocks(self, trajectory: list[dict]) -> list[str]:
        """从轨迹中提取所有代码块

        🔥 修复：mini-swe-agent 使用 OpenAI tool_calls 格式
        代码在 tool_calls[0]["function"]["arguments"] 的 command 字段里
        而不是在 content 的 ```python 代码块里
        """
        import re
        import json as json_lib
        code_blocks = []

        for step in trajectory:
            role = step.get("role", "")
            if role == "assistant":
                # 新格式：从 tool_calls 提取
                tool_calls = step.get("tool_calls", [])
                for tc in tool_calls:
                    func = tc.get("function", {})
                    if func.get("name") == "bash":
                        try:
                            # arguments 是 JSON 字符串：{"command": "python3 << 'EOF'\n...code...\nEOF"}
                            args_str = func.get("arguments", "{}")
                            args = json_lib.loads(args_str)
                            cmd = args.get("command", "")

                            # 提取 heredoc 中的 Python 代码
                            # 兼容两种格式：
                            #   有结束符: python3 << 'EOF'\n...code...\nEOF
                            #   无结束符: python3 << 'EOF'\n...code...  (到字符串末尾)
                            heredoc_match = re.search(
                                r"python3\s*<<\s*['\"]?EOF['\"]?\s*\n(.*?)(?:\nEOF\s*$|\Z)",
                                cmd, re.DOTALL
                            )
                            if heredoc_match:
                                code_blocks.append(heredoc_match.group(1).rstrip())
                            # 也处理直接的 python3 -c '...' 格式
                            elif "python3 -c" in cmd:
                                code_match = re.search(r"python3\s+-c\s+['\"](.+?)['\"]", cmd, re.DOTALL)
                                if code_match:
                                    code_blocks.append(code_match.group(1))
                        except (json_lib.JSONDecodeError, KeyError, AttributeError):
                            continue

                # 兼容旧格式：从 content 里的 ```python 代码块提取（以防万一）
                content = step.get("content", "")
                if content:
                    matches = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
                    code_blocks.extend(matches)

        return code_blocks

    def _llm_analyze_success(
        self,
        problem: str,
        trajectory_summary: str,
        code_blocks: str,
        answer: str
    ) -> dict:
        """🔥 新增：调用 GLM 分析成功案例

        返回格式（Markdown 键值对）：
        [Key_Steps]: 步骤1 | 步骤2 | 步骤3
        [Core_Insight]: 为什么这个方法有效（1句话）
        [Tags]: tag1, tag2
        """
        system_prompt = """\
You are a math solution analyst. Extract reusable techniques from successful solutions.

You may think, but your response MUST end with EXACTLY these four lines (no markdown, no backticks):

[Problem_Type]: abstract 1-sentence description of the mathematical structure (NOT specific numbers, focus on the technique/domain)
[Key_Steps]: step1 | step2 | step3
[Core_Insight]: why this approach worked (1 sentence, max 20 words)
[Tags]: tag1, tag2

EXAMPLE output ending:
[Problem_Type]: divisor enumeration over base representations to satisfy divisibility constraint
[Key_Steps]: convert to base-10 | find divisors | filter valid bases
[Core_Insight]: divisor enumeration directly yields valid bases satisfying the divisibility constraint
[Tags]: number_theory, divisibility"""

        user_prompt = f"""Problem: {problem[:300]}

Trajectory:
{trajectory_summary[:600]}

Code:
{code_blocks[:400]}

Answer: {answer}"""

        try:
            # 使用 BaseAgent._llm_call()，与 AnalyzerAgent 保持一致
            llm_output = self._llm_call(
                system=system_prompt,
                user=user_prompt,
                temperature=0.0,
                max_tokens=200,  # 关闭 thinking 后只需三行输出
                extra_body={"thinking": {"type": "disabled"}},  # 关闭 GLM thinking，节省 token
            )

            if not llm_output:
                print(f"    [SuccessAnalyzer] ⚠️  GLM 返回空内容", flush=True)
                return {"error": "empty_response"}

            print(f"    [SuccessAnalyzer] GLM 原始输出:\n{llm_output}", flush=True)

            # 使用与 AnalyzerAgent 相同的解析逻辑
            parsed = self._parse_markdown(llm_output)

            # 验证必需字段
            if "key_steps" not in parsed or "core_insight" not in parsed:
                print(f"    [SuccessAnalyzer] ⚠️  解析结果缺少必需字段: {list(parsed.keys())}", flush=True)
                return {"error": "missing_required_fields"}

            return parsed

        except Exception as exc:
            print(f"    [SuccessAnalyzer] ✗ GLM 调用失败: {exc}", flush=True)
            return {"error": str(exc)}

    def _parse_markdown(self, text: str) -> dict:
        """解析 Markdown 键值对格式（与 AnalyzerAgent._parse_markdown 保持一致）

        格式：
        [Key_Name]: value
        [Another_Key]: another value
        """
        import re
        result = {}

        # 找到第一个 [Key]:，从这里开始提取（丢弃前面的思考）
        first_bracket = text.find('[')
        if first_bracket > 0:
            text = text[first_bracket:]

        # 正则表达式：支持 Key_Steps 或 Key-Steps 或 KeySteps
        pattern = re.compile(r"\[([A-Za-z_-]+)\]:\s*(.+?)(?=\n\[|\Z)", re.DOTALL | re.MULTILINE)
        for match in pattern.finditer(text):
            key_raw = match.group(1)
            # 统一转换：Key_Steps -> key_steps, Key-Steps -> key_steps
            key = key_raw.lower().replace('-', '_')
            value = match.group(2).strip()
            result[key] = value

        return result if result else {"error": "failed to parse", "raw": text[:300]}

    def _fallback_rule_based_extraction(
        self,
        problem: str,
        trajectory: list[dict],
        answer: str
    ) -> dict:
        """Fallback：基于规则的关键步骤提取（原逻辑）"""
        print(f"    [SuccessAnalyzer] 使用规则提取作为 fallback...", flush=True)

        # 提取代码块
        code_blocks = self._extract_code_blocks(trajectory)
        all_code = "\n".join(code_blocks)

        if not code_blocks:
            return self._fallback_simple_summary(problem, answer)

        # 分析代码模式
        techniques = self._identify_techniques(all_code, problem)

        if not techniques:
            return self._fallback_simple_summary(problem, answer)

        # 组合成最终格式
        key_steps = " | ".join(techniques[:5])
        core_insight = self._generate_insight(techniques, problem)
        tags = self._infer_tags(problem, all_code)

        result = {
            "key_steps": key_steps,
            "core_insight": core_insight,
            "tags": ", ".join(tags),
        }

        print(f"    [SuccessAnalyzer] ✓ 规则提取完成", flush=True)
        return result

    def _identify_techniques(self, code: str, problem: str) -> list[str]:
        """从代码中识别关键技术和步骤"""
        techniques = []

        # 模式 1: 数论相关
        if "%" in code or "mod" in code.lower():
            techniques.append("apply modular arithmetic")

        if "pow(" in code:
            # pow(a, b, m) 表示模幂（三参数）
            pow_idx = code.find("pow(")
            if pow_idx >= 0:
                snippet = code[pow_idx:pow_idx+50]
                if snippet.count(",") >= 2:
                    techniques.append("use modular exponentiation")

        # 模式 2: 进制转换
        if "int(" in code and "," in code:
            techniques.append("convert between bases")

        # 模式 3: 循环搜索
        if "for " in code and "range(" in code:
            techniques.append("iterate through candidates")

        # 模式 4: 整除性检查
        if "==" in code and "0" in code:
            techniques.append("check divisibility")

        # 模式 5: 求和/计数
        if "sum(" in code:
            techniques.append("aggregate results")
        elif ".append(" in code:
            techniques.append("collect valid solutions")

        # 模式 6: 组合数学
        if "factorial" in code or "comb" in code:
            techniques.append("compute combinatorial terms")

        # 模式 7: 数学库函数
        if "math." in code or "from math import" in code:
            techniques.append("use math library functions")

        # 如果没有识别到任何技术，给一个通用的
        if not techniques:
            techniques.append("implement direct computation")

        return techniques

    def _generate_insight(self, techniques: list[str], problem: str) -> str:
        """根据识别的技术生成核心洞察"""
        # 简单模板：根据第一个技术生成
        first = techniques[0] if techniques else "systematic approach"

        if "modular" in first:
            return "modular arithmetic simplifies the computation"
        elif "base" in first or "convert" in first:
            return "base conversion reveals the pattern"
        elif "iterate" in first or "search" in first:
            return "systematic enumeration finds all solutions"
        elif "combinatorial" in first:
            return "combinatorial formula gives direct answer"
        else:
            return "direct computation solves the problem"

    def _infer_tags(self, problem: str, code: str) -> list[str]:
        """从问题和代码推断标签"""
        tags = []

        # 从问题文本推断
        problem_lower = problem.lower()
        if any(kw in problem_lower for kw in ["base", "divisor", "modulo", "remainder"]):
            tags.append("number_theory")
        if any(kw in problem_lower for kw in ["combinations", "permutations", "choose"]):
            tags.append("combinatorics")
        if any(kw in problem_lower for kw in ["probability", "expected"]):
            tags.append("probability")
        if any(kw in problem_lower for kw in ["sequence", "series", "recursive"]):
            tags.append("sequences")

        # 从代码推断
        if "%" in code or "mod" in code:
            if "number_theory" not in tags:
                tags.append("modular_arithmetic")

        # 默认标签
        if not tags:
            tags.append("general")

        return tags

    def _fallback_simple_summary(self, problem: str, answer: str) -> dict:
        """简单的 fallback 总结（不依赖代码分析）"""
        print(f"    [SuccessAnalyzer] 使用 fallback 简单总结", flush=True)

        return {
            "key_steps": "analyze problem | compute solution | verify answer",
            "core_insight": "systematic problem solving",
            "tags": ", ".join(self._infer_tags(problem, "")),
        }

    def _parse_json(self, text: str) -> dict:
        """解析 Markdown 键值对格式

        兼容智谱 GLM 的推理模式：
        1. 先输出思考过程（"The user wants..."）
        2. 再输出最终答案（[Key]: Value）

        我们只提取最终答案部分。

        如果完全找不到格式，尝试 fallback 简单提取。
        """
        result = {}

        # 🔥 智谱推理模式：找到第一个 [Key]:，从这里开始提取
        first_bracket = text.find('[')
        if first_bracket > 0:
            # 截取从第一个 [ 开始的内容，丢弃前面的思考
            text = text[first_bracket:]
        elif first_bracket < 0:
            # 完全找不到 [，说明 LLM 没按格式输出
            print(f"    [_parse_json] ⚠️  响应中找不到 [Key]: 格式，尝试 fallback", flush=True)
            return self._fallback_parse(text)

        # 🔥 改进的正则表达式：
        # 1. 支持 Key_Steps 或 Key-Steps 或 KeySteps
        # 2. 值部分匹配到下一个 [Key] 或字符串结尾
        # 3. 移除值末尾的空白字符
        pattern = re.compile(r"\[([A-Za-z_-]+)\]:\s*(.+?)(?=\n\[|\Z)", re.DOTALL | re.MULTILINE)
        for match in pattern.finditer(text):
            key_raw = match.group(1)
            # 统一转换：Key_Steps -> key_steps, Key-Steps -> key_steps
            key = key_raw.lower().replace('-', '_')
            value = match.group(2).strip()
            result[key] = value

        # 打印调试信息
        if result:
            print(f"    [_parse_json] 成功解析 {len(result)} 个字段: {list(result.keys())}", flush=True)
        else:
            print(f"    [_parse_json] 正则解析失败，尝试 fallback", flush=True)
            print(f"    [_parse_json] 文本片段:\n{text[:500]}", flush=True)
            return self._fallback_parse(text)

        return result if result else {"error": "failed to parse", "raw": text[:300]}

    def _fallback_parse(self, text: str) -> dict:
        """Fallback: 当 LLM 完全不按格式输出时的简单提取

        尝试从自然语言中提取关键信息：
        - 寻找 "steps:", "key steps:", "reasoning:" 等关键词
        - 提取问题标签（从 problem 文本推断）
        """
        print(f"    [_fallback_parse] 启动 fallback 解析", flush=True)

        # 🔥 策略：直接放弃，返回错误
        # 因为如果 LLM 不遵循格式，提取出的信息质量也不可靠
        return {
            "error": "LLM did not follow format",
            "raw": text[:300],
            "suggestion": "Consider using a stronger model or adjusting temperature to 0"
        }


