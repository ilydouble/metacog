"""SuccessAnalyzer - Success case analyzer

Subscribes to TrajectoryEvent (successful ones), extracts key reasoning steps and core insights,
and stores them in episodic memory.

Design Principles
-----------------
1. Only process successful trajectories (passed=True)
2. Extract 3-5 key reasoning steps (not all steps)
3. Distill core insights (why this approach worked)
4. Store in episodic memory for future retrieval
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
    """Success case analyzer

    Subscribes to successful TrajectoryEvents, extracts key steps and insights, stores in episodic memory
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
        """Handle trajectory events (only successful ones)"""
        data = event.data

        print(f"  [SuccessAnalyzer] Received trajectory event: id={data.get('id')}, passed={data.get('passed')}", flush=True)

        # Only process successful trajectories
        if not data.get("passed", False):
            print(f"  [SuccessAnalyzer] Skipped (not successful)", flush=True)
            return

        problem_id = data.get("id", "unknown")
        problem_text = data.get("problem", "")
        answer = data.get("extracted_answer", "")
        traj_path = data.get("traj_path")

        print(f"  [SuccessAnalyzer] passed=True, traj_path={traj_path}", flush=True)

        if not traj_path or not Path(traj_path).exists():
            print(f"  [SuccessAnalyzer] ✗ Trajectory file does not exist: {traj_path}", flush=True)
            return

        print(f"  [SuccessAnalyzer] Analyzing success case: {problem_id}", flush=True)

        # Read trajectory
        print(f"  [SuccessAnalyzer] Reading trajectory file...", flush=True)
        try:
            traj_data = json.loads(Path(traj_path).read_text())
            print(f"  [SuccessAnalyzer] Trajectory file parsed successfully", flush=True)
        except Exception as e:
            print(f"  [SuccessAnalyzer] ✗ Failed to read trajectory file: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return

        # mini-swe-agent 1.1 trajectories are in the "messages" field
        trajectory = traj_data.get("messages") or traj_data.get("trajectory") or traj_data.get("history", [])
        print(f"  [SuccessAnalyzer] Raw message count: {len(trajectory)} (including system/user/assistant/tool/exit)", flush=True)

        if not trajectory:
            print(f"  [SuccessAnalyzer] ✗ Trajectory is empty", flush=True)
            return

        # Filter: steps < 2 means the model didn't write any code (pure text answer), not worth storing
        # Use n_steps from event passed by Executor, consistent with log display
        # (Avoid inconsistency from recounting assistant tool_call messages:
        #   PoT-injected user reflection messages are counted as steps in logs but don't produce assistant messages)
        n_steps = data.get("n_steps", 0)
        if n_steps < 2:
            print(f"  [SuccessAnalyzer] ⚠️  Too few steps ({n_steps} steps), model didn't write code, skipping episodic memory storage", flush=True)
            return

        # Extract key steps
        try:
            analysis = self._extract_key_steps(
                problem=problem_text,
                trajectory=trajectory,
                answer=answer,
            )
        except Exception as exc:
            print(f"  [SuccessAnalyzer] ✗ Extraction failed: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            return

        if not analysis or "error" in analysis:
            return

        # Episodic memory — 4 flat fields from LLM
        problem_type = analysis.get("problem_type", "")
        key_insight  = analysis.get("key_insight", "")
        tags         = analysis.get("tags") or ["general"]
        approach     = analysis.get("approach") or []

        solution_steps = approach  # approach is already a list of steps

        # Validate required fields
        missing_fields = []
        if not solution_steps:
            missing_fields.append("approach")
        if not key_insight:
            missing_fields.append("key_insight")

        if missing_fields:
            print(f"  [SuccessAnalyzer] ⚠️  Missing required fields: {missing_fields}, skipping storage", flush=True)
            return

        try:
            memory_id = self.episodic_memory.add_success_case(
                problem_id=problem_id,
                problem_text=problem_text,
                solution_steps=solution_steps,
                key_insight=key_insight,
                tags=tags,
                answer=answer,
                problem_type=problem_type,
            )
            print(f"  [SuccessAnalyzer] ✓ Stored in episodic memory: {memory_id[:12]} | tags={tags}", flush=True)
        except Exception as exc:
            print(f"  [SuccessAnalyzer] ✗ Storage failed: {exc}", flush=True)
            import traceback
            traceback.print_exc()
    
    def _extract_key_steps(
        self,
        problem: str,
        trajectory: list[dict],
        answer: str,
    ) -> dict:
        """🔥 Improvement: Use GLM to analyze success cases, extract deep insights

        Strategy:
        1. Extract first 5 steps of key operations (including thinking and code)
        2. Call GLM to analyze "why this method worked"
        3. Parse structured output (consistent format with failure analysis)
        4. If GLM fails, fallback to rule-based extraction
        """
        print(f"    [SuccessAnalyzer] 🔥 Using GLM to analyze success case...", flush=True)
        print(f"    [SuccessAnalyzer] Raw message count: {len(trajectory)}", flush=True)

        # 🔥 Step 1: Extract key content from first 5 steps (thinking + code)
        trajectory_summary = self._extract_trajectory_summary(trajectory[:5])

        # 🔥 Step 2: Extract code blocks (as auxiliary info)
        code_blocks = self._extract_code_blocks(trajectory)
        all_code = "\n".join(code_blocks[:3]) if code_blocks else "No code"  # Max 3 code blocks

        print(f"    [SuccessAnalyzer] Trajectory summary length: {len(trajectory_summary)} chars", flush=True)
        print(f"    [SuccessAnalyzer] Code block count: {len(code_blocks)}", flush=True)

        # 🔥 Step 3: Call GLM analysis
        try:
            analysis = self._llm_analyze_success(
                problem=problem,
                trajectory_summary=trajectory_summary,
                code_blocks=all_code,
                answer=answer
            )

            if analysis and "error" not in analysis:
                print(f"    [SuccessAnalyzer] ✓ GLM analysis successful", flush=True)
                print(f"      - key_steps: {analysis.get('key_steps', '')[:80]}...", flush=True)
                print(f"      - core_insight: {analysis.get('core_insight', '')[:80]}...", flush=True)
                return analysis

            print(f"    [SuccessAnalyzer] ⚠️  GLM analysis failed, fallback to rule extraction", flush=True)
        except Exception as exc:
            print(f"    [SuccessAnalyzer] ⚠️  GLM call exception: {exc}, fallback to rule extraction", flush=True)
            import traceback
            traceback.print_exc()

        # Step 4: fallback to rule-based extraction
        return self._fallback_rule_based_extraction(problem, trajectory, answer)
    
    def _extract_trajectory_summary(self, first_steps: list[dict]) -> str:
        """Extract key operations from the first few steps (thinking + code)

        Fixed: mini-swe-agent assistant message content is empty
        Actual thinking and commands are in tool_calls, need to extract summary from there
        """
        import json as json_lib
        summaries = []

        for i, step in enumerate(first_steps, 1):
            role = step.get("role", "")

            if role == "assistant":
                # New format: extract command summary from tool_calls
                tool_calls = step.get("tool_calls", [])
                cmd_summary = ""

                for tc in tool_calls:
                    func = tc.get("function", {})
                    if func.get("name") == "bash":
                        try:
                            args_str = func.get("arguments", "{}")
                            args = json_lib.loads(args_str)
                            cmd = args.get("command", "")

                            # Extract key information: if Python code, take comments; otherwise take first 100 characters of command
                            if "python3" in cmd and "<<" in cmd:
                                # Extract first few lines of comments from heredoc
                                lines = cmd.split('\n')
                                comments = [l.strip() for l in lines if l.strip().startswith('#')]
                                cmd_summary = ' '.join(comments[:3])[:200]  # First 3 lines of comments
                            else:
                                cmd_summary = cmd[:100]  # Direct command
                        except (json_lib.JSONDecodeError, KeyError):
                            pass

                # Compatible with old format: if content has content, also extract
                content = step.get("content", "")
                if content:
                    cmd_summary = content[:200]

                if cmd_summary:
                    summaries.append(f"Step {i}: {cmd_summary}")

            elif role == "tool":
                # Extract key part of tool output (first 100 characters)
                output = step.get("content", "")
                if output:
                    output_snippet = output[:100].replace('\n', ' ')
                    summaries.append(f"  Output: {output_snippet}")

        return "\n\n".join(summaries[:10])  # Max 10 items (5 steps + 5 outputs)

    def _extract_code_blocks(self, trajectory: list[dict]) -> list[str]:
        """Extract all code blocks from trajectory

        Fixed: mini-swe-agent uses OpenAI tool_calls format
        Code is in the command field of tool_calls[0]["function"]["arguments"]
        Instead of in the ```python code block in content
        """
        import re
        import json as json_lib
        code_blocks = []

        for step in trajectory:
            role = step.get("role", "")
            if role == "assistant":
                # New format: extract from tool_calls
                tool_calls = step.get("tool_calls", [])
                for tc in tool_calls:
                    func = tc.get("function", {})
                    if func.get("name") == "bash":
                        try:
                            # arguments is JSON string: {"command": "python3 << 'EOF'\n...code...\nEOF"}
                            args_str = func.get("arguments", "{}")
                            args = json_lib.loads(args_str)
                            cmd = args.get("command", "")

                            # Extract Python code from heredoc
                            # Compatible with two formats:
                            #   With terminator: python3 << 'EOF'\n...code...\nEOF
                            #   Without terminator: python3 << 'EOF'\n...code... (to end of string)
                            heredoc_match = re.search(
                                r"python3\s*<<\s*['\"]?EOF['\"]?\s*\n(.*?)(?:\nEOF\s*$|\Z)",
                                cmd, re.DOTALL
                            )
                            if heredoc_match:
                                code_blocks.append(heredoc_match.group(1).rstrip())
                            # Also handle direct python3 -c '...' format
                            elif "python3 -c" in cmd:
                                code_match = re.search(r"python3\s+-c\s+['\"](.+?)['\"]", cmd, re.DOTALL)
                                if code_match:
                                    code_blocks.append(code_match.group(1))
                        except (json_lib.JSONDecodeError, KeyError, AttributeError):
                            continue

                # Compatible with old format: extract from ```python code blocks in content (just in case)
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
        """Call GLM to analyze success case (episodic memory only).

        Returns parsed JSON with exactly 3 keys:
          problem_type, verified_cot_trajectory, state_evaluation_metric
        """
        system_prompt = """\
# Role
You are an AI Core Evaluator specializing in "Episodic Trajectory Distillation". Your task is to analyze an agent's successful execution trajectory and distill it EXCLUSIVELY into abstract, structured Episodic Memory to serve as a high-quality few-shot Chain-of-Thought (CoT) example.

# Objective 1: The Filtering Gate
You must only process trajectories that successfully solved the problem. If the trajectory is a failure or incomplete, output EXACTLY AND ONLY: {"status": "discard"}

# Objective 2: Structural De-parameterization & Answer Stripping (CRITICAL)
1. Replace all specific numerical values, explicit coordinates, and raw entities with generic mathematical variables (e.g., Target Sum S, Plane Equation P, Bounding Box B). Do not leak specific numbers from the prompt.
2. ABSOLUTELY NO FINAL ANSWERS: You must permanently delete the final numeric or algebraic answer from the memory. We are saving the *method*, not the *solution*.

# Output Format (JSON)
If the trajectory is a valid success, output a standard JSON block inside ```json tags. The keys MUST exactly match the schema below. ALL generated content MUST be entirely in English.

```json
{
  "problem_type": "<Abstract mathematical or algorithmic description of the problem category (e.g., 'Optimization of bounded planar regions in 3D')>",
  "key_insight": "<What specific geometric/algebraic feature, theorem, or dimensionality reduction strategy simplified the problem and gave confidence?>",
  "tags": ["<tag1>", "<tag2>", "<tag3>"],
  "approach": [
    "<Step 1: Abstract algorithmic action, e.g., 'Substitute the linear constraint to eliminate variables'>",
    "<Step 2: Abstract algorithmic action, e.g., 'Project the system into a lower-dimensional plane'>",
    "<Step 3: Abstract algorithmic action, e.g., 'Calculate the bounded area using generic vertex coordinates'>"
  ]
}
```

# Execution Constraints
- The `tags` array should contain 2-4 lowercase snake_case strings representing the math domains (e.g., "number_theory", "combinatorics").
- The `approach` array should contain 3-4 highly abstracted, imperative action steps showing the flow of logic.
- Ensure absolutely no original numbers, target answers, or specific equations leak into the output."""

        user_prompt = f"""Your task: Distill the following successful trajectory into episodic memory JSON. Output ONLY the ```json block. ALL content must be in English.

<problem>
{problem[:400]}
</problem>

<trajectory>
{trajectory_summary[:800]}
</trajectory>

<code>
{code_blocks[:400]}
</code>"""

        try:
            llm_output = self._llm_call(
                system=system_prompt,
                user=user_prompt,
                temperature=0.0,
                max_tokens=500,
                extra_body={"thinking": {"type": "disabled"}},
            )

            if not llm_output:
                print(f"    [SuccessAnalyzer] ⚠️  GLM returned empty content", flush=True)
                return {"error": "empty_response"}

            print(f"    [SuccessAnalyzer] GLM raw output:\n{llm_output[:300]}...", flush=True)

            import json as json_parser

            # Try JSON first (```json block or bare JSON), then Markdown fallback
            parsed = None
            clean = llm_output.strip()
            m = re.search(r"```json\s*(.*?)\s*```", clean, re.DOTALL)
            if m:
                try:
                    parsed = json_parser.loads(m.group(1))
                except json_parser.JSONDecodeError:
                    pass
            if parsed is None:
                bare = re.sub(r"^```[a-z]*\n?", "", clean)
                bare = re.sub(r"\n?```$", "", bare.strip())
                if bare.startswith("{"):
                    try:
                        parsed = json_parser.loads(bare)
                    except json_parser.JSONDecodeError:
                        pass
            if parsed is None:
                # Markdown fallback: [Problem_Type]: / [Key_Insight]: / [Tags]: / [Approach]:
                parsed = self._parse_markdown(llm_output)
                if "error" in parsed:
                    parsed = None
            if parsed is None:
                print(f"    [SuccessAnalyzer] ✗ Both JSON and Markdown parse failed\nRaw: {llm_output[:200]}", flush=True)
                return {"error": "parse_failed"}

            # Handle discard signal
            if parsed.get("status") == "discard":
                print(f"    [SuccessAnalyzer] ⏭  Discarded by filtering gate", flush=True)
                return {"error": "discarded"}

            # Validate fields
            required = ["problem_type", "key_insight", "tags", "approach"]
            missing = [k for k in required if not parsed.get(k)]
            if missing:
                print(f"    [SuccessAnalyzer] ⚠️  Missing required fields: {missing}", flush=True)
                return {"error": "missing_required_fields"}

            # Ensure tags and approach are lists
            tags     = parsed["tags"] if isinstance(parsed["tags"], list) else [t.strip() for t in str(parsed["tags"]).split(",") if t.strip()]
            approach = parsed["approach"] if isinstance(parsed["approach"], list) else [parsed["approach"]]

            return {
                "problem_type": parsed["problem_type"],
                "key_insight":  parsed["key_insight"],
                "tags":         tags,
                "approach":     approach,
            }

        except Exception as exc:
            print(f"    [SuccessAnalyzer] ✗ GLM call failed: {exc}", flush=True)
            return {"error": str(exc)}

    def _parse_markdown(self, text: str) -> dict:
        """Parse Markdown key-value pair format (consistent with AnalyzerAgent._parse_markdown)

        Format:
        [Key_Name]: value
        [Another_Key]: another value
        """
        import re
        result = {}

        # Find the first [Key]:, start extraction from here (discard preceding thinking)
        first_bracket = text.find('[')
        if first_bracket > 0:
            text = text[first_bracket:]

        # Regular expression: support Key_Steps or Key-Steps or KeySteps
        pattern = re.compile(r"\[([A-Za-z_-]+)\]:\s*(.+?)(?=\n\[|\Z)", re.DOTALL | re.MULTILINE)
        for match in pattern.finditer(text):
            key_raw = match.group(1)
            # Unified conversion: Key_Steps -> key_steps, Key-Steps -> key_steps
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
        """Fallback: rule-based key step extraction (original logic)"""
        print(f"    [SuccessAnalyzer] Using rule-based extraction as fallback...", flush=True)

        # Extract code blocks
        code_blocks = self._extract_code_blocks(trajectory)
        all_code = "\n".join(code_blocks)

        if not code_blocks:
            return self._fallback_simple_summary(problem, answer)

        # Analyze code patterns
        techniques = self._identify_techniques(all_code, problem)

        if not techniques:
            return self._fallback_simple_summary(problem, answer)

        # Combine into final format
        key_steps = " | ".join(techniques[:5])
        core_insight = self._generate_insight(techniques, problem)
        tags = self._infer_tags(problem, all_code)

        result = {
            "key_steps": key_steps,
            "core_insight": core_insight,
            "tags": ", ".join(tags),
        }

        print(f"    [SuccessAnalyzer] ✓ Rule-based extraction completed", flush=True)
        return result

    def _identify_techniques(self, code: str, problem: str) -> list[str]:
        """Identify key techniques and steps from code"""
        techniques = []

        # Pattern 1: Number theory related
        if "%" in code or "mod" in code.lower():
            techniques.append("apply modular arithmetic")

        if "pow(" in code:
            # pow(a, b, m) represents modular exponentiation (three parameters)
            pow_idx = code.find("pow(")
            if pow_idx >= 0:
                snippet = code[pow_idx:pow_idx+50]
                if snippet.count(",") >= 2:
                    techniques.append("use modular exponentiation")

        # Pattern 2: Base conversion
        if "int(" in code and "," in code:
            techniques.append("convert between bases")

        # Pattern 3: Loop search
        if "for " in code and "range(" in code:
            techniques.append("iterate through candidates")

        # Pattern 4: Divisibility check
        if "==" in code and "0" in code:
            techniques.append("check divisibility")

        # Pattern 5: Summation/counting
        if "sum(" in code:
            techniques.append("aggregate results")
        elif ".append(" in code:
            techniques.append("collect valid solutions")

        # Pattern 6: Combinatorics
        if "factorial" in code or "comb" in code:
            techniques.append("compute combinatorial terms")

        # Pattern 7: Math library functions
        if "math." in code or "from math import" in code:
            techniques.append("use math library functions")

        # If no techniques are identified, give a general one
        if not techniques:
            techniques.append("implement direct computation")

        return techniques

    def _generate_insight(self, techniques: list[str], problem: str) -> str:
        """Generate core insight based on identified techniques"""
        # Simple template: generate based on the first technique
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
        """Infer tags from problem and code"""
        tags = []

        # Infer from problem text
        problem_lower = problem.lower()
        if any(kw in problem_lower for kw in ["base", "divisor", "modulo", "remainder"]):
            tags.append("number_theory")
        if any(kw in problem_lower for kw in ["combinations", "permutations", "choose"]):
            tags.append("combinatorics")
        if any(kw in problem_lower for kw in ["probability", "expected"]):
            tags.append("probability")
        if any(kw in problem_lower for kw in ["sequence", "series", "recursive"]):
            tags.append("sequences")

        # Infer from code
        if "%" in code or "mod" in code:
            if "number_theory" not in tags:
                tags.append("modular_arithmetic")

        # Default tags
        if not tags:
            tags.append("general")

        return tags

    def _infer_tags_from_problem_type(self, problem_type: str) -> list[str]:
        """Infer tags from problem_type keywords"""
        tags = []
        text = problem_type.lower()

        tag_keywords = {
            "number_theory":      ["divisor", "modulo", "remainder", "prime", "gcd", "lcm", "base representation"],
            "combinatorics":      ["combination", "permutation", "choose", "counting", "subset", "arrangement"],
            "probability":        ["probability", "expected", "random", "distribution"],
            "geometry":           ["polygon", "area", "coordinate", "plane", "angle", "triangle", "circle", "projection"],
            "algebra":            ["equation", "inequality", "polynomial", "factorize", "solve for"],
            "sequences":          ["sequence", "series", "recursive", "recurrence", "arithmetic", "geometric"],
            "optimization":       ["minimize", "maximize", "optimal", "constraint", "linear programming"],
            "dimensionality":     ["dimensionality reduction", "substitution", "projection", "reduce"],
        }

        for tag, keywords in tag_keywords.items():
            if any(kw in text for kw in keywords):
                tags.append(tag)

        return tags[:3]  # Cap at 3 tags to avoid noise

    def _fallback_simple_summary(self, problem: str, answer: str) -> dict:
        """Simple fallback summary (does not depend on code analysis)"""
        print(f"    [SuccessAnalyzer] Using simple fallback summary", flush=True)

        return {
            "key_steps": "analyze problem | compute solution | verify answer",
            "core_insight": "systematic problem solving",
            "tags": ", ".join(self._infer_tags(problem, "")),
        }

    def _parse_json(self, text: str) -> dict:
        """Parse Markdown key-value pair format

        Compatible with Zhipu GLM's reasoning mode:
        1. First output thinking process ("The user wants...")
        2. Then output final answer ([Key]: Value)

        We only extract the final answer part.

        If format is completely not found, try fallback simple extraction.
        """
        result = {}

        # Zhipu reasoning mode: find the first [Key]:, start extraction from here
        first_bracket = text.find('[')
        if first_bracket > 0:
            # Extract content starting from the first [, discard preceding thinking
            text = text[first_bracket:]
        elif first_bracket < 0:
            # Completely cannot find [, means LLM did not output in format
            print(f"    [_parse_json] ⚠️  Cannot find [Key]: format in response, trying fallback", flush=True)
            return self._fallback_parse(text)

        # Improved regular expression:
        # 1. Support Key_Steps or Key-Steps or KeySteps
        # 2. Value part matches to next [Key] or end of string
        # 3. Remove trailing whitespace from value
        pattern = re.compile(r"\[([A-Za-z_-]+)\]:\s*(.+?)(?=\n\[|\Z)", re.DOTALL | re.MULTILINE)
        for match in pattern.finditer(text):
            key_raw = match.group(1)
            # Unified conversion: Key_Steps -> key_steps, Key-Steps -> key_steps
            key = key_raw.lower().replace('-', '_')
            value = match.group(2).strip()
            result[key] = value

        # Print debug information
        if result:
            print(f"    [_parse_json] Successfully parsed {len(result)} fields: {list(result.keys())}", flush=True)
        else:
            print(f"    [_parse_json] Regex parsing failed, trying fallback", flush=True)
            print(f"    [_parse_json] Text fragment:\n{text[:500]}", flush=True)
            return self._fallback_parse(text)

        return result if result else {"error": "failed to parse", "raw": text[:300]}

    def _fallback_parse(self, text: str) -> dict:
        """Fallback: simple extraction when LLM does not output in format at all

        Try to extract key information from natural language:
        - Look for keywords like "steps:", "key steps:", "reasoning:"
        - Extract problem tags (inferred from problem text)
        """
        print(f"    [_fallback_parse] Starting fallback parsing", flush=True)

        # Strategy: give up directly, return error
        # Because if LLM does not follow format, extracted information quality is also unreliable
        return {
            "error": "LLM did not follow format",
            "raw": text[:300],
            "suggestion": "Consider using a stronger model or adjusting temperature to 0"
        }


