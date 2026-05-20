import json
import re
from typing import List, Optional, Union

from generators.generator_types import Generator
from generators.model import Message, ModelBase
from utils import print_v
from rich.panel import Panel


# 第一次调用llm让它写代码
DS1000_SIMPLE_CHAT_INSTRUCTION = """You are a Python data science coding assistant.
You will be given a DS1000 problem whose prompt ends with `BEGIN SOLUTION` and an opening `<code>` tag.
Complete only the missing Python code snippet. The snippet must assign the final answer to a variable named `result`.
Return only Python code, without explanations."""



# Original free-text reflection prompt, kept for comparison/rollback:
# DS1000_SELF_REFLECTION_CHAT_INSTRUCTION = """You are a Python data science coding assistant.
# You will be given a failed DS1000 code snippet and test feedback.
# Briefly explain the likely mistake and how to fix it. Do not write code."""

# 写错后诊断阶段，让它分析错误并总结成结构化的json
DS1000_SELF_REFLECTION_CHAT_INSTRUCTION = """You are a Python data science coding assistant.
You will be given a library hint, a failed DS1000 code snippet, and compact test feedback.
Diagnose the failure for future memory retrieval.
Return exactly one valid JSON object and nothing else. Do not use markdown.
The JSON object must have exactly these string fields:
{
  "error_type": "short error category, e.g. shape mismatch, index alignment error, dtype error, missing result assignment",
  "trigger_condition": "specific condition that triggered the failure",
  "repair_action": "concrete reusable repair action",
  "library": "main Python/data-science library involved, e.g. pandas, numpy, scipy, sklearn, matplotlib, unknown"
}"""



# 有了上一步诊断报告后系统再次尝试解决问题
DS1000_REFLEXION_CHAT_INSTRUCTION = """You are a Python data science coding assistant.
You will be given a DS1000 problem, your previous code snippet, test feedback, and a JSON reflection.
Write an improved Python code snippet that assigns the final answer to `result`.
Return only Python code, without explanations."""




DS1000_SIMPLE_COMPLETION_INSTRUCTION = """Complete the DS1000 Python code snippet.
The snippet must assign the final answer to a variable named `result`.
Return only Python code.

Problem:
"""

# Original free-text completion reflection prompt, kept for comparison/rollback:
# DS1000_SELF_REFLECTION_COMPLETION_INSTRUCTION = """Briefly explain why this DS1000 Python code snippet failed and how to fix it.
# Do not write code.
# """
DS1000_SELF_REFLECTION_COMPLETION_INSTRUCTION = """Diagnose this failed DS1000 Python code snippet for future memory retrieval.
Return exactly one valid JSON object and nothing else. Do not use markdown.
The JSON object must have exactly these string fields: error_type, trigger_condition, repair_action, library.
"""


DS1000_REFLEXION_COMPLETION_INSTRUCTION = """Improve the DS1000 Python code snippet using the feedback and JSON reflection.
The snippet must assign the final answer to a variable named `result`.
Return only Python code.
"""




class DS1000Generator(Generator):
    def self_reflection(
        self,
        func: str,
        feedback: str,
        model: ModelBase,
        problem_prompt: Optional[str] = None,
        library: Optional[str] = None,
    ) -> str:
        from generators.generator_utils import print_messages
        compact_feedback = compact_ds1000_feedback(feedback)
        if model.is_chat:
            system_msg = DS1000_SELF_REFLECTION_CHAT_INSTRUCTION
            user_msg = (
                f"[library hint]:\n{library or 'unknown'}\n\n"
                # Original full-problem context, kept for comparison/rollback:
                # f"[problem]:\n{problem_prompt or ''}\n\n"
                f"[previous code]:\n```python\n{func}\n```\n\n"
                f"[test feedback]:\n{compact_feedback}\n\n"
                "[json reflection]:"
            )
            print_messages(system_msg, user_msg)
            output = model.generate_chat(
                messages=[
                    Message(role="system", content=system_msg),
                    Message(role="user", content=user_msg),
                ],
                max_tokens=512,
                temperature=0.0,
            )
        else:
            output = model.generate(
                f"{DS1000_SELF_REFLECTION_COMPLETION_INSTRUCTION}\n"
                f"[library hint]:\n{library or 'unknown'}\n\n"
                # Original full-problem context, kept for comparison/rollback:
                # f"[problem]:\n{problem_prompt or ''}\n\n"
                f"[previous code]:\n{func}\n\n[test feedback]:\n{compact_feedback}\n\n[json reflection]:",
                max_tokens=512,
                temperature=0.0,
            )
        assert isinstance(output, str)
        reflection = parse_ds1000_reflection_json(output, library)
        print_v(Panel(reflection, title="Self reflection output", border_style="yellow"))
        return reflection

    def func_impl(
        self,
        func_sig: str,
        model: ModelBase,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.0,
    ) -> Union[str, List[str]]:
        if strategy not in ("simple", "reflexion"):
            raise ValueError(f"Invalid strategy: {strategy}")
        if strategy == "reflexion" and (
            prev_func_impl is None or feedback is None or self_reflection is None
        ):
            raise ValueError("DS1000 reflexion requires previous code, feedback, and reflection")

        from generators.generator_utils import print_messages, print_generated_func_body
        compact_feedback = compact_ds1000_feedback(feedback) if feedback is not None else None

        if model.is_chat:
            if strategy == "simple":
                sys_content = DS1000_SIMPLE_CHAT_INSTRUCTION
                user_content = func_sig
                messages = [
                    Message(role="system", content=sys_content),
                    Message(role="user", content=user_content),
                ]
                print_messages(sys_content, user_content)
            else:
                sys_content = DS1000_REFLEXION_CHAT_INSTRUCTION
                user_content = (
                    f"[problem]:\n{func_sig}\n\n"
                    f"[previous code]:\n```python\n{prev_func_impl}\n```\n\n"
                    # Original full feedback, kept for comparison/rollback:
                    # f"[test feedback]:\n{feedback}\n\n"
                    f"[test feedback]:\n{compact_feedback}\n\n"
                    f"[reflection]:\n{self_reflection}\n\n"
                    "[improved code]:"
                )
                messages = [
                    Message(role="system", content=sys_content),
                    Message(role="user", content=user_content),
                ]
                print_messages(sys_content, user_content)
            output = model.generate_chat(
                messages=messages,
                num_comps=num_comps,
                temperature=temperature,
                max_tokens=1024,
            )
        else:
            if strategy == "simple":
                prompt = f"{DS1000_SIMPLE_COMPLETION_INSTRUCTION}{func_sig}\n\nCode:"
            else:
                prompt = (
                    f"{DS1000_REFLEXION_COMPLETION_INSTRUCTION}\n"
                    f"[problem]:\n{func_sig}\n\n"
                    f"[previous code]:\n{prev_func_impl}\n\n"
                    # Original full feedback, kept for comparison/rollback:
                    # f"[test feedback]:\n{feedback}\n\n"
                    f"[test feedback]:\n{compact_feedback}\n\n"
                    f"[reflection]:\n{self_reflection}\n\n"
                    "[improved code]:"
                )
            print_v(Panel(prompt, title="Completion Prompt", border_style="magenta")) 
            output = model.generate(prompt, num_comps=num_comps, temperature=temperature)

        if num_comps == 1:
            assert isinstance(output, str)
            print_generated_func_body(output)
            return parse_ds1000_code(output)

        assert isinstance(output, list)
        print_generated_func_body("\n\n".join(output))
        return [parse_ds1000_code(item) for item in output]

    def internal_tests(
        self,
        func_sig: str,
        model: ModelBase,
        max_num_tests: int = 5,
    ) -> List[str]:
        return []


def parse_ds1000_code(output: str) -> str:
    fenced = re.search(r"```(?:python)?\n(.*?)\n```", output, re.DOTALL)
    if fenced:
        output = fenced.group(1)

    output = output.strip()
    output = re.sub(r"^<code>\s*", "", output)
    output = re.sub(r"\s*</code>$", "", output)
    output = output.replace("BEGIN SOLUTION", "").strip()
    return output


def parse_ds1000_reflection_json(output: str, library: Optional[str] = None) -> str:
    output = output.strip()
    fenced = re.search(r"```(?:json)?\n(.*?)\n```", output, re.DOTALL)
    if fenced:
        output = fenced.group(1).strip()

    json_match = re.search(r"\{.*\}", output, re.DOTALL)
    if json_match:
        output = json_match.group(0)

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = {
            "error_type": "unstructured_reflection",
            "trigger_condition": output[:500],
            "repair_action": "Review the raw reflection and test feedback, then produce a targeted code fix.",
            "library": library or "unknown",
        }

    normalized = {
        "error_type": str(parsed.get("error_type", "unknown")).strip() or "unknown",
        "trigger_condition": str(parsed.get("trigger_condition", "unknown")).strip() or "unknown",
        "repair_action": str(parsed.get("repair_action", "unknown")).strip() or "unknown",
        "library": str(parsed.get("library", library or "unknown")).strip() or (library or "unknown"),
    }
    return json.dumps(normalized, ensure_ascii=False)


def compact_ds1000_feedback(feedback: str) -> str:
    lines = []
    for line in feedback.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Traceback"):
            break
        if stripped.startswith("File "):
            continue
        if stripped.startswith("self.ret =") or stripped.startswith("thread.join"):
            continue
        lines.append(stripped)

    if not lines:
        return feedback.strip()

    return "\n".join(lines)
