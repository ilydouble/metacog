import json
import os
import re
import textwrap
import ast
from typing import List, Optional, Union

from generators.generator_types import Generator
from generators.model import Message, ModelBase
from utils import print_v
from rich.panel import Panel


DS1000_CODE_MAX_TOKENS = int(os.getenv("DS1000_CODE_MAX_TOKENS", "1024"))
DS1000_REFLECTION_MAX_TOKENS = int(os.getenv("DS1000_REFLECTION_MAX_TOKENS", "1024"))


# 第一次调用llm让它写代码
# DS1000_SIMPLE_CHAT_INSTRUCTION = """You are a Python data science coding assistant.
# You will be given a DS1000 problem whose prompt ends with `BEGIN SOLUTION` and an opening `<code>` tag.
# Complete only the missing Python code snippet. The snippet must assign the final answer to a variable named `result`.
# If the problem asks you to complete the body of an existing Python function, output only the missing function body.
# For function-body completions, every non-blank line must be indented by exactly four spaces, assign the answer to `result`, then return it with `return result`.
# For top-level snippet completions, assign the final answer to `result` and do not use `return`.
# If the problem description includes explicit input-output examples (e.g., `test_data`), your code must logically transform the input into EXACTLY the expected output.
# If no explicit expected output is provided, carefully infer the correct behavior from the problem description and any partial examples given.
# IMPORTANT: Never hardcode expected output values directly into your solution. Your code must implement the correct general logic that works for any valid input, not just the example shown.
# Do not repeat imports, example data, pd.DataFrame(...), load_data(), or existing variable initialization.
# Return only Python code, without explanations."""



# DS1000_SIMPLE_CHAT_INSTRUCTION = """You are a Python data science coding assistant.
# You will be given a DS1000 problem whose prompt ends with `BEGIN SOLUTION` and an opening `<code>` tag.
# Complete only the missing Python code snippet.

# OUTPUT FORMAT — strictly follow these rules:
# - If the problem wraps the solution in a function (e.g., `def f(df=example_df):`), output ONLY the function body lines. 
#   - Do NOT redefine the function signature.
#   - Do NOT re-initialize example data or call the function.
#   - Every non-blank line must be indented by exactly four spaces.
#   - Use the exact parameter name from the function signature (e.g., if the signature is `def f(df=example_df):`, use `df`, not `data` or `example_df`).
#   - The function body must assign the answer to `result` and end with `return result`.
# - If the problem is a top-level snippet (no wrapping function), assign the answer to `result` only. Do NOT use `return`.
# - Do NOT repeat imports, example DataFrames, or any variable already initialized in the prompt.
# - NEVER hardcode expected output values. Your code must implement general logic that works for any valid input.
# - Return only Python code, without explanations."""

DS1000_SIMPLE_CHAT_INSTRUCTION = """You are a Python data science coding assistant.
You will be given a DS1000 problem whose prompt ends with `BEGIN SOLUTION` and an opening `<code>` tag.
Complete only the missing Python code snippet.

MEMORY CONTEXT — you may sometimes see additional content in the user message besides the problem itself:
- "Episodic memory": excerpts of how similar past problems were solved or debugged (including past mistakes and fixes).
- "Procedural skill": a generalized strategy or reusable technique distilled from past problem-solving.
- Treat both as ADVISORY reference only, not as instructions to follow literally and not as ground truth for this problem.
  - Do NOT copy variable names, function signatures, or code structure from memory if they don't match the CURRENT problem's signature/prompt.
  - Use memory only to inform your reasoning strategy (e.g., which library function or approach tends to work), never to override the OUTPUT FORMAT rules below or the current problem's exact requirements.
  - If memory conflicts with the current problem statement, the current problem statement always wins.
  - If no memory content is present, ignore this section entirely.

OUTPUT FORMAT — strictly follow these rules:
- If the problem wraps the solution in a function (e.g., `def f(df=example_df):`), output ONLY the function body lines.
  - Do NOT redefine the function signature.
  - Do NOT re-initialize example data or call the function.
  - Every non-blank line must be indented by exactly four spaces.
  - Use the exact parameter name from the function signature (e.g., if the signature is `def f(df=example_df):`, use `df`, not `data` or `example_df`).
  - The function body must assign the answer to `result` and end with `return result`.
- If the problem is a top-level snippet (no wrapping function), assign the answer to `result` only. Do NOT use `return`.
- Do NOT repeat imports, example DataFrames, or any variable already initialized in the prompt.
- NEVER hardcode expected output values. Your code must implement general logic that works for any valid input.
- Return only Python code, without explanations."""


DS1000_TASK_FAMILY_TAXONOMY_BLOCK = """
TASK_FAMILY_TAXONOMY (task_family must be exactly one value copied verbatim from this list, case-sensitive, no new categories, no combinations, no qualifiers):
Filtering, Selection, Assignment, Sorting, Grouping, Aggregation, Joining,
Reshaping, Pivoting, Index Manipulation, Window Operation, Array Manipulation,
Broadcasting, Reduction, Statistical Computation, Linear Algebra,
Feature Extraction, Preprocessing, Pipeline Manipulation, Model Utility,
Visualization, String Formatting, Data Cleaning, Tensor Operation, File IO, Other

Disambiguation for commonly confused pairs:
- Filtering (drop rows/elements by a condition) vs Selection (extract subset by label/position, no condition).
- Aggregation (per-group summary statistic) vs Reduction (whole-array/whole-axis collapse, no grouping).
- Grouping (partition only, no summary yet) vs Aggregation (partition + summarize) — if both occur, choose Aggregation.
- Reshaping (layout change, e.g. melt/stack/transpose, values unchanged) vs Pivoting (long-to-wide or wide-to-long using index/columns/values).
- Feature Extraction (derive new columns/features from raw data) vs Preprocessing (scale/encode/impute/normalize before modeling).
- Array Manipulation (numpy-level shape/dtype/slicing) vs Tensor Operation (torch/tf-specific operations).
- Assignment (write/overwrite values conditionally or via mapping) vs Filtering (only removes/keeps, does not overwrite values).

Choose the category matching the DOMINANT operation being verified (the final transformation that determines pass/fail), not a secondary or setup step.
If uncertain, or no category clearly applies, output "Other".
"""

DS1000_ERROR_TYPE_TAXONOMY_BLOCK = """
ERROR_TYPE_TAXONOMY (error_type must be exactly one value copied verbatim from this list):

Wrong API Usage
Missing Parameter
Wrong Parameter Value
Axis Confusion
Shape Mismatch
Index Alignment
MultiIndex Mismatch
Wrong Return Object
Missing Return Statement
Wrong Output Type
Column Loss
Grouping Semantics
Aggregation Semantics
Broadcasting Error
Data Leakage
Deprecated API
Version Compatibility
Visualization Semantics
Environment Dependency
Numerical Precision
Logic Error
Other

Disambiguation:
- Wrong API Usage: wrong function chosen entirely.
- Missing Parameter: correct API but omitted critical argument.
- Wrong Parameter Value: parameter exists but value is incorrect.
- Axis Confusion: wrong axis/dim specification.
- Shape Mismatch: output dimensions differ from expected.
- Index Alignment: index alignment behavior misunderstood.
- MultiIndex Mismatch: wrong number/order/names of index levels.
- Wrong Return Object: returned estimator/result/container instead of desired value.
- Missing Return Statement: function computes result but does not return it.
- Wrong Output Type: correct content but wrong type (Series vs DataFrame, ndarray vs scalar).
- Column Loss: required columns removed accidentally.
- Grouping Semantics: misunderstanding of groupby behavior.
- Aggregation Semantics: aggregation result differs from expected statistic.
- Broadcasting Error: array broadcasting misunderstood.
- Data Leakage: target/test information improperly used.
- Deprecated API: obsolete API should be replaced.
- Version Compatibility: API behavior changed across versions.
- Visualization Semantics: wrong plotting object/plot behavior expected.
- Environment Dependency: backend/platform/runtime assumption caused failure.
- Numerical Precision: tolerance/rounding/float comparison issue.
- Logic Error: algorithmic reasoning mistake not covered above.

Choose the SINGLE dominant failure mode.
If uncertain, output "Other".
"""


# DS1000_SELF_REFLECTION_CHAT_INSTRUCTION = """You are a Python data science coding assistant. Diagnose why a DS1000 code snippet failed and how to fix it.
# You will be given a library hint, problem, failed code, and test feedback.

# Write a CONCISE diagnosis (max 2-3 sentences per field). This text will guide the next code generation attempt — be specific and actionable.

# Return exactly one valid JSON object and nothing else. Do not use markdown. Do not use code fences.

# {
#   "library": "main library, e.g. pandas, numpy, scipy, sklearn",
#   "task_family": "exactly one category from TASK_FAMILY_TAXONOMY below",
#   "task_intent": "what the code was trying to accomplish — the specific data transformation goal, 1 sentence",
#   "error_type": "exactly one category from ERROR_TYPE_TAXONOMY below",
#   "api_concepts": "1-3 key APIs involved, e.g. groupby.apply, boolean mask, merge",
#   "evidence_from_feedback": "Expected output contains NaN group, actual output does not.",
#   "root_cause": "the specific logic error that caused the failure — reference actual vs expected from feedback",
#   "failure_signature": "the key failure pattern that could repeat in similar problems — a compact fingerprint of this mistake, 1 sentence",
#   "failed_assumption": "the incorrect assumption the failed code made",
#   "repair_action": "exactly what the next code must do differently — name specific API parameters (e.g. drop=False, axis=1, level=0) that are critical to the fix, 1-2 sentences",
#   "preserve_constraints": "what must remain unchanged (output shape, columns, types, ordering)",
#   "confidence": "low, medium, or high"
# }

# """ + DS1000_TASK_FAMILY_TAXONOMY_BLOCK + """
# """ + DS1000_ERROR_TYPE_TAXONOMY_BLOCK + """

# Important:
# 1. task_intent must describe the concrete data transformation goal, not the task_family label.
# Good:
# - "Group rows by 'id' and compute the mean of each numeric column within each group."
# - "Rotate x-axis tick labels by 45 degrees counterclockwise."
# Bad:
# - "Grouping"
# - "Visualization"
# 2. failure_signature must be a compact fingerprint of this mistake that helps retrieve similar past failures.
# Good:
# - "groupby drops NaN keys by default, losing rows in output."
# - "plt.xticks rotation direction is clockwise for positive values."
# Bad:
# - "AssertionError"
# - "Wrong output"
# 3. evidence_from_feedback must quote the most specific evidence available from the test feedback.
# Good:
# - actual_result shape=(6,1)
# - Expected index levels=2, actual levels=3
# - Returned type OptimizeResult instead of ndarray
# Bad:
# - AssertionError
# - Tests failed
# - Output mismatch
# 4. root_cause must reference the specific mismatch from test feedback (actual vs expected).
# 5. repair_action must be a concrete instruction the code gen model can directly follow.
# 6. Do NOT write a memory card. Do NOT mention future retrieval or reusability.
# 7. Each field max 2-3 sentences. Brevity is critical — the code gen model needs clear guidance.
# 8. task_family must come from the fixed taxonomy above.
# 9. error_type must come from the fixed taxonomy above

# """


# 写错后诊断阶段，让它分析错误并总结成结构化的json
# DS1000_SELF_REFLECTION_CHAT_INSTRUCTION = """You are a Python data science coding assistant.
# You will be given a library hint, a DS1000 problem, a failed DS1000 code snippet, and test feedback.
# Diagnose the deepest root cause of the failure.

# Requirements:
# - Focus on the specific logic error that directly causes the test failure.
# - Ignore stylistic issues or implementation choices unless they are responsible for the failure.
# - The repair_action should be sufficient to make the code pass if applied correctly.
# - Before outputting, check whether the proposed repair would likely resolve the observed failure.
# - If the repair would not resolve the failure, continue searching for a deeper root cause.

# Return exactly one JSON object:

# {
#   "error_type": "...",
#   "root_cause": "...",
#   "failed_assumption": "...",
#   "repair_action": "...",
#   "library": "..."
# }"""


# DS1000_SELF_REFLECTION_CHAT_INSTRUCTION = """You are a Python data science coding assistant.
# You will be given a failed DS1000 code snippet and test feedback.
# Briefly explain the likely mistake and how to fix it. Do not write code."""


DS1000_SELF_REFLECTION_CHAT_INSTRUCTION = """You are a Python data science debugging assistant. You are given a problem, the code that just failed, and test feedback.

Your goal is to produce a direct, executable repair instruction for the NEXT code generation attempt. Do not explain background, do not classify the task type, do not summarize lessons learned. Just answer: what went wrong, and what must change next.

Return exactly one valid JSON object and nothing else. Do not use markdown. Do not use code fences.

{
  "root_cause": "the specific mismatch between actual and expected from the test feedback, 1 sentence",
  "repair_action": "the exact change the next code must make, naming specific APIs/parameters (e.g. reset_index(drop=False), axis=1), 1-2 sentences",
  "preserve_constraints": "what must NOT be broken by the fix (output shape, columns, types, ordering), 1 sentence",
  "confidence": "low, medium, or high"
}

Requirements:
1. root_cause must cite concrete evidence from the feedback (e.g. shape=(5,2) vs (5,3)), not just "test failed".
2. repair_action must be a directly actionable instruction, not a diagnostic description.
3. Keep the entire response under 4 sentences total. Brevity is critical — the next code-gen call only needs this to act, not to understand the full history.
"""



# 有了上一步诊断报告后系统再次尝试解决问题
# DS1000_REFLEXION_CHAT_INSTRUCTION = """You are a Python data science coding assistant.
# You will be given a DS1000 problem, your previous code snippet, test feedback, and a reflection.
# Write an improved Python code snippet based on the reflection.

# OUTPUT FORMAT — strictly follow these rules:
# - If the problem wraps the solution in a function (e.g., `def f(df=example_df):`), output ONLY the function body lines. 
#   - Do NOT redefine the function signature.
#   - Do NOT re-initialize example data or call the function.
#   - Every non-blank line must be indented by exactly four spaces.
#   - Use the exact parameter name from the function signature (e.g., if the signature is `def f(df=example_df):`, use `df`, not `data` or `example_df`).
#   - The function body must assign the answer to `result` and end with `return result`.
# - If the problem is a top-level snippet (no wrapping function), assign the answer to `result` only. Do NOT use `return`.
# - Do NOT repeat imports, example DataFrames, or any variable already initialized in the prompt.
# - NEVER hardcode expected output values. Your code must implement general logic that works for any valid input.
# - Return only Python code, without explanations."""

DS1000_REFLEXION_CHAT_INSTRUCTION = """You are a Python data science coding assistant.
You will be given a DS1000 problem, your previous code snippet for THIS problem, test feedback, and a reflection on THIS problem's failure.
Write an improved Python code snippet based on the reflection.

MEMORY CONTEXT — you may sometimes see additional content in the user message besides the above:
- "Episodic memory": excerpts of how DIFFERENT, past problems were solved or debugged.
- "Procedural skill": a generalized strategy or reusable technique distilled from past problem-solving across problems.
- These are ADVISORY reference only — distinct from your own previous code/test feedback/reflection for THIS problem, which you should treat as authoritative and directly actionable.
  - Do NOT copy variable names, function signatures, or code structure from memory if they don't match the CURRENT problem's signature/prompt.
  - Use memory only to inform strategy (e.g., a library function or pattern that worked before), never to override the OUTPUT FORMAT rules below, the current problem's requirements, or your own reflection for this problem.
  - If memory conflicts with the current problem or your own reflection, the current problem and your own reflection always win.
  - If no memory content is present, ignore this section entirely.

OUTPUT FORMAT — strictly follow these rules:
- If the problem wraps the solution in a function (e.g., `def f(df=example_df):`), output ONLY the function body lines.
  - Do NOT redefine the function signature.
  - Do NOT re-initialize example data or call the function.
  - Every non-blank line must be indented by exactly four spaces.
  - Use the exact parameter name from the function signature (e.g., if the signature is `def f(df=example_df):`, use `df`, not `data` or `example_df`).
  - The function body must assign the answer to `result` and end with `return result`.
- If the problem is a top-level snippet (no wrapping function), assign the answer to `result` only. Do NOT use `return`.
- Do NOT repeat imports, example DataFrames, or any variable already initialized in the prompt.
- NEVER hardcode expected output values. Your code must implement general logic that works for any valid input.
- Return only Python code, without explanations."""


DS1000_EPISODIC_MEMORY_CHAT_INSTRUCTION = """You are building an episodic memory entry for a Reflexion-style framework. You are given a problem, the failed code, test feedback, and an ALREADY-DETERMINED repair_action from a prior step.

Do NOT re-diagnose the failure. Only add classification labels and a distilled retrieval-oriented summary on top of the existing diagnosis.

Known diagnosis (treat as ground truth, do not contradict):
repair_action: {repair_action_from_stage_A}

Your main task is DISTILLATION: repair_action above is problem-specific (it names exact APIs/parameters/column names). You must abstract it into a generalized principle that could apply to a DIFFERENT problem with a similar underlying mistake, even if that problem uses completely different APIs or data.

Distillation rules — both injection_card and failed_assumption must:
1. NOT contain any specific API name, method name, parameter name/value, column name, variable name, or library-specific call (e.g. write "the operation that changes index alignment" instead of "reset_index(drop=False)"; write "the axis controlling row-vs-column direction" instead of "axis=1").
2. Describe the CATEGORY of mistake and the CATEGORY of fix, not the literal fix. Ask yourself: "if someone hits a different problem with the same underlying misconception, would this sentence still make sense and still help, without me having named today's specific API or column?"
2.5. Preserve the STRUCTURAL TYPE mentioned in repair_action — do not swap it for a different structural concept during abstraction. If repair_action is about the index, keep the abstraction about the index (not "a column"); if about column order, keep it about column order (not row order); if about dtype, keep it about dtype (not shape); if about value content, keep it about value content (not structure). Abstracting means removing the specific NAME (e.g. 'cokey', 'reset_index()'), not changing WHICH structural concept (index vs column vs row order vs dtype vs shape vs value content) the fix operates on.
3. Be phrased as a transferable rule of thumb, not as an instruction tied to today's code (avoid "the next code must call X"; prefer "when Y is the goal, Z is often the thing that gets missed").
4. injection_card must stay under 20 words. failed_assumption should be one concise sentence.
5. If you genuinely cannot abstract the repair_action without losing all meaning (rare — e.g. the mistake IS an API-specific gotcha with no generalizable pattern), output "NONE" for injection_card instead of forcing a bad abstraction.

Example of correct abstraction (structural type + names both handled correctly):
repair_action: "Reindex columns to [...] and call reset_index(drop=True) after dropna()."
Good injection_card: "When reshaping changes column order or leaves gaps in the index, explicitly reorder columns and reset the index."
Good failed_assumption: "Assumed reshaping operations preserve the desired column order and produce a clean sequential index after row removal."
Bad injection_card: "After dropping rows, reindex columns to ['user', 'date', 'value'] and call reset_index."
Bad failed_assumption: "Assumed dropna() keeps the column order and index clean."
(Bad versions fail because they keep literal column names — not because the structural concepts are wrong.)

Example of structural type WRONGLY swapped during abstraction (avoid this):
repair_action: "Change group_keys to True so the result retains the MultiIndex with the original index preserved."
Bad injection_card: "The original ordering info may need to be explicitly re-attached as a column."
(WRONG — repair_action is about preserving the INDEX, but this rewrites it as adding a COLUMN. These are different structural concepts; this abstraction silently changed the meaning, not just removed a name.)
Good injection_card: "When grouping may collapse or drop the original index, check whether it needs to be explicitly preserved rather than reset."

Self-check before finalizing injection_card AND failed_assumption:
1. Scan both for any token that looks like a function/method call (contains "()", ".", or matches a known pandas/numpy/sklearn API name), an exact column/variable name, or a literal parameter value. If found, rewrite until none remain in either field.
2. Identify which structural concept(s) repair_action operates on (index / column / row order / dtype / shape / value content). Confirm your abstracted sentence still refers to the SAME structural concept(s), just without the specific names. If your abstraction silently switched the structural concept, rewrite it to correct this.

Return exactly one valid JSON object and nothing else. Do not use markdown. Do not use code fences.
{{
  "task_family": "exactly one category copied verbatim from TASK_FAMILY_TAXONOMY below",
  "error_type": "exactly one category copied verbatim from ERROR_TYPE_TAXONOMY below",
  "failed_assumption": "the incorrect assumption the failed code made, phrased as a general misconception per the Distillation rules above, 1 sentence",
  "retrieval_key": "short semantic description for embedding retrieval, 5-10 words",
  "injection_card": "the distilled, generalized principle per the Distillation rules above, max ~20 words, or 'NONE' if not abstractable"
}}

""" + DS1000_TASK_FAMILY_TAXONOMY_BLOCK + """
""" + DS1000_ERROR_TYPE_TAXONOMY_BLOCK + """

Requirements:
1. Do NOT alter the substance of repair_action — it is already final from the prior step; it is given only as context for your distillation, not as an output field.
2. task_family and error_type must come from the fixed taxonomies above, copied verbatim.
3. If uncertain on task_family or error_type, output "Other".
"""



DS1000_SKILL_EXTRACTION_CHAT_INSTRUCTION = """
You are a Python data science coding assistant responsible for extracting reusable procedural skills from DS1000 repair trajectories.
You will be given:

- A DS1000 problem
- The failed code
- A structured reflection
- The successful patch
- The patch diff
- Success verification

Your goal is NOT to summarize this repair.
Your goal is to extract a reusable procedural skill that can generalize to future DS1000 problems.

Return exactly one valid JSON object and nothing else.
Do NOT use markdown.
Do NOT use code fences.

The JSON object must contain exactly these string fields:
{
  "task_family": "exactly one category copied verbatim from TASK_FAMILY_TAXONOMY below",
  "error_type": "exactly one category from ERROR_TYPE_TAXONOMY below",
  "retrieval_key": "short semantic description for memory retrieval, 5-10 words",
  "api_concepts": "comma-separated APIs or concepts involved, without example-specific values",
  "procedure": "numbered algorithm-level reusable procedure describing semantic operations instead of API-specific edits",
  "avoid_assumption": "general incorrect assumption that this procedure helps avoid",
  "applicability_conditions": "high-level situations where this skill should be retrieved",
  "non_applicability_conditions": "situations where this procedure should NOT be applied",
  "preserve_constraints": "semantic invariants that must remain unchanged, such as output structure, shape, index alignment, unaffected values, ordering, assignments, etc.",
  "verification": "general properties that should hold after successful execution, not references to this specific test case",
  "confidence": "low, medium, or high"
}

""" + DS1000_TASK_FAMILY_TAXONOMY_BLOCK + """
""" + DS1000_ERROR_TYPE_TAXONOMY_BLOCK + """

Important rules:

1. Extract a reusable procedural skill, NOT a bug repair log.
If reflection diagnosis conflicts with the successful patch, prioritize evidence from the patch diff and successful execution. The reflection is an hypothesis; the successful patch is empirical evidence. When they disagree, the patch diff and verified execution are the authoritative signal for determining both the error_type and the underlying reusable procedure.

2. Generalize beyond this specific DS1000 example.
Do NOT mention specific variable names, column names, values, thresholds, row indices, datasets, public examples, or literal patches unless they are part of a generally reusable concept.

3. The procedure should describe semantic solving operations instead of implementation details.

Good example:

1. Identify the target object to transform.
2. Compute the selection criterion.
3. Construct the transformation selector.
4. Apply the transformation while preserving unaffected elements.
5. Validate structural consistency after transformation.

Bad example:

1. Replace data with df.
2. Call value_counts().
3. Use loc().

4. Prefer algorithm-level reasoning over API-level implementation whenever possible.

5. Use the reflection, patch diff, successful patch, and verification only as evidence for extracting a broader reusable procedure.

6. The extracted procedure should remain useful even if:
- variable names change,
- datasets change,
- thresholds change,
- column names change,
- APIs differ,
- implementation details differ.

7. The retrieval_key should summarize the skill in a compact semantic form suitable for vector retrieval. It carries the fine-grained, example-specific nuance that task_family deliberately omits.

Examples:

"frequency-based conditional replacement"

"verify dataframe object consistency"

"preserve dataframe alignment during assignment"

"axis-aware reduction operation"

8. task_family must be chosen from the fixed taxonomy above based on the DOMINANT operation in the procedure — do not invent finer-grained labels here; that granularity belongs in retrieval_key.
9. error_type must be chosen verbatim from ERROR_TYPE_TAXONOMY above based on the DOMINANT root-cause error evidenced by the patch diff and successful patch — not the symptom the reflection describes. 

10. The avoid_assumption field should describe the incorrect assumption that this procedure prevents.

Good examples:

"Assuming the referenced object exists without verification."

"Assuming reduction axes match the intended semantic dimension."

"Assuming exception rules apply uniformly across all columns."

11. The preserve_constraints field should describe semantic invariants rather than implementation details.

Examples:

"Preserve output shape and index alignment."

"Preserve unaffected elements."

"Preserve column ordering."

"Preserve assignment semantics."

12. The verification field should describe reusable correctness properties rather than test outcomes.

Good example:

"The selected elements are transformed correctly while preserving structural consistency and unaffected values."

Bad example:

"Tests passed."

The extracted skill should maximize transferability across heterogeneous DS1000 problems and represent reusable procedural knowledge instead of a one-time repair record.
"""



DS1000_QUERY_EXTRACTION_CHAT_INSTRUCTION = """You are a Python data science coding assistant.
You will be given a DS1000 problem description.
Your job is to extract a lightweight QUERY for retrieving relevant procedural memories
from a skill database.

IMPORTANT: You are ONLY extracting retrieval keys. Do NOT reason about the solution,
do NOT infer algorithmic steps, do NOT guess pitfalls. The query is used for embedding
similarity search — a later LLM judge will compare the full problem text against
retrieved candidates.

Return exactly one valid JSON object and nothing else.
Do NOT use markdown.
Do NOT use code fences.

The JSON object must contain exactly these string fields:

{
  "task_family": "exactly one category copied verbatim from TASK_FAMILY_TAXONOMY below",
  "retrieval_key": "short semantic description for embedding retrieval, 5-10 words"
}

""" + DS1000_TASK_FAMILY_TAXONOMY_BLOCK + """

Important:
- task_family must come from the fixed taxonomy above.
- retrieval_key should be a compact semantic description suitable for vector retrieval.
- Do NOT solve the problem or infer procedure/assumptions/constraints — those are the
  responsibility of the downstream LLM Applicability Judge, which reads the full problem text.
"""




DS1000_SIMPLE_COMPLETION_INSTRUCTION = """Complete the DS1000 Python code snippet.
The snippet must assign the final answer to a variable named `result`.
If completing an existing function body, indent every non-blank line by exactly four spaces and end with `return result`.
If completing a top-level snippet, assign to `result` and do not use `return`.
Do not repeat imports, example data, pd.DataFrame(...), load_data(), or existing variable initialization.
Return only Python code.

Problem:
"""

DS1000_SELF_REFLECTION_COMPLETION_INSTRUCTION = """Diagnose this failed DS1000 Python code snippet and write a concise repair guide.
Return exactly one valid JSON object and nothing else. Do not use markdown.
Fields: root_cause, repair_action, preserve_constraints, confidence.
"""


DS1000_REFLEXION_COMPLETION_INSTRUCTION = """Improve the DS1000 Python code snippet using the feedback and reflection.
The snippet must assign the final answer to a variable named `result`.
Return only Python code.
"""

DS1000_SKILL_EXTRACTION_COMPLETION_INSTRUCTION = """Extract a reusable DS1000 procedural repair skill.
Return exactly one valid JSON object and nothing else.
The JSON object must have exactly these string fields: task_family, retrieval_key, api_concepts, procedure, avoid_assumption, applicability_conditions, non_applicability_conditions, preserve_constraints, verification, confidence.
"""



DS1000_QUERY_EXTRACTION_COMPLETION_INSTRUCTION = """Extract a lightweight query from this DS1000 problem for memory retrieval.
Return exactly one valid JSON object and nothing else.
Fields: task_family, retrieval_key.
"""

DS1000_EPISODIC_MEMORY_COMPLETION_INSTRUCTION = """Build an episodic memory entry from the DS1000 failure. A repair_action is already known.
Add classification labels and retrieval metadata. Do NOT re-diagnose.
Return exactly one valid JSON object and nothing else.
Fields: task_family, error_type, failed_assumption, retrieval_key, injection_card.
"""


class DS1000Generator(Generator):
    def self_reflection(
        self,
        func: str,                        # 运行失败的源代码实现
        feedback: str,                    # 测试报错信息或堆栈日志
        model: ModelBase,
        problem_prompt: Optional[str] = None, # 原始问题描述
        library: Optional[str] = None,       # 涉及的库名（如 pandas）
    ) -> str:

        from generators.generator_utils import print_messages
        # compact_feedback = compact_ds1000_feedback(feedback)
        if model.is_chat:
            system_msg = DS1000_SELF_REFLECTION_CHAT_INSTRUCTION
            user_msg = (
                f"[library hint]:\n{library or 'unknown'}\n\n"
                f"[problem]:\n{problem_prompt or ''}\n\n"
                f"[previous code]:\n```python\n{func}\n```\n\n"
                # f"[test feedback]:\n{compact_feedback}\n\n"
                f"[test feedback]:\n{feedback}\n\n"
                "[json reflection]:"
            )
            print_messages(system_msg, user_msg)
            output = generate_chat_with_empty_retry(
                model=model,
                messages=[
                    Message(role="system", content=system_msg),
                    Message(role="user", content=user_msg),
                ],
                max_tokens=DS1000_REFLECTION_MAX_TOKENS,
                temperature=0.0,
                num_comps=1,
                retry_content=(
                    "Your previous response was empty. Return exactly one valid JSON "
                    "object with the requested fields and nothing else."
                ),
                extra_body=get_glm_disable_thinking_body(model),
            )
        else:
            output = model.generate(
                f"{DS1000_SELF_REFLECTION_COMPLETION_INSTRUCTION}\n"
                f"[library hint]:\n{library or 'unknown'}\n\n"
                # Original full-problem context, kept for comparison/rollback:
                # f"[problem]:\n{problem_prompt or ''}\n\n"
                # f"[previous code]:\n{func}\n\n[test feedback]:\n{compact_feedback}\n\n[json reflection]:",
                f"[previous code]:\n{func}\n\n[test feedback]:\n{feedback}\n\n[json reflection]:",
                max_tokens=DS1000_REFLECTION_MAX_TOKENS,
                temperature=0.0,
            )
        assert isinstance(output, str)       # 判断模型输出是否为字符串
        reflection = parse_ds1000_reflection_json(output, library)
        print_v(Panel(reflection, title="Self reflection output", border_style="yellow"))
        return reflection

    def skill_extraction(
        self,
        problem_prompt: str,
        reflection: str,
        failed_code: str,
        successful_code: str,
        patch_diff: str,
        success_feedback: str,
        model: ModelBase,
        library: Optional[str] = None,
    ) -> str:
        from generators.generator_utils import print_messages

        if model.is_chat:
            system_msg = DS1000_SKILL_EXTRACTION_CHAT_INSTRUCTION
            user_msg = (
                f"[library hint]:\n{library or 'unknown'}\n\n"
                f"[problem]:\n{problem_prompt}\n\n"
                f"[failed code]:\n```python\n{failed_code}\n```\n\n"
                f"[reflection]:\n{reflection}\n\n"
                f"[successful patch]:\n```python\n{successful_code}\n```\n\n"
                f"[patch diff]:\n```diff\n{patch_diff}\n```\n\n"
                f"[success verification]:\n{success_feedback}\n\n"
                "[procedural skill json]:"
            )
            print_messages(system_msg, user_msg)
            output = generate_chat_with_empty_retry(
                model=model,
                messages=[
                    Message(role="system", content=system_msg),
                    Message(role="user", content=user_msg),
                ],
                max_tokens=DS1000_REFLECTION_MAX_TOKENS,
                temperature=0.0,
                num_comps=1,
                retry_content=(
                    "Your previous response was empty. Return exactly one valid JSON "
                    "object with the requested procedural skill fields and nothing else."
                ),
                extra_body=get_glm_disable_thinking_body(model),
            )
        else:
            output = model.generate(
                f"{DS1000_SKILL_EXTRACTION_COMPLETION_INSTRUCTION}\n"
                f"[library hint]:\n{library or 'unknown'}\n\n"
                f"[problem]:\n{problem_prompt}\n\n"
                f"[reflection]:\n{reflection}\n\n"
                f"[failed code]:\n{failed_code}\n\n"
                f"[successful patch]:\n{successful_code}\n\n"
                f"[patch diff]:\n{patch_diff}\n\n"
                f"[success verification]:\n{success_feedback}\n\n"
                "[procedural skill json]:",
                max_tokens=DS1000_REFLECTION_MAX_TOKENS,
                temperature=0.0,
            )

        assert isinstance(output, str)
        skill = parse_ds1000_skill_json(output, library)
        print_v(Panel(skill, title="Procedural skill output", border_style="green"))
        return skill

    def extract_query_skill(
        self,
        problem_prompt: str,
        model: ModelBase,
        library: Optional[str] = None,
    ) -> str:
        """从问题描述中提取轻量 query skill，用于检索 procedural memory。"""
        from generators.generator_utils import print_messages

        if model.is_chat:
            system_msg = DS1000_QUERY_EXTRACTION_CHAT_INSTRUCTION
            user_msg = (
                f"[library hint]:\n{library or 'unknown'}\n\n"
                f"[problem]:\n{problem_prompt}\n\n"
                "[query skill json]:"
            )
            print_messages(system_msg, user_msg)
            output = generate_chat_with_empty_retry(
                model=model,
                messages=[
                    Message(role="system", content=system_msg),
                    Message(role="user", content=user_msg),
                ],
                max_tokens=DS1000_REFLECTION_MAX_TOKENS,
                temperature=0.0,
                num_comps=1,
                retry_content=(
                    "Your previous response was empty. Return exactly one valid JSON "
                    "object with the requested query skill fields and nothing else."
                ),
                extra_body=get_glm_disable_thinking_body(model),
            )
        else:
            output = model.generate(
                f"{DS1000_QUERY_EXTRACTION_COMPLETION_INSTRUCTION}\n"
                f"[library hint]:\n{library or 'unknown'}\n\n"
                f"[problem]:\n{problem_prompt}\n\n"
                "[query skill json]:",
                max_tokens=DS1000_REFLECTION_MAX_TOKENS,
                temperature=0.0,
            )

        assert isinstance(output, str)
        skill = parse_ds1000_query_json(output, library)
        print_v(Panel(skill, title="Query skill output", border_style="cyan"))
        return skill

    def build_episodic_memory(
        self,
        problem_prompt: str,
        func: str,
        feedback: str,
        reflection_json: str,
        model: ModelBase,
        library: Optional[str] = None,
    ) -> str:
        """Stage B: 基于 Stage A 的 repair_action 生成 episodic memory。

        reflection_json 是 self_reflection 的输出，包含 repair_action。
        此方法仅调用 LLM 补充分类标签和检索元数据，不重新诊断错误。
        """
        from generators.generator_utils import print_messages

        # 从 Stage A reflection 提取 repair_action
        try:
            reflection_parsed = json.loads(reflection_json)
        except (json.JSONDecodeError, TypeError):
            reflection_parsed = {}
        repair_action = reflection_parsed.get("repair_action", "unknown")

        # 用 Stage A 的值填充 episodic memory 提示词
        formatted_instruction = DS1000_EPISODIC_MEMORY_CHAT_INSTRUCTION.format(
            repair_action_from_stage_A=repair_action,
        )

        if model.is_chat:
            system_msg = formatted_instruction
            user_msg = (
                f"[library hint]:\n{library or 'unknown'}\n\n"
                f"[problem]:\n{problem_prompt}\n\n"
                f"[failed code]:\n```python\n{func}\n```\n\n"
                f"[test feedback]:\n{feedback}\n\n"
                "[episodic memory json]:"
            )
            print_messages(system_msg, user_msg)
            output = generate_chat_with_empty_retry(
                model=model,
                messages=[
                    Message(role="system", content=system_msg),
                    Message(role="user", content=user_msg),
                ],
                max_tokens=DS1000_REFLECTION_MAX_TOKENS,
                temperature=0.0,
                num_comps=1,
                retry_content=(
                    "Your previous response was empty. Return exactly one valid JSON "
                    "object with the requested episodic memory fields and nothing else."
                ),
                extra_body=get_glm_disable_thinking_body(model),
            )
        else:
            output = model.generate(
                f"{DS1000_EPISODIC_MEMORY_COMPLETION_INSTRUCTION}\n"
                f"Known repair_action: {repair_action}\n\n"
                f"[library hint]:\n{library or 'unknown'}\n\n"
                f"[problem]:\n{problem_prompt}\n\n"
                f"[failed code]:\n{func}\n\n"
                f"[test feedback]:\n{feedback}\n\n"
                "[episodic memory json]:",
                max_tokens=DS1000_REFLECTION_MAX_TOKENS,
                temperature=0.0,
            )

        assert isinstance(output, str)
        # 解析 Stage B 分类标签
        stage_b_labels = parse_ds1000_episodic_memory_json(output, library)
        # 合并 Stage A (repair_action) + Stage B (分类标签)
        try:
            labels = json.loads(stage_b_labels)
        except (json.JSONDecodeError, TypeError):
            labels = {}
        merged = {
            "repair_action": repair_action,
        }
        merged.update(labels)
        episodic_memory = json.dumps(merged, ensure_ascii=False)
        print_v(Panel(episodic_memory, title="Episodic memory output", border_style="green"))
        return episodic_memory

    def func_impl(
        self,
        func_sig: str,                    # 数据集中的problem
        model: ModelBase,
        strategy: str,                    # 生成策略："simple" (初试) 或 "reflexion" (改进)
        prev_func_impl: Optional[str] = None, # (仅限 reflexion) 之前的错误实现
        feedback: Optional[str] = None,      # (仅限 reflexion) 测试反馈信息
        self_reflection: Optional[str] = None, # (仅限 reflexion) JSON 格式的反思诊断报告
        injected_memory: Optional[str] = None, # (可选) 注入的 procedural memory prompt 文本
        num_comps: int = 1,      # 模型一次生成的回复数量
        temperature: float = 0.0,
    ) -> Union[str, List[str]]:

        if strategy not in ("simple", "reflexion"):
            raise ValueError(f"Invalid strategy: {strategy}")
        if strategy == "reflexion" and (
            prev_func_impl is None or feedback is None or self_reflection is None
        ):
            raise ValueError("DS1000 reflexion requires previous code, feedback, and reflection")

        from generators.generator_utils import print_messages, print_generated_func_body
        # compact_feedback = compact_ds1000_feedback(feedback) if feedback is not None else None
        is_function_body = is_ds1000_function_body(func_sig)
        context_instruction = get_ds1000_context_instruction(func_sig)

        memory_prefix = f"{injected_memory}\n\n" if injected_memory else ""

        if model.is_chat:
            if strategy == "simple":
                sys_content = join_instructions(
                    DS1000_SIMPLE_CHAT_INSTRUCTION, context_instruction
                )
                user_content = f"{memory_prefix}{func_sig}"
                messages = [
                    Message(role="system", content=sys_content),
                    Message(role="user", content=user_content),
                ]
                print_messages(sys_content, user_content)
            else:
                sys_content = join_instructions(
                    DS1000_REFLEXION_CHAT_INSTRUCTION, context_instruction
                )
                user_content = (
                    f"{memory_prefix}"
                    f"[problem]:\n{func_sig}\n\n"
                    f"[previous code]:\n```python\n{prev_func_impl}\n```\n\n"
                    f"[test feedback]:\n{feedback}\n\n"
                    f"[reflection]:\n{self_reflection}\n\n"
                    "[improved code]:"
                )
                messages = [
                    Message(role="system", content=sys_content),
                    Message(role="user", content=user_content),
                ]
                print_messages(sys_content, user_content)
            output = generate_chat_with_empty_retry(
                model=model,
                messages=messages,
                num_comps=num_comps,
                temperature=temperature,
                max_tokens=DS1000_CODE_MAX_TOKENS,
                retry_content=DS1000_CODE_RETRY_INSTRUCTION,
            )
        else:
            if strategy == "simple":
                prompt = (
                    f"{memory_prefix}"
                    f"{DS1000_SIMPLE_COMPLETION_INSTRUCTION}"
                    f"{context_instruction}\n\n"
                    f"{func_sig}\n\nCode:"
                )
            else:
                prompt = (
                    f"{memory_prefix}"
                    f"{DS1000_REFLEXION_COMPLETION_INSTRUCTION}\n"
                    f"{context_instruction}\n\n"
                    f"[problem]:\n{func_sig}\n\n"
                    f"[previous code]:\n{prev_func_impl}\n\n"
                    f"[test feedback]:\n{feedback}\n\n"
                    f"[reflection]:\n{self_reflection}\n\n"
                    "[improved code]:"
                )
            print_v(Panel(prompt, title="Completion Prompt", border_style="magenta")) 
            output = generate_completion_with_empty_retry(
                model=model,
                prompt=prompt,
                num_comps=num_comps,
                temperature=temperature,
            )

        # 单回复模式的后处理
        if num_comps == 1:
            assert isinstance(output, str)
            print_generated_func_body(output)
            parsed_output = parse_ds1000_code(output, is_function_body=is_function_body)
            invalid_reason = get_invalid_ds1000_code_reason(parsed_output, is_function_body)
            print_v(Panel(
                f"is_function_body={is_function_body}\n"
                f"invalid_reason={invalid_reason or 'None'}",
                title="DS1000 AST validation",
                border_style="blue",
            ))
            if model.is_chat:
                parsed_output = retry_invalid_ds1000_code_once(
                    parsed_code=parsed_output,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=DS1000_CODE_MAX_TOKENS,
                    is_function_body=is_function_body,
                    invalid_reason=invalid_reason,
                )
            return parsed_output

        assert isinstance(output, list)
        print_generated_func_body("\n\n".join(output))
        return [parse_ds1000_code(item, is_function_body=is_function_body) for item in output]

    def internal_tests(
        self,
        func_sig: str,
        model: ModelBase,
        max_num_tests: int = 5,
    ) -> List[str]:
        return []


# 清洗模型输出，提取代码块中的内容，移除常见的干扰标签，处理缩进
def parse_ds1000_code(output: str, is_function_body: bool = False) -> str:

    fenced = re.search(r"```(?:python)?\n(.*?)\n```", output, re.DOTALL)
    if fenced:
        output = fenced.group(1)

    output = strip_blank_edge_lines(output)
    output = re.sub(r"^```(?:python)?\s*\n?", "", output)
    output = re.sub(r"\n?```\s*$", "", output)
    output = re.sub(r"^<code>\s*", "", output)
    output = re.sub(r"\n?</code>$", "", output)
    output = output.replace("BEGIN SOLUTION", "")
    output = strip_blank_edge_lines(output)
    output = textwrap.dedent(output)   # 消除所有行共有的前导空白字符（缩进）
    output = normalize_ds1000_returns(output, is_function_body=is_function_body)
    if is_function_body:
        return indent_function_body(output)
    return output


def is_ds1000_function_body(func_sig: str) -> bool:
    # Only check the code template (between "A:" and "BEGIN SOLUTION").
    # Even within this section, some `def`s are complete helper functions
    # (with real code in the body) — those do NOT indicate FUNCTION_BODY mode.
    # Only `def`s with placeholder bodies (comments, `...`, `pass`, `###`)
    # represent functions that the model must complete.
    template = func_sig
    if "BEGIN SOLUTION" in template:
        template = template.split("BEGIN SOLUTION")[0]
    if "A:" in template:
        template = template.split("A:")[-1]

    # Find the last `def` in the template — this determines the mode.
    def_matches = list(re.finditer(r"def\s+(\w+)\s*\([^)]*\)\s*:", template))
    if not def_matches:
        return False

    last_def = def_matches[-1]
    after_def = template[last_def.end():]

    # Collect indented body lines under the def.
    body_stmts: List[str] = []
    for line in after_def.split("\n"):
        stripped = line.strip()
        if not stripped:
            body_stmts.append("")
            continue
        if line[0] in (" ", "\t"):
            body_stmts.append(stripped)
        else:
            break  # non-indented, non-blank — body ended

    # Filter to real executable statements (not comments / placeholders).
    real_code = [
        s
        for s in body_stmts
        if s
        and not s.startswith("#")
        and s not in ("...", "pass")
        and not s.startswith("###")
    ]

    # Only placeholder content → the def body is incomplete → FUNCTION_BODY.
    # Real code in body → this is a helper function → TOP_LEVEL snippet.
    return len(real_code) == 0 and len(body_stmts) > 0
def get_ds1000_context_instruction(func_sig: str) -> str:

    if is_ds1000_function_body(func_sig):
        return (
            "MODE: FUNCTION_BODY.\n"
            "Output only the indented function body. End with `return result`."
        )
    return (
        "MODE: TOP_LEVEL_SNIPPET.\n"
        "A `return` statement is invalid here. Assign the final answer to `result` only."
    )


# 将多个指令片段联结在一起，中间用双换行符分隔，并过滤掉空片段
def join_instructions(*parts: str) -> str:
    return "\n\n".join(part for part in parts if part)


# 移除字符串开头和结尾的空白行（不影响中间的空白行）
def strip_blank_edge_lines(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


# 为生成的代码块增加缩进。先移除两端空白行，然后每行前方增加 4 个空格（如果该行不为空）
def indent_function_body(text: str) -> str:
    lines = strip_blank_edge_lines(text).splitlines()
    return "\n".join(f"    {line}" if line.strip() else "" for line in lines)


# 规范化返回语句
def normalize_ds1000_returns(code: str, is_function_body: bool) -> str:
    lines = strip_blank_edge_lines(code).splitlines()
    if is_function_body:
        # 如果最后一行是 "result"，替换为 "return result"
        if lines and lines[-1].strip() == "result":
            indent = lines[-1][:len(lines[-1]) - len(lines[-1].lstrip())]
            lines[-1] = f"{indent}return result"
        # 如果没有任何 return 语句，添加 "return result"
        if not any(line.strip().startswith("return ") for line in lines):
            lines.append("return result")
        return "\n".join(lines)

    fixed_lines = []
    for line in lines:
        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())]
        # 移除 "return result" 行
        if stripped == "return result":
            continue
        # 将 "return xxx" 转换为 "result = xxx"
        if stripped.startswith("return "):
            fixed_lines.append(f"{indent}result = {stripped[len('return '):]}")
            continue
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


# 第一次生成结果为空时，重试一次并加入催促指令
def generate_chat_with_empty_retry(
    model: ModelBase,
    messages: List[Message],
    max_tokens: int,
    temperature: float,
    num_comps: int,
    retry_content: str = (
        "Your previous response was empty. Output executable Python code only. "
        "Do not leave the answer blank."
    ),
    extra_body: Optional[dict] = None,
) -> Union[str, List[str]]:
    
    output = model.generate_chat(
        messages=messages,
        num_comps=num_comps,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    if num_comps != 1:
        return output
    assert isinstance(output, str)
    if output.strip():
        return output

    print_v(Panel("Empty model output; retrying once.", title="Generation warning", border_style="red"))
    retry_messages = messages + [
        Message(
            role="user",
            content=retry_content,
        )
    ]
    return model.generate_chat(
        messages=retry_messages,
        num_comps=num_comps,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body or get_glm_disable_thinking_body(model),
    )


# 针对 GLM 系列模型的特殊配置：禁用其内置 'thinking'（思考）过程
def get_glm_disable_thinking_body(model: ModelBase) -> Optional[dict]:

    if model.name.lower().startswith("glm-"):
        return {"thinking": {"type": "disabled"}}
    return None


DS1000_CODE_RETRY_INSTRUCTION = """Your previous response was empty or unusable.
Output a short executable Python snippet only.
Do not use markdown code fences.
Do not repeat imports, example data, pd.DataFrame(...), load_data(), or existing variable initialization.
Use the variables already provided by the problem context and assign the final answer to `result`.
For top-level snippets, never output a `return` statement."""


# 生成的代码被验证为无效时，自动进行一次重试生成
def retry_invalid_ds1000_code_once(
    parsed_code: str,
    model: ModelBase,
    messages: List[Message],
    temperature: float,
    max_tokens: int,
    is_function_body: bool,
    invalid_reason: Optional[str] = None,
) -> str:
    reason = invalid_reason or get_invalid_ds1000_code_reason(parsed_code, is_function_body)
    if reason is None:
        return parsed_code

    print_v(Panel(reason, title="Generated code rejected; retrying once", border_style="red"))
    retry_messages = messages + [
        Message(
            role="user",
            content=(
                f"The previous code was rejected because: {reason}\n"
                f"{DS1000_CODE_RETRY_INSTRUCTION}"
            ),
        )
    ]
    retry_output = model.generate_chat(
        messages=retry_messages,
        num_comps=1,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=get_glm_disable_thinking_body(model),
    )
    assert isinstance(retry_output, str)
    print_v(Panel(retry_output, title="Retry generated func body", border_style="cyan"))
    retry_code = parse_ds1000_code(retry_output, is_function_body=is_function_body)
    retry_reason = get_invalid_ds1000_code_reason(retry_code, is_function_body)
    if retry_reason is not None:
        print_v(Panel(retry_reason, title="Retry code still suspicious", border_style="yellow"))
    return retry_code


# 获取无效的代码原因
def get_invalid_ds1000_code_reason(code: str, is_function_body: bool) -> Optional[str]:
    if not code.strip():
        return "empty generated code"
    if "```" in code:
        return "markdown code fence remained in parsed code"
    if re.search(r"\b\w+\s*=\s*pd\.DataFrame\s*\(", code):
        return "code repeats example DataFrame initialization instead of using test input variables"
    if re.search(r"\b\w+\s*=\s*load_data\s*\(", code):
        return "code repeats data loading instead of using test input variables"

    syntax_code = f"def _ds1000_generated():\n{code}\n" if is_function_body else code
    try:
        ast.parse(syntax_code)
        compile(syntax_code, "<ds1000_generated>", "exec")
    except SyntaxError as exc:
        return f"syntax error after parsing: {exc.msg}"
    return None


# 生成完成时，如果输出为空，则自动重试一次
def generate_completion_with_empty_retry(
    model: ModelBase,
    prompt: str,
    temperature: float,
    num_comps: int,
) -> Union[str, List[str]]:
    output = model.generate(prompt, num_comps=num_comps, temperature=temperature)
    if num_comps != 1:
        return output
    assert isinstance(output, str)
    if output.strip():
        return output

    print_v(Panel("Empty model output; retrying once.", title="Generation warning", border_style="red"))
    return model.generate(
        prompt + "\n\nYour previous response was empty. Output executable Python code only.",
        num_comps=num_comps,
        temperature=temperature,
    )


# 确保输出格式符合预期的JSON结构
def parse_ds1000_reflection_json(output: str, library: Optional[str] = None) -> str:
    output = output.strip()
    # return output

    fenced = re.search(r"```(?:json)?\n(.*?)\n```", output, re.DOTALL)
    if fenced:
        output = fenced.group(1).strip()

    json_match = re.search(r"\{.*\}", output, re.DOTALL)
    if json_match:
        output = json_match.group(0)

    fields = [
        "root_cause",
        "repair_action",
        "preserve_constraints",
        "confidence",
    ]
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = {
            "root_cause": output[:500],
            "repair_action": "Review the test feedback, identify the root cause, and produce a targeted code fix.",
            "preserve_constraints": "unknown",
            "confidence": "low",
        }

    normalized = {}
    for field in fields:
        normalized[field] = str(parsed.get(field, "unknown")).strip() or "unknown"
    return json.dumps(normalized, ensure_ascii=False)


def parse_ds1000_episodic_memory_json(output: str, library: Optional[str] = None) -> str:
    """解析 episodic memory (Stage B) 的 JSON 输出，校验 9 个分类标签字段。"""
    output = output.strip()

    fenced = re.search(r"```(?:json)?\n(.*?)\n```", output, re.DOTALL)
    if fenced:
        output = fenced.group(1).strip()

    json_match = re.search(r"\{.*\}", output, re.DOTALL)
    if json_match:
        output = json_match.group(0)

    fields = [
        "task_family",
        "error_type",
        "failed_assumption",
        "retrieval_key",
        "injection_card",
    ]
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = {
            "task_family": "unknown",
            "error_type": "unknown",
            "failed_assumption": "unknown",
            "retrieval_key": output[:500],
            "injection_card": "unknown",
        }

    normalized = {}
    for field in fields:
        normalized[field] = str(parsed.get(field, "unknown")).strip() or "unknown"
    # library 直接来自数据集，不由 LLM 生成
    normalized["library"] = library or "unknown"
    return json.dumps(normalized, ensure_ascii=False)


def parse_ds1000_query_json(output: str, library: Optional[str] = None) -> str:
    """解析 query extraction 的 JSON 输出，仅校验 3 个字段。"""
    output = output.strip()

    fenced = re.search(r"```(?:json)?\n(.*?)\n```", output, re.DOTALL)
    if fenced:
        output = fenced.group(1).strip()

    json_match = re.search(r"\{.*\}", output, re.DOTALL)
    if json_match:
        output = json_match.group(0)

    fields = [
        "task_family",
        "retrieval_key",
    ]
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = {
            "task_family": "unknown",
            "retrieval_key": "unknown",
        }

    normalized = {}
    for field in fields:
        normalized[field] = str(parsed.get(field, "unknown")).strip() or "unknown"
    # library 直接来自数据集，不由 LLM 生成
    normalized["library"] = library or "unknown"
    return json.dumps(normalized, ensure_ascii=False)


def parse_ds1000_skill_json(output: str, library: Optional[str] = None) -> str:
    output = output.strip()

    fenced = re.search(r"```(?:json)?\n(.*?)\n```", output, re.DOTALL)
    if fenced:
        output = fenced.group(1).strip()

    json_match = re.search(r"\{.*\}", output, re.DOTALL)
    if json_match:
        output = json_match.group(0)

    fields = [
        "task_family",
        "error_type",
        "retrieval_key",
        "api_concepts",
        "procedure",
        "avoid_assumption",
        "applicability_conditions",
        "non_applicability_conditions",
        "preserve_constraints",
        "verification",
        "confidence",
    ]
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = {
            "task_family": "unknown",
            "error_type": "unknown",
            "retrieval_key": "unknown",
            "api_concepts": "unknown",
            "procedure": "Review the reflection, patch diff, successful patch, and success verification to infer a reusable operational procedure.",
            "avoid_assumption": "unknown",
            "applicability_conditions": "unknown",
            "non_applicability_conditions": "unknown",
            "preserve_constraints": "unknown",
            "verification": output[:500],
            "confidence": "low",
        }

    normalized = {}
    for field in fields:
        if field == "procedure":
            procedure = parsed.get("procedure")
            if procedure is None:
                procedure = parsed.get("implementation_steps")
            if procedure is None:
                procedure = parsed.get("repair_rule")
            normalized[field] = str(procedure or "unknown").strip() or "unknown"
            continue
        normalized[field] = str(parsed.get(field, "unknown")).strip() or "unknown"
    # library 直接来自数据集，不由 LLM 生成
    normalized["library"] = library or "unknown"
    return json.dumps(normalized, ensure_ascii=False)

