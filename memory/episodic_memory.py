import ast
import re
from typing import Any, Dict, List, Optional

from .common import (
    MEMORY_SCHEMA_VERSION,
    _field,
    _first_success_index,
    _is_passing_feedback,
    _parse_json_or_raw,
    _reflection_by_attempt,
    _reflection_for_attempt,
    _stable_id,
    _utc_now,
)


# 构建episode id
def build_episode_id(
    item: Dict[str, Any],
    item_index: int,
    implementations: List[str],
    test_feedback: List[str],
) -> str:
    metadata = item.get("metadata", {})
    return _stable_id(
        "episode",
        item_index,
        metadata.get("problem_id"),
        item.get("prompt", ""),
        implementations,
        test_feedback,
    )


# 构建episodic memory
def build_ds1000_episodic_memory(
    *,
    episode_id: str,
    item: Dict[str, Any],
    item_index: int,
    implementations: List[str],
    test_feedback: List[str],
    reflections: List[str],
    reflection_attempt_indices: Optional[List[int]] = None,
    is_solved: bool,
    final_solution: str,
) -> Dict[str, Any]:
    metadata = item.get("metadata", {})

    # 步骤 1：定位最有价值的失败
    decisive_failure_index = _decisive_failure_index(test_feedback)

    #步骤 2：提取对应反思与代码
    decisive_reflection = _reflection_for_attempt(
        reflections, reflection_attempt_indices, decisive_failure_index
    )
    reflection = _parse_json_or_raw(decisive_reflection)
    feedback = test_feedback[decisive_failure_index] if decisive_failure_index is not None else ""
    failed_code = (
        implementations[decisive_failure_index]
        if decisive_failure_index is not None and decisive_failure_index < len(implementations)
        else ""
    )

    # 步骤 3：从反思/反馈中结构化提炼知识
    library = metadata.get("library") or _field(reflection, "library", "unknown")
    task_family = _field(reflection, "task_family", "unknown")
    repair_action = _field(
        reflection,
        "repair_action",
        _field(reflection, "repair_pattern", "unknown"),
    )
    repair_success = bool(is_solved)

    # 步骤 4：组装成最终记录
    record = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "memory_type": "episodic",
        "created_at": _utc_now(),
        "episode_id": episode_id,
        "retrieval_key": _build_problem_embedding_text(
            library=library,
            task_family=task_family,
            repair_action=repair_action,
        ),
        "library": library,
        "task_family": task_family,
        "error_type": _extract_error_type(feedback),
        "failed_assumption": _field(reflection, "failed_assumption", "unknown"),
        "repair_action": repair_action,
        "failure_snapshot": _extract_failure_snapshot(feedback, failed_code),
        "repair_success": repair_success,
        "archive_ref": episode_id,
        "item_index": item_index,
        "problem_id": metadata.get("problem_id"),
    }
    if repair_success:
        record["fixed_code"] = final_solution
    return record


def build_ds1000_episodic_memory_from_llm(
    *,
    episode_id: str,
    item: Dict[str, Any],
    item_index: int,
    implementations: List[str],
    test_feedback: List[str],
    reflections: List[str],
    reflection_attempt_indices: Optional[List[int]] = None,
    is_solved: bool,
    final_solution: str,
    llm_episodic_memory: str,
) -> Dict[str, Any]:
    """基于 LLM (Stage B) 生成的 episodic memory JSON 组装最终记录。

    repair_action 来自 Stage A (reflection)，已由 build_episodic_memory 合并到
    llm_episodic_memory 中。此函数仅追加元数据字段（schema_version, episode_id 等）。
    """
    metadata = item.get("metadata", {})
    parsed = _parse_json_or_raw(llm_episodic_memory)
    if not isinstance(parsed, dict):
        parsed = {}

    # 找到决定性失败索引，用于提取 feedback / failed_code
    decisive_failure_index = _decisive_failure_index(test_feedback)
    feedback = test_feedback[decisive_failure_index] if decisive_failure_index is not None else ""
    failed_code = (
        implementations[decisive_failure_index]
        if decisive_failure_index is not None and decisive_failure_index < len(implementations)
        else ""
    )

    # library 直接来自数据集，不由 LLM 生成
    library = metadata.get("library") or "unknown"
    task_family = _field(parsed, "task_family", "unknown")
    repair_action = _field(parsed, "repair_action", "unknown")
    repair_success = bool(is_solved)

    record = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "memory_type": "episodic",
        "created_at": _utc_now(),
        "episode_id": episode_id,
        "retrieval_key": _field(
            parsed,
            "retrieval_key",
            _build_problem_embedding_text(
                library=library,
                task_family=task_family,
                repair_action=repair_action,
            ),
        ),
        "library": library,
        "task_family": task_family,
        "error_type": _field(parsed, "error_type", _extract_error_type(feedback)),
        "failed_assumption": _field(parsed, "failed_assumption", "unknown"),
        "repair_action": repair_action,
        "injection_card": _field(parsed, "injection_card", "unknown"),
        "failure_snapshot": _extract_failure_snapshot(feedback, failed_code),
        "repair_success": repair_success,
        "archive_ref": episode_id,
        "item_index": item_index,
        "problem_id": metadata.get("problem_id"),
    }
    if repair_success:
        record["fixed_code"] = final_solution
    return record


def build_ds1000_episodic_archive(
    *,
    episode_id: str,
    item: Dict[str, Any],
    item_index: int,
    implementations: List[str],
    test_feedback: List[str],
    reflections: List[str],
    reflection_attempt_indices: Optional[List[int]] = None,
    is_solved: bool,
    final_solution: str,
) -> Dict[str, Any]:
    metadata = item.get("metadata", {})
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "memory_type": "episodic_archive",
        "created_at": _utc_now(),
        "episode_id": episode_id,
        "dataset": "ds1000",
        "item_index": item_index,
        "problem_id": metadata.get("problem_id"),
        "library_problem_id": metadata.get("library_problem_id"),
        "library": metadata.get("library", "unknown"),
        "perturbation_type": metadata.get("perturbation_type"),
        "perturbation_origin_id": metadata.get("perturbation_origin_id"),
        "prompt": item.get("prompt", ""),
        "code_context": item.get("code_context", ""),
        "reference_code": item.get("reference_code", ""),
        "is_solved": is_solved,
        "final_solution": final_solution,
        "attempts": _build_attempts(
            implementations,
            test_feedback,
            reflections,
            reflection_attempt_indices,
            is_solved=is_solved,
        ),
    }


def build_ds1000_successful_example(
    *,
    item: Dict[str, Any],
    item_index: int,
    implementation: str,
    success_feedback: str,
) -> Dict[str, Any]:
    metadata = item.get("metadata", {})
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "memory_type": "successful_example",
        "created_at": _utc_now(),
        "example_id": _stable_id(
            "successful_example",
            item_index,
            metadata.get("problem_id"),
            item.get("prompt", ""),
            implementation,
        ),
        "dataset": "ds1000",
        "item_index": item_index,
        "problem_id": metadata.get("problem_id"),
        "library_problem_id": metadata.get("library_problem_id"),
        "library": metadata.get("library", "unknown"),
        "prompt": item.get("prompt", ""),
        "successful_solution": implementation,
        "reference_code": item.get("reference_code", ""),
        "success_feedback": success_feedback,
    }

# 每个元素代表一次完整的"编码—测试—反思"循环
def _build_attempts(
    implementations: List[str],
    test_feedback: List[str],
    reflections: List[str],
    reflection_attempt_indices: Optional[List[int]] = None,
    *,
    is_solved: bool = False,
) -> List[Dict[str, Any]]:
    reflection_by_attempt = _reflection_by_attempt(reflections, reflection_attempt_indices)
    attempts = []
    for index, implementation in enumerate(implementations):
        has_feedback = index < len(test_feedback)
        feedback = test_feedback[index] if has_feedback else ""
        reflection = reflection_by_attempt.get(index)
        is_passing = _is_passing_feedback(feedback) or (
            is_solved and index == len(implementations) - 1 and not has_feedback
        )
        attempts.append(
            {
                "attempt_index": index,
                "implementation": implementation,
                "feedback": feedback,
                "is_passing": is_passing,
                "reflection_after_failure": _parse_json_or_raw(reflection)
                if reflection
                else None,
            }
        )
    return attempts

# 找到决定性的失败索引
def _decisive_failure_index(test_feedback: List[str]) -> Optional[int]:
    success_index = _first_success_index(test_feedback)
    # 如果最终成功了 → 取"成功前最后一次失败"
    if success_index is not None and success_index > 0:
        return success_index - 1
    # 如果从未成功过 → 取"最后一次失败"
    for index in range(len(test_feedback) - 1, -1, -1):
        if not _is_passing_feedback(test_feedback[index]):
            return index
    return None

# def _extract_error_type(feedback: str) -> str:
#     for line in feedback.splitlines():
#         stripped = line.strip()
#         if not stripped:
#             continue
#         if stripped in ("Tests passed:", "Tests failed:"):
#             continue
#         if stripped.startswith("DS1000 diagnostic outputs"):
#             continue
#         if stripped.startswith("TestCase "):
#             continue
#         if stripped.startswith("File "):
#             continue
#         if stripped.startswith("Traceback"):
#             continue
#         if re.match(r"^[A-Za-z_][A-Za-z0-9_]*(Error|Exception|Warning|Interrupt)\b", stripped):
#             return stripped
#         if stripped == "AssertionError":
#             return stripped
#     for line in feedback.splitlines():
#         stripped = line.strip()
#         if stripped and stripped not in ("Tests passed:", "Tests failed:"):
#             return stripped
#     return "unknown"

# 从测试反馈中提取异常类型名
def _extract_error_type(feedback: str) -> str:
    error_type_pattern = re.compile(
        r"^((?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning|Interrupt))\b"
    )
    for line in feedback.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in ("Tests passed:", "Tests failed:"):
            continue
        if stripped.startswith("DS1000 diagnostic outputs"):
            continue
        if stripped.startswith("TestCase "):
            continue
        if stripped.startswith("File "):
            continue
        if stripped.startswith("Traceback"):
            continue
        match = error_type_pattern.match(stripped)
        if match:
            return match.group(1).split(".")[-1]
    return "unknown"


# 把多个结构化字段组装成一段用于向量检索（embedding）的纯文本描述
def _build_problem_embedding_text(
    *,
    library: str,
    task_family: str,
    repair_action: str,
) -> str:
    # 与 procedural 对齐：embedding 使用统一的 retrieval_key 格式
    # 当 episodic 记录缺少 LLM 生成的 retrieval_key 时，用 task_family +
    # repair_action 拼接作为 fallback
    return "\n".join(
        [
            f"retrieval_key: task_family: {task_family}",
            f"repair_action: {repair_action}",
        ]
    )


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _snapshot_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or default
    text = _compact_repr(value)
    return text if text else default


def _combine_location(*parts: Any) -> str:
    cleaned = []
    for part in parts:
        text = _snapshot_text(part, "")
        if text:
            cleaned.append(text)
    return ", ".join(cleaned) if cleaned else "unknown"


def _default_expected_for_failure_type(snapshot_type: str) -> str:
    defaults = {
        "assertion_error": "passing assertion",
        "wrong_output": "expected output",
        "key_error": "available key or dataframe column",
        "name_error": "defined variable or symbol",
        "unbound_local_error": "assigned local variable",
        "attribute_error": "existing attribute",
        "index_error": "valid positional index",
        "shape_error": "matching shape or length",
        "value_error": "valid value or dtype-compatible input",
        "type_error": "compatible types or function call signature",
        "import_error": "available import or dependency",
        "syntax_error": "syntactically valid code",
        "numeric_error": "valid numeric computation",
        "resource_error": "terminating computation within resource limits",
        "file_error": "accessible file or environment resource",
        "linear_algebra_error": "solvable linear algebra input",
        "runtime_error": "successful execution",
    }
    return defaults.get(snapshot_type, "successful execution")


def _normalize_failure_type(error_type: str, reason: str = "", *, has_output_diagnostics: bool = False) -> str:
    if error_type == "AssertionError":
        return "wrong_output" if has_output_diagnostics else "assertion_error"
    if error_type == "KeyError":
        return "key_error"
    if error_type == "NameError":
        return "name_error"
    if error_type == "UnboundLocalError":
        return "unbound_local_error"
    if error_type == "AttributeError":
        return "attribute_error"
    if error_type == "IndexError":
        return "index_error"
    if error_type == "AxisError":
        return "shape_error"
    if error_type == "ValueError":
        reason_lower = reason.lower()
        if any(
            token in reason_lower
            for token in ("shape", "broadcast", "length", "size of axis", "indices are out-of-bounds")
        ):
            return "shape_error"
        return "value_error"
    if error_type == "TypeError":
        return "type_error"
    if error_type in ("ImportError", "ModuleNotFoundError"):
        return "import_error"
    if error_type in ("SyntaxError", "IndentationError"):
        return "syntax_error"
    if error_type in ("ZeroDivisionError", "OverflowError", "FloatingPointError"):
        return "numeric_error"
    if error_type in ("MemoryError", "TimeoutError", "RecursionError"):
        return "resource_error"
    if error_type in ("FileNotFoundError", "OSError", "PermissionError"):
        return "file_error"
    if error_type == "LinAlgError":
        return "linear_algebra_error"
    if error_type in (
        "MergeError",
        "ParserError",
        "EmptyDataError",
        "InvalidIndexError",
        "OptionError",
        "SpecificationError",
    ):
        return f"pandas_{_camel_to_snake(error_type)}"
    return "runtime_error"


def _make_failure_snapshot(
    snapshot_type: str,
    location: Any,
    observed: Any,
    expected: Any,
) -> Dict[str, str]:
    normalized_type = _snapshot_text(snapshot_type, "runtime_error")
    return {
        "type": normalized_type,
        "location": _snapshot_text(location),
        "observed": _snapshot_text(observed),
        "expected": _snapshot_text(expected, _default_expected_for_failure_type(normalized_type)),
    }


# 把"原始错误信息"转化为"可分析的结构化数据"
def _extract_failure_snapshot(feedback: str, failed_code: str) -> Dict[str, str]:
    error_type = _extract_error_type(feedback)
    if error_type != "AssertionError":
        return _extract_runtime_failure_snapshot(feedback, failed_code)

    actual = _extract_feedback_value(feedback, "actual_result")
    expected = _extract_feedback_value(feedback, "expected_result")
    if actual and expected:
        mismatch = _extract_first_output_mismatch(actual, expected)
        if mismatch:
            return mismatch
        return _make_failure_snapshot(
            "wrong_output",
            "output",
            actual,
            expected,
        )

    if actual:
        return _make_failure_snapshot(
            "wrong_output",
            "output",
            actual,
            "unknown",
        )

    return _make_failure_snapshot(
        "assertion_error",
        _extract_traceback_location(feedback),
        "AssertionError",
        "captured actual and expected output",
    )

# 从运行时错误的反馈中提取一个结构化的failure_snapshot
def _extract_runtime_failure_snapshot(feedback: str, failed_code: str) -> Dict[str, str]:
    error_type = _extract_error_type(feedback)
    message = _extract_exception_message(feedback, error_type)
    location = _extract_traceback_location(feedback)

    name_match = re.search(r"NameError: name '([^']+)' is not defined", feedback)
    if name_match:
        symbol = name_match.group(1)
        return _make_failure_snapshot(
            "name_error",
            _find_symbol_location(failed_code, symbol),
            message or symbol,
            _default_expected_for_failure_type("name_error"),
        )

    attr_match = re.search(r"AttributeError: .* has no attribute '([^']+)'", feedback)
    if attr_match:
        symbol = attr_match.group(1)
        return _make_failure_snapshot(
            "attribute_error",
            _find_symbol_location(failed_code, symbol),
            message or symbol,
            _default_expected_for_failure_type("attribute_error"),
        )

    unbound_match = re.search(
        r"UnboundLocalError: .*local variable '([^']+)' referenced before assignment",
        feedback,
    )
    if unbound_match:
        symbol = unbound_match.group(1)
        return _make_failure_snapshot(
            "unbound_local_error",
            _find_symbol_location(failed_code, symbol),
            message or symbol,
            _default_expected_for_failure_type("unbound_local_error"),
        )

    symbol = _extract_key_error_symbol(message or feedback) if error_type == "KeyError" else ""
    if symbol:
        return _make_failure_snapshot(
            "key_error",
            _find_symbol_location(failed_code, symbol),
            message or symbol,
            _default_expected_for_failure_type("key_error"),
        )

    index_match = re.search(r"IndexError: (.+)", feedback)
    if index_match:
        detail = index_match.group(1).strip()
        return _make_failure_snapshot(
            "index_error",
            location,
            message or detail,
            _default_expected_for_failure_type("index_error"),
        )

    axis_match = re.search(r"(?:AxisError|ValueError): .*axis\s+(-?\d+).*", feedback, re.I)
    if axis_match:
        axis_text = f"axis {axis_match.group(1)}"
        return _make_failure_snapshot(
            "shape_error",
            axis_text,
            message or axis_text,
            _default_expected_for_failure_type("shape_error"),
        )

    shape_match = re.search(
        r"(?:ValueError|IndexError): .*(shape|broadcast|Length mismatch|same length|size of axis|indices are out-of-bounds).*$",
        feedback,
        re.I | re.M,
    )
    if shape_match:
        return _make_failure_snapshot(
            "shape_error",
            location,
            message or error_type,
            _default_expected_for_failure_type("shape_error"),
        )

    dtype_match = re.search(
        r"(?:TypeError|ValueError|DTypePromotionError): .*(dtype|could not convert|astype|ufunc|numeric|datetime|categorical).*$",
        feedback,
        re.I | re.M,
    )
    if dtype_match:
        return _make_failure_snapshot(
            "value_error",
            location,
            message or error_type,
            _default_expected_for_failure_type("value_error"),
        )

    pandas_match = re.search(
        r"(?:pandas\.errors\.)?(MergeError|ParserError|EmptyDataError|InvalidIndexError|OptionError|SpecificationError): (.+)",
        feedback,
    )
    if pandas_match:
        pandas_error_type = pandas_match.group(1)
        detail = pandas_match.group(2).strip()
        snapshot_type = f"pandas_{_camel_to_snake(pandas_error_type)}"
        return _make_failure_snapshot(
            snapshot_type,
            location,
            detail,
            _default_expected_for_failure_type(snapshot_type),
        )

    type_match = re.search(r"TypeError: (.+)", feedback)
    if type_match:
        detail = type_match.group(1).strip()
        return _make_failure_snapshot(
            "type_error",
            _extract_type_error_operation(type_match.group(1)),
            detail,
            _default_expected_for_failure_type("type_error"),
        )

    value_match = re.search(r"ValueError: (.+)", feedback)
    if value_match:
        detail = value_match.group(1).strip()
        return _make_failure_snapshot(
            "value_error",
            location,
            detail,
            _default_expected_for_failure_type("value_error"),
        )

    import_match = re.search(
        r"(?:ImportError|ModuleNotFoundError): (?:No module named |cannot import name )['\"]?([^'\"\n]+)",
        feedback,
    )
    if import_match:
        symbol = import_match.group(1)
        return _make_failure_snapshot(
            "import_error",
            symbol,
            message or symbol,
            _default_expected_for_failure_type("import_error"),
        )

    syntax_match = re.search(r"(SyntaxError|IndentationError): (.+)", feedback)
    if syntax_match:
        detail = syntax_match.group(2).strip()
        return _make_failure_snapshot(
            "syntax_error",
            location,
            detail,
            _default_expected_for_failure_type("syntax_error"),
        )

    numeric_match = re.search(r"(ZeroDivisionError|OverflowError|FloatingPointError): (.+)", feedback)
    if numeric_match:
        detail = numeric_match.group(2).strip()
        return _make_failure_snapshot(
            "numeric_error",
            location,
            detail,
            _default_expected_for_failure_type("numeric_error"),
        )

    resource_match = re.search(r"(MemoryError|TimeoutError|RecursionError): ?(.*)", feedback)
    if resource_match:
        detail = resource_match.group(2).strip() or resource_match.group(1)
        return _make_failure_snapshot(
            "resource_error",
            location,
            detail,
            _default_expected_for_failure_type("resource_error"),
        )

    file_match = re.search(r"(FileNotFoundError|OSError|PermissionError): (.+)", feedback)
    if file_match:
        detail = file_match.group(2).strip()
        return _make_failure_snapshot(
            "file_error",
            location,
            detail,
            _default_expected_for_failure_type("file_error"),
        )
    
    linalg_match = re.search(r"LinAlgError: (.+)", feedback)
    if linalg_match:
        detail = linalg_match.group(1).strip()
        return _make_failure_snapshot(
            "linear_algebra_error",
            location,
            detail,
            _default_expected_for_failure_type("linear_algebra_error"),
        )

    return _make_failure_snapshot(
        _normalize_failure_type(error_type, message),
        location,
        message or error_type,
        _default_expected_for_failure_type(_normalize_failure_type(error_type, message)),
    )


# 从测试反馈文本中提取异常消息正文（即冒号后面的具体描述）
def _extract_exception_message(feedback: str, error_type: str) -> str:
    if error_type == "unknown":
        return ""
    pattern = re.compile(
        rf"^(?:[A-Za-z_][A-Za-z0-9_]*\.)*{re.escape(error_type)}:?\s*(.*)$"
    )
    for line in feedback.splitlines():
        stripped = line.strip()
        match = pattern.match(stripped)
        if match:
            return match.group(1).strip()
    return ""


def _extract_key_error_symbol(message: str) -> str:
    stripped = message.strip().strip('"').strip("'")
    list_match = re.search(r"\[([^\]]+)\]\s+not in index", stripped)
    if list_match:
        keys = re.findall(r"['\"]([^'\"]+)['\"]", list_match.group(1))
        return ", ".join(keys) if keys else list_match.group(1).strip()

    direct_match = re.search(r"KeyError:\s*['\"]?([^'\"\n]+)['\"]?", stripped)
    if direct_match:
        candidate = direct_match.group(1).strip()
        if candidate:
            return candidate

    if stripped and "\n" not in stripped:
        return stripped
    return ""


# 把 pandas 特有的英文异常类名翻译成人类可读的原因描述
def _pandas_error_reason(error_type: str) -> str:
    reasons = {
        "MergeError": "invalid dataframe merge or join",
        "ParserError": "failed to parse tabular input",
        "EmptyDataError": "empty tabular input",
        "InvalidIndexError": "invalid or non-unique dataframe index",
        "OptionError": "invalid pandas option",
        "SpecificationError": "invalid pandas aggregation specification",
    }
    return reasons.get(error_type, error_type)


# 从 Python traceback 中定位崩溃的具体位置（文件名、行号、函数名）
def _extract_traceback_location(feedback: str) -> str:
    file_lines = [line.strip() for line in feedback.splitlines() if line.strip().startswith("File ")]
    if not file_lines:
        return "unknown"
    match = re.search(r'File "([^"]+)", line (\d+), in ([^\n]+)', file_lines[-1])
    if not match:
        return file_lines[-1]
    filename, line_no, func_name = match.groups()
    return f"{filename}:{line_no} in {func_name}"


# 从 TypeError 的异常消息中提取具体是什么操作导致了类型错误
def _extract_type_error_operation(message: str) -> str:
    # 模式 1：函数调用参数错误
    call_match = re.search(r"([A-Za-z_][A-Za-z0-9_\.]*)\(\) (?:got|takes|missing)", message)
    if call_match:
        return call_match.group(1)
    # 模式 2：运算符不支持
    operand_match = re.search(r"unsupported operand type\(s\) for ([^:]+):", message)
    if operand_match:
        return operand_match.group(1).strip()
    # 模式 3：对象不支持下标访问
    subscript_match = re.search(r"'([^']+)' object is not subscriptable", message)
    if subscript_match:
        return f"{subscript_match.group(1)} subscript"
    return "unknown"

# 在失败的代码中搜索某个符号（变量名、属性名）第一次出现的完整表达式
def _find_symbol_location(code: str, symbol: str) -> str:
    symbol_pattern = re.compile(rf"\b{re.escape(symbol)}\b(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
    for line in code.splitlines():
        match = symbol_pattern.search(line)
        if match:
            return match.group(0)
    return symbol


# 解析 actual vs. expected 的 DS1000 诊断文本，尽量定位第一个具体不匹配点
def _extract_first_output_mismatch(actual_block: str, expected_block: str) -> Optional[Dict[str, str]]:
    dataframe_mismatch = _extract_first_dataframe_mismatch(actual_block, expected_block)
    if dataframe_mismatch:
        return dataframe_mismatch

    type_mismatch = _extract_type_or_shape_mismatch(actual_block, expected_block)
    if type_mismatch:
        return type_mismatch

    actual_parsed, actual_value = _parse_output_value(actual_block)
    expected_parsed, expected_value = _parse_output_value(expected_block)
    if actual_parsed and expected_parsed:
        value_mismatch = _find_first_value_mismatch(actual_value, expected_value)
        if value_mismatch:
            return value_mismatch

    return _extract_first_text_mismatch(actual_block, expected_block)


# 解析两边（actual vs. expected）的 DataFrame 预览文本，逐行逐列比较，找到第一个单元格不匹配的位置并返回结构化描述
def _extract_first_dataframe_mismatch(actual_block: str, expected_block: str) -> Optional[Dict[str, str]]:
    actual_table = _parse_dataframe_preview(actual_block)
    expected_table = _parse_dataframe_preview(expected_block)
    if not actual_table or not expected_table:
        return None

    actual_columns, actual_rows = actual_table
    expected_columns, expected_rows = expected_table
    shared_columns = [col for col in actual_columns if col in expected_columns]
    for row_id, actual_values in actual_rows.items():
        expected_values = expected_rows.get(row_id)
        if expected_values is None:
            continue
        for column in shared_columns:
            actual_value = actual_values.get(column)
            expected_value = expected_values.get(column)
            if actual_value != expected_value:
                return _make_failure_snapshot(
                    "wrong_output",
                    _combine_location(f"row {row_id}", f"field {column}"),
                    actual_value or "",
                    expected_value or "",
                )
    return None


# 提取 type/shape 级别的不匹配，适合 ndarray/tensor/list 等无法进一步解析时兜底
def _extract_type_or_shape_mismatch(actual_block: str, expected_block: str) -> Optional[Dict[str, str]]:
    actual_type = _extract_output_metadata(actual_block, "type")
    expected_type = _extract_output_metadata(expected_block, "type")
    if actual_type and expected_type and actual_type != expected_type:
        return _make_failure_snapshot(
            "wrong_output",
            "output type",
            actual_type,
            expected_type,
        )

    actual_shape = _extract_output_metadata(actual_block, "shape")
    expected_shape = _extract_output_metadata(expected_block, "shape")
    if actual_shape and expected_shape and actual_shape != expected_shape:
        return _make_failure_snapshot(
            "shape_error",
            "output shape",
            actual_shape,
            expected_shape,
        )

    return None


def _extract_output_metadata(block: str, key: str) -> str:
    if key == "type":
        match = re.search(r"^[A-Za-z_]+_result:\s*type=([^\n]+)", block)
    else:
        match = re.search(rf"(?:^|\n){re.escape(key)}=([^\n]+)", block)
    return match.group(1).strip() if match else ""


def _parse_output_value(block: str) -> tuple[bool, Any]:
    text = _extract_output_value_text(block)
    if not text:
        return False, None

    literal_text = _strip_common_value_wrapper(text)
    literal_text = _normalize_literal_tokens(literal_text)
    try:
        return True, ast.literal_eval(literal_text)
    except (SyntaxError, ValueError):
        scalar = _parse_plain_scalar(literal_text)
        if scalar is not None:
            return True, scalar
    return False, None


def _extract_output_value_text(block: str) -> str:
    value_match = re.search(r"(?:^|\n)value=(.*)", block, flags=re.DOTALL)
    if not value_match:
        return ""
    value_text = value_match.group(1).strip()
    return value_text.replace("... <truncated>", "").strip()


def _strip_common_value_wrapper(text: str) -> str:
    stripped = text.strip()
    for name in (
        "tensor",
        "array",
        "matrix",
        "MaskedArray",
        "np.float64",
        "np.float32",
        "np.int64",
        "np.int32",
        "float64",
        "float32",
        "int64",
        "int32",
    ):
        call_arg = _extract_first_call_argument(stripped, name)
        if call_arg:
            return call_arg.strip()
    return stripped


def _extract_first_call_argument(text: str, name: str) -> str:
    prefix = f"{name}("
    if not text.startswith(prefix) or not text.endswith(")"):
        return ""

    content = text[len(prefix):-1]
    depth = 0
    quote = ""
    escaped = False
    for index, char in enumerate(content):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char in "([{":
            depth += 1
            continue
        if char in ")]}":
            depth -= 1
            continue
        if char == "," and depth == 0:
            return content[:index]
    return content


def _normalize_literal_tokens(text: str) -> str:
    normalized = re.sub(r"\bnan\b", "'nan'", text, flags=re.I)
    normalized = re.sub(r"\binf\b", "'inf'", normalized, flags=re.I)
    return normalized


def _parse_plain_scalar(text: str) -> Optional[Any]:
    stripped = text.strip()
    if stripped in ("True", "False"):
        return stripped == "True"
    if stripped == "None":
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return None


def _find_first_value_mismatch(actual: Any, expected: Any, path: str = "") -> Optional[Dict[str, str]]:
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key, actual_value in actual.items():
            child_path = _join_mismatch_path(path, key)
            if key not in expected:
                return _build_value_mismatch(child_path, actual_value, "<missing>")
            mismatch = _find_first_value_mismatch(actual_value, expected[key], child_path)
            if mismatch:
                return mismatch
        for key, expected_value in expected.items():
            if key not in actual:
                return _build_value_mismatch(_join_mismatch_path(path, key), "<missing>", expected_value)
        return None

    if _is_sequence(actual) and _is_sequence(expected):
        shared_len = min(len(actual), len(expected))
        for index in range(shared_len):
            mismatch = _find_first_value_mismatch(
                actual[index],
                expected[index],
                f"{path}[{index}]" if path else f"[{index}]",
            )
            if mismatch:
                return mismatch
        if len(actual) != len(expected):
            return _make_failure_snapshot(
                "shape_error",
                path or "length",
                str(len(actual)),
                str(len(expected)),
            )
        return None

    if isinstance(actual, set) and isinstance(expected, set):
        missing = sorted(expected - actual, key=repr)
        extra = sorted(actual - expected, key=repr)
        if missing or extra:
            return _make_failure_snapshot(
                "wrong_output",
                path or "set",
                {"extra": extra},
                {"missing": missing},
            )
        return None

    if actual != expected:
        return _build_value_mismatch(path or "value", actual, expected)
    return None


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _join_mismatch_path(path: str, key: Any) -> str:
    key_text = str(key)
    if not path:
        return key_text
    return f"{path}.{key_text}"


def _build_value_mismatch(path: str, actual: Any, expected: Any) -> Dict[str, str]:
    return _make_failure_snapshot(
        "wrong_output",
        path,
        actual,
        expected,
    )


def _extract_first_text_mismatch(actual_block: str, expected_block: str) -> Optional[Dict[str, str]]:
    actual_value = _extract_output_value_text(actual_block) or actual_block
    expected_value = _extract_output_value_text(expected_block) or expected_block
    if actual_value == expected_value:
        return None

    actual_lines = actual_value.splitlines()
    expected_lines = expected_value.splitlines()
    for line_no, (actual_line, expected_line) in enumerate(zip(actual_lines, expected_lines), start=1):
        if actual_line != expected_line:
            return _make_failure_snapshot(
                "wrong_output",
                f"line {line_no}",
                actual_line,
                expected_line,
            )

    return _make_failure_snapshot(
        "shape_error",
        "line count",
        str(len(actual_lines)),
        str(len(expected_lines)),
    )


def _compact_repr(value: Any, max_len: int = 240) -> str:
    text = str(value) if isinstance(value, str) else repr(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 15] + "... <truncated>"


# 把一段 DS1000 测试反馈中的 DataFrame 预览文本，解析成可编程可对比的结构化数据
def _parse_dataframe_preview(block: str) -> Optional[tuple[List[str], Dict[str, Dict[str, str]]]]:
    columns = _extract_dataframe_columns(block)
    value_text = _extract_dataframe_value_text(block)
    if not columns or not value_text:
        return None

    rows: Dict[str, Dict[str, str]] = {}
    for line in value_text.splitlines():
        parts = line.split()
        if len(parts) != len(columns) + 1:
            continue
        row_id = parts[0]
        rows[row_id] = dict(zip(columns, parts[1:]))

    if not rows:
        return None
    return columns, rows

# 从 DataFrame 预览文本中提取列名列表
def _extract_dataframe_columns(block: str) -> List[str]:
    columns_match = re.search(r"columns=\[(.*?)\]", block, flags=re.DOTALL)
    if not columns_match:
        return []
    return [
        single_quoted or double_quoted
        for single_quoted, double_quoted in re.findall(
            r"'([^']+)'|\"([^\"]+)\"", columns_match.group(1)
        )
    ]

#从 DataFrame 预览文本中提取数据正文（即去掉表头的表格数据行）
def _extract_dataframe_value_text(block: str) -> str:
    value_match = re.search(r"\nvalue=(.*)", block, flags=re.DOTALL)
    if not value_match:
        return ""
    lines = value_match.group(1).strip().splitlines()
    if len(lines) <= 1:
        return "\n".join(lines)
    return "\n".join(lines[1:])


# feedback 文本中的 label: 字段值提取器
def _extract_feedback_value(feedback: str, label: str) -> str:
    pattern = rf"{re.escape(label)}:.*?(?=\n(?:actual_result|expected_result):|\nTraceback|\n\s*File |\Z)"
    match = re.search(pattern, feedback, flags=re.DOTALL)
    if not match:
        return ""
    return match.group(0).strip()


# ============================================================================
# Episodic Memory Retrieval & Injection
# ============================================================================

import math
import os

EPISODIC_RETRIEVAL_TOP_K = int(os.getenv("DS1000_EPISODIC_RETRIEVAL_TOP_K", "3"))
EPISODIC_RETRIEVAL_CANDIDATE_K = int(
    os.getenv("DS1000_EPISODIC_RETRIEVAL_CANDIDATE_K", "15")
)
EPISODIC_RERANK_TOP_K = int(
    os.getenv("DS1000_EPISODIC_RERANK_TOP_K", "5")
)
EPISODIC_RERANK_ALPHA = float(
    os.getenv("DS1000_EPISODIC_RERANK_ALPHA", "0.3")
)
EPISODIC_EMBEDDING_MODEL = os.getenv(
    "DS1000_EPISODIC_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

EPISODIC_APPLICABILITY_JUDGE_INSTRUCTION = """
You are an applicability judge for DS1000 episodic memory records.

You will receive:
- A QUERY summary (task_family + retrieval_key extracted from the current problem).
- A PROBLEM description (the current DS1000 problem to solve).
- A list of CANDIDATE episodic memory records from past problem-solving attempts.

Each candidate records a specific failure experience — what went wrong, why, and how it was fixed. Your job is to determine which past failure experiences are relevant enough that the code generation model should be warned about them before writing a solution.

Judge a candidate as applicable when:
1. The candidate's task_family matches or overlaps with the query's task_family.
2. The candidate's repair_action provides actionable guidance that could help avoid a similar mistake.

Do NOT judge as applicable when:
- The candidate's failure is in a completely different domain or library.
- The repair_action is too specific to a different problem context.

When in doubt, INCLUDE the candidate. Warning the model about a marginally relevant past failure is less harmful than missing a highly relevant one.

Return exactly one valid JSON object and nothing else.
Do NOT use markdown.
Do NOT use code fences.

{
  "applicable_indices": [0, 2]
}

applicable_indices: 0-based indices of applicable candidates. Empty [] if truly none apply.
"""


def retrieve_applicable_episodes(
    query_skill: Dict[str, Any],
    existing_records: List[Dict[str, Any]],
    problem_prompt: str,
    model: Any,
) -> List[Dict[str, Any]]:
    """Memory Injection 入口 — 检索适用的 episodic memory 记录。

    管线（与 procedural 前三阶段统一，增加阶段四）：
      阶段1: library 过滤 → embedding 检索（retrieval_key）→ top-15
      阶段2: 重排 — score = embedding_similarity + α × task_family_match → top-5
      阶段3: LLM 批量 Applicability Judge
      阶段4: repair_success=true 优先，不足时用 false 补足 top-K

    返回 judged applicable 的 episodic 记录列表（可能为空）。
    """
    if not existing_records or model is None:
        return []

    # 用 query_skill 构造一个轻量 candidate record（仅用于检索）
    candidate = _normalize_episodic_candidate(query_skill)

    records = [_normalize_episodic_record(r) for r in existing_records]

    # 阶段1: library 过滤 + embedding 检索 → top-K_candidate（默认15）
    candidate_indices = _episodic_top_retrieval_candidate_indices(records, candidate)
    if not candidate_indices:
        return []

    # 阶段2: 重排 — score = embedding_similarity + α × task_family_match → top-K（默认5）
    reranked_indices = _episodic_rerank_candidates(records, candidate, candidate_indices)
    if not reranked_indices:
        return []

    # 阶段3: LLM 批量 Applicability Judge
    reranked_records = [records[i] for i in reranked_indices]
    applicable_indices = _episodic_llm_batch_applicability_judge(
        reranked_records, problem_prompt, query_skill, model
    )

    # 阶段4: repair_success=true 优先排序，截断到 top-K
    applicable_records = [reranked_records[i] for i in applicable_indices]
    # 成功的（有 fixed_code 可参考）排在前面
    success_records = [r for r in applicable_records if r.get("repair_success")]
    failure_records = [r for r in applicable_records if not r.get("repair_success")]
    # 优先取成功案例，不足时用失败案例补足
    top_k = EPISODIC_RETRIEVAL_TOP_K
    result = success_records[:top_k]
    if len(result) < top_k:
        result.extend(failure_records[: top_k - len(result)])
    return result


def build_episodic_injection_prompt(
    applicable_episodes: List[Dict[str, Any]],
) -> str:
    """将适用的 episodic memories 格式化为注入 prompt 的文本。
    成功修复的记录展示 fixed_code，失败的记录仅展示修复建议。
    """
    if not applicable_episodes:
        return ""

    blocks = []
    for idx, record in enumerate(applicable_episodes):
        repair_success = record.get("repair_success", False)
        status_label = "✅ Solved" if repair_success else "❌ Unsolved"

        # failure_snapshot = record.get("failure_snapshot", {})
        # if isinstance(failure_snapshot, dict):
        #     snapshot_lines = [
        #         f"  - type: {failure_snapshot.get('type', 'unknown')}",
        #         f"  - observed: {failure_snapshot.get('observed', 'unknown')}",
        #         f"  - expected: {failure_snapshot.get('expected', 'unknown')}",
        #     ]
        #     snapshot_text = "\n".join(snapshot_lines)
        # else:
        #     snapshot_text = f"  {failure_snapshot}"

        block = (
            f"### Past Failure {idx + 1} [{status_label}]\n"
            f"- **Error Type**: {_field(record, 'error_type', 'N/A')}\n"
            f"- **Wrong Assumption**: {_field(record, 'failed_assumption', 'N/A')}\n"
            f"- **Key Takeaway**: {_field(record, 'injection_card', 'N/A')}"
        )
        # # 成功修复的记录展示 fixed_code 作为正例参考
        # fixed_code = record.get("fixed_code", "")
        # if repair_success and fixed_code:
        #     block += f"\n- **Fixed Code**:\n```python\n{fixed_code}\n```"

        blocks.append(block)

    header = (
        "## Past Error Experiences from Similar Problems\n\n"
        "The following failures were encountered when solving similar DS1000 problems. "
        "Records marked ✅ Solved include the corrected code — use them as positive examples. "
        "Records marked ❌ Unsolved describe mistakes to avoid.\n\n"
    )
    return header + "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Episodic retrieval helpers
# ---------------------------------------------------------------------------


def _normalize_episodic_candidate(query_skill: Dict[str, Any]) -> Dict[str, Any]:
    """用 query_skill 构造轻量候选记录，仅用于 embedding 检索。

    与 procedural 完全对齐：query 端和 memory 端都使用 retrieval_key 做 embedding，
    格式统一为 "retrieval_key: {value}"（与 _procedural_retrieval_text 一致）。
    """
    retrieval_key = _field(query_skill, "retrieval_key", "unknown")
    task_family = _field(query_skill, "task_family", "unknown")
    return {
        "memory_type": "episodic",
        "library": _field(query_skill, "library", "unknown"),
        "task_family": task_family,
        "retrieval_key": f"retrieval_key: {retrieval_key}",
    }


def _normalize_episodic_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """规范化 episodic 记录，确保 retrieval_key 可用（与 procedural 对齐）。

    若已有 retrieval_key 字段（LLM 生成的 5-10 词语义描述），则封装为
    "retrieval_key: {value}" 格式用于 embedding。
    否则用 _build_problem_embedding_text 作为 fallback。
    """
    normalized = dict(record)
    raw_key = normalized.get("retrieval_key")
    if raw_key and isinstance(raw_key, str) and raw_key.strip():
        # LLM 已生成 retrieval_key，封装为与 procedural 一致的嵌入文本
        normalized["retrieval_key"] = f"retrieval_key: {raw_key.strip()}"
    else:
        normalized["retrieval_key"] = _build_problem_embedding_text(
            library=_field(normalized, "library", "unknown"),
            task_family=_field(normalized, "task_family", "unknown"),
            repair_action=_field(normalized, "repair_action", "unknown"),
        )
    return normalized


def _episodic_top_retrieval_candidate_indices(
    records: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    top_k: int = EPISODIC_RETRIEVAL_CANDIDATE_K,
) -> List[int]:
    """阶段1: library 过滤 + embedding 检索（retrieval_key），返回 top-K 候选索引。"""
    library_index = _episodic_build_library_index(records)
    filtered_indices = library_index.get(
        _normalized_index_value_episodic(candidate.get("library")), []
    )
    if len(filtered_indices) <= top_k:
        return filtered_indices

    try:
        retrieval_records = [candidate] + [records[i] for i in filtered_indices]
        _ensure_episodic_embeddings(retrieval_records)
    except Exception:
        return filtered_indices[:top_k]

    candidate_embedding = candidate.get("retrieval_embedding")
    scored = [
        (
            _episodic_cosine_similarity(
                candidate_embedding,
                records[i].get("retrieval_embedding"),
            ),
            i,
        )
        for i in filtered_indices
    ]

    scored.sort(key=lambda item: item[0], reverse=True)
    return [index for _, index in scored[:top_k]]


def _episodic_rerank_candidates(
    records: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    candidate_indices: List[int],
    alpha: float = EPISODIC_RERANK_ALPHA,
    top_k: int = EPISODIC_RERANK_TOP_K,
) -> List[int]:
    """阶段2: 重排 — score = embedding_similarity + α × (task_family 是否匹配)，选出 top-K。

    与 procedural 的 _rerank_candidates 逻辑完全一致：
    先对候选计算 embedding 相似度，再叠加 task_family 匹配加分。
    episodic 的 task_family 位于记录顶层（非嵌套 skill 字段）。
    """
    candidate_task_family = _normalized_index_value_episodic(
        candidate.get("task_family", "")
    )

    try:
        retrieval_records = [candidate] + [records[i] for i in candidate_indices]
        _ensure_episodic_embeddings(retrieval_records)
    except Exception:
        return candidate_indices[:top_k]

    candidate_embedding = candidate.get("retrieval_embedding")
    scored = []
    for idx in candidate_indices:
        cos_sim = _episodic_cosine_similarity(
            candidate_embedding, records[idx].get("retrieval_embedding")
        )
        existing_task_family = _normalized_index_value_episodic(
            records[idx].get("task_family", "")
        )
        task_family_match = 1.0 if candidate_task_family == existing_task_family else 0.0
        rerank_score = cos_sim + alpha * task_family_match
        scored.append((rerank_score, idx))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [index for _, index in scored[:top_k]]


def _episodic_build_library_index(
    records: List[Dict[str, Any]],
) -> Dict[str, List[int]]:
    """构建 episodic 记录的 library → indices 映射。"""
    library_index: Dict[str, List[int]] = {}
    for record_index, record in enumerate(records):
        if record.get("memory_type") not in ("episodic", None):
            continue
        library = _normalized_index_value_episodic(record.get("library"))
        library_index.setdefault(library, []).append(record_index)
    return library_index


def _normalized_index_value_episodic(value: Any) -> str:
    return str(value or "unknown").strip().lower()


def _episodic_cosine_similarity(left: Any, right: Any) -> float:
    if not isinstance(left, list) or not isinstance(right, list):
        return float("-inf")
    if len(left) != len(right) or not left:
        return float("-inf")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for lv, rv in zip(left, right):
        lf = float(lv)
        rf = float(rv)
        dot += lf * rf
        left_norm += lf * lf
        right_norm += rf * rf
    if left_norm <= 0 or right_norm <= 0:
        return float("-inf")
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def _ensure_episodic_embeddings(records: List[Dict[str, Any]]) -> None:
    """为 episodic 记录计算 embedding，使用 retrieval_key（与 procedural 对齐）。"""
    pending_records = []
    pending_texts = []
    for record in records:
        text = str(record.get("retrieval_key") or "")
        embedding_meta = record.get("retrieval_embedding_meta")
        if (
            isinstance(record.get("retrieval_embedding"), list)
            and isinstance(embedding_meta, dict)
            and embedding_meta.get("model") == EPISODIC_EMBEDDING_MODEL
            and embedding_meta.get("text") == text
        ):
            continue
        pending_records.append(record)
        pending_texts.append(text)

    if not pending_records:
        return

    from generators.model import gpt_embedding

    embeddings = gpt_embedding(EPISODIC_EMBEDDING_MODEL, pending_texts)
    for record, text, embedding in zip(pending_records, pending_texts, embeddings):
        record["retrieval_key"] = text
        record["retrieval_embedding"] = embedding
        record["retrieval_embedding_meta"] = {
            "model": EPISODIC_EMBEDDING_MODEL,
            "text": text,
        }


def _episodic_llm_batch_applicability_judge(
    candidate_records: List[Dict[str, Any]],
    problem_prompt: str,
    query_skill: Dict[str, Any],
    model: Any,
) -> List[int]:
    """LLM Applicability Judge（批量）— 判断哪些 episodic 记录与当前问题相关。"""
    if not candidate_records:
        return []

    query_summary = (
        f"task_family: {_field(query_skill, 'task_family', 'unknown')}\n"
        f"retrieval_key: {_field(query_skill, 'retrieval_key', 'unknown')}"
    )

    candidates_text = ""
    for idx, record in enumerate(candidate_records):
        candidates_text += (
            f"[candidate {idx}]:\n"
            f"task_family: {_field(record, 'task_family', 'unknown')}\n"
            f"error_type: {_field(record, 'error_type', 'unknown')}\n"
            f"failed_assumption: {_field(record, 'failed_assumption', 'unknown')}\n"
            f"repair_action: {_field(record, 'repair_action', 'unknown')}\n\n"
        )

    problem_text = str(problem_prompt)[:2000]

    user_msg = (
        f"[query summary]:\n{query_summary}\n\n"
        f"[problem]:\n{problem_text}\n\n"
        f"{candidates_text}"
        "[applicability judge json]:"
    )
    try:
        output = _generate_episodic_judge_response(model, user_msg)
    except Exception:
        return []
    return _parse_episodic_batch_judge_response(output, len(candidate_records))


def _generate_episodic_judge_response(model: Any, user_msg: str) -> str:
    if getattr(model, "is_chat", False):
        from generators.model import Message

        output = model.generate_chat(
            messages=[
                Message(role="system", content=EPISODIC_APPLICABILITY_JUDGE_INSTRUCTION),
                Message(role="user", content=user_msg),
            ],
            max_tokens=512,
            temperature=0.0,
            num_comps=1,
        )
    else:
        output = model.generate(
            f"{EPISODIC_APPLICABILITY_JUDGE_INSTRUCTION}\n\n{user_msg}",
            max_tokens=512,
            temperature=0.0,
            num_comps=1,
        )
    assert isinstance(output, str)
    return output


def _parse_episodic_batch_judge_response(
    output: str, num_candidates: int
) -> List[int]:
    import json as _json

    output = str(output).strip()
    fenced = re.search(r"```(?:json)?\n(.*?)\n```", output, re.DOTALL)
    if fenced:
        output = fenced.group(1).strip()
    json_match = re.search(r"\{.*\}", output, re.DOTALL)
    if json_match:
        output = json_match.group(0)
    try:
        parsed = _json.loads(output)
    except _json.JSONDecodeError:
        return []

    if not isinstance(parsed, dict):
        return []
    indices = parsed.get("applicable_indices")
    if not isinstance(indices, list):
        return []
    valid = []
    for idx in indices:
        try:
            i = int(idx)
            if 0 <= i < num_candidates:
                valid.append(i)
        except (ValueError, TypeError):
            continue
    return valid
