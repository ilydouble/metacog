import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


MEMORY_SCHEMA_VERSION = "ds1000-memory-v2"


# 通过"代码尝试的索引"快速查找对应的反思内容
def _reflection_for_attempt(
    reflections: List[str],
    reflection_attempt_indices: Optional[List[int]],
    attempt_index: Optional[int],
) -> Optional[str]:
    if attempt_index is None:
        return None
    return _reflection_by_attempt(reflections, reflection_attempt_indices).get(attempt_index)


# 建立"代码尝试索引 → 对应反思内容"的映射字典
def _reflection_by_attempt(
    reflections: List[str], reflection_attempt_indices: Optional[List[int]] = None
) -> Dict[int, str]:
    if reflection_attempt_indices is None:
        reflection_attempt_indices = list(range(len(reflections)))
    return {
        attempt_index: reflections[reflection_index]
        for reflection_index, attempt_index in enumerate(reflection_attempt_indices)
        if reflection_index < len(reflections)
    }

# 找到第一次测试通过的索引
def _first_success_index(test_feedback: List[str]) -> Optional[int]:
    for index, feedback in enumerate(test_feedback):
        if _is_passing_feedback(feedback):
            return index
    return None


# 判断单次执行反馈是否通过了测试
def _is_passing_feedback(feedback: str) -> bool:
    return (
        "DS1000 code_context test" in feedback       # 至少有一个测试通过
        and "Tests failed:" in feedback
        and feedback.rstrip().endswith("Tests failed:")  # Tests failed: 后面为空意味着没有失败的测试
    )


# 安全地从字典中提取字段
def _field(value: Any, key: str, default: str) -> str:
    if isinstance(value, dict):
        field_value = value.get(key)
        if field_value is not None and str(field_value).strip():
            return str(field_value).strip()
    return default


# 安全解析 JSON 字符串
def _parse_json_or_raw(value: Optional[str]) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {"raw": str(value)}


# 传入的内容相同，生成的id就相同
def _stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


# 获取当前UTC时间的ISO格式字符串
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

# 列表中取出最后一条反馈，用于证明"最终这次尝试确实通过了测试"
def _last_feedback(test_feedback: List[str]) -> str:
    return test_feedback[-1] if test_feedback else ""
