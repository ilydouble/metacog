import difflib
import json
import math
import os
import re
from typing import Any, Dict, List, Optional

from .common import (
    MEMORY_SCHEMA_VERSION,
    _field,
    _first_success_index,
    _parse_json_or_raw,
    _reflection_for_attempt,
    _stable_id,
    _utc_now,
)


PROCEDURAL_CONSOLIDATION_MAX_TOKENS = int(
    os.getenv("DS1000_PROCEDURAL_CONSOLIDATION_MAX_TOKENS", "1024")
)
PROCEDURAL_RETRIEVAL_TOP_K = int(os.getenv("DS1000_PROCEDURAL_RETRIEVAL_TOP_K", "5"))
PROCEDURAL_RETRIEVAL_CANDIDATE_K = int(
    os.getenv("DS1000_PROCEDURAL_RETRIEVAL_CANDIDATE_K", "15")
)
PROCEDURAL_RERANK_ALPHA = float(
    os.getenv("DS1000_PROCEDURAL_RERANK_ALPHA", "0.3")
)
PROCEDURAL_EMBEDDING_MODEL = os.getenv(
    "DS1000_PROCEDURAL_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

PROCEDURAL_SKILL_FIELDS = [
    "library",
    "task_family",
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

DS1000_SKILL_CONSOLIDATION_CHAT_INSTRUCTION = """
You are consolidating DS1000 procedural memory skills.

You will receive an existing procedural skill and a new candidate procedural skill.
Decide whether they describe the same reusable skill. Your only job is to make a
merge/no-merge decision. Do not rewrite, improve, summarize, or consolidate the
skill text.

Only merge when ALL of the following are true:

1. Same task intent.
2. Same applicability conditions.
3. Same reusable algorithmic procedure.
4. Same core reasoning pattern.

Do NOT merge skills that only share a broad library, task family, or superficial
API names while requiring different reasoning or different applicability
conditions.

Different algorithmic strategies should NOT be merged even if they solve similar
tasks.

When in doubt, return false. False positives are more harmful than false
negatives because over-merging causes skill drift.

Return exactly one valid JSON object and nothing else.
Do NOT use markdown.
Do NOT use code fences.

The JSON object must contain exactly these fields:

{
  "same_skill": true
}
"""

DS1000_APPLICABILITY_JUDGE_INSTRUCTION = """
You are an applicability judge for DS1000 procedural memory skills.

You will receive:
- A QUERY summary (task_family + retrieval_key extracted from the current problem).
- A PROBLEM description (the current DS1000 problem to solve).
- A list of CANDIDATE procedural skills retrieved from memory.

The candidates were retrieved by embedding similarity on retrieval_key — your job
is to confirm which ones are genuinely applicable. The query summary provides a
semantic bridge between the concrete problem and the abstract skills.

For EACH candidate, check whether its task_family matches the query's task_family
and whether its procedure is relevant to the query's retrieval_key. Then verify
against the problem description.

Judge a candidate as applicable when:
1. The candidate's task_family is the same as the query's task_family, OR the
   procedure clearly addresses the same algorithmic goal described in the query.
2. The avoid_assumption addresses a pitfall that could plausibly occur.
3. The preserve_constraints do not conflict with the problem requirements.

Do NOT judge as applicable ONLY when there is a clear conflict:
- The candidate targets a different algorithmic goal entirely.
- The preserve_constraints directly contradict the problem requirements.

When in doubt, INCLUDE the candidate. The code generation model has strong judgment
and can ignore irrelevant advice — but missing a relevant skill causes the same
mistake to be repeated unnecessarily.

Return exactly one valid JSON object and nothing else.
Do NOT use markdown.
Do NOT use code fences.

{
  "applicable_indices": [0, 2]
}

applicable_indices: 0-based indices of applicable candidates. Empty [] if truly none apply.
"""


def build_ds1000_procedural_memory(
    *,
    item: Dict[str, Any],
    item_index: int,
    implementations: List[str],
    test_feedback: List[str],
    reflections: List[str],
    reflection_attempt_indices: Optional[List[int]] = None,
    final_solution: str,
    procedural_skill: str,
) -> Dict[str, Any]:
    metadata = item.get("metadata", {})
    success_index = _first_success_index(test_feedback)
    if success_index is not None:
        failed_index = success_index - 1 if success_index > 0 else None
    else:
        failed_index = len(test_feedback) - 1 if test_feedback else None
    failed_code = implementations[failed_index] if failed_index is not None else ""
    reflection = _reflection_for_attempt(reflections, reflection_attempt_indices, failed_index)
    skill = _sanitize_procedural_skill(_parse_json_or_raw(procedural_skill))
    evidence = _build_procedural_evidence(
        item=item,
        item_index=item_index,
        reflection=reflection,
        failed_code=failed_code,
        final_solution=final_solution,
        test_feedback=test_feedback,
    )
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "memory_type": "procedural",
        "created_at": _utc_now(),
        "skill_id": _procedural_skill_id(skill),
        "support_count": 1,
        "dataset": "ds1000",
        "item_index": item_index,
        "problem_id": metadata.get("problem_id"),
        "library_problem_id": metadata.get("library_problem_id"),
        "library": metadata.get("library") or _field(skill, "library", "unknown"),
        "skill": skill,
        "evidence": [evidence],
    }


def retrieve_applicable_skills(
    query_skill: Dict[str, Any],
    existing_records: List[Dict[str, Any]],
    problem_prompt: str,
    model: Any,
) -> List[Dict[str, Any]]:
    """Memory Injection 入口 — 检索适用的 procedural memory skills。

    管线：library 过滤 → embedding 检索 top15 → 重排 top5 → LLM 批量 Applicability Judge。
    Judge 阶段直接使用 problem_prompt 原文与候选 skill 做全文比较，避免 LLM 提前推理。
    返回 judged applicable 的 skill 记录列表（可能为空）。
    """
    if not existing_records or model is None:
        return []

    # 用 query_skill 构造一个轻量 candidate record（仅用于检索和重排）
    candidate = _normalize_procedural_record({
        "memory_type": "procedural",
        "library": _field(query_skill, "library", "unknown"),
        "skill": _sanitize_procedural_skill(query_skill),
    })

    records = [_normalize_procedural_record(r) for r in existing_records]

    # 阶段1: library 过滤 + embedding 检索 → top15
    candidate_indices = _top_retrieval_candidate_indices(records, candidate)
    if not candidate_indices:
        return []

    # 阶段2: 重排 → top5
    reranked_indices = _rerank_candidates(records, candidate, candidate_indices)
    if not reranked_indices:
        return []

    # 阶段3: LLM 批量 Applicability Judge（query_skill 作语义桥梁 + problem 原文验证）
    reranked_records = [records[i] for i in reranked_indices]
    applicable_indices = _llm_batch_applicability_judge(
        reranked_records, problem_prompt, query_skill, model
    )

    return [reranked_records[i] for i in applicable_indices]


def build_memory_injection_prompt(
    applicable_skills: List[Dict[str, Any]],
) -> str:
    """将适用的 procedural skills 格式化为注入 prompt 的文本。"""
    if not applicable_skills:
        return ""

    blocks = []
    for idx, record in enumerate(applicable_skills):
        skill = record.get("skill", {})
        blocks.append(
            f"### Reusable Skill {idx + 1}\n"
            f"- **Procedure**: {_field(skill, 'procedure', 'N/A')}\n"
            f"- **Avoid**: {_field(skill, 'avoid_assumption', 'N/A')}\n"
            f"- **Preserve**: {_field(skill, 'preserve_constraints', 'N/A')}\n"
            f"- **Verification**: {_field(skill, 'verification', 'N/A')}"
        )

    header = (
        "## Reusable Procedural Skills from Past Repairs\n\n"
        "The following skills were extracted from previously solved DS1000 problems "
        "and judged applicable to the current problem. Use them to guide your solution "
        "and avoid known pitfalls.\n\n"
    )
    return header + "\n\n".join(blocks)


def consolidate_ds1000_procedural_memory(
    existing_records: List[Dict[str, Any]],
    candidate_record: Dict[str, Any],
    model: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    candidate = _normalize_procedural_record(candidate_record)
    records = [_normalize_procedural_record(record) for record in existing_records]
    if model is None:
        return records + [candidate]

    # 阶段1: library 过滤 + embedding 检索（仅用 retrieval_key），选出 top-K_candidate（默认15）
    candidate_indices = _top_retrieval_candidate_indices(records, candidate)

    # 阶段2: 重排 — score = embedding_similarity + α * task_family_match，选出 top-K（默认5）
    reranked_indices = _rerank_candidates(records, candidate, candidate_indices)

    for index in reranked_indices:
        existing = records[index]
        if existing.get("memory_type") != "procedural":
            continue

        # 阶段3: LLM Applicability Judge（全文判断 procedure + avoid_assumption + preserve_constraints）
        judge_result = _llm_applicability_judge(existing, candidate, model)
        if not judge_result.get("applicable"):
            continue

        # 阶段4: LLM Consolidation 决定是否合并
        decision = _llm_consolidation_decision(existing, candidate, model)
        if not decision.get("same_skill"):
            continue
        merged = _merge_procedural_records(existing, candidate)
        records[index] = merged
        return records

    return records + [candidate]


def _build_procedural_evidence(
    *,
    item: Dict[str, Any],
    item_index: int,
    reflection: Optional[str],
    failed_code: str,
    final_solution: str,
    test_feedback: List[str],
) -> Dict[str, Any]:
    metadata = item.get("metadata", {})
    success_verification = "All DS1000 tests passed."
    patch_diff = build_patch_diff(failed_code, final_solution)
    return {
        "evidence_id": _stable_id(
            "procedural_evidence",
            item_index,
            metadata.get("problem_id"),
            failed_code,
            final_solution,
            success_verification,
        ),
        "item_index": item_index,
        "problem_id": metadata.get("problem_id"),
        "library_problem_id": metadata.get("library_problem_id"),
        "reflection": _parse_json_or_raw(reflection),
        "success_patch": final_solution,
        "patch_diff": patch_diff,
        "success_verification": success_verification,
    }


def _procedural_skill_id(skill: Any) -> str:
    return _stable_id(
        "procedural",
        _field(skill, "library", "unknown"),
        _field(skill, "task_family", "unknown"),
        _field(skill, "retrieval_key", "unknown"),
        _field(skill, "procedure", "unknown"),
    )


def _normalize_procedural_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(record)
    skill = _sanitize_procedural_skill(normalized.get("skill"))
    normalized["skill"] = skill
    normalized["skill_id"] = normalized.get("skill_id") or _procedural_skill_id(skill)
    normalized["retrieval_text"] = _procedural_retrieval_text(skill)
    evidence = _dedupe_evidence(_evidence_entries(normalized))
    normalized["evidence"] = evidence
    normalized["support_count"] = len(evidence) if evidence else int(
        normalized.get("support_count") or 0
    )
    return normalized


def _top_retrieval_candidate_indices(
    records: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    top_k: int = PROCEDURAL_RETRIEVAL_CANDIDATE_K,
) -> List[int]:
    """阶段1: library 过滤 + embedding 检索（仅 retrieval_key），返回 top-K 候选索引。"""
    library_index = _build_procedural_library_index(records)
    filtered_indices = library_index.get(_normalized_index_value(candidate.get("library")), [])
    if len(filtered_indices) <= top_k:
        return filtered_indices

    try:
        retrieval_records = [candidate] + [records[index] for index in filtered_indices]
        _ensure_retrieval_embeddings(retrieval_records)
    except Exception:
        return filtered_indices[:top_k]

    candidate_embedding = candidate.get("retrieval_embedding")
    scored = [
        (
            _cosine_similarity(candidate_embedding, records[index].get("retrieval_embedding")),
            index,
        )
        for index in filtered_indices
    ]

    scored.sort(key=lambda item: item[0], reverse=True)
    return [index for _, index in scored[:top_k]]


def _rerank_candidates(
    records: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    candidate_indices: List[int],
    alpha: float = PROCEDURAL_RERANK_ALPHA,
    top_k: int = PROCEDURAL_RETRIEVAL_TOP_K,
) -> List[int]:
    """阶段2: 重排 — score = embedding_similarity + α * (task_family 是否匹配)，选出 top-K。

    先对候选计算 embedding 相似度，再叠加 task_family 匹配加分。
    """
    candidate_task_family = _normalized_index_value(
        _field(candidate.get("skill"), "task_family", "")
    )

    try:
        retrieval_records = [candidate] + [records[index] for index in candidate_indices]
        _ensure_retrieval_embeddings(retrieval_records)
    except Exception:
        return candidate_indices[:top_k]

    candidate_embedding = candidate.get("retrieval_embedding")
    scored = []
    for index in candidate_indices:
        cos_sim = _cosine_similarity(
            candidate_embedding, records[index].get("retrieval_embedding")
        )
        existing_skill = records[index].get("skill")
        existing_task_family = _normalized_index_value(
            _field(existing_skill, "task_family", "")
        )
        task_family_match = 1.0 if candidate_task_family == existing_task_family else 0.0
        rerank_score = cos_sim + alpha * task_family_match
        scored.append((rerank_score, index))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [index for _, index in scored[:top_k]]


def _build_procedural_library_index(
    records: List[Dict[str, Any]],
) -> Dict[str, List[int]]:
    library_index: Dict[str, List[int]] = {}
    for record_index, record in enumerate(records):
        if record.get("memory_type") != "procedural":
            continue
        library = _normalized_index_value(record.get("library"))
        library_index.setdefault(library, []).append(record_index)
    return library_index


def _normalized_index_value(value: Any) -> str:
    return str(value or "unknown").strip().lower()


def _procedural_retrieval_text(skill: Any) -> str:
    """仅使用 retrieval_key 做 embedding 检索，task_family 留到重排阶段使用。"""
    return f"retrieval_key: {_field(skill, 'retrieval_key', 'unknown')}"


def _procedural_applicability_judge_text(skill: Any) -> str:
    """包含完整判断字段，留给 LLM Applicability Judge 阶段做全文判断。"""
    return "\n".join(
        [
            f"procedure: {_field(skill, 'procedure', 'unknown')}",
            f"avoid_assumption: {_field(skill, 'avoid_assumption', 'unknown')}",
            f"preserve_constraints: {_field(skill, 'preserve_constraints', 'unknown')}",
        ]
    )


def _ensure_retrieval_embeddings(records: List[Dict[str, Any]]) -> None:
    pending_records = []
    pending_texts = []
    for record in records:
        text = str(record.get("retrieval_text") or _procedural_retrieval_text(record.get("skill")))
        embedding_meta = record.get("retrieval_embedding_meta")
        if (
            isinstance(record.get("retrieval_embedding"), list)
            and isinstance(embedding_meta, dict)
            and embedding_meta.get("model") == PROCEDURAL_EMBEDDING_MODEL
            and embedding_meta.get("text") == text
        ):
            continue
        pending_records.append(record)
        pending_texts.append(text)

    if not pending_records:
        return

    from generators.model import gpt_embedding

    embeddings = gpt_embedding(PROCEDURAL_EMBEDDING_MODEL, pending_texts)
    for record, text, embedding in zip(pending_records, pending_texts, embeddings):
        record["retrieval_text"] = text
        record["retrieval_embedding"] = embedding
        record["retrieval_embedding_meta"] = {
            "model": PROCEDURAL_EMBEDDING_MODEL,
            "text": text,
        }


def _cosine_similarity(left: Any, right: Any) -> float:
    if not isinstance(left, list) or not isinstance(right, list):
        return float("-inf")
    if len(left) != len(right) or not left:
        return float("-inf")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        left_float = float(left_value)
        right_float = float(right_value)
        dot += left_float * right_float
        left_norm += left_float * left_float
        right_norm += right_float * right_float
    if left_norm <= 0 or right_norm <= 0:
        return float("-inf")
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def _llm_batch_applicability_judge(
    candidate_records: List[Dict[str, Any]],
    problem_prompt: str,
    query_skill: Dict[str, Any],
    model: Any,
) -> List[int]:
    """LLM Applicability Judge（批量）— query_skill 作语义桥梁 + problem 原文做全文判断。"""
    if not candidate_records:
        return []

    query_summary = (
        f"task_family: {_field(query_skill, 'task_family', 'unknown')}\n"
        f"retrieval_key: {_field(query_skill, 'retrieval_key', 'unknown')}"
    )

    candidates_text = ""
    for idx, record in enumerate(candidate_records):
        existing_skill = record.get("skill")
        candidate_task_family = _field(existing_skill, "task_family", "unknown")
        candidate_text = _procedural_applicability_judge_text(existing_skill)
        candidates_text += (
            f"[candidate {idx}]:\n"
            f"task_family: {candidate_task_family}\n"
            f"{candidate_text}\n\n"
        )

    problem_text = str(problem_prompt)[:2000]

    user_msg = (
        f"[query summary]:\n{query_summary}\n\n"
        f"[problem]:\n{problem_text}\n\n"
        f"{candidates_text}"
        "[applicability judge json]:"
    )
    try:
        output = _generate_judge_response(model, user_msg)
    except Exception:
        return []
    return _parse_batch_judge_response(output, len(candidate_records))


def _parse_batch_judge_response(output: str, num_candidates: int) -> List[int]:
    parsed = _parse_json_object(output)
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


def _llm_applicability_judge(
    existing_record: Dict[str, Any],
    candidate_record: Dict[str, Any],
    model: Any,
) -> Dict[str, Any]:
    """Consolidation 专用 skill-to-skill 比较 — 判断两个 skill 是否描述同一可复用技能。"""
    existing_skill = existing_record.get("skill")
    candidate_skill = candidate_record.get("skill")
    existing_text = _procedural_applicability_judge_text(existing_skill)
    candidate_text = _procedural_applicability_judge_text(candidate_skill)
    user_msg = (
        "[existing skill]:\n"
        f"{existing_text}\n\n"
        "[candidate skill]:\n"
        f"{candidate_text}\n\n"
        "[consolidation judge json]:"
    )
    try:
        output = _generate_consolidation_judge_response(model, user_msg)
    except Exception:
        return {"applicable": False}
    return _parse_consolidation_judge_response(output)


CONSOLIDATION_JUDGE_INSTRUCTION = """
You are a consolidation judge for DS1000 procedural memory skills.
Compare an existing skill with a candidate skill. Judge whether the candidate
describes the same reusable skill and should be merged with the existing one.

Judge as applicable (merge) when ALL of the following are true:
1. The core algorithmic procedure is fundamentally the same.
2. The avoid_assumption describes the same class of pitfall.
3. The preserve_constraints are compatible.

Do NOT judge as applicable when:
- The procedures differ in algorithmic approach.
- The skills target different sub-problems.
- The avoid_assumption or preserve_constraints conflict.

When in doubt, return false.

Return exactly one valid JSON object and nothing else.
Do NOT use markdown. Do NOT use code fences.
{"applicable": true}
"""


def _generate_consolidation_judge_response(model: Any, user_msg: str) -> str:
    if getattr(model, "is_chat", False):
        from generators.model import Message

        output = model.generate_chat(
            messages=[
                Message(role="system", content=CONSOLIDATION_JUDGE_INSTRUCTION),
                Message(role="user", content=user_msg),
            ],
            max_tokens=PROCEDURAL_CONSOLIDATION_MAX_TOKENS,
            temperature=0.0,
            num_comps=1,
        )
    else:
        output = model.generate(
            f"{CONSOLIDATION_JUDGE_INSTRUCTION}\n\n{user_msg}",
            max_tokens=PROCEDURAL_CONSOLIDATION_MAX_TOKENS,
            temperature=0.0,
            num_comps=1,
        )
    assert isinstance(output, str)
    return output


def _parse_consolidation_judge_response(output: str) -> Dict[str, Any]:
    parsed = _parse_json_object(output)
    if not isinstance(parsed, dict):
        return {"applicable": False}
    return {"applicable": _coerce_bool(parsed.get("applicable"))}


def _generate_judge_response(model: Any, user_msg: str) -> str:
    if getattr(model, "is_chat", False):
        from generators.model import Message

        output = model.generate_chat(
            messages=[
                Message(role="system", content=DS1000_APPLICABILITY_JUDGE_INSTRUCTION),
                Message(role="user", content=user_msg),
            ],
            max_tokens=PROCEDURAL_CONSOLIDATION_MAX_TOKENS,
            temperature=0.0,
            num_comps=1,
        )
    else:
        output = model.generate(
            f"{DS1000_APPLICABILITY_JUDGE_INSTRUCTION}\n\n{user_msg}",
            max_tokens=PROCEDURAL_CONSOLIDATION_MAX_TOKENS,
            temperature=0.0,
            num_comps=1,
        )
    assert isinstance(output, str)
    return output


def _llm_consolidation_decision(
    existing_record: Dict[str, Any],
    candidate_record: Dict[str, Any],
    model: Any,
) -> Dict[str, Any]:
    existing_skill = _skill_for_prompt(existing_record)
    candidate_skill = _skill_for_prompt(candidate_record)
    user_msg = (
        "[existing skill]:\n"
        f"{json.dumps(existing_skill, ensure_ascii=False, indent=2)}\n\n"
        "[candidate skill]:\n"
        f"{json.dumps(candidate_skill, ensure_ascii=False, indent=2)}\n\n"
        "[consolidation json]:"
    )
    try:
        output = _generate_consolidation_response(model, user_msg)
    except Exception:
        return {"same_skill": False}
    return _parse_consolidation_response(output)


def _generate_consolidation_response(model: Any, user_msg: str) -> str:
    if getattr(model, "is_chat", False):
        from generators.model import Message

        output = model.generate_chat(
            messages=[
                Message(role="system", content=DS1000_SKILL_CONSOLIDATION_CHAT_INSTRUCTION),
                Message(role="user", content=user_msg),
            ],
            max_tokens=PROCEDURAL_CONSOLIDATION_MAX_TOKENS,
            temperature=0.0,
            num_comps=1,
        )
    else:
        output = model.generate(
            f"{DS1000_SKILL_CONSOLIDATION_CHAT_INSTRUCTION}\n\n{user_msg}",
            max_tokens=PROCEDURAL_CONSOLIDATION_MAX_TOKENS,
            temperature=0.0,
            num_comps=1,
        )
    assert isinstance(output, str)
    return output


def _parse_consolidation_response(output: str) -> Dict[str, Any]:
    parsed = _parse_json_object(output)
    if not isinstance(parsed, dict):
        return {"same_skill": False}

    return {"same_skill": _coerce_bool(parsed.get("same_skill"))}


def _parse_json_object(output: str) -> Any:
    output = str(output).strip()
    fenced = re.search(r"```(?:json)?\n(.*?)\n```", output, re.DOTALL)
    if fenced:
        output = fenced.group(1).strip()
    json_match = re.search(r"\{.*\}", output, re.DOTALL)
    if json_match:
        output = json_match.group(0)
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1", "same", "merge"}


def _skill_for_prompt(record: Dict[str, Any]) -> Dict[str, str]:
    skill = record.get("skill")
    if not isinstance(skill, dict):
        return {"library": str(record.get("library", "unknown")), "procedure": str(skill)}
    return _normalize_skill_fields(skill)


def _normalize_skill_fields(
    skill: Dict[str, Any],
    defaults: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    defaults = defaults or {}
    normalized = {}
    for field in PROCEDURAL_SKILL_FIELDS:
        default = defaults.get(field, "unknown")
        if field == "library":
            default = defaults.get(field) or "unknown"
        normalized[field] = str(skill.get(field, default)).strip() or default
    return normalized


def _merge_procedural_records(
    existing: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(existing)
    merged["skill"] = existing.get("skill")
    merged["library"] = existing.get("library", "unknown")
    merged["updated_at"] = _utc_now()
    merged["evidence"] = _dedupe_evidence(
        _evidence_entries(existing) + _evidence_entries(candidate)
    )
    merged["support_count"] = len(merged["evidence"])
    merged["merged_skill_ids"] = _dedupe_strings(
        [existing.get("skill_id"), candidate.get("skill_id")]
        + list(existing.get("merged_skill_ids") or [])
        + list(candidate.get("merged_skill_ids") or [])
    )
    return merged


def _evidence_entries(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence = record.get("evidence")
    if isinstance(evidence, list):
        entries = evidence
    elif isinstance(evidence, dict):
        entries = [evidence]
    elif evidence:
        entries = [{"raw": str(evidence)}]
    else:
        entries = []

    normalized = []
    for entry in entries:
        if not isinstance(entry, dict):
            entry = {"raw": str(entry)}
        entry = dict(entry)
        entry.setdefault("evidence_id", _stable_id("procedural_evidence", entry))
        normalized.append(entry)
    return normalized


def _dedupe_evidence(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped = []
    seen = set()
    for entry in entries:
        key = str(entry.get("evidence_id") or _stable_id("procedural_evidence", entry))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _dedupe_strings(values: List[Any]) -> List[str]:
    deduped = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped

# 构建patch diff，失败代码到成功代码的改动
def build_patch_diff(before_code: str, after_code: str) -> str:
    return "".join(
        difflib.unified_diff(
            before_code.splitlines(keepends=True),
            after_code.splitlines(keepends=True),
            fromfile="failed_code",
            tofile="successful_code",
            lineterm="",
        )
    )



# 构建 procedural memory 记录里的 skill 字段
def _sanitize_procedural_skill(skill: Any) -> Any:
    if not isinstance(skill, dict):
        return skill

    normalized = dict(skill)
    # 移除 task_intent 字段
    normalized.pop("task_intent", None)

    # 检查 procedure 是否为空
    if not str(normalized.get("procedure", "")).strip():
        # 若为空，尝试从 implementation_steps 取内容
        procedure = normalized.pop("implementation_steps", None)
        # 若仍为空，再尝试从 repair_rule 取内容
        if procedure is None:
            procedure = normalized.pop("repair_rule", None)
        # 如果找到了内容，写入 normalized["procedure"]
        if procedure is not None:
            normalized["procedure"] = str(procedure).strip()

    # 统一移除冗余字段：failure_signature、failed_assumption、repair_rule、implementation_steps
    normalized.pop("failure_signature", None)
    normalized.pop("failed_assumption", None)
    normalized.pop("repair_rule", None)
    normalized.pop("implementation_steps", None)
    return normalized
