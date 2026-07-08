import os
import json
from typing import Any, Dict, List, Optional

from memory.common import _is_passing_feedback
from memory.episodic_memory import (
    build_episode_id,
    build_ds1000_episodic_archive,
    build_ds1000_episodic_memory,
    build_ds1000_episodic_memory_from_llm,
    build_ds1000_successful_example,
    retrieve_applicable_episodes,
    build_episodic_injection_prompt,
)
from memory.procedural_memory import build_patch_diff, build_ds1000_procedural_memory
from memory.procedural_memory import consolidate_ds1000_procedural_memory
from memory.procedural_memory import retrieve_applicable_skills, build_memory_injection_prompt


def get_ds1000_memory_paths(log_path: str) -> Dict[str, str]:
    log_dir = os.path.dirname(log_path)
    log_name = os.path.splitext(os.path.basename(log_path))[0]
    memory_dir = os.path.join(log_dir, "memory")
    return {
        "episodic": os.path.join(memory_dir, f"{log_name}.episodic_memory.json"),
        "episodic_archive": os.path.join(memory_dir, f"{log_name}.episodic_archive.json"),
        "procedural": os.path.join(memory_dir, f"{log_name}.procedural_memory.json"),
        "successful_examples": os.path.join(memory_dir, f"{log_name}.successful_examples.json"),
    }


def store_ds1000_memories(
    *,
    log_path: str,
    item: Dict[str, Any],
    item_index: int,
    implementations: List[str],
    test_feedback: List[str],
    reflections: List[str],
    reflection_attempt_indices: Optional[List[int]] = None,
    is_solved: bool,
    final_solution: str,
    procedural_skill: Optional[str] = None,
    procedural_consolidation_model: Optional[Any] = None,
    episodic_memory: Optional[str] = None,
    episodic_memory_enabled: bool = True,
    procedural_skill_enabled: bool = True,
) -> Dict[str, Optional[str]]:
    paths = get_ds1000_memory_paths(log_path)
    os.makedirs(os.path.dirname(paths["episodic"]), exist_ok=True)
    written: Dict[str, Optional[str]] = {key: None for key in paths}

    has_failure = _has_failed_attempt(test_feedback)
    # 如果有失败，且 episodic memory 开关开启
    if has_failure and episodic_memory_enabled:
        episode_id = build_episode_id(item, item_index, implementations, test_feedback)
        if episodic_memory:
            # 使用 LLM 生成的 episodic memory（包含 Stage A 的 root_cause/repair_action + Stage B 的分类标签）
            episodic_record = build_ds1000_episodic_memory_from_llm(
                episode_id=episode_id,
                item=item,
                item_index=item_index,
                implementations=implementations,
                test_feedback=test_feedback,
                reflections=reflections,
                reflection_attempt_indices=reflection_attempt_indices,
                is_solved=is_solved,
                final_solution=final_solution,
                llm_episodic_memory=episodic_memory,
            )
        else:
            episodic_record = build_ds1000_episodic_memory(
                episode_id=episode_id,
                item=item,
                item_index=item_index,
                implementations=implementations,
                test_feedback=test_feedback,
                reflections=reflections,
                reflection_attempt_indices=reflection_attempt_indices,
                is_solved=is_solved,
                final_solution=final_solution,
            )
        _write_jsonl(
            paths["episodic"],
            [episodic_record],
        )
        _write_jsonl(
            paths["episodic_archive"],
            [
                build_ds1000_episodic_archive(
                    episode_id=episode_id,
                    item=item,
                    item_index=item_index,
                    implementations=implementations,
                    test_feedback=test_feedback,
                    reflections=reflections,
                    reflection_attempt_indices=reflection_attempt_indices,
                    is_solved=is_solved,
                    final_solution=final_solution,
                )
            ],
        )
        written["episodic"] = paths["episodic"]
        written["episodic_archive"] = paths["episodic_archive"]

    if is_solved and not has_failure:
        _write_jsonl(
            paths["successful_examples"],
            [
                build_ds1000_successful_example(
                    item=item,
                    item_index=item_index,
                    implementation=final_solution,
                    success_feedback="All DS1000 tests passed.",
                )
            ],
        )
        written["successful_examples"] = paths["successful_examples"]

    if is_solved and has_failure and procedural_skill and procedural_skill_enabled:
        candidate_record = build_ds1000_procedural_memory(
            item=item,
            item_index=item_index,
            implementations=implementations,
            test_feedback=test_feedback,
            reflections=reflections,
            reflection_attempt_indices=reflection_attempt_indices,
            final_solution=final_solution,
            procedural_skill=procedural_skill,
        )
        consolidated_records = consolidate_ds1000_procedural_memory(
            _read_json_records(paths["procedural"]),
            candidate_record,
            model=procedural_consolidation_model,
        )
        _write_json_records(paths["procedural"], consolidated_records)
        written["procedural"] = paths["procedural"]

    return written


# # 列表中取出最后一条反馈，用于证明"最终这次尝试确实通过了测试"
# def _last_feedback(test_feedback: List[str]) -> str:
#     return test_feedback[-1] if test_feedback else ""


# 判断是否存在至少一次失败的代码尝试
def _has_failed_attempt(test_feedback: List[str]) -> bool:
    # 只要存在任何一条不通过的反馈，any() 就会短路并返回 True
    return any(not _is_passing_feedback(feedback) for feedback in test_feedback)


def _write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    if path.endswith(".json"):
        existing = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                existing = json.load(file)
            if not isinstance(existing, list):
                raise ValueError(f"File `{path}` must contain a JSON list.")
        existing.extend(records)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(existing, file, ensure_ascii=False, indent=2)
            file.write("\n")
        return

    with open(path, "a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_json_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as file:
        if path.endswith(".json"):
            records = json.load(file)
        else:
            records = [json.loads(line) for line in file if line.strip()]

    if not isinstance(records, list):
        raise ValueError(f"File `{path}` must contain a JSON list.")
    return records


def _write_json_records(path: str, records: List[Dict[str, Any]]) -> None:
    if path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)
            file.write("\n")
        return

    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
