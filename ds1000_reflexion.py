import os
from typing import Any, List

from executors.ds1000_executor import DS1000Executor
from generators.ds1000_generate import DS1000Generator
from generators import model_factory
from memory.episodic_memory import _decisive_failure_index
from memory_store import (
    build_patch_diff,
    build_memory_injection_prompt,
    build_episodic_injection_prompt,
    get_ds1000_memory_paths,
    retrieve_applicable_skills,
    retrieve_applicable_episodes,
    store_ds1000_memories,
    _read_json_records,
)
from memory.procedural_memory import _parse_json_or_raw
from utils import enumerate_resume, make_printv, resume_success_count, write_jsonl


def run_ds1000_simple(
    dataset: List[dict],
    model_name: str,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    episodic_memory: bool = True,
    procedural_skill: bool = True,
) -> None:
    exe = DS1000Executor()
    gen = DS1000Generator()
    model = model_factory(model_name)
    stdout_log_path = os.path.splitext(log_path)[0] + ".log"
    # print_v = make_printv(verbose)
    print_v = make_printv(verbose, stdout_log_path)


    num_items = len(dataset)
    num_success = resume_success_count(dataset, log_path)
    for i, item in enumerate_resume(dataset, log_path):
        print_v(f"\n[bold]Solving Item {i+1}...[/bold]")
        cur_pass = 0
        is_solved = False
        cur_impl = ""

        while cur_pass < pass_at_k and not is_solved:
            cur_impl = gen.func_impl(item["prompt"], model, "simple")
            assert isinstance(cur_impl, str)
            is_solved = exe.evaluate("", cur_impl, item["code_context"], timeout=10)
            if is_solved:
                print_v(f"[success]Item {i+1} SOLVED[/success]")
            else:
                print_v(f"[danger]Item {i+1} FAILED[/danger]")
            num_success += int(is_solved)
            cur_pass += 1

        item["solution"] = cur_impl
        item["is_solved"] = is_solved
        write_jsonl(log_path, [item], append=True)
        print_v(f"completed {i+1}/{num_items}: solved={num_success}/{i+1}, acc = {round(num_success/(i+1), 4)}")


def run_ds1000_reflexion(
    dataset: List[dict],
    model_name: str,
    reflection_model_name: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    episodic_memory: bool = True,
    procedural_skill: bool = True,
) -> None:
    exe = DS1000Executor()
    gen = DS1000Generator()
    model = model_factory(model_name)
    reflection_model = model_factory(reflection_model_name)
    stdout_log_path = os.path.splitext(log_path)[0] + ".log"
    # print_v = make_printv(verbose)
    print_v = make_printv(verbose, stdout_log_path)

    # 加载已有的 procedural memory 和 episodic memory 记录
    memory_paths = get_ds1000_memory_paths(log_path)
    procedural_records = _read_json_records(memory_paths["procedural"])
    episodic_records = _read_json_records(memory_paths["episodic"])

    # 保存 CLI 开关状态，避免被局部变量遮蔽
    episodic_memory_enabled = episodic_memory
    procedural_skill_enabled = procedural_skill

    num_items = len(dataset)
    num_success = resume_success_count(dataset, log_path)
    for i, item in enumerate_resume(dataset, log_path):
        print_v(f"\n[bold]Solving Item {i+1}...[/bold]")

        # ====== Memory Retrieval + Injection ======
        injected_memory_parts = []
        query_skill_json = None
        query_skill = None

        # 是否需要检索：同时由 CLI 开关和已有记录共同决定
        need_procedural = procedural_skill_enabled and bool(procedural_records)
        need_episodic = episodic_memory_enabled and bool(episodic_records)

        # 提取 query skill（procedural 和 episodic 共用）
        if need_procedural or need_episodic:
            query_skill_json = gen.extract_query_skill(
                problem_prompt=item["prompt"],
                model=reflection_model,
                library=item.get("metadata", {}).get("library"),
            )
            query_skill = _parse_json_or_raw(query_skill_json)

        # --- Procedural skill injection ---
        if need_procedural and isinstance(query_skill, dict):
            applicable_skills = retrieve_applicable_skills(
                query_skill=query_skill,
                existing_records=procedural_records,
                problem_prompt=item["prompt"],
                model=reflection_model,
            )
            if applicable_skills:
                injected_memory_parts.append(
                    build_memory_injection_prompt(applicable_skills)
                )
                print_v(
                    f"[bold cyan]Injected {len(applicable_skills)} procedural skill(s)[/bold cyan]"
                )
            else:
                print_v("[dim]No applicable procedural skills found[/dim]")
        elif procedural_skill_enabled and not procedural_records:
            print_v("[dim]Procedural skill enabled but no records yet; skipping[/dim]")

        # --- Episodic memory injection ---
        if need_episodic and isinstance(query_skill, dict):
            applicable_episodes = retrieve_applicable_episodes(
                query_skill=query_skill,
                existing_records=episodic_records,
                problem_prompt=item["prompt"],
                model=reflection_model,
            )
            if applicable_episodes:
                injected_memory_parts.append(
                    build_episodic_injection_prompt(applicable_episodes)
                )
                print_v(
                    f"[bold yellow]Injected {len(applicable_episodes)} episodic memory record(s)[/bold yellow]"
                )
            else:
                print_v("[dim]No applicable episodic memories found[/dim]")
        elif episodic_memory_enabled and not episodic_records:
            print_v("[dim]Episodic memory enabled but no records yet; skipping[/dim]")

        injected_memory = "\n\n".join(injected_memory_parts) if injected_memory_parts else ""
        # =========================================

        cur_pass = 0
        is_solved = False
        reflections = []
        reflection_attempt_indices = []
        implementations = []
        test_feedback = []
        procedural_skill = None
        cur_impl = ""

        while cur_pass < pass_at_k and not is_solved:
            cur_impl = gen.func_impl(
                item["prompt"], model, "simple",
                injected_memory=injected_memory or None,
            )
            implementations.append(cur_impl)
            assert isinstance(cur_impl, str)

            is_passing, feedback, _ = exe.execute(cur_impl, [item["code_context"]], timeout=10)
            if is_passing:
                is_solved = True
                num_success += 1
                break

            test_feedback.append(feedback)
            cur_iter = 1    # 执行次数
            cur_feedback = feedback
            while cur_iter < max_iters:
                reflection_attempt_index = len(implementations) - 1
                failed_impl = cur_impl
                reflection = gen.self_reflection(
                    cur_impl,
                    cur_feedback,
                    reflection_model,
                    problem_prompt=item.get("prompt"),
                    library=item.get("metadata", {}).get("library"),
                )
                reflections.append(reflection)
                reflection_attempt_indices.append(reflection_attempt_index)

                cur_impl = gen.func_impl(
                    item["prompt"],
                    model,
                    "reflexion",
                    prev_func_impl=cur_impl,
                    feedback=cur_feedback,
                    self_reflection=reflection,
                    injected_memory=injected_memory or None,
                )
                implementations.append(cur_impl)
                assert isinstance(cur_impl, str)

                is_passing, cur_feedback, _ = exe.execute(
                    cur_impl, [item["code_context"]], timeout=10
                )
                if is_passing:
                    is_solved = True
                    num_success += 1
                    if procedural_skill_enabled:
                        procedural_skill = gen.skill_extraction(
                            problem_prompt=item.get("prompt", ""),
                            reflection=reflection,
                            failed_code=failed_impl,
                            successful_code=cur_impl,
                            patch_diff=build_patch_diff(failed_impl, cur_impl),
                            success_feedback="All DS1000 tests passed.",
                            model=reflection_model,
                            library=item.get("metadata", {}).get("library"),
                        )
                    break

                test_feedback.append(cur_feedback)
                cur_iter += 1

            cur_pass += 1

        _ensure_final_failure_reflection(
            gen=gen,
            reflection_model=reflection_model,
            item=item,
            is_solved=is_solved,
            implementations=implementations,
            test_feedback=test_feedback,
            reflections=reflections,
            reflection_attempt_indices=reflection_attempt_indices,
        )

        # 如果存在失败尝试，用 Stage A reflection 的 root_cause / repair_action
        # 调用 LLM 生成 episodic memory（Stage B）
        episodic_memory = None
        if episodic_memory_enabled and test_feedback:
            decisive_index = _decisive_failure_index(test_feedback)
            if decisive_index is not None:
                decisive_reflection = (
                    reflections[decisive_index]
                    if decisive_index < len(reflections)
                    else reflections[-1] if reflections else ""
                )
                decisive_failed_code = (
                    implementations[decisive_index]
                    if decisive_index < len(implementations)
                    else ""
                )
                decisive_feedback = (
                    test_feedback[decisive_index]
                    if decisive_index < len(test_feedback)
                    else ""
                )
                if decisive_reflection:
                    episodic_memory = gen.build_episodic_memory(
                        problem_prompt=item.get("prompt", ""),
                        func=decisive_failed_code,
                        feedback=decisive_feedback,
                        reflection_json=decisive_reflection,
                        model=reflection_model,
                        library=item.get("metadata", {}).get("library"),
                    )

        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["reflection_attempt_indices"] = reflection_attempt_indices
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["procedural_skill"] = procedural_skill
        item["solution"] = cur_impl
        item["memory_store_paths"] = store_ds1000_memories(
            log_path=log_path,
            item=item,
            item_index=i,
            implementations=implementations,
            test_feedback=test_feedback,
            reflections=reflections,
            reflection_attempt_indices=reflection_attempt_indices,
            is_solved=is_solved,
            final_solution=cur_impl,
            procedural_skill=procedural_skill,
            procedural_consolidation_model=reflection_model,
            episodic_memory=episodic_memory,
            episodic_memory_enabled=episodic_memory_enabled,
            procedural_skill_enabled=procedural_skill_enabled,
        )
        write_jsonl(log_path, [item], append=True)

        # 更新内存中的 procedural / episodic records（新写入的已合并到文件，重新加载）
        new_paths = item.get("memory_store_paths", {})
        if new_paths.get("procedural"):
            procedural_records = _read_json_records(new_paths["procedural"])
        if new_paths.get("episodic"):
            episodic_records = _read_json_records(new_paths["episodic"])

        if is_solved:
            print_v(f"[success]Item {i+1} SOLVED[/success]")
        else:
            print_v(f"[danger]Item {i+1} FAILED[/danger]")

        print_v(f"completed {i+1}/{num_items}: solved={num_success}/{i+1}, acc = {round(num_success/(i+1), 4)}")


def _ensure_final_failure_reflection(
    *,
    gen: DS1000Generator,
    reflection_model: Any,
    item: dict,
    is_solved: bool,
    implementations: List[str],
    test_feedback: List[str],
    reflections: List[str],
    reflection_attempt_indices: List[int],
) -> None:
    if is_solved or not implementations or not test_feedback:
        return

    final_attempt_index = len(implementations) - 1
    final_feedback = (
        test_feedback[final_attempt_index]
        if final_attempt_index < len(test_feedback)
        else test_feedback[-1]
    )
    if final_attempt_index in set(reflection_attempt_indices):
        return

    final_reflection = gen.self_reflection(
        implementations[final_attempt_index],
        final_feedback,
        reflection_model,
        problem_prompt=item.get("prompt"),
        library=item.get("metadata", {}).get("library"),
    )
    reflections.append(final_reflection)
    reflection_attempt_indices.append(final_attempt_index)
