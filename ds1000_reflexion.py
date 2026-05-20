from typing import List

from executors.ds1000_executor import DS1000Executor
from generators.ds1000_generate import DS1000Generator
from generators import model_factory
from utils import enumerate_resume, make_printv, resume_success_count, write_jsonl


def run_ds1000_simple(
    dataset: List[dict],
    model_name: str,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
) -> None:
    exe = DS1000Executor()
    gen = DS1000Generator()
    model = model_factory(model_name)
    stdout_log_path = log_path.replace(".jsonl", ".log")
    # print_v = make_printv(verbose)
    print_v = make_printv(verbose, stdout_log_path)


    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    for i, item in enumerate_resume(dataset, log_path):
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
        print_v(f"completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}")


def run_ds1000_reflexion(
    dataset: List[dict],
    model_name: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
) -> None:
    exe = DS1000Executor()
    gen = DS1000Generator()
    model = model_factory(model_name)
    stdout_log_path = log_path.replace(".jsonl", ".log")
    # print_v = make_printv(verbose)
    print_v = make_printv(verbose, stdout_log_path)


    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    for i, item in enumerate_resume(dataset, log_path):
        cur_pass = 0
        is_solved = False
        reflections = []
        implementations = []
        test_feedback = []
        cur_impl = ""

        while cur_pass < pass_at_k and not is_solved:
            cur_impl = gen.func_impl(item["prompt"], model, "simple")
            implementations.append(cur_impl)
            assert isinstance(cur_impl, str)

            is_passing, feedback, _ = exe.execute(cur_impl, [item["code_context"]], timeout=10)
            test_feedback.append(feedback)

            if is_passing:
                is_solved = True
                num_success += 1
                break

            cur_iter = 1
            cur_feedback = feedback
            while cur_iter < max_iters:
                reflection = gen.self_reflection(
                    cur_impl,
                    cur_feedback,
                    model,
                    problem_prompt=item.get("prompt"),
                    library=item.get("metadata", {}).get("library"),
                )
                reflections.append(reflection)

                cur_impl = gen.func_impl(
                    item["prompt"],
                    model,
                    "reflexion",
                    prev_func_impl=cur_impl,
                    feedback=cur_feedback,
                    self_reflection=reflection,
                )
                implementations.append(cur_impl)
                assert isinstance(cur_impl, str)

                is_passing, cur_feedback, _ = exe.execute(
                    cur_impl, [item["code_context"]], timeout=10
                )
                test_feedback.append(cur_feedback)
                if is_passing:
                    is_solved = True
                    num_success += 1
                    break

                cur_iter += 1

            cur_pass += 1

        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["solution"] = cur_impl
        write_jsonl(log_path, [item], append=True)
        print_v(f"completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}")
