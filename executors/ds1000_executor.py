from typing import List
import traceback

from .executor_types import ExecuteResult, Executor
from .executor_utils import function_with_timeout


class DS1000Executor(Executor):
    """
    Executes DS1000-style snippets.

    DS1000 problems ask the model to complete a Python code fragment that
    assigns the final answer to `result`. Each dataset item carries a
    `code_context` defining `test_execution(solution: str)`.
    """

    def execute(self, func: str, tests: List[str], timeout: int = 10) -> ExecuteResult:
        success_tests = []
        failed_tests = []
        state = []

        for code_context in tests:
            is_passing, feedback = self._run_test(func, code_context, timeout)
            state.append(is_passing)
            if is_passing:
                success_tests.append("DS1000 code_context test")
            else:
                failed_tests.append(feedback)

        feedback = "Tests passed:"
        for test in success_tests:
            feedback += f"\n{test}"
        feedback += "\n\nTests failed:"
        for test in failed_tests:
            feedback += f"\n{test}"

        return ExecuteResult(all(state), feedback, tuple(state))

    def evaluate(self, name: str, func: str, test: str, timeout: int = 10) -> bool:
        is_passing, _ = self._run_test(func, test, timeout)
        return is_passing

    def _run_test(self, solution: str, code_context: str, timeout: int):
        try:
            test_env = {}
            function_with_timeout(exec, (code_context, test_env), timeout)
            test_execution = test_env["test_execution"]
            function_with_timeout(test_execution, (solution,), timeout)
            return True, "passed"
        except Exception as exc:
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            stack = traceback.format_exc(limit=4)
            return False, f"{error}\n{stack}"
