from typing import Any, Dict, List
import ast
import traceback

from .executor_types import ExecuteResult, Executor
from .executor_utils import function_with_timeout


class DS1000Executor(Executor):

    # 批量执行多个测试用例，返回整体执行结果
    def execute(self, func: str, tests: List[str], timeout: int = 10) -> ExecuteResult:
        failed_tests = []
        state = []

        for code_context in tests:
            is_passing, feedback = self._run_test(func, code_context, timeout)
            state.append(is_passing)
            if not is_passing:
                failed_tests.append(feedback)

        feedback = ""
        if failed_tests:
            feedback = "Tests failed:"
            for test in failed_tests:
                feedback += f"\n{test}"

        return ExecuteResult(all(state), feedback, tuple(state))

    # 简化的评估接口，仅判断单个测试是否通过
    def evaluate(self, name: str, func: str, test: str, timeout: int = 10) -> bool:
        is_passing, _ = self._run_test(func, test, timeout)
        return is_passing

    def _run_test(self, solution: str, code_context: str, timeout: int):
        test_env = {}
        test_failures: List[Dict[str, Any]] = []
        try:
            code_context = self._make_exec_test_raise(code_context)  #失败时抛出异常而非返回False
            function_with_timeout(exec, (code_context, test_env), timeout)  
            self._wrap_exec_test_to_capture_diagnostics(test_env, test_failures)  #获取实际输出和期望输出
            # 执行测试
            test_execution = test_env["test_execution"]
            function_with_timeout(test_execution, (solution,), timeout)
            return True, "passed"
        except Exception as exc:
            if isinstance(exc, AssertionError):
                error = "AssertionError"
                stack_list = traceback.format_list(traceback.extract_tb(exc.__traceback__, limit=4))  #限制堆栈信息长度，4层，避免过长输出
                stack = "".join(stack_list) + "AssertionError\n"
            else:
                error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                stack = traceback.format_exc(limit=4)
            output_feedback = self._format_test_failure_feedback(test_failures)
            # 测试未通过时返回：错误类型、堆栈信息（异常位置和调用栈）和测试反馈（实际输出和期望输出）
            return False, f"{error}\n{stack}{output_feedback}"

    # DS1000 测试代码通常包含 try-except 块，失败时返回 False。此方法将其改为抛出异常，便于捕获详细错误信息
    def _make_exec_test_raise(self, code_context: str) -> str:
        try:
            tree = ast.parse(code_context)
        except SyntaxError:
            return code_context

        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "exec_test":
                continue
            if len(node.body) != 1 or not isinstance(node.body[0], ast.Try):
                continue

            try_node = node.body[0]
            if not self._exec_test_handlers_return_false(try_node):
                continue

            node.body = try_node.body
            changed = True

        if not changed:
            return code_context

        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    # 检查 try-except 块中的所有异常处理器是否都只包含一个返回 False 或 0 的语句
    def _exec_test_handlers_return_false(self, try_node: ast.Try) -> bool:
        if not try_node.handlers:
            return False

        for handler in try_node.handlers:
            if len(handler.body) != 1 or not isinstance(handler.body[0], ast.Return):
                return False
            value = handler.body[0].value
            if not isinstance(value, ast.Constant) or value.value not in (0, False):
                return False

        return True

    # 包装exec_test 函数，捕获测试用例的执行结果和预期结果
    def _wrap_exec_test_to_capture_diagnostics(
        self, test_env: dict, test_failures: List[Dict[str, Any]]
    ) -> None:
        exec_test = test_env.get("exec_test")
        if not callable(exec_test):
            return

        def wrapped_exec_test(result: Any, ans: Any) -> Any:
            test_case_id = len(test_failures) + 1
            test_failures.append(
                {
                    "test_case_id": test_case_id,
                    "actual": result,
                    "expected": ans,
                }
            )
            return exec_test(result, ans)     # 调用ds1000中原函数

        test_env["exec_test"] = wrapped_exec_test

    # 格式化测试失败反馈，包含实际输出和期望输出
    def _format_test_failure_feedback(self, test_failures: List[Dict[str, Any]]) -> str:
        if not test_failures:
            return ""

        failure = test_failures[-1]
        test_case_id = failure["test_case_id"]
        actual = failure["actual"]
        lines = ["\nDS1000 diagnostic outputs:"]

        if test_case_id == 1:
            lines.append(f"TestCase 1 failed (public example).")
            lines.append(self._format_value("actual_result", actual))
            lines.append(self._format_value("expected_result", failure["expected"]))
        else:
            lines.append("在未公开的测试集上失败了。")
            lines.append(self._format_value("actual_result", actual))

        return "\n" + "\n".join(lines) + "\n"

    # 格式化任意值的详细信息，包括类型、形状、列名等
    def _format_value(self, label: str, value: Any) -> str:
        parts = [f"{label}: type={type(value).__name__}"]

        shape = getattr(value, "shape", None)
        if shape is not None:
            parts.append(f"shape={shape}")

        columns = getattr(value, "columns", None)
        if columns is not None:
            parts.append(f"columns={self._truncate_repr(list(columns))}")

        dtypes = getattr(value, "dtypes", None)
        if dtypes is not None:
            parts.append(f"dtypes={self._truncate_repr(dtypes)}")

        preview = value
        head = getattr(value, "head", None)
        if callable(head):
            try:
                preview = head(10)
            except Exception:
                preview = value

        parts.append(f"value={self._truncate_repr(preview)}")
        return "\n".join(parts)

    # 截断过长字符串表示，避免输出日志过大
    def _truncate_repr(self, value: Any, max_len: int = 2000) -> str:
        text = repr(value)
        if len(text) <= max_len:
            return text
        return text[:max_len] + "... <truncated>"
