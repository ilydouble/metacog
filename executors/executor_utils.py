
def timeout_handler(_, __):
    raise TimeoutError()

import os, json
def to_jsonl(dict_data, file_path):
    with open(file_path, 'a') as file:
        json_line = json.dumps(dict_data)
        file.write(json_line + os.linesep)

from threading import Thread

# 全局集合，跟踪所有仍在运行的执行线程，防止僵尸线程堆积
import weakref
_active_threads = weakref.WeakSet()


class PropagatingThread(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daemon = True  # 守护线程不会阻止进程退出
        self.exc = None
        self._completed = False

    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e
        finally:
            self._completed = True

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if not self._completed and timeout is not None:
            # 线程尚未完成 —— 不要访问 self.ret，因为它还没被设置。
            # 返回一个标记值；调用者必须检查 is_alive()。
            return _NOT_COMPLETED
        if self.exc:
            raise self.exc
        return self.ret if self._completed else _NOT_COMPLETED


# join() 用于表示线程仍在运行且 self.ret 不可用时的标记值
_NOT_COMPLETED = object()


# 带超时控制的函数执行工具，用于限制任意函数的执行时间，防止代码无限循环或长时间阻塞
def function_with_timeout(func, args, timeout):
    result_container = []

    def wrapper():
        result_container.append(func(*args))

    thread = PropagatingThread(target=wrapper)
    _active_threads.add(thread)
    try:
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # 线程仍在运行 —— 不要留下僵尸线程。
            # 等待一小段额外时间让它清理，然后通过引发 TimeoutError 强制失败。
            thread.join(5)  # 再给 5 秒作为清理宽限期
            if thread.is_alive():
                raise TimeoutError(
                    f"Function execution timed out after {timeout}s "
                    f"and did not stop after an additional 5s grace period."
                )
            # 如果它在宽限期内完成，我们可以使用其结果
            if result_container:
                return result_container[0]
            raise TimeoutError(f"Function execution timed out after {timeout}s.")

        return result_container[0]
    finally:
        _active_threads.discard(thread)
    
# Py tests

# if __name__ == "__main__":
#     formatter = PySubmissionFormatter()
#     leetcode_1 = 'class Solution:\n    def solveSudoku(self, board: List[List[str]]) -> None:\n        """\n        Do not return anything, modify board in-place instead.\n        """\n        '
#     humaneval_1 = 'def solveSudoku(self, board: List[List[str]]) -> None:\n        """\n        Do not return anything, modify board in-place instead.\n        """\n'

#     assert leetcode_1 == formatter.to_leetcode(humaneval_1)
#     assert humaneval_1 == formatter.to_humaneval(leetcode_1)




