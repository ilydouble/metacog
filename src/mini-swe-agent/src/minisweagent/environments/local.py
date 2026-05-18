import os
import platform
import subprocess
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class LocalEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30


class LocalEnvironment:
    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        """This class executes bash commands directly on the local machine."""
        self.config = config_class(**kwargs)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the local environment and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or os.getcwd()
        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                cwd=cwd,
                env=os.environ | self.config.env,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion.

        Scans all output lines for the COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT signal,
        not just the first line, so the model can print intermediate results before submitting.
        """
        if output.get("returncode") != 0:
            return
        lines = output.get("output", "").splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
                # 不只是合并后面的所有行，我们希望确保后面真的有"有意义的输出"（数字）
                submission = "\n".join(lines[i + 1:]).strip()
                if not submission:
                    continue
                # 如果 submission 中包含数字，就认定它提交了，防止提交标点符号或纯字符串如 "]"
                # 增加了对 "ANSWER" 的特殊处理
                if any(c.isdigit() for c in submission) or "ANSWER" in submission:
                    raise Submitted(
                        {
                            "role": "exit",
                            "content": submission,
                            "extra": {"exit_status": "Submitted", "submission": submission},
                        }
                    )
                else:
                    # 空提交说明答案计算失败，不接受，让 agent 继续尝试
                    continue

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), os.environ, kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
