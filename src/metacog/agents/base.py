"""BaseAgent - 所有 metacog 智能体的基类。

持有 LitellmModel，提供统一的 LLM 调用接口，
并通过 EventBus 与其他 Agent 通信。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 添加 mini-swe-agent 路径（相对于本文件向上四层到项目根）
_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_root / "src" / "mini-swe-agent" / "src"))

from minisweagent.models.litellm_model import LitellmModel

from ..bus import Event, EventBus


class BaseAgent:
    """metacog Agent 基类。

    属性:
        name:       Agent 名称（用于日志）
        model:      LitellmModel 实例
        bus:        EventBus 实例（与其他 Agent 共享）
    """

    name: str = "base_agent"

    def __init__(self, model: LitellmModel, bus: EventBus) -> None:
        self.model = model
        self.bus = bus
        self._register_handlers()

    def _register_handlers(self) -> None:
        """子类覆盖此方法，向 bus 注册自己关心的事件。"""

    # ------------------------------------------------------------------ #
    # LLM 工具方法
    # ------------------------------------------------------------------ #

    def _llm_call(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """封装一次 LLM completion，返回文本内容。"""
        import litellm

        model_name = self.model.model_name
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "drop_params": True,
        }
        # 复用 model 的 model_kwargs（含 api_base 等）
        for k, v in (self.model.model_kwargs or {}).items():
            if k not in kwargs:
                kwargs[k] = v

        response = litellm.completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def _publish(self, event: Event) -> None:
        self.bus.publish(event)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"

