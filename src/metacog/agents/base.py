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
        **extra_kwargs,
    ) -> str:
        """封装一次 LLM completion，返回文本内容。"""
        import litellm

        # 修复：LitellmModel 的 model_name 在 config 中
        model_name = self.model.config.model_name
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "drop_params": True,
        }
        # 复用 model 的 model_kwargs（含 api_base 等）
        model_kwargs = self.model.config.model_kwargs or {}
        for k, v in model_kwargs.items():
            if k not in kwargs:
                kwargs[k] = v

        # 🔥 修复：移除工具相关参数，避免 LM Studio 报错
        # LM Studio 不支持 tool_choice，需要显式移除
        kwargs.pop('tool_choice', None)
        kwargs.pop('tools', None)
        kwargs.pop('parallel_tool_calls', None)

        # 合并额外参数（如 extra_body 用于控制 thinking 等）
        kwargs.update(extra_kwargs)

        response = litellm.completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )

        # 提取内容（兼容智谱 GLM 的推理模式）
        message = response.choices[0].message
        content = message.content

        # 🔥 智谱 GLM-4.7 的推理内容可能在 reasoning_content 或其他字段
        if not content:
            # 尝试从 message 的其他属性中提取
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content
            elif hasattr(message, 'tool_calls') and message.tool_calls:
                # 可能在 tool_calls 中
                content = str(message.tool_calls)
            else:
                # 打印完整响应用于调试
                print(f"  [BaseAgent] ⚠️  LLM 返回空内容", flush=True)
                print(f"  [BaseAgent] Message 对象: {message}", flush=True)
                print(f"  [BaseAgent] Message 属性: {dir(message)}", flush=True)

        return content or ""

    def _publish(self, event: Event) -> None:
        self.bus.publish(event)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"

