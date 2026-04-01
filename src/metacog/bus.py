"""事件总线 - 同步发布/订阅，用于多智能体之间的消息传递。"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Event:
    """基础事件。"""
    type: str
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# 预定义事件类型常量
class EventType:
    TRAJECTORY = "trajectory"          # ExecutorAgent 跑完一道题
    ANALYSIS = "analysis"              # AnalyzerAgent 分析失败轨迹
    MEMORY_UPDATED = "memory_updated"  # MemoryManagerAgent 更新了记忆
    SUCCESS_ANALYSIS = "success_analysis"  # AnalyzerAgent 分析成功轨迹
    SKILL_CREATED = "skill_created"        # SkillAgent 生成了新 skill


class EventBus:
    """轻量同步事件总线。

    用法::

        bus = EventBus()

        @bus.on(EventType.TRAJECTORY)
        def handle(event: Event):
            ...

        bus.publish(Event(type=EventType.TRAJECTORY, data={...}))
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[Event], None]]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """注册事件处理函数。"""
        self._handlers[event_type].append(handler)

    def on(self, event_type: str) -> Callable:
        """装饰器形式的订阅。"""
        def decorator(fn: Callable[[Event], None]) -> Callable:
            self.subscribe(event_type, fn)
            return fn
        return decorator

    def publish(self, event: Event) -> None:
        """发布事件，同步调用所有订阅者。"""
        for handler in self._handlers.get(event.type, []):
            handler(event)

    def clear(self, event_type: str | None = None) -> None:
        """清除订阅（测试用）。"""
        if event_type:
            self._handlers.pop(event_type, None)
        else:
            self._handlers.clear()

