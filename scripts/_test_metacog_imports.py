"""Quick sanity check for metacog module imports."""
import sys
sys.path.insert(0, 'src/mini-swe-agent/src')
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')

from metacog.bus import EventBus, EventType, Event
from metacog.memory.store import MemoryStore, MemoryEntry
from metacog.skills.base import StructuredSkill, SkillResult
from metacog.skills.composite import CompositionalSkill
from metacog.skills.registry import SkillRegistry
from metacog.agents.base import BaseAgent
from metacog.agents.executor import ExecutorAgent
from metacog.agents.analyzer import AnalyzerAgent
from metacog.agents.memory_manager import MemoryManagerAgent
print("All imports OK")

# EventBus test
bus = EventBus()
received = []
bus.subscribe(EventType.MEMORY_UPDATED, lambda e: received.append(e))
bus.publish(Event(type=EventType.MEMORY_UPDATED, data={"action": "test"}))
assert len(received) == 1
print("EventBus OK")

# MemoryEntry / MemoryStore (without file I/O)
entry = MemoryEntry(title="Test", content="Use sympy.", tags=["algebra"])
assert "mem_" in entry.id
store = MemoryStore.__new__(MemoryStore)
store._entries = [entry]
text = store.as_prompt_text()
assert "Test" in text and "sympy" in text
print("MemoryStore OK:", len(store), "entries")

# SkillRegistry
registry = SkillRegistry()
assert len(registry) == 0
print("SkillRegistry OK")

print("\nAll checks passed!")

