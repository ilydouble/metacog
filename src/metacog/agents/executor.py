"""ExecutorAgent - Runs Solver, emits TrajectoryEvent after each problem.

Key Points
----------
1. Before assembling system prompt, query memU for Top-K retrieval with current problem
2. Only inject the most relevant 1-2 memories (extremely concise)
3. No longer load all memories from entire MemoryStore
4. 🔥 Integrated ExecutionMonitor: smart brake and mid-execution reflection
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel


class VerboseAgent(DefaultAgent):
    """Real-time print every step's model output and execution results for debugging."""

    def query(self) -> dict:
        print(f"\n=== Step {self.n_calls + 1} · Calling model... ===", flush=True)
        message = super().query()
        content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls") or message.get("extra", {}).get("actions", [])
        if content:
            preview = content[:800] if len(content) <= 800 else content[:800] + "\n...(truncated)"
            print("--- Model response ---")
            print(preview)
            print("--- End of response ---", flush=True)
        elif tool_calls:
            # Empty content with tool_calls is normal (model outputs tool call directly)
            print(f"[Tool call only, no text content (normal)]", flush=True)
        else:
            print("!!! Model response is empty and no tool call, possible network error or model not responding !!!", flush=True)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        for action in actions:
            cmd = action.get("command", "")
            print(f">>> Execute command: {cmd[:300]}", flush=True)
        observations = super().execute_actions(message)
        for obs in observations:
            output = obs.get("extra", {}).get("raw_output", "") or obs.get("content", "")
            if output:
                preview = output[:400] if len(output) <= 400 else output[:400] + "\n...(truncated)"
                print(f"<<< Execution result:\n{preview}", flush=True)
        return observations

from ..bus import Event, EventBus, EventType
from ..memory.memu_client import MemUClient
from ..memory.store import MemoryStore
from ..skills.registry import SkillRegistry
from .base import BaseAgent
from .pot_sandbox_wrapper import create_pot_sandbox_wrapper


class ExecutorAgent(BaseAgent):
    """Solver executor agent (memU RAG version).

    For each problem:
    1. Use current problem to do Top-K retrieval in memU (only take the most relevant 1-2 items)
    2. Generate skill descriptions from SkillRegistry, inject into system_template
    3. Run DefaultAgent
    4. Publish TrajectoryEvent (for AnalyzerAgent consumption)
    """

    name = "executor"

    def __init__(
        self,
        model: LitellmModel,
        bus: EventBus,
        scaffold: dict,
        memory_store: MemoryStore | None = None,  # Retain compatibility but no longer used
        skill_registry: SkillRegistry | None = None,
        output_dir: Path | None = None,
        skills_dir: Path | None = None,
        memu_client: MemUClient | None = None,  # memU client (semantic memory)
        rag_top_k: int = 2,  # Top-K retrieval count (key parameter!)
        procedural_memory = None,  # ProceduralMemory instance (procedural memory)
        skill_top_k: int = 3,  # How many skills to retrieve
        episodic_memory = None,  # EpisodicMemory instance (episodic memory)
        case_top_k: int = 1,  # How many success cases to retrieve
    ) -> None:
        super().__init__(model, bus)
        self.scaffold = scaffold
        self.memory_store = memory_store  # Retain but no longer used for prompt injection
        self.skill_registry = skill_registry
        self.output_dir = output_dir
        self.skills_dir = skills_dir
        self.memu = memu_client  # memU client (semantic memory)
        self.rag_top_k = rag_top_k  # Control how many memories to inject
        self.procedural_memory = procedural_memory  # Procedural memory
        self.skill_top_k = skill_top_k  # Control how many skills to inject
        self.episodic_memory = episodic_memory  # Episodic memory
        self.case_top_k = case_top_k  # Control how many success cases to inject

    def run_problem(self, problem_data: dict) -> dict:
        """Run a single problem, return result dict, and publish TrajectoryEvent."""
        from utils.answer_extraction import extract_final_answer, normalize_answer
        from utils.evaluation import compare_answers

        problem_id = str(problem_data.get("id", "unknown"))
        problem = problem_data.get("problem") or problem_data.get("question", "")
        expected_answer = str(
            problem_data.get("expected_answer") or problem_data.get("answer", "")
        ).strip()

        # ==========================================
        # Core modification: Top-K RAG retrieval based on current problem
        # ==========================================
        system_template = self.scaffold.get("system_template", "")

        # Record used memory IDs (for subsequent statistics update)
        used_memory_ids = []

        # 1. memU RAG retrieval (only inject most relevant memories)
        if self.memu:
            try:
                retrieved_memories = self.memu.search(
                    query=problem,  # Use problem text as query
                    top_k=self.rag_top_k  # Only take the most relevant K items (default 2)
                )

                # Threshold filtering: memories with distance > 0.75 (similarity < 25%) are not injected
                # Semantic memory query (original problem text) vs document (Problem type + Error + Fix)
                # Types are close but still have differences, so threshold is looser than episodic memory (0.6)
                retrieved_memories = [m for m in retrieved_memories if m.distance <= 0.75]

                if retrieved_memories:
                    # Calculate average similarity (displayed in title)
                    avg_similarity = 1 - sum(m.distance for m in retrieved_memories) / len(retrieved_memories)

                    # Assemble memory injection
                    memory_injection = (
                        f"\n## 📚 Retrieved Semantic Memory (Avg Relevance: {avg_similarity:.0%})\n"
                        "The following lessons were extracted from cognitively similar failed trajectories.\n"
                        "Use them as **reference warnings**, not rigid rules.\n\n"
                    )

                    for idx, mem in enumerate(retrieved_memories, 1):
                        sim = 1 - mem.distance
                        meta = mem.metadata or {}
                        is_hint_only = "hint_only" in meta.get("tags", [])

                        if is_hint_only:
                            memory_injection += f"### Hint {idx} (Relevance: {sim:.0%})\n{mem.content}\n\n"
                        else:
                            # Render structured fields from metadata when available,
                            # otherwise fall back to raw content
                            problem_type   = meta.get("problem_type", "")
                            error          = meta.get("error", "")
                            fix            = meta.get("fix", "")
                            solution_steps = meta.get("solution_steps", "")
                            common_traps   = meta.get("common_traps", "")

                            if fix:
                                memory_injection += f"### Lesson {idx} (Relevance: {sim:.0%})\n"
                                if problem_type:
                                    memory_injection += f"**Problem type**: {problem_type}\n"
                                memory_injection += f"**Error**: {error}\n"
                                memory_injection += f"**Fix**: {fix}\n"
                                if solution_steps:
                                    memory_injection += f"**Solution steps**: {solution_steps}\n"
                                if common_traps:
                                    memory_injection += f"**Common traps**: {common_traps}\n"
                                memory_injection += "\n"
                            else:
                                # Fallback for old-format memories
                                memory_injection += f"### Lesson {idx} (Relevance: {sim:.0%})\n{mem.content}\n\n"

                        used_memory_ids.append(("semantic", mem.id))

                    system_template = system_template.rstrip() + "\n\n" + memory_injection
                    print(f"  [Executor] Injected {len(retrieved_memories)} semantic memories (distances: {[f'{m.distance:.3f}' for m in retrieved_memories]})", flush=True)
                else:
                    print(f"  [Executor] Semantic memory: no sufficiently relevant entries (threshold distance≤0.75), skipping injection", flush=True)
            except Exception as exc:
                print(f"  [Executor] memU retrieval failed: {exc}", flush=True)

        # 2. Skill retrieval (procedural memory)
        if self.procedural_memory:
            try:
                retrieved_skills = self.procedural_memory.search_skills(
                    query=problem,
                    top_k=self.skill_top_k  # Only take the most relevant K skills (default 3)
                )

                if retrieved_skills:
                    skill_text = self.procedural_memory.format_skills_for_prompt(
                        retrieved_skills,
                        include_code=False  # Only include metadata, not complete code
                    )
                    system_template = system_template.rstrip() + "\n\n" + skill_text
                    print(f"  [Executor] Injected {len(retrieved_skills)} relevant skills (procedural memory)", flush=True)
            except Exception as exc:
                print(f"  [Executor] Procedural memory retrieval failed: {exc}", flush=True)
        elif self.skill_registry:
            # Fallback: inject all skills if no procedural memory available
            skill_text = self.skill_registry.as_prompt_text()
            if skill_text:
                system_template = system_template.rstrip() + "\n\n" + skill_text
                print(f"  [Executor] ⚠️  Injecting all skills (consider enabling procedural memory)", flush=True)

        # 3. Episodic memory retrieval (success cases)
        if self.episodic_memory:
            try:
                similar_cases = self.episodic_memory.search_similar_cases(
                    query=problem,
                    top_k=self.case_top_k  # Only take the most similar K cases (default 1)
                )

                # Similarity threshold filtering: cases with distance > 0.6 (similarity < 40%) are not injected to avoid irrelevant interference
                similar_cases = [c for c in similar_cases if c.distance <= 0.6]

                if similar_cases:
                    case_text = self.episodic_memory.format_cases_for_prompt(
                        similar_cases,
                        max_cases=self.case_top_k
                    )
                    system_template = system_template.rstrip() + "\n\n" + case_text
                    print(f"  [Executor] Injected {len(similar_cases)} similar success cases (episodic memory, similarity≥40%)", flush=True)
                    for case in similar_cases:
                        used_memory_ids.append(("episodic", case.id))
                else:
                    print(f"  [Executor] Episodic memory: no sufficiently similar cases (threshold 40%), skipping injection", flush=True)
            except Exception as exc:
                print(f"  [Executor] Episodic memory retrieval failed: {exc}", flush=True)

        start_time = time.time()
        traj_path = (self.output_dir / f"{problem_id}.traj.json") if self.output_dir else None

        # Construct PYTHONPATH: append skills_dir to the front of current environment's PYTHONPATH
        import os as _os
        env_extra: dict[str, str] = {}
        if self.skills_dir and self.skills_dir.exists():
            existing = _os.environ.get("PYTHONPATH", "")
            env_extra["PYTHONPATH"] = f"{self.skills_dir}:{existing}" if existing else str(self.skills_dir)

        local_env = LocalEnvironment(env=env_extra) if env_extra else LocalEnvironment()
        step_limit = self.scaffold.get("step_limit", 10)

        agent = VerboseAgent(
            model=self.model,
            env=local_env,
            system_template=system_template,
            instance_template=self.scaffold.get("instance_template", "{{task}}"),
            step_limit=step_limit,
            cost_limit=self.scaffold.get("cost_limit", 2.0),
            output_path=traj_path,
        )

        # PoT sandbox wrapper: monitor errors → inject reflection prompt → continue linear execution
        agent = create_pot_sandbox_wrapper(agent)

        error = None
        extracted_answer = None
        passed = False
        submission = ""

        try:
            result = agent.run(task=problem)
            submission = result.get("submission", "")

            # Extract answer (prioritize extraction from submission)
            extracted_answer = extract_final_answer(submission)
            if extracted_answer:
                extracted_answer = normalize_answer(extracted_answer)

            # Fallback 1: when submission is empty, search in assistant message content
            if not extracted_answer:
                for msg in reversed(agent.messages):
                    role = msg.get("role", "")
                    content = msg.get("content", "") or ""
                    if role == "assistant" and content:
                        fallback = extract_final_answer(content)
                        if fallback:
                            extracted_answer = normalize_answer(fallback)
                            break

            # Fallback 2: scan tool (bash execution result) messages, look for COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT signal
            if not extracted_answer:
                for msg in reversed(agent.messages):
                    if msg.get("role") != "tool":
                        continue
                    raw = msg.get("extra", {}).get("raw_output", "") or msg.get("content", "") or ""
                    lines = raw.splitlines()
                    for i, line in enumerate(lines):
                        if line.strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
                            # Look down for the first non-empty line containing numbers
                            for j in range(i + 1, len(lines)):
                                candidate_line = lines[j].strip()
                                if candidate_line:
                                    candidate = normalize_answer(candidate_line)
                                    if candidate and any(c.isdigit() for c in candidate):
                                        extracted_answer = candidate
                                        break
                            if extracted_answer:
                                break
                    if extracted_answer:
                        break

            if extracted_answer:
                passed = compare_answers(extracted_answer, expected_answer)
        except Exception as exc:
            error = str(exc)

        record = {
            "id": problem_id,
            "problem": problem[:300],
            "expected_answer": expected_answer,
            "extracted_answer": extracted_answer,
            "submission": submission[:500],
            "passed": passed,
            "n_steps": agent.n_calls,
            "cost": agent.cost,
            "time": round(time.time() - start_time, 2),
            "error": error,
            "traj_path": str(traj_path) if traj_path else None,
        }

        # Update usage statistics for memories
        if used_memory_ids:
            for layer, mem_id in used_memory_ids:
                try:
                    if layer == "semantic" and self.memu:
                        self.memu.update_usage_stats(mem_id, success=passed)
                    elif layer == "procedural" and self.procedural_memory:
                        self.procedural_memory.memu.update_usage_stats(mem_id, success=passed)
                    elif layer == "episodic" and self.episodic_memory:
                        self.episodic_memory.memu.update_usage_stats(mem_id, success=passed)
                except Exception:
                    pass  # Silent failure, does not affect main flow

        # Publish trajectory event
        self._publish(Event(type=EventType.TRAJECTORY, data=record))
        return record

