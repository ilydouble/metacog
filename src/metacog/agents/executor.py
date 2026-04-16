"""ExecutorAgent - 跑 Solver，每题结束后发 TrajectoryEvent。

改造要点
--------
1. 在组装 system prompt 前，拿当前题目去 memU 做 Top-K 检索
2. 只注入最相关的 1-2 条记忆（极度精简）
3. 不再加载整个 MemoryStore 的所有记忆
4. 🔥 集成 ExecutionMonitor：智能刹车和中途反思
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel


class VerboseAgent(DefaultAgent):
    """实时打印每一步模型输出和执行结果，方便调试。"""

    def query(self) -> dict:
        print(f"\n=== Step {self.n_calls + 1} · 正在调用模型... ===", flush=True)
        message = super().query()
        content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls") or message.get("extra", {}).get("actions", [])
        if content:
            preview = content[:800] if len(content) <= 800 else content[:800] + "\n...(截断)"
            print("--- 模型回复 ---")
            print(preview)
            print("--- 回复结束 ---", flush=True)
        elif tool_calls:
            # content 为空但有 tool_calls 是正常行为（Qwen3 关闭 thinking 后直接输出 tool call）
            print(f"[纯 tool call，无文字说明（正常）]", flush=True)
        else:
            print("!!! 模型回复为空且无 tool call，可能是网络错误或模型未响应 !!!", flush=True)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        for action in actions:
            cmd = action.get("command", "")
            print(f">>> 执行命令: {cmd[:300]}", flush=True)
        observations = super().execute_actions(message)
        for obs in observations:
            output = obs.get("extra", {}).get("raw_output", "") or obs.get("content", "")
            if output:
                preview = output[:400] if len(output) <= 400 else output[:400] + "\n...(截断)"
                print(f"<<< 执行结果:\n{preview}", flush=True)
        return observations

from ..bus import Event, EventBus, EventType
from ..memory.memu_client import MemUClient
from ..memory.store import MemoryStore
from ..skills.registry import SkillRegistry
from .base import BaseAgent
from .pot_sandbox_wrapper import create_pot_sandbox_wrapper


class ExecutorAgent(BaseAgent):
    """Solver 执行智能体（memU RAG 版本）。

    每道题：
    1. 用当前题目去 memU 做 Top-K 检索（只取最相关的 1-2 条）
    2. 从 SkillRegistry 生成技能描述，注入 system_template
    3. 跑 DefaultAgent
    4. 发布 TrajectoryEvent（供 AnalyzerAgent 消费）
    """

    name = "executor"

    def __init__(
        self,
        model: LitellmModel,
        bus: EventBus,
        scaffold: dict,
        memory_store: MemoryStore | None = None,  # 保留兼容性，但不再使用
        skill_registry: SkillRegistry | None = None,
        output_dir: Path | None = None,
        skills_dir: Path | None = None,
        memu_client: MemUClient | None = None,  # memU 客户端（语义记忆）
        rag_top_k: int = 2,  # Top-K 检索数量（关键参数！）
        procedural_memory = None,  # ProceduralMemory 实例（程序记忆）
        skill_top_k: int = 3,  # 检索多少个 skill
        episodic_memory = None,  # EpisodicMemory 实例（情景记忆）
        case_top_k: int = 1,  # 检索多少个成功案例
    ) -> None:
        super().__init__(model, bus)
        self.scaffold = scaffold
        self.memory_store = memory_store  # 保留但不再用于 prompt 注入
        self.skill_registry = skill_registry
        self.output_dir = output_dir
        self.skills_dir = skills_dir
        self.memu = memu_client  # memU 客户端（语义记忆）
        self.rag_top_k = rag_top_k  # 控制注入多少条记忆
        self.procedural_memory = procedural_memory  # 程序记忆
        self.skill_top_k = skill_top_k  # 控制注入多少个 skill
        self.episodic_memory = episodic_memory  # 情景记忆
        self.case_top_k = case_top_k  # 控制注入多少个成功案例

    def run_problem(self, problem_data: dict) -> dict:
        """运行单道题，返回结果 dict，并发布 TrajectoryEvent。"""
        from utils.answer_extraction import extract_final_answer, normalize_answer
        from utils.evaluation import compare_answers

        problem_id = str(problem_data.get("id", "unknown"))
        problem = problem_data.get("problem") or problem_data.get("question", "")
        expected_answer = str(
            problem_data.get("expected_answer") or problem_data.get("answer", "")
        ).strip()

        # ==========================================
        # 核心改造：基于当前题目进行 Top-K RAG 检索
        # ==========================================
        system_template = self.scaffold.get("system_template", "")

        # 🔥 记录使用的记忆 ID（用于后续更新统计）
        used_memory_ids = []

        # 1. memU RAG 检索（只注入最相关的记忆）
        if self.memu:
            try:
                retrieved_memories = self.memu.search(
                    query=problem,  # 用题目文本做查询
                    top_k=self.rag_top_k  # 只取最相关的 K 条（默认 2）
                )

                # 阈值过滤：distance > 0.75（相似度 < 25%）的记忆不注入
                # 语义记忆 query（题目原文）与 document（Problem type + Error + Fix）
                # 类型相近但仍有差异，故阈值比情景记忆（0.6）宽松
                retrieved_memories = [m for m in retrieved_memories if m.distance <= 0.75]

                if retrieved_memories:
                    # 组装极度精简的记忆注入
                    memory_injection = "\n## 📚 历史避坑指南\n在处理类似问题时，你曾经犯过错，请务必遵守以下教训：\n\n"
                    for idx, mem in enumerate(retrieved_memories, 1):
                        memory_injection += f"{idx}. {mem.content}\n"
                        used_memory_ids.append(("semantic", mem.id))

                    system_template = system_template.rstrip() + "\n\n" + memory_injection

                    print(f"  [Executor] 注入 {len(retrieved_memories)} 条相关记忆 (距离: {[f'{m.distance:.3f}' for m in retrieved_memories]})", flush=True)
                else:
                    print(f"  [Executor] 语义记忆：无足够相关的记忆（阈值 distance≤0.75），跳过注入", flush=True)
            except Exception as exc:
                print(f"  [Executor] memU 检索失败: {exc}", flush=True)

        # 2. 技能检索（使用程序记忆）
        if self.procedural_memory:
            try:
                retrieved_skills = self.procedural_memory.search_skills(
                    query=problem,
                    top_k=self.skill_top_k  # 只取最相关的 K 个 skill（默认 3）
                )

                if retrieved_skills:
                    skill_text = self.procedural_memory.format_skills_for_prompt(
                        retrieved_skills,
                        include_code=False  # 只包含元数据，不包含完整代码
                    )
                    system_template = system_template.rstrip() + "\n\n" + skill_text
                    print(f"  [Executor] 注入 {len(retrieved_skills)} 个相关技能 (程序记忆)", flush=True)
            except Exception as exc:
                print(f"  [Executor] 程序记忆检索失败: {exc}", flush=True)
        elif self.skill_registry:
            # 兼容：如果没有程序记忆，使用旧方式（全量注入）
            skill_text = self.skill_registry.as_prompt_text()
            if skill_text:
                system_template = system_template.rstrip() + "\n\n" + skill_text
                print(f"  [Executor] ⚠️  全量注入技能（建议启用程序记忆）", flush=True)

        # 🔥 3. 情景记忆检索（成功案例）
        if self.episodic_memory:
            try:
                similar_cases = self.episodic_memory.search_similar_cases(
                    query=problem,
                    top_k=self.case_top_k  # 只取最相似的 K 个案例（默认 1）
                )

                # 相似度阈值过滤：distance > 0.6（相似度 < 40%）的案例不注入，避免不相关干扰
                similar_cases = [c for c in similar_cases if c.distance <= 0.6]

                if similar_cases:
                    case_text = self.episodic_memory.format_cases_for_prompt(
                        similar_cases,
                        max_cases=self.case_top_k
                    )
                    system_template = system_template.rstrip() + "\n\n" + case_text
                    print(f"  [Executor] 注入 {len(similar_cases)} 个类似成功案例 (情景记忆, 相似度≥40%)", flush=True)
                    for case in similar_cases:
                        used_memory_ids.append(("episodic", case.id))
                else:
                    print(f"  [Executor] 情景记忆：无足够相似的案例（阈值 40%），跳过注入", flush=True)
            except Exception as exc:
                print(f"  [Executor] 情景记忆检索失败: {exc}", flush=True)

        start_time = time.time()
        traj_path = (self.output_dir / f"{problem_id}.traj.json") if self.output_dir else None

        # 构造 PYTHONPATH：把 skills_dir 追加到当前环境的 PYTHONPATH 前面
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

        # 🔥 PoT 沙箱包装：监控报错 → 注入反思 prompt → 继续线性执行
        agent = create_pot_sandbox_wrapper(agent)

        error = None
        extracted_answer = None
        passed = False
        submission = ""

        try:
            result = agent.run(task=problem)
            submission = result.get("submission", "")
            extracted_answer = extract_final_answer(submission)
            if extracted_answer:
                extracted_answer = normalize_answer(extracted_answer)
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

        # 🔥 更新使用记忆的统计
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
                    pass  # 静默失败，不影响主流程

        # 发布轨迹事件
        self._publish(Event(type=EventType.TRAJECTORY, data=record))
        return record

