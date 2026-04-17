"""ExecutionMonitor - Execution monitoring and smart brake mechanism

Core Features
-------------
1. **Early Stopping**: Detect dead loops and ineffective repetition
2. **Mid-Task Reflection**: Force mid-execution reflection
3. **Timeout Protection**: Limit ineffective brute-force attempts

Design Principles
-----------------
- Don't let the model hang itself on the wrong tree
- Early detection, early interruption, early reflection
- Save compute, improve efficiency
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecutionState:
    """Single step execution state"""
    step_num: int
    command: str
    output: str
    timestamp: float
    state_hash: str  # For detecting repetition


@dataclass
class MonitoringStats:
    """Monitoring statistics"""
    total_steps: int = 0
    repeated_actions: int = 0
    error_count: int = 0
    timeout_count: int = 0
    forced_stops: int = 0

    # State hash frequency statistics
    state_hashes: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Consecutive error count
    consecutive_errors: int = 0


class ExecutionMonitor:
    """Execution monitor
    
    Responsibilities:
    1. Detect dead loops (state hash repetition)
    2. Detect ineffective repetition (consecutive identical operations)
    3. Trigger mid-task reflection
    4. Timeout protection
    """
    
    def __init__(
        self,
        max_repeated_states: int = 3,  # Maximum of 3 identical states allowed
        max_consecutive_errors: int = 3,  # Maximum of 3 consecutive errors allowed
        mid_task_reflection_threshold: float = 0.5,  # Trigger reflection at 50% of steps
        code_timeout: int = 10,  # Code execution timeout (seconds)
    ) -> None:
        self.max_repeated_states = max_repeated_states
        self.max_consecutive_errors = max_consecutive_errors
        self.mid_task_reflection_threshold = mid_task_reflection_threshold
        self.code_timeout = code_timeout
        
        # Execution history
        self.execution_history: list[ExecutionState] = []
        self.stats = MonitoringStats()
    
    def reset(self) -> None:
        """Reset monitoring state (called when new problem starts)"""
        self.execution_history.clear()
        self.stats = MonitoringStats()
    
    def record_step(
        self,
        step_num: int,
        command: str,
        output: str,
        is_error: bool = False,
    ) -> None:
        """Record one execution step"""
        # Compute state hash
        state_hash = self._compute_state_hash(command, output)
        
        # Record execution state
        state = ExecutionState(
            step_num=step_num,
            command=command,
            output=output,
            timestamp=time.time(),
            state_hash=state_hash
        )
        self.execution_history.append(state)
        
        # Update statistics
        self.stats.total_steps += 1
        self.stats.state_hashes[state_hash] += 1
        
        if is_error:
            self.stats.error_count += 1
            self.stats.consecutive_errors += 1
        else:
            self.stats.consecutive_errors = 0  # Reset if successful
    
    def should_early_stop(self) -> tuple[bool, str]:
        """Check whether early termination should be triggered
        
        Returns
        -------
        (should_stop, reason)
        """
        # Check 1: Too many repeated states (dead loop)
        for state_hash, count in self.stats.state_hashes.items():
            if count >= self.max_repeated_states:
                return True, f"Dead loop detected: same state repeated {count} times"
        
        # Check 2: Too many consecutive errors
        if self.stats.consecutive_errors >= self.max_consecutive_errors:
            return True, f"{self.stats.consecutive_errors} consecutive errors, current path is invalid"
        
        # Check 3: Detect "meaningless repetitive operations"
        if self._detect_meaningless_repetition():
            return True, "Detected meaningless repetitive operations"
        
        return False, ""
    
    def should_trigger_mid_reflection(
        self,
        current_step: int,
        step_limit: int
    ) -> bool:
        """Check whether mid-task reflection should be triggered
        
        Trigger when reaching 50% of the step limit
        """
        threshold_step = int(step_limit * self.mid_task_reflection_threshold)
        
        # Only trigger once when exactly reaching the threshold
        if current_step == threshold_step:
            return True
        
        return False
    
    def get_mid_reflection_prompt(self, current_step: int) -> str:
        """Generate mid-task reflection prompt"""
        recent_errors = self._get_recent_errors(n=3)
        
        prompt = f"""⚠️ CRITICAL CHECKPOINT (Step {current_step}) ⚠️

You have been working on this problem for {current_step} steps with limited progress.
STOP your current approach immediately and conduct a deep reflection:

1. Review the problem statement again - are you solving the RIGHT question?
2. Analyze your recent attempts:
{self._format_recent_errors(recent_errors)}

3. Identify the FUNDAMENTAL issue with your current path
4. Propose a COMPLETELY DIFFERENT approach

If you continue the same failing strategy, this attempt will be terminated.
Choose a new method NOW."""
        
        return prompt
    
    def get_execution_summary(self) -> dict:
        """Get execution summary (for AnalyzerAgent analysis)"""
        return {
            "total_steps": self.stats.total_steps,
            "repeated_actions": self.stats.repeated_actions,
            "error_count": self.stats.error_count,
            "consecutive_errors": self.stats.consecutive_errors,
            "forced_stops": self.stats.forced_stops,
            "unique_states": len(self.stats.state_hashes),
            "most_repeated_state_count": max(self.stats.state_hashes.values()) if self.stats.state_hashes else 0,
        }
    
    # ------------------------------------------------------------------ #
    # Internal utility methods
    # ------------------------------------------------------------------ #
    
    def _compute_state_hash(self, command: str, output: str) -> str:
        """Compute state hash (for detecting repetition)
        
        Only take core parts of command and output, ignore dynamic info like timestamps
        """
        # Normalize command (remove spaces, newlines)
        cmd_normalized = command.strip().lower()
        
        # Normalize output (only take first 200 characters, ignore specific values)
        out_normalized = output[:200].strip().lower()
        
        # Compute hash
        content = f"{cmd_normalized}||{out_normalized}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _detect_meaningless_repetition(self) -> bool:
        """Detect meaningless repetitive operations
        
        For example: executing the same code 3 times consecutively but all failed
        """
        if len(self.execution_history) < 3:
            return False
        
        # Check the last 3 steps
        recent = self.execution_history[-3:]
        
        # If the commands of the last 3 steps are completely identical
        commands = [s.command.strip() for s in recent]
        if len(set(commands)) == 1:
            # And all outputs contain error information
            outputs = [s.output.lower() for s in recent]
            error_keywords = ["error", "traceback", "exception", "failed"]
            
            if all(any(kw in out for kw in error_keywords) for out in outputs):
                return True
        
        return False
    
    def _get_recent_errors(self, n: int = 3) -> list[ExecutionState]:
        """Get the most recent n error steps"""
        errors = [
            state for state in self.execution_history
            if any(kw in state.output.lower() for kw in ["error", "traceback", "exception"])
        ]
        return errors[-n:] if errors else []
    
    def _format_recent_errors(self, errors: list[ExecutionState]) -> str:
        """Format recent errors"""
        if not errors:
            return "   No recent errors (but no progress either)"
        
        lines = []
        for err in errors:
            lines.append(f"   Step {err.step_num}: {err.command[:60]}...")
            lines.append(f"   → Error: {err.output[:100]}...")
        
        return "\n".join(lines)
