"""TrajectoryAnalyzer - Post-hoc trajectory analyzer

Analyzes completed trajectories to identify dead loops and ineffective repetition patterns.
While it cannot prevent them in real-time (which would require modifying mini-swe-agent internals), it can:
1. Mark these patterns in AnalyzerAgent
2. Store "avoid dead loops" lessons in memU
3. Provide early warnings when similar situations are encountered next time
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional


@dataclass
class LoopPattern:
    """Detected loop pattern"""
    start_step: int
    end_step: int
    loop_count: int
    action_signature: str
    description: str


class TrajectoryAnalyzer:
    """Post-hoc trajectory analyzer

    Analyzes completed trajectories to identify dead loops and anti-patterns
    """
    
    def __init__(self) -> None:
        pass
    
    def analyze_trajectory_for_loops(
        self,
        steps: list
    ) -> Optional[LoopPattern]:
        """Analyze dead loop patterns in the trajectory

        Parameters
        ----------
        steps : list
            List of trajectory steps, can be dict or _Step objects

        Returns
        -------
        LoopPattern | None
            If an obvious loop is detected, returns the pattern description
        """
        if len(steps) < 3:
            return None  # Need at least 3 steps to detect consecutive repeats

        # Compute signature for each step (hash of command + output)
        signatures = []
        for step in steps:
            sig = self._compute_step_signature(step)
            signatures.append(sig)

        # Detect consecutive repeats
        consecutive_repeats = self._find_consecutive_repeats(signatures)
        if consecutive_repeats:
            return consecutive_repeats

        # Detect alternating patterns (A-B-A-B-A-B)
        alternating_pattern = self._find_alternating_pattern(signatures, steps)
        if alternating_pattern:
            return alternating_pattern

        return None
    
    def detect_inefficient_approach(
        self,
        steps: list,
        step_limit: int
    ) -> Optional[str]:
        """Detect inefficient solution approaches

        Criteria: Detect repetition in Python code content itself
        - Extract actual Python code executed in each step
        - Normalize (remove comments, numbers, whitespace) and compute hash
        - If >60% of code blocks have the same hash → logic is highly repetitive, no substantive improvement
        """
        if len(steps) < step_limit * 0.7:
            return None  # Not enough steps to judge

        # Extract Python code signature from each step
        code_sigs = []
        for step in steps:
            cmd = getattr(step, 'command', '') if hasattr(step, 'command') else step.get('command', '')
            code = self._extract_python_code(cmd)
            if code:
                sig = self._normalize_code_signature(code)
                code_sigs.append(sig)

        if len(code_sigs) < 3:
            return None  # Too few code blocks to judge

        # Check proportion of most common code signature
        counts = Counter(code_sigs)
        most_common_sig, most_common_count = counts.most_common(1)[0]

        if most_common_count / len(code_sigs) > 0.6:
            return (f"Inefficient approach: {most_common_count}/{len(code_sigs)} code blocks "
                    f"are highly similar (identical after normalization), no substantive improvement in solution logic")

        return None

    # ------------------------------------------------------------------ #
    # Internal utility methods
    # ------------------------------------------------------------------ #
    
    def _compute_step_signature(self, step) -> str:
        """Compute step signature (for detecting repetition)

        Parameter can be:
        - dict: {'action': ..., 'observation': ...}
        - _Step: dataclass with .command and .output
        """
        # Handle different step formats
        if hasattr(step, 'command'):  # _Step object
            command = getattr(step, 'command', '').strip().lower()
            output = getattr(step, 'output', '')[:200].strip().lower()
        elif isinstance(step, dict):  # Dictionary
            command = step.get("action", step.get("command", "")).strip().lower()
            output = step.get("observation", step.get("output", ""))[:200].strip().lower()
        else:
            # Other types, try converting to string
            command = str(step).lower()
            output = ""

        # Simplify: remove numbers, spaces
        import re
        command = re.sub(r'\d+', 'N', command)
        output = re.sub(r'\d+', 'N', output)

        content = f"{command}::{output}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _find_consecutive_repeats(
        self,
        signatures: list[str]
    ) -> Optional[LoopPattern]:
        """Find consecutive repetition patterns"""
        if len(signatures) < 3:
            return None

        # Check if the most recent 3-5 steps are repeated
        for window_size in [3, 4, 5]:
            if len(signatures) < window_size:
                continue

            recent = signatures[-window_size:]
            if len(set(recent)) == 1:  # All the same
                return LoopPattern(
                    start_step=len(signatures) - window_size + 1,
                    end_step=len(signatures),
                    loop_count=window_size,
                    action_signature=recent[0],
                    description=f"Consecutive {window_size} steps executing the same action"
                )

        return None

    def _find_alternating_pattern(
        self,
        signatures: list[str],
        steps: list[dict]
    ) -> Optional[LoopPattern]:
        """Find alternating patterns (A-B-A-B-A-B)"""
        if len(signatures) < 4:
            return None

        # Check if the most recent 4-6 steps show A-B alternation
        recent_sigs = signatures[-6:]

        if len(recent_sigs) >= 4:
            # Check for A-B-A-B pattern
            if (len(set(recent_sigs[::2])) == 1 and  # Even positions same
                len(set(recent_sigs[1::2])) == 1 and  # Odd positions same
                recent_sigs[0] != recent_sigs[1]):    # Both different

                return LoopPattern(
                    start_step=len(signatures) - len(recent_sigs) + 1,
                    end_step=len(signatures),
                    loop_count=len(recent_sigs) // 2,
                    action_signature=f"{recent_sigs[0]}<->{recent_sigs[1]}",
                    description=f"Detected A-B alternating pattern, repeated {len(recent_sigs)//2} times"
                )

        return None
    
    def _extract_python_code(self, cmd: str) -> str:
        """Extract actual Python code content from shell command

        Supports two formats:
        1. heredoc: python3 << 'EOF'\n...code...\nEOF (with or without end marker)
        2. python3 -c '...'
        """
        # heredoc format (compatible with or without EOF ending)
        m = re.search(
            r"python3\s*<<\s*['\"]?EOF['\"]?\s*\n(.*?)(?:\nEOF\s*$|\Z)",
            cmd, re.DOTALL
        )
        if m:
            return m.group(1).strip()
        # python3 -c format
        m = re.search(r"python3\s+-c\s+['\"](.+?)['\"]", cmd, re.DOTALL)
        if m:
            return m.group(1).strip()
        return ""

    def _normalize_code_signature(self, code: str) -> str:
        """Normalize Python code and compute hash, to judge if logic is repetitive

        Normalization rules:
        1. Remove comments (# lines)
        2. Replace all numbers with N (different parameters considered same logic)
        3. Replace string literals with STR
        4. Merge consecutive whitespace
        5. Convert to lowercase and take first 8 chars of MD5
        """
        code = re.sub(r'#.*', '', code)                     # Remove comments
        code = re.sub(r'["\'].*?["\']', 'STR', code)       # Remove strings
        code = re.sub(r'\b\d+\.?\d*\b', 'N', code)         # Numbers → N
        code = re.sub(r'\s+', ' ', code).strip().lower()    # Compress whitespace + lowercase
        return hashlib.md5(code.encode()).hexdigest()[:8]
