"""Program-of-Thoughts (PoT) Reflector - Program-assisted reflection module

Core Idea
--------
1. Detect mathematical calculation errors in trajectories
2. Generate Python/SymPy verification code
3. Actually execute the verification code
4. Extract reusable code patterns
5. Store code logic (not text descriptions) in actionable_advice

This way, when retrieving next time, you get:
  "For this type of equation, use sympy.solve(Eq(x**2 + 3*x - 10, 0), x)"
Instead of:
  "When solving quadratic equations, pay attention to the discriminant's sign"
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PoTVerification:
    """Program-assisted verification result"""
    code: str  # Verification code
    output: str  # Execution output
    success: bool  # Whether execution was successful
    correct_result: Optional[str] = None  # Extracted correct result
    reusable_pattern: Optional[str] = None  # Reusable code pattern


class PoTReflector:
    """Program-assisted reflector
    
    Responsibilities:
    1. Analyze failed trajectories, identify mathematical calculation errors
    2. Generate verification code
    3. Execute and extract correct logic
    """
    
    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout
    
    def generate_verification_code(
        self,
        problem: str,
        failed_output: str,
        summaries: list[str],
    ) -> str:
        """Generate verification code prompt
        
        This method will be called by LLM to generate a Python code snippet for mathematical verification
        """
        prompt = f"""You are a math verification code generator.

Given a failed solution attempt, generate a SHORT Python code snippet (max 20 lines) that:
1. Uses sympy to solve the problem CORRECTLY
2. Prints the correct result
3. Is self-contained (no imports needed except sympy)

Problem: {problem[:300]}

Failed attempt summary:
{chr(10).join(summaries[:3])}

Generate ONLY the Python code, no explanation. Start with:
```python
from sympy import *
```"""
        return prompt
    
    def execute_verification_code(self, code: str) -> PoTVerification:
        """Execute verification code and return results
        
        Parameters
        ----------
        code : str
            The Python code to execute
            
        Returns
        -------
        PoTVerification
            Contains execution results and reusable patterns
        """
        # Clean code (remove markdown code block markers)
        code = self._clean_code(code)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)
        
        try:
            # Execute code
            result = subprocess.run(
                ['python3', str(temp_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            
            # Extract correct result (from last line of output)
            correct_result = None
            if success and output.strip():
                lines = output.strip().split('\n')
                correct_result = lines[-1] if lines else None
            
            # Extract reusable code pattern
            reusable_pattern = self._extract_reusable_pattern(code, success)
            
            return PoTVerification(
                code=code,
                output=output[:500],  # Limit output length
                success=success,
                correct_result=correct_result,
                reusable_pattern=reusable_pattern
            )
        
        except subprocess.TimeoutExpired:
            return PoTVerification(
                code=code,
                output=f"Timeout after {self.timeout}s",
                success=False
            )
        except Exception as exc:
            return PoTVerification(
                code=code,
                output=f"Error: {exc}",
                success=False
            )
        finally:
            # Clean up temporary file
            try:
                temp_path.unlink()
            except Exception:
                pass
    
    def _clean_code(self, code: str) -> str:
        """Clean code, remove markdown markers"""
        # Remove ```python ... ``` wrapper
        code = re.sub(r'^```python\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*\n', '', code, flags=re.MULTILINE)
        return code.strip()
    
    def _extract_reusable_pattern(self, code: str, success: bool) -> Optional[str]:
        """Extract reusable patterns from successful code
        
        For example:
        - sympy.solve(...) calls
        - pow(a, -1, m) modular inverse calculation
        - factorial / binomial combinatorial calculations
        """
        if not success:
            return None
        
        patterns = []
        
        # Detect sympy.solve usage
        if 'solve(' in code:
            patterns.append("Use sympy.solve() for equation solving")
        
        # Detect modular arithmetic
        if 'pow(' in code and '-1' in code:
            patterns.append("Use pow(a, -1, m) for modular inverse")
        
        # Detect combinatorics
        if 'factorial' in code or 'binomial' in code:
            patterns.append("Use sympy.factorial() or binomial() for combinatorics")
        
        # Detect simplification
        if 'simplify' in code or 'expand' in code:
            patterns.append("Use sympy.simplify() to reduce complex expressions")
        
        return "; ".join(patterns) if patterns else None
