"""Program-of-Thoughts (PoT) Reflector - 程序辅助反思模块

核心思路
--------
1. 检测轨迹中的数学计算错误
2. 生成 Python/SymPy 验证代码
3. 实际执行验证代码
4. 提取可复用的代码模式
5. 将代码逻辑（而非文字描述）存入 actionable_advice

这样下次检索时得到的是：
  "遇到此类方程，请使用 sympy.solve(Eq(x**2 + 3*x - 10, 0), x)"
而不是：
  "求解二次方程时要注意判别式的正负"
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
    """程序辅助验证结果"""
    code: str  # 验证代码
    output: str  # 执行输出
    success: bool  # 是否成功执行
    correct_result: Optional[str] = None  # 提取的正确结果
    reusable_pattern: Optional[str] = None  # 可复用的代码模式


class PoTReflector:
    """程序辅助反思器
    
    负责：
    1. 分析失败轨迹，识别数学计算错误
    2. 生成验证代码
    3. 执行并提取正确逻辑
    """
    
    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout
    
    def generate_verification_code(
        self,
        problem: str,
        failed_output: str,
        summaries: list[str],
    ) -> str:
        """生成验证代码的 prompt
        
        这个方法会被 LLM 调用，生成一段 Python 代码来验证数学计算
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
        """执行验证代码并返回结果
        
        参数
        ----
        code : str
            要执行的 Python 代码
            
        返回
        ----
        PoTVerification
            包含执行结果和可复用模式
        """
        # 清理代码（去除 markdown 代码块标记）
        code = self._clean_code(code)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)
        
        try:
            # 执行代码
            result = subprocess.run(
                ['python3', str(temp_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            
            # 提取正确结果（从输出的最后一行）
            correct_result = None
            if success and output.strip():
                lines = output.strip().split('\n')
                correct_result = lines[-1] if lines else None
            
            # 提取可复用的代码模式
            reusable_pattern = self._extract_reusable_pattern(code, success)
            
            return PoTVerification(
                code=code,
                output=output[:500],  # 限制输出长度
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
            # 清理临时文件
            try:
                temp_path.unlink()
            except Exception:
                pass
    
    def _clean_code(self, code: str) -> str:
        """清理代码，去除 markdown 标记"""
        # 去除 ```python ... ``` 包装
        code = re.sub(r'^```python\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*\n', '', code, flags=re.MULTILINE)
        return code.strip()
    
    def _extract_reusable_pattern(self, code: str, success: bool) -> Optional[str]:
        """从成功的代码中提取可复用的模式
        
        例如：
        - sympy.solve(...) 调用
        - pow(a, -1, m) 模逆元计算
        - factorial / binomial 组合计算
        """
        if not success:
            return None
        
        patterns = []
        
        # 检测 sympy.solve 使用
        if 'solve(' in code:
            patterns.append("Use sympy.solve() for equation solving")
        
        # 检测模运算
        if 'pow(' in code and '-1' in code:
            patterns.append("Use pow(a, -1, m) for modular inverse")
        
        # 检测组合数学
        if 'factorial' in code or 'binomial' in code:
            patterns.append("Use sympy.factorial() or binomial() for combinatorics")
        
        # 检测简化
        if 'simplify' in code or 'expand' in code:
            patterns.append("Use sympy.simplify() to reduce complex expressions")
        
        return "; ".join(patterns) if patterns else None
