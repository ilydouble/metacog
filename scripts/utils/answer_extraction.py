"""答案提取工具

从 agent 输出中提取数学答案，支持多种格式。
"""

import re
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """从文本中提取 \\boxed{} 格式的答案
    
    Args:
        text: 包含答案的文本
        
    Returns:
        提取的答案字符串，如果未找到则返回 None
    """
    # 匹配 \boxed{...} 格式
    # 使用非贪婪匹配处理嵌套大括号
    pattern = r'\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        # 返回最后一个 \boxed{} 中的内容（通常是最终答案）
        return matches[-1].strip()
    
    # 尝试更宽松的匹配
    pattern2 = r'\\boxed\s*\{([^}]+)\}'
    matches2 = re.findall(pattern2, text)
    if matches2:
        return matches2[-1].strip()
    
    return None


def extract_final_answer(text: str) -> Optional[str]:
    """从文本中提取最终答案，尝试多种格式
    
    支持的格式：
    - \\boxed{答案}
    - 答案是：XXX
    - 最终答案：XXX
    - 答案：XXX
    - The answer is XXX
    
    Args:
        text: 包含答案的文本
        
    Returns:
        提取的答案字符串，如果未找到则返回 None
    """
    # 1. 首先尝试 \boxed{} 格式
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    
    # 2. 尝试 "答案是"、"最终答案" 等格式
    patterns = [
        r'答案[是为][：:]\s*([^\n]+)',
        r'最终答案[是为]?[：:]\s*([^\n]+)',
        r'The answer is\s*[:：]?\s*([^\n]+)',
        r'answer[:：]\s*([^\n]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    
    # 3. 尝试匹配最后一行的数字（作为最后手段）
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        # 匹配纯数字或简单表达式
        if re.match(r'^-?\d+\.?\d*$', line):
            return line
        # 匹配分数
        if re.match(r'^\d+/\d+$', line):
            return line
    
    return None


def normalize_answer(answer: str) -> str:
    """标准化答案格式
    
    处理：
    - 去除前后空格
    - 统一大小写
    - 处理前导零（如 073 -> 73）
    - 处理分数格式
    - 处理小数格式
    
    Args:
        answer: 原始答案字符串
        
    Returns:
        标准化后的答案字符串
    """
    if not answer:
        return ""
    
    # 去除空格和换行
    answer = answer.strip()
    
    # 去除 LaTeX 格式符号
    answer = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)
    
    # 去除 $ 符号
    answer = answer.replace('$', '')
    
    # 去除前后空格
    answer = answer.strip()
    
    # 尝试处理数值
    try:
        # 检查是否是整数（可能带前导零）
        if re.match(r'^-?0*\d+$', answer):
            return str(int(answer))
        
        # 检查是否是小数
        if re.match(r'^-?\d+\.\d+$', answer):
            # 保留原始精度，但去除尾部多余的零
            return str(float(answer)).rstrip('0').rstrip('.')
        
        # 检查是否是分数
        if re.match(r'^-?\d+/\d+$', answer):
            # 保持分数格式，但简化
            parts = answer.split('/')
            num, den = int(parts[0]), int(parts[1])
            # 约分
            from math import gcd
            g = gcd(abs(num), abs(den))
            return f"{num // g}/{den // g}"
    except (ValueError, ZeroDivisionError):
        pass
    
    return answer


def extract_and_normalize(text: str) -> Optional[str]:
    """提取并标准化答案
    
    Args:
        text: 包含答案的文本
        
    Returns:
        标准化后的答案字符串，如果未找到则返回 None
    """
    answer = extract_final_answer(text)
    if answer:
        return normalize_answer(answer)
    return None
