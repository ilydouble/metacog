"""答案评估工具

比较提取的答案与预期答案是否匹配。
"""

import re
from fractions import Fraction
from typing import Optional


def evaluate_numeric(extracted: str, expected: str) -> bool:
    """数值比较
    
    支持整数、小数、分数的比较，允许一定的容差。
    
    Args:
        extracted: 提取的答案
        expected: 预期的答案
        
    Returns:
        是否匹配
    """
    try:
        # 尝试解析为数值
        extracted_val = _parse_number(extracted)
        expected_val = _parse_number(expected)
        
        if extracted_val is None or expected_val is None:
            return False
        
        # 比较数值（允许小误差）
        if isinstance(extracted_val, Fraction) and isinstance(expected_val, Fraction):
            return extracted_val == expected_val
        
        # 浮点数比较
        diff = abs(float(extracted_val) - float(expected_val))
        return diff < 1e-9
    except (ValueError, TypeError, ZeroDivisionError):
        return False


def _parse_number(s: str) -> Optional[Fraction]:
    """将字符串解析为分数（精确数值）
    
    支持格式：
    - 整数: "73", "-5"
    - 小数: "3.14", "-0.5"
    - 分数: "1/2", "-3/4"
    
    Args:
        s: 数字字符串
        
    Returns:
        Fraction 对象，解析失败返回 None
    """
    s = s.strip()
    
    # 去除前导零
    if re.match(r'^-?0+\d', s):
        s = s.lstrip('0') if not s.startswith('-') else '-' + s[1:].lstrip('0')
    
    try:
        # 整数
        if re.match(r'^-?\d+$', s):
            return Fraction(int(s), 1)
        
        # 小数
        if re.match(r'^-?\d+\.\d+$', s):
            return Fraction(s)
        
        # 分数
        if re.match(r'^-?\d+/\d+$', s):
            parts = s.split('/')
            return Fraction(int(parts[0]), int(parts[1]))
        
        return None
    except (ValueError, ZeroDivisionError):
        return None


def evaluate_string(extracted: str, expected: str) -> bool:
    """字符串精确匹配
    
    比较两个字符串是否完全相同（忽略大小写和前后空格）。
    
    Args:
        extracted: 提取的答案
        expected: 预期的答案
        
    Returns:
        是否匹配
    """
    if not extracted or not expected:
        return False
    
    # 标准化：去除空格、统一小写
    ext = extracted.strip().lower()
    exp = expected.strip().lower()
    
    return ext == exp


def compare_answers(extracted: str, expected: str) -> bool:
    """比较提取的答案与预期答案
    
    按优先级尝试不同的比较方法：
    1. 数值比较（支持整数、小数、分数）
    2. 字符串精确匹配
    
    Args:
        extracted: 提取的答案
        expected: 预期的答案
        
    Returns:
        是否匹配
    """
    if not extracted or not expected:
        return False
    
    # 标准化答案
    extracted = extracted.strip()
    expected = expected.strip()
    
    # 1. 尝试数值比较
    if evaluate_numeric(extracted, expected):
        return True
    
    # 2. 尝试字符串匹配
    if evaluate_string(extracted, expected):
        return True
    
    # 3. 特殊处理：前导零
    # 例如 "073" 应该等于 "73"
    try:
        ext_num = int(extracted)
        exp_num = int(expected)
        if ext_num == exp_num:
            return True
    except ValueError:
        pass
    
    return False


def compute_accuracy(results: list[dict]) -> dict:
    """计算准确率统计
    
    Args:
        results: 结果列表，每个元素包含 'passed' 字段
        
    Returns:
        统计信息字典
    """
    if not results:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
        }
    
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    failed = total - passed
    pass_rate = passed / total if total > 0 else 0.0
    
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
    }
