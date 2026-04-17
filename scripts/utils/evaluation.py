"""Answer Evaluation Utilities

Compare extracted answers with expected answers.
"""

import re
from fractions import Fraction
from typing import Optional


def evaluate_numeric(extracted: str, expected: str) -> bool:
    """Numeric comparison

    Supports integer, decimal, and fraction comparison, with some tolerance.

    Args:
        extracted: The extracted answer
        expected: The expected answer

    Returns:
        True if they match, False otherwise
    """
    try:
        # Try parsing as numbers
        extracted_val = _parse_number(extracted)
        expected_val = _parse_number(expected)

        if extracted_val is None or expected_val is None:
            return False

        # Compare numeric values (allowing small tolerance)
        if isinstance(extracted_val, Fraction) and isinstance(expected_val, Fraction):
            return extracted_val == expected_val

        # Floating point comparison
        diff = abs(float(extracted_val) - float(expected_val))
        return diff < 1e-9
    except (ValueError, TypeError, ZeroDivisionError, AttributeError):
        return False


def _parse_number(s: str) -> Optional[Fraction]:
    """Parse string to fraction (exact numeric value).

    Supported formats:
    - Integer: "73", "-5"
    - Decimal: "3.14", "-0.5"
    - Fraction: "1/2", "-3/4"

    Args:
        s: Numeric string

    Returns:
        Fraction object, or None if parsing fails
    """
    s = s.strip()

    # Remove leading zeros
    if re.match(r'^-?0+\d', s):
        s = s.lstrip('0') if not s.startswith('-') else '-' + s[1:].lstrip('0')
    
    try:
        # Integer
        if re.match(r'^-?\d+$', s):
            return Fraction(int(s), 1)

        # Decimal
        if re.match(r'^-?\d+\.\d+$', s):
            return Fraction(s)

        # Fraction
        if re.match(r'^-?\d+/\d+$', s):
            parts = s.split('/')
            return Fraction(int(parts[0]), int(parts[1]))

        return None
    except (ValueError, ZeroDivisionError):
        return None


def evaluate_string(extracted: str, expected: str) -> bool:
    """String exact match

    Compare two strings to see if they are identical (ignoring case and whitespace).

    Args:
        extracted: The extracted answer
        expected: The expected answer

    Returns:
        True if they match, False otherwise
    """
    if not extracted or not expected:
        return False

    # Normalize: strip whitespace, convert to lowercase
    ext = extracted.strip().lower()
    exp = expected.strip().lower()

    return ext == exp


def compare_answers(extracted: str, expected: str) -> bool:
    """Compare extracted answer with expected answer

    Try different comparison methods in order of priority:
    1. Numeric comparison (supports integers, decimals, fractions)
    2. String exact match

    Args:
        extracted: The extracted answer
        expected: The expected answer

    Returns:
        True if they match, False otherwise
    """
    if not extracted or not expected:
        return False

    # Normalize answers
    extracted = extracted.strip()
    expected = expected.strip()

    # 1. Try numeric comparison
    if evaluate_numeric(extracted, expected):
        return True

    # 2. Try string match
    if evaluate_string(extracted, expected):
        return True

    # 3. Special handling: leading zeros
    # e.g., "073" should equal "73"
    try:
        ext_num = int(extracted)
        exp_num = int(expected)
        if ext_num == exp_num:
            return True
    except ValueError:
        pass

    # 4. If expected is purely numeric, extract the single number from extracted to compare
    if re.match(r'^-?\d+\.?\d*|-?\d+/\d+$', expected):
        # Avoid extracting multiple numbers (if only one is extracted, it's likely the answer)
        nums = re.findall(r'(?<!\d)-?\d+\.?\d*(?!\d)|(?<!\d)-?\d+/\d+(?!\d)', extracted)
        if len(nums) == 1:
            return evaluate_numeric(nums[0], expected)

    return False


def compute_accuracy(results: list[dict]) -> dict:
    """Calculate accuracy statistics

    Args:
        results: List of results, each containing a 'passed' field

    Returns:
        Dictionary with statistical info
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
