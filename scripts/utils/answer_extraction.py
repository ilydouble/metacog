"""Answer Extraction Utilities

Extracts mathematical answers from agent outputs, supporting multiple formats.
"""

import re
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answers in \\boxed{} format from text.

    Args:
        text: Text containing the answer

    Returns:
        The extracted answer string, or None if not found
    """
    # Match \boxed{...} format
    # Use non-greedy matching to handle nested braces
    pattern = r'\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)

    if matches:
        # Return the content of the last \boxed{} (usually the final answer)
        return matches[-1].strip()

    # Try more lenient matching
    pattern2 = r'\\boxed\s*\{([^}]+)\}'
    matches2 = re.findall(pattern2, text)
    if matches2:
        return matches2[-1].strip()

    return None


def extract_final_answer(text: str) -> Optional[str]:
    """Extract the final answer from text, trying multiple formats.

    Supported formats:
    - \\boxed{answer}
    - The answer is: XXX
    - Final answer: XXX
    - Answer: XXX

    Args:
        text: The text containing the answer

    Returns:
        The extracted answer string, or None if not found
    """
    if not text:
        return None

    # If it's purely a number, return it directly
    if re.match(r'^\s*-?\d+\.?\d*\s*$', text) or re.match(r'^\s*-?\d+/\d+\s*$', text):
        return text.strip()

    # 1. First try \boxed{} format
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    # 2. Try "The answer is", "Final answer" etc.
    patterns = [
        r'Final answer(?: is)?\s*[:：]?\s*([^\n]+)',
        r'The answer is\s*[:：]?\s*([^\n]+)',
        r'answer\s*[:：]\s*([^\n]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    
    # 3. Try to match numbers in the last line (as a fallback)
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue

        # First check if the line is purely a number or fraction
        if re.match(r'^-?\d+\.?\d*$', line) or re.match(r'^-?\d+/\d+$', line):
            return line

        # Extract the only number in the line
        # e.g. "The answer is 123" or "=> 123"
        # Avoid extracting if there are multiple numbers
        # Use word boundaries to ensure matching complete numbers
        nums = re.findall(r'(?<!\d)-?\d+\.?\d*(?!\d)|(?<!\d)-?\d+/\d+(?!\d)', line)
        if len(nums) == 1:
            return nums[0]

    return None


def normalize_answer(answer: str) -> str:
    """Normalize the answer format.

    Processing steps:
    - Strip leading/trailing whitespaces and newlines
    - Remove LaTeX formatting symbols (\\boxed{x} -> x etc.)
    - Remove $ symbols
    - Remove common punctuation at ends (periods, commas, etc.)
    - Handle leading zeros (e.g., 073 -> 73)
    - Handle decimal format (remove trailing zeros)
    - Handle fractions (simplify)

    Note: Makes no dataset-specific assumptions about answer content
    (e.g., "answer must be an integer"), keeping it general for integers,
    decimals, fractions, and text answers.

    Args:
        answer: Original answer string

    Returns:
        Normalized answer string
    """
    if not answer:
        return ""

    # Strip whitespaces
    answer = answer.strip()

    # Remove LaTeX symbols
    answer = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)

    # Remove $ symbols
    answer = answer.replace('$', '')

    # Remove common punctuation at ends, then strip again
    answer = answer.strip('.,;:!?')
    answer = answer.strip()

    # Try processing numeric values
    try:
        # Check if it's an integer (possibly with leading zeros)
        if re.match(r'^-?0*\d+$', answer):
            return str(int(answer))

        # Check if it's a decimal
        if re.match(r'^-?\d+\.\d+$', answer):
            # Remove trailing zeros
            return str(float(answer)).rstrip('0').rstrip('.')

        # Check if it's a fraction
        if re.match(r'^-?\d+/\d+$', answer):
            parts = answer.split('/')
            num, den = int(parts[0]), int(parts[1])
            from math import gcd
            g = gcd(abs(num), den)
            return f"{num // g}/{den // g}"

        # Fallback: try to extract the only integer in the answer string
        # (Handles cases with messy text like "The answer is 123")
        nums = re.findall(r'(?<!\d)-?\d+(?!\d)', answer)
        if len(nums) == 1:
            return str(int(nums[0]))

        # Last resort: if there is only one number (integer or decimal), extract it
        nums = re.findall(r'(?<!\d)-?\d+\.?\d*(?!\d)', answer)
        if len(nums) == 1:
            return str(float(nums[0])).rstrip('0').rstrip('.')

    except (ValueError, ZeroDivisionError):
        pass

    return answer


def extract_and_normalize(text: str) -> Optional[str]:
    """Extract and normalize the answer.

    Args:
        text: Text containing the answer

    Returns:
        The normalized answer string, or None if not found
    """
    answer = extract_final_answer(text)
    if answer:
        return normalize_answer(answer)
    return None
