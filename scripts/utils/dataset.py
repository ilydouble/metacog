"""Dataset Loading Utilities

Unified dataset loading for all math test scripts.
Supports both JSONL (one JSON object per line) and JSON array formats.
"""

import json
from pathlib import Path


def load_dataset(
    data_source: str,
    base_path: str,
    max_instances: int | None = None,
    start: int = 0,
) -> list[dict]:
    """Load a math dataset from disk.

    Supports both JSONL format (one JSON object per line) and JSON array format.

    Args:
        data_source: Dataset name, e.g. "aime24", "aime25", "amc23"
        base_path: Directory that contains ``{data_source}.json``
        max_instances: Maximum number of problems to load (None = all)
        start: Zero-based index of the first problem to include

    Returns:
        List of problem dicts

    Raises:
        FileNotFoundError: If the data file does not exist
    """
    data_file = Path(base_path) / f"{data_source}.json"
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    problems: list[dict] = []

    # Try JSONL first (one object per line)
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                problems.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # Fallback: whole file is a JSON array or single object
    if not problems:
        content = data_file.read_text()
        data = json.loads(content)
        if isinstance(data, list):
            problems = data
        elif isinstance(data, dict):
            problems = [data]

    # Apply start offset and limit
    problems = problems[start:]
    if max_instances is not None:
        problems = problems[:max_instances]

    return problems
