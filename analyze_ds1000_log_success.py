#!/usr/bin/env python3
"""Count DS1000 solved items from experiment result files.

This script matches the current DS1000 logging logic:
`test_feedback` only contains failed attempts. Therefore:

- `is_solved == True` means the item was eventually solved.
- `len(test_feedback) == 0` means it was solved on the first attempt.
- `len(test_feedback) == 1` means the first attempt failed and the second
  attempt solved it.

You may pass either result files (`.json` / `.jsonl`) or the matching stdout
`.log` files. When a `.log` path is passed, the script automatically looks for
a same-name `.json` or `.jsonl` file in the same directory.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass
class ItemResult:
    item_id: int
    is_solved: bool
    failed_attempts: int


@dataclass
class FileSummary:
    input_path: str
    result_path: str
    items: List[ItemResult] = field(default_factory=list)

    @property
    def solved_items(self) -> List[ItemResult]:
        return [item for item in self.items if item.is_solved]

    @property
    def first_attempt_solved_items(self) -> List[ItemResult]:
        return [
            item
            for item in self.solved_items
            if item.failed_attempts == 0
        ]

    @property
    def second_attempt_solved_items(self) -> List[ItemResult]:
        return [
            item
            for item in self.solved_items
            if item.failed_attempts == 1
        ]


def expand_paths(patterns: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches or [pattern])
    return sorted(dict.fromkeys(paths))


def result_path_for(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    if ext in {".json", ".jsonl"}:
        return input_path
    if ext == ".log":
        for candidate in (base + ".json", base + ".jsonl"):
            if os.path.exists(candidate):
                return candidate
    return input_path


def load_records(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def item_id_for(record: Dict[str, Any], fallback_index: int) -> int:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        problem_id = metadata.get("problem_id")
        if isinstance(problem_id, int):
            return problem_id
    return fallback_index + 1


def parse_result_file(input_path: str) -> FileSummary:
    result_path = result_path_for(input_path)
    records = load_records(result_path)
    summary = FileSummary(input_path=input_path, result_path=result_path)

    for index, record in enumerate(records):
        test_feedback = record.get("test_feedback") or []
        if not isinstance(test_feedback, list):
            test_feedback = []

        summary.items.append(
            ItemResult(
                item_id=item_id_for(record, index),
                is_solved=bool(record.get("is_solved")),
                failed_attempts=len(test_feedback),
            )
        )

    return summary


def format_ids(items: List[ItemResult]) -> str:
    return ", ".join(str(item.item_id) for item in items) if items else "-"


def print_summary(summary: FileSummary, show_items: bool) -> None:
    solved = summary.solved_items
    first = summary.first_attempt_solved_items
    second = summary.second_attempt_solved_items
    failed = [item for item in summary.items if not item.is_solved]

    print(f"\n{summary.input_path}")
    if summary.result_path != summary.input_path:
        print(f"  result_file: {summary.result_path}")
    print(f"  total_items: {len(summary.items)}")
    print(f"  solved: {len(solved)}")
    print(f"  first_attempt_solved: {len(first)}")
    print(f"  first_wrong_second_correct: {len(second)}")
    print(f"  failed: {len(failed)}")

    if show_items:
        print(f"  solved_items: {format_ids(solved)}")
        print(f"  first_attempt_solved_items: {format_ids(first)}")
        print(f"  first_wrong_second_correct_items: {format_ids(second)}")
        print(f"  failed_items: {format_ids(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count DS1000 solved items using current test_feedback logic."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Result .json/.jsonl files, matching .log files, or glob patterns.",
    )
    parser.add_argument(
        "--show-items",
        action="store_true",
        help="Print item ids for each category.",
    )
    args = parser.parse_args()

    summaries: List[FileSummary] = []
    for input_path in expand_paths(args.paths):
        result_path = result_path_for(input_path)
        if not os.path.exists(result_path):
            print(f"\n{input_path}")
            print("  error: matching .json/.jsonl result file not found")
            continue
        summaries.append(parse_result_file(input_path))

    for summary in summaries:
        print_summary(summary, args.show_items)

    if len(summaries) <= 1:
        return

    print("\nTOTAL")
    print(f"  files: {len(summaries)}")
    print(f"  total_items: {sum(len(summary.items) for summary in summaries)}")
    print(f"  solved: {sum(len(summary.solved_items) for summary in summaries)}")
    print(
        "  first_attempt_solved: "
        f"{sum(len(summary.first_attempt_solved_items) for summary in summaries)}"
    )
    print(
        "  first_wrong_second_correct: "
        f"{sum(len(summary.second_attempt_solved_items) for summary in summaries)}"
    )
    print(
        "  failed: "
        f"{sum(len([item for item in summary.items if not item.is_solved]) for summary in summaries)}"
    )


if __name__ == "__main__":
    main()
