import os
import gzip
import json
import openai
import jsonlines
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.theme import Theme

from typing import List

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

# 自定义主题
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "success": "bold green"
})

# 全局配置
console = Console(theme=custom_theme)
_log_file = None
_verbose = True

def setup_logger(verbose: bool, log_file: str = None):
    global _log_file, _verbose
    _verbose = verbose
    _log_file = log_file
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

def print_v(*args, **kwargs):
    if _verbose:
        console.print(*args, **kwargs)
    
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            file_console = Console(file=f, force_terminal=False, width=120)
            file_console.print(*args, **kwargs)

def make_printv(verbose: bool, log_file: str = None):
    # def print_v(*args, **kwargs):
    #     if verbose:
    #         kwargs["flush"] = True
    #         print(*args, **kwargs)
    #     else:
    #         pass
    # return print_v
    setup_logger(verbose, log_file)
    return print_v



def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items += [item]
    return items


def write_jsonl(path: str, data: List[dict], append: bool = False):
    with jsonlines.open(path, mode='a' if append else 'w') as writer:
        for item in data:
            writer.write(item)


def read_jsonl_gz(path: str) -> List[dict]:
    if not path.endswith(".jsonl.gz"):
        raise ValueError(f"File `{path}` is not a jsonl.gz file.")
    with gzip.open(path, "rt") as f:
        data = [json.loads(line) for line in f]
    return data


# generator that returns the item and the index in the dataset.
# if the results_path exists, it will skip all items that have been processed
# before.
def enumerate_resume(dataset, results_path):
    if not os.path.exists(results_path):
        for i, item in enumerate(dataset):
            yield i, item
    else:
        count = 0
        with jsonlines.open(results_path) as reader:
            for item in reader:
                count += 1

        for i, item in enumerate(dataset):
            # skip items that have been processed before
            if i < count:
                continue
            yield i, item


def resume_success_count(dataset) -> int:
    count = 0
    for item in dataset:
        if "is_solved" in item and item["is_solved"]:
            count += 1
    return count

