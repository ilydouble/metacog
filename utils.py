import os
import gzip
import json
import openai
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.theme import Theme

from typing import List

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

# Embedding 专用端点（始终与 chat 分离，不管是否同一提供商）
# 默认走 OpenAI 原生端点；使用其他提供商时通过环境变量覆盖
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.openai.com/v1")

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
    from rich.markup import escape
    
    # 辅助函数，用来将传入的字符/Panel 内部的中括号进行转义，防止被当成格式标签
    def _escape_args(args):
        escaped_args = []
        for arg in args:
            if isinstance(arg, Panel) and isinstance(arg.renderable, str):
                arg.renderable = escape(arg.renderable)
                escaped_args.append(arg)
            elif isinstance(arg, str):
                # 只有当它不是带有 [bold] 之类已知 rich 标签的内容时，才进行转义
                # 为了防止把正常的 [success] 等终端变色标签也干掉，
                # 最安全的办法是对 panel content 做单独 escape（已在上面处理）
                escaped_args.append(arg)
            else:
                escaped_args.append(arg)
        return tuple(escaped_args)

    escaped_args = _escape_args(args)

    if _verbose:
        console.print(*escaped_args, **kwargs)
    
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            file_console = Console(file=f, force_terminal=False, width=120)
            file_console.print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
            file_console.print(*escaped_args, **kwargs)

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
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"File `{path}` must contain a JSON list.")
        return data
    elif path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    else:
        raise ValueError(f"File `{path}` is not a json/jsonl file.")


def write_jsonl(path: str, data: List[dict], append: bool = False):
    if path.endswith(".json"):
        records = []
        if append and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                records = json.load(file)
            if not isinstance(records, list):
                raise ValueError(f"File `{path}` must contain a JSON list.")
        records.extend(data)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)
            file.write("\n")
        return

    with open(path, "a" if append else "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


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
        if results_path.endswith(".json"):
            count = len(read_jsonl(results_path))
        else:
            count = 0
            with open(results_path, "r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        count += 1

        for i, item in enumerate(dataset):
            # skip items that have been processed before
            if i < count:
                continue
            yield i, item


def resume_success_count(dataset, results_path: str = None) -> int:
    # 如果提供了 results_path 且文件存在，从结果文件中统计已解决的题目数
    # 这修复了重新运行时准确率归零的问题：原来的实现从 dataset（原始题目集）
    # 中统计 is_solved，但原始题目集不包含 is_solved 字段，导致重启后 num_success 归零
    if results_path and os.path.exists(results_path):
        count = 0
        try:
            results = read_jsonl(results_path)
            for item in results:
                if item.get("is_solved"):
                    count += 1
        except (json.JSONDecodeError, ValueError):
            pass
        return count

    count = 0
    for item in dataset:
        if "is_solved" in item and item["is_solved"]:
            count += 1
    return count
