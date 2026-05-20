#  稳定性（Flakiness）测试。它将同一段代码循环运行 100 次，检查是否每次都能通过。
# 这通常是为了排查环境（如 Pandas/Numpy 状态）是否存在非确定性导致的结果波动


from executors.ds1000_executor import DS1000Executor
import pandas as pd
import numpy as np
import copy
from utils import read_jsonl

dataset = read_jsonl('./ds1000_data/ds1000-single.jsonl')
item = dataset[1]
exe = DS1000Executor()
code = "df_new = df.reindex(List)\nresult = sum(df_new['Type'] != df['Type'])"
passes = 0
for i in range(100):
    is_passing, feedback, _ = exe.execute(code, [item['code_context']], timeout=10)
    if is_passing: passes += 1
print('Pass times out of 100:', passes)
