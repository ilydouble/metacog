# 手动指定了一段 Pandas 代码并调用 DS1000Executor 来运行，
# 目的是验证执行器能否正确处理 DS1000 的题目格式，看是否能返回预期的 is_passing 结果


from utils import read_jsonl
from executors.ds1000_executor import DS1000Executor

dataset = read_jsonl('./ds1000_data/ds1000-single.jsonl')
item = dataset[1]
exe = DS1000Executor()
code = "df_new = df.reindex(List)\nresult = sum(df_new['Type'] != df['Type'])"
is_passing, feedback, _ = exe.execute(code, [item['code_context']], timeout=10)
print('Passing:', is_passing)
print('Feedback:', feedback)
