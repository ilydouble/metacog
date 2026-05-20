# Pandas 逻辑调试。它脱离了执行器框架，直接运行 reindex 和相关的比较操作。
# 从代码看，它是在调试为什么 df_new['Type'] != df['Type'] 会报错或结果不符合预期（通常是因为 reindex 后索引不一致导致无法直接比较）

import pandas as pd
import numpy as np

df = pd.DataFrame({'Type': [1,1,2,2,3,3]})
List = [2, 4, 0, 3, 1, 5]
df_new = df.reindex(List)
print("df_new index:", df_new.index)
print("df index:", df.index)
try:
    res = df_new['Type'] != df['Type']
    print("Success! Res:", res.values)
except Exception as e:
    print("Error:", repr(e))
