Reading from ds1000_data/ds1000.jsonl...
Found 7 question types:
  Pandas: 291 problems
  Numpy: 220 problems
  Matplotlib: 155 problems
  Tensorflow: 45 problems
  Scipy: 106 problems
  Sklearn: 115 problems
  Pytorch: 68 problems
Selected 70 problems total
Writing to ds1000_data/ds10.jsonl...
Done!

todo:



Episodic Memory：
                Success?
               /        \
             Yes        No
              │          │
Store last failure   Store final failure
before success         episode
              │          │
     repair_success=True repair_success=False
检索时优先返回 repair_success=true 的经验，没有时再退化到失败经验。


Procedural Memory：

Retrieval Pipeline

                 New Procedural Skill
                         │
                         ▼
             Library Filter (Exact Match)
                         │
                         ▼
     text-embedding-3-small Semantic Retrieval
(task_family + retrieval_key + procedure + avoid_assumption )
                         │
                  Top-5 Candidates
                         │
       (similarity < threshold → directly append)
                         │
                         ▼
           LLM Same-Skill Verification
                         │
                ┌────────┴────────┐
                │                 │
             Merge             Append

模型是 text-embedding-3-small
