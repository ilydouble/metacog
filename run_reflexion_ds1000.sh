# export OPENAI_API_KEY="sk-9kNPPcNiyBmPOvZIh5jQ7apzLLPhRDmPj83PIx8dtcFJ8tRI"
# export OPENAI_API_BASE="https://yunwu.ai/v1"

# ---- Chat 模型 (智谱 GLM) ----
export OPENAI_API_KEY="2ba769a5762a4659adf51acd19ac5e58.Ej983tTE8sW1rZay"
export OPENAI_API_BASE="https://open.bigmodel.cn/api/paas/v4"

# ---- Embedding 模型 (OpenAI) ----
# 使用 OpenAI 原生端点调用 text-embedding-3-small
export EMBEDDING_API_KEY="sk-aFMshaJhFliKasNwz1ohWwanrEBZSgSfdZYeQepXn9iFqTf2"
export EMBEDDING_API_BASE="https://yunwu.ai/v1"


python main.py \
  --log_name "7_07_no" \
  --run_name "glm-5.1-memory/7_07_no" \
  --root_dir "root3" \
  --dataset_path ./ds1000_data/ds1000.jsonl \
  --dataset_type "ds1000" \
  --strategy "reflexion" \
  --language "py" \
  --model "glm-5.1" \
  --embedding_model "text-embedding-3-small" \
  --reflection_model "glm-5.1" \
  --pass_at_k 1 \
  --max_iters 2 \
  --episodic_memory "false" \
  --procedural_skill "false" \
  --verbose