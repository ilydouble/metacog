export OPENAI_API_KEY="sk-IpDxN2TFJC61y5MPyZhH3KTjDLhV4bjWLMN7gOGbWeSl4HFA"
export OPENAI_API_BASE="https://yunwu.ai/v1"

python main.py \
  --log_name "test3" \
  --run_name "reflexion_ds1000" \
  --root_dir "root" \
  --dataset_path ./ds1000_data/ds10002.jsonl \
  --dataset_type "ds1000" \
  --strategy "reflexion" \
  --language "py" \
  --model "gpt-4" \
  --pass_at_k 1 \
  --max_iters 2 \
  --verbose