#调用
  curl https://u155745-827f-dc6ad2dd.westc.gpuhub.com:8443/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/autodl-tmp/huggingface/Qwen3.5-9B",
    "api_key": "sk_123456",
    "prompt": "Hello, my name is",
    "max_tokens": 20,
    "temperature": 0
  }'
#查看模型列表
curl https://u155745-827f-dc6ad2dd.westc.gpuhub.com:8443/v1/models

nohup python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/huggingface/Qwen3.5-9B \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 3 \
    --gpu-memory-utilization 0.95 \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    > vllm.log 2>&1 &