docker run --gpus all -it --rm --shm-size=8g -e MODEL=facebook/opt-125m -v ~/models:/models -p 8192:8000 huangyingting/vllm

curl http://localhost:8192/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "How are you?",
        "max_tokens": 1024,
        "temperature": 0.7
    }'

curl http://localhost:8192/v1/chat/completions -H "Content-Type: application/json" -d '{
     "model": "facebook/opt-125m",
     "messages": [{"role": "user", "content": "What is large language model?"}],
     "temperature": 0.9 
   }'