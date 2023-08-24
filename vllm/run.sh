docker run --gpus all -it --rm --shm-size=8g -e MODEL=lmsys/vicuna-7b-v1.5 -v ~/models:/models -p 8192:8000 huangyingting/vllm

curl http://localhost:8192/v1/chat/completions -H "Content-Type: application/json" -d '{
     "model": "lmsys/vicuna-7b-v1.5",
     "max_tokens": 1024,
     "messages": [{"role": "user", "content": "What is large language model?"}],
     "temperature": 0.9 
   }'
