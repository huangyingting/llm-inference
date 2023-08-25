#!/bin/bash
docker run -d -it --rm --gpus all -v ~/models:/models -p 8192:8080 -e PRELOAD_MODELS='[{"url":"github:huangyingting/llm-inference/localai/orca_mini_v3_7b.yaml"}]' huangyingting/localai

SECONDS=0

curl http://localhost:8192/v1/chat/completions -H "Content-Type: application/json" -d '{
     "model": "orca_mini_v3_7b",
     "messages": [{"role": "user", "content": "What is large language model?"}],
     "temperature": 0.9 
   }'

duration=$SECONDS
echo ""
echo "$duration seconds elapsed."
