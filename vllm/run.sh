#!/bin/bash

# Start the container
MODEL=lmsys/vicuna-7b-v1.5
VOLUME=~/models
GPU_MEMORY_UTILIZATION=0.2
docker run -d --gpus all -it --rm --shm-size=8g -v $VOLUME:/models -e MODEL=$MODEL -e GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION -p 8192:8000 huangyingting/vllm

# Test model
SECONDS=0
curl http://localhost:8192/v1/chat/completions -H "Content-Type: application/json" -d '{
     "model": "lmsys/vicuna-7b-v1.5",
     "max_tokens": 1024,
     "messages": [{"role": "user", "content": "What is large language model?"}],
     "temperature": 0.9 
   }'
duration=$SECONDS
echo ""
echo "$duration seconds elapsed."