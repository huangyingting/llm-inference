#!/bin/bash

# LLAMA
docker run -d -it --rm --gpus all -v ~/models:/models -p 8192:8080 -e PRELOAD_MODELS='[{"url":"github:huangyingting/llm-inference/localai/vicuna-7b-v1.5.yaml"}]' huangyingting/localai


SECONDS=0
curl http://localhost:8192/v1/chat/completions -H "Content-Type: application/json" -d '{
     "model": "vicuna-7b-v1.5",
     "messages": [{"role": "user", "content": "What is large language model?"}],
     "temperature": 0.9 
   }'

duration=$SECONDS
echo ""
echo "$duration seconds elapsed."

# Whisper
docker run -d -it --rm --gpus all -v ~/models:/models -p 8192:8080 -e PRELOAD_MODELS='[{"url":"github:huangyingting/llm-inference/localai/whisper.yaml"}]' huangyingting/localai

wget --quiet --show-progress -O gb.ogg https://upload.wikimedia.org/wikipedia/commons/1/1f/George_W_Bush_Columbia_FINAL.ogg

curl http://localhost:8192/v1/audio/transcriptions -H "Content-Type: multipart/form-data" -F file="@$PWD/gb.ogg" -F model="whisper"    