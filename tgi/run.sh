#!/bin/bash

# Start the container
MODEL=lmsys/vicuna-7b-v1.5
VOLUME=~/models
MAX_INPUT_LENGTH=1024
MAX_TOTAL_TOKENS=2048
docker run --gpus all --shm-size 1g -p 8192:80 -v $VOLUME:/models huangyingting/tgi --model-id $MODEL --max-input-length $MAX_INPUT_LENGTH --max-total-tokens $MAX_TOTAL_TOKENS

# Test model
SECONDS=0
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is large language model?","parameters":{"max_new_tokens":512}}' \
    -H 'Content-Type: application/json'
duration=$SECONDS
echo ""
echo "$duration seconds elapsed."    