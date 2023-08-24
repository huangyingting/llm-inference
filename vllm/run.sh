docker run --gpus all -it --rm --shm-size=8g -e MODEL=lmsys/vicuna-13b-v1.5 -v ~/models:/models -p 8192:8000 huangyingting/vllm
