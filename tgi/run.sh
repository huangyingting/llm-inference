model=tiiuae/falcon-7b-instruct
volume=$PWD/models # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8192:80 -v $volume:/models huangyingting/tgi --model-id $model