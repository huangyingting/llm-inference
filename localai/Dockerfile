ARG CUDA_VERSION="11.8.0"
ARG CUDNN_VERSION="8"
ARG UBUNTU_VERSION="22.04"
ARG GO_VERSION="1.21.0"
ARG BUILD_TYPE="cublas"
ARG GO_TAGS=""
ARG LOCALAI_VERSION="v1.25.0"

# Base NVidia CUDA Ubuntu image
FROM nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION as builder

ARG GO_VERSION
ARG BUILD_TYPE
ARG GO_TAGS
ARG LOCALAI_VERSION

ENV BUILD_TYPE=${BUILD_TYPE}
ENV GO_TAGS=${GO_TAGS}

ENV PATH="/usr/local/cuda/bin:${PATH}:/usr/local/go/bin"
WORKDIR /app
RUN apt update -y && \
    apt install -y git wget cmake && \
    wget https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz && \
    tar -xf go${GO_VERSION}.linux-amd64.tar.gz -C /usr/local && \
    rm -f go${GO_VERSION}.linux-amd64.tar.gz && \
    git clone https://github.com/go-skynet/LocalAI && \
    cd LocalAI && \
    git checkout ${LOCALAI_VERSION} && \
    make get-sources && \
    make prepare && \
    make build

#FROM nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-runtime-ubuntu$UBUNTU_VERSION
FROM nvidia/cuda:$CUDA_VERSION-runtime-ubuntu$UBUNTU_VERSION
WORKDIR /app
ENV THREADS=2
ENV GALLERIES='[{"name":"model-gallery", "url":"github:go-skynet/model-gallery/index.yaml"}, {"url": "github:go-skynet/model-gallery/huggingface.yaml","name":"huggingface"}]'
ENV MODELS_PATH=/models
ENV DEBUG=true
COPY --from=builder /app/LocalAI/local-ai .
RUN mkdir /models && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 8000
ENTRYPOINT [ "./local-ai" ]