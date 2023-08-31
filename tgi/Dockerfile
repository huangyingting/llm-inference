ARG CUDA_VERSION="11.8.0"
ARG CUDNN_VERSION="8"
ARG UBUNTU_VERSION="22.04"
ARG TGI_VERSION="v1.0.2"

# Base NVidia CUDA Ubuntu image
FROM nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION as builder
ARG VLLM_VERSION
ENV PATH="/usr/local/cuda/bin:/root/.cargo/bin:${PATH}"
WORKDIR /build
RUN apt update -y && \
    apt install -y python3 python3-pip git curl pkg-config libssl-dev protobuf-compiler python-is-python3 && \
    python -m pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu118 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    git clone https://github.com/huggingface/text-generation-inference && \
    cd text-generation-inference && \
    git checkout ${TGI_VERSION} && \
    cargo build --release

FROM builder as flash-att-builder
WORKDIR /build
COPY --from=builder /build/text-generation-inference/server/Makefile-flash-att Makefile
RUN make build-flash-attention

# Build Flash Attention v2 CUDA kernels
FROM builder as flash-att-v2-builder
WORKDIR /build
COPY --from=builder /build/text-generation-inference/server/Makefile-flash-att-v2 Makefile
# Build specific version of flash attention v2
RUN make build-flash-attention-v2

# Build Transformers exllama kernels
FROM builder as exllama-kernels-builder
WORKDIR /build
COPY --from=builder /build/text-generation-inference/server/exllama_kernels/ .
# Build specific version of transformers
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" python setup.py build

# Build Transformers CUDA kernels
FROM builder as custom-kernels-builder
WORKDIR /build
COPY --from=builder /build/text-generation-inference/server/custom_kernels/ .
# Build specific version of transformers
RUN python setup.py build

# Build vllm CUDA kernels
FROM builder as vllm-builder
WORKDIR /build
COPY --from=builder /build/text-generation-inference/server/Makefile-vllm Makefile
# Build specific version of vllm
RUN pip install packaging && make build-vllm

# Text Generation Inference base image
FROM nvidia/cuda:$CUDA_VERSION-runtime-ubuntu$UBUNTU_VERSION
WORKDIR /app
# Text Generation Inference base env
ENV HUGGINGFACE_HUB_CACHE=/models \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

# Copy build artifacts from flash attention builder
COPY --from=flash-att-builder /build/flash-attention/build/lib.linux-x86_64-3.10 /usr/local/lib/python3.10/dist-packages
COPY --from=flash-att-builder /build/flash-attention/csrc/layer_norm/build/lib.linux-x86_64-3.10 /usr/local/lib/python3.10/dist-packages
COPY --from=flash-att-builder /build/flash-attention/csrc/rotary/build/lib.linux-x86_64-3.10 /usr/local/lib/python3.10/dist-packages

# Copy build artifacts from flash attention v2 builder
COPY --from=flash-att-v2-builder /build/flash-attention-v2/build/lib.linux-x86_64-3.10 /usr/local/lib/python3.10/dist-packages

# Copy build artifacts from custom kernels builder
COPY --from=custom-kernels-builder /build/build/lib.linux-x86_64-3.10 /usr/local/lib/python3.10/dist-packages
# Copy build artifacts from exllama kernels builder
COPY --from=exllama-kernels-builder /build/build/lib.linux-x86_64-3.10 /usr/local/lib/python3.10/dist-packages

# Copy builds artifacts from vllm builder
COPY --from=vllm-builder /build/vllm/build/lib.linux-x86_64-3.10 /usr/local/lib/python3.10/dist-packages

COPY --from=builder /build/text-generation-inference/proto proto
COPY --from=builder /build/text-generation-inference/server server
COPY --from=builder /build/text-generation-inference/server/Makefile server/Makefile
COPY --from=builder /build/text-generation-inference/server/requirements.txt server/requirements.txt

RUN apt update -y && \
    apt install -y python3 python3-pip python-is-python3 && \
    python -m pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir && \
    pip install einops --no-cache-dir && \
    cd server && \
    make gen-server && \
    pip install -r requirements.txt --no-cache-dir && \
    pip install ".[bnb, accelerate, quantize]" --no-cache-dir && \
    apt remove -y python3-pip && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install benchmarker
COPY --from=builder /build/text-generation-inference/target/release/text-generation-benchmark /usr/local/bin/text-generation-benchmark
# Install router
COPY --from=builder /build/text-generation-inference/target/release/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=builder /build/text-generation-inference/target/release/text-generation-launcher /usr/local/bin/text-generation-launcher

ENTRYPOINT ["text-generation-launcher"]
CMD ["--json-output"]