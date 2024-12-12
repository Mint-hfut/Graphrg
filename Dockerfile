ARG BASE_IMAGE=ubuntu:22.04
ARG PYTHON_VERSION=3.10
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 将当前目录所有文件（除了.dockerignore中排除的rag目录）复制进容器
COPY . /app

EXPOSE 8084 80 443 8000

# 使用 environment.yml 构建 rag 环境
# 如果没有 environment.yml，请根据你的依赖手动安装，比如：RUN conda create -n rag python=3.9 ...
RUN conda env create -f environment.yml -n rag
RUN pip install -U huggingface_hub
ENV HF_ENDPOINT=https://hf-mirror.com
RUN huggingface-cli download --resume-download BAAI/bge-large-zh-v1.5 --local-dir bge-large-zh-v1.5
# 确保后续命令运行在rag环境中
# 一种方式是使用SHELL改变后续指令的前置命令
SHELL ["conda", "run", "-n", "rag", "/bin/bash", "-c"]

# 启动容器时执行的命令
# 使用 & 符号在后台启动vllm服务，然后启动本地server
CMD python use_local_api_server.py
