ARG BASE_IMAGE=ubuntu:22.04
ARG PYTHON_VERSION=3.10
FROM continuumio/miniconda3:latest

WORKDIR /app

COPY . /app

EXPOSE 8084 80 443 8000

RUN conda env create -f environment.yml -n rag
RUN pip install -U huggingface_hub
ENV HF_ENDPOINT=https://hf-mirror.com
RUN huggingface-cli download --resume-download BAAI/bge-large-zh-v1.5 --local-dir bge-large-zh-v1.5

SHELL ["conda", "run", "-n", "rag", "/bin/bash", "-c"]

# CMD python use_local_api_server.py
