FROM nvcr.io/nvidia/pytorch:23.02-py3

LABEL maintainer="David Morrison"

RUN apt update \
    && apt -y upgrade \ 
    && apt install -y python3-pip \
    && apt install -y build-essential libssl-dev libffi-dev python3-dev \
    && apt install python3.8-venv

COPY . /workspace/repos/sisap2023
WORKDIR /workspace/repos/sisap2023
RUN make environment

# add the venv to the path
ENV PATH="/workspace/repos/sisap2023/venv/bin:$PATH"
