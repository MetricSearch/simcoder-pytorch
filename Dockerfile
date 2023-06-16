FROM nvcr.io/nvidia/pytorch:23.02-py3

LABEL maintainer="David Morrison"

COPY . /workspace/repos/sisap2023
RUN make environment
WORKDIR /workspace/repos/sisap2023

# add the venv to the path
ENV PATH="/workspace/repos/sisap2023/venv/bin:$PATH"
