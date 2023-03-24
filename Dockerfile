FROM nvcr.io/nvidia/pytorch:23.02-py3

LABEL maintainer="David Morrison"

RUN pip install tqdm

COPY . /workspace/simcoder-pytorch

WORKDIR /workspace/simcoder-pytorch
# ENTRYPOINT [ "python", "simcoder" ]