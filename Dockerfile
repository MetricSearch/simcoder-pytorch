FROM nvcr.io/nvidia/pytorch:23.02-py3

LABEL maintainer="David Morrison"

RUN pip install tqdm
RUN pip install torchinfo

COPY . /workspace/repos/simcoder-pytorch

WORKDIR /workspace/repos/simcoder-pytorch
# ENTRYPOINT [ "python", "simcoder" ]
