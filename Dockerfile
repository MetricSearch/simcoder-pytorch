FROM nvcr.io/nvidia/pytorch:23.02-py3

LABEL maintainer="David Morrison"

RUN pip install tqdm
RUN pip install torchinfo

COPY . /workspace/repos/sisap2023-pytorch

WORKDIR /workspace/repos/sisap2023-pytorch
# ENTRYPOINT [ "python", "sisap2023" ]
