FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=0

RUN apt-get update --fix-missing && \
    apt-get upgrade -y && \
    apt-get install -y git curl nvtop neovim

RUN apt-get install -y python3-pip
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /home/sites

RUN git clone https://github.com/octo-models/octo.git

RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN cd octo && pip install -v -e . && pip install -r requirements.txt
RUN pip install jaxlib==0.4.20
RUN pip install scipy==1.9.0

COPY modifier/dataset.py ./octo/octo/data/dataset.py
WORKDIR /home/sites/octo

CMD ["bash"]