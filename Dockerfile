FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt -y update
RUN apt -y install \
    automake \
    build-essential \
    cmake \
    curl \
    git \
    htop \
    openssh-server \
    python3 \
    python3-dev \
    python3-pip \
    sudo \
    tmux \
    unzip \
    vim \
    wget \
    zip
RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /root

RUN git clone https://github.com/microsoft/nni.git

WORKDIR /root/nni

RUN pip install --upgrade setuptools pip wheel

RUN python setup.py develop

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

RUN pip install lightning tensorflow

WORKDIR /root

CMD ["/bin/bash"]
