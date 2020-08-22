FROM gitpod/workspace-full:latest

ARG DEBIAN_FRONTEND=noninteractive
RUN sudo apt-get update -qq \
    && sudo apt-get install -yq \
      graphviz \
      xdg-utils \
      build-essential \
      g++ \
      git \
      openssh-client \
      python3 \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-virtualenv \
      python3-wheel \
      pkg-config \
      libopenblas-base \
      python3-numpy \
      python3-scipy \
      python3-h5py \
      python3-yaml \
      python3-pydot \
      cuda \
    && sudo rm -rf /var/lib/apt/lists/*
