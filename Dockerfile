FROM nvidia/cudagl:10.1-base-ubuntu18.04
  
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-key list

RUN apt-get update && apt-get install -y \
  python3.6-dev \
  python3-pip \
  python3-virtualenv \
  libeigen3-dev \
  libopencv-dev \
  libzmqpp-dev \
  libblas-dev \
  ffmpeg \
  cmake \
  tmux \
  vim \
  nano

RUN python3.6 -m pip install -U virtualenv jupyter

ENTRYPOINT jupyter notebook --generate-config && \
    echo 'c.NotebookApp.ip="192.168.86.45"' >> /root/.jupyter/jupyter_notebook_config.py && \
    echo 'c.NotebookApp.allow_root = True' >> /root/.jupyter/jupyter_notebook_config.py && \
    cd /root/mfbo_trajectory && \
    /bin/bash
