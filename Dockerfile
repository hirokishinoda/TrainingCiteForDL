FROM nvcr.io/nvidia/pytorch:22.02-py3

ARG USERNAME=user
ARG GROUPNAME=user
ARG UID=1000
ARG GID=1000
ARG PASSWORD=passwd

# apt-getを更新，sudoをインストール
RUN apt-get update && apt-get install -y sudo

# 使うライブラリをアップデート
RUN apt update \
    && apt install -y \
    wget \
    bzip2 \
    git \
    curl \
    unzip \
    file \
    xz-utils \
    sudo \
    python3 \
    python3-pip

# 必要のないライブラリを削除
RUN apt-get autoremove -y && apt-get clean && \
  rm -rf /usr/local/src/*

# ユーザーの作成
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    echo "$USERNAME   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME
WORKDIR /home/$USERNAME/

COPY requirements.txt /home/$USERNAME/
RUN pip install -r requirements.txt