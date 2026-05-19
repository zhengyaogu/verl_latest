#!/usr/bin/env bash

# Exit immediately if any command fails
set -e
apt update

# Install ssh
apt install -y openssh-client
apt-get install git-lfs

# Change into the rllm directory
cd "/workspace/mnt/verl"

# Install verl in editable mode
pip install -e . --no-deps

git config --global user.email "zhengyao.gu30@gmail.com"
git config --global user.name "Zhengyao Gu"