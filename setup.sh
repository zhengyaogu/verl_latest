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

# Extra reward-scoring deps not included in verlai/verl image (needed by ifeval scorer)
pip install langdetect nltk immutabledict
python -m nltk.downloader punkt punkt_tab stopwords

git config --global user.email "zhengyao.gu30@gmail.com"
git config --global user.name "Zhengyao Gu"