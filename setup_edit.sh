#!/usr/bin/env bash

# Exit immediately if any command fails
set -e
apt update

# Install ssh
apt install -y openssh-client
apt-get install git-lfs

git config --global user.email "zhengyao.gu30@gmail.com"
git config --global user.name "Zhengyao Gu"
echo "Git config set"