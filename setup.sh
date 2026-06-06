#!/usr/bin/env bash

# Exit immediately if any command fails
set -e
apt update

# Install ssh
apt install -y openssh-client rsync
apt-get install git-lfs

# Persist Claude Code session history across cluster spins by symlinking
# ~/.claude -> /workspace/mnt/.claude. Idempotent: safe to re-run.
CLAUDE_PERSIST="/workspace/mnt/.claude"
CLAUDE_LOCAL="$HOME/.claude"
mkdir -p "$CLAUDE_PERSIST"
if [ -L "$CLAUDE_LOCAL" ]; then
    if [ "$(readlink "$CLAUDE_LOCAL")" != "$CLAUDE_PERSIST" ]; then
        echo "WARNING: ~/.claude is a symlink to $(readlink "$CLAUDE_LOCAL"); leaving as-is."
    fi
elif [ -d "$CLAUDE_LOCAL" ]; then
    BACKUP="$CLAUDE_LOCAL.bak.$(date +%s)"
    echo "Merging $CLAUDE_LOCAL into $CLAUDE_PERSIST (newer files win), backing up to $BACKUP"
    rsync -a --update "$CLAUDE_LOCAL/" "$CLAUDE_PERSIST/"
    mv "$CLAUDE_LOCAL" "$BACKUP"
    ln -s "$CLAUDE_PERSIST" "$CLAUDE_LOCAL"
else
    rm -f "$CLAUDE_LOCAL"
    ln -s "$CLAUDE_PERSIST" "$CLAUDE_LOCAL"
fi

# Change into the rllm directory
cd "/workspace/mnt/verl_latest"

# SPaCe Phase 1 preprocessing (recipe/deepscaler/space_preprocess.py) — not pulled in
# by any of the verl extras above.
pip install --no-cache-dir sentence-transformers

# Install verl in editable mode
pip install -e .[test,geo,gpu,math,vllm]

git config --global user.email "zhengyao.gu30@gmail.com"
git config --global user.name "Zhengyao Gu"