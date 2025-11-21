#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR=${1:-.}   # 默认当前目录

# GNU sed (Linux) 用这个
find "$TARGET_DIR" -type f -name '*.py' -print0 | \
  xargs -0 sed -i 's/jax\.tree_leaves/jax\.tree\.leaves/g'
