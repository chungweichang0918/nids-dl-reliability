#!/bin/bash
# 正式訓練。用法:
#   ./run_mac.sh              跑 RNN 種子 4 5 (建議的分工，筆電負責 1-3)
#   ./run_mac.sh 4 5          指定種子
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo '找不到可執行的 .venv/bin/python。請先在這台 Mac 執行 bash setup_mac.sh。' >&2
    exit 1
fi
export KERAS_BACKEND=torch
export KERAS_TORCH_DEVICE="${KERAS_TORCH_DEVICE:-cpu}"

if [ "$#" -eq 0 ]; then
    set -- 4 5
fi
echo "RNN 種子: $*"
# caffeinate 防止 Mac 在訓練時進入睡眠
caffeinate -i .venv/bin/python multiseed.py --models RNN --seeds "$@"
