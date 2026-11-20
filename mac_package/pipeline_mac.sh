#!/bin/bash
# Mac mini: 依序執行所有實驗階段 (SimpleRNN 與 DNN)。可隨時 Ctrl+C，再執行一次會接續。
cd "$(dirname "$0")"
source .venv/bin/activate
export KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cpu

if [ ! -f data/cicids2017/improved.npz ]; then
  python cicids.py || exit 1
fi
caffeinate -i python pipeline_v2.py --models RNN DNN "$@" && ./collect_v2.sh
