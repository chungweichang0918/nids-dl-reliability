#!/bin/bash
# Mac mini: 公平調參 (SimpleRNN 與 DNN)。可隨時 Ctrl+C，再執行一次會接續。
cd "$(dirname "$0")"
source .venv/bin/activate
export KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cpu

# 第一次執行時先整理 CIC-IDS2017 (約 2-5 分鐘)
if [ ! -f data/cicids2017/improved.npz ]; then
  python cicids.py || exit 1
fi
caffeinate -i python tune.py --models RNN DNN "$@" && ./collect_v2.sh
