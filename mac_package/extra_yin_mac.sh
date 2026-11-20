#!/bin/bash
# Mac mini 空檔: 幫 Yin et al. 2017 的「122 個特徵當序列」解讀補跑種子 4、5 (二元 + 五分類)
# 原本只有 3 顆種子，五分類其中 1 顆發散；補到 5 顆才能估計發散比例。約 2 小時。
# 可隨時 Ctrl+C，再執行一次會接續 (已完成的種子會跳過)。
cd "$(dirname "$0")"
source .venv/bin/activate
export KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cpu

caffeinate -i python yin_repro.py --interp featseq --seeds 4 5 && ./collect_v2.sh
