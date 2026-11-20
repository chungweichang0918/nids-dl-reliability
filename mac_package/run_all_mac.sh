#!/bin/bash
# 在 Mac 上一次跑完所有剩下的訓練 (可隨時 Ctrl+C，再執行一次會接續):
#   1) NSL-KDD 的 RNN 種子 1-5 (二元 + 五類，全部在同一台機器上跑)
#   2) UNSW-NB15 全部模型 (預設: RNN 以外各 3 個種子、RNN 1 個種子)
#   3) 全部成功才自動打包結果
cd "$(dirname "$0")"
source .venv/bin/activate
export KERAS_BACKEND=torch
# Apple GPU (MPS) 不支援循環層初始化用到的 QR 分解，改用 CPU (實測 RNN 約 6.7 秒/epoch)
export KERAS_TORCH_DEVICE=cpu

caffeinate -i bash -c '
python multiseed.py --models RNN --seeds 1 2 3 4 5 &&
python unsw.py
' && ./collect.sh
