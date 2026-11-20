#!/bin/bash
# CSE-CIC-IDS2018 — Mac mini CPU 這一條: SimpleRNN、DNN 的全部階段，接著樹模型，最後打包結果。
# GRU、LSTM、CNN-LSTM、CNN 在雲端 GPU 上跑；轉移攻擊在筆電合併所有結果後執行。
# 可隨時 Ctrl+C，再執行一次就會接續。(cic18_gpu.sh 已不使用)
cd "$(dirname "$0")"
source .venv/bin/activate
export KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cpu
python -c "import xgboost" 2>/dev/null || pip install xgboost || exit 1

caffeinate -i python pipeline_v2.py --models RNN DNN --datasets cic18 || exit 1
caffeinate -i python baselines_v2.py --datasets cic18 --jobs 8 || exit 1
./collect_v2.sh
echo "Mac 的 CSE-CIC-IDS2018 部分完成，請把 mac_v2_results.zip 傳回筆電。"
