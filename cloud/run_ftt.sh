#!/bin/bash
# FT-Transformer (第 9 個模型): 四個資料集的公平調參 (20 次試驗 x 2 任務)，
# 接著主要條件 (多種子、二元與五類) 與時序切分 (CIC-IDS2017、CSE-CIC-IDS2018)。在背景執行。
cd "$(dirname "$0")"
export KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cuda PYTHONIOENCODING=utf-8 NIDS_TF32=1
mkdir -p logs
if pgrep -f "models FTT" >/dev/null; then echo "FTT 已經在跑了"; exit 0; fi
nohup bash -c '
.venv/bin/python tune.py --datasets nsl unsw cic cic18 --models FTT &&
.venv/bin/python run_v2.py --datasets nsl unsw cic cic18 --models FTT --groups main temporal &&
echo "FTT 全部完成"
' >> logs/FTT.log 2>&1 &
echo "已啟動 FTT (記錄檔 logs/FTT.log)"
