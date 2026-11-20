#!/bin/bash
# 跨硬體對照: NSL-KDD 二元 GRU 與 DNN，5 個種子，用筆電調好的超參數 (device_check_hp.json)。背景執行。
cd "$(dirname "$0")"
export KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cuda PYTHONIOENCODING=utf-8
mkdir -p logs
if pgrep -f "device_check.py" >/dev/null; then echo "跨硬體對照已經在跑了"; exit 0; fi
nohup .venv/bin/python device_check.py --label cloud_rtx5090 --models GRU DNN \
    --hp-file device_check_hp.json >> logs/device_check.log 2>&1 &
echo "已啟動跨硬體對照 (記錄檔 logs/device_check.log)"
