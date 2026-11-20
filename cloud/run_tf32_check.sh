#!/bin/bash
# TF32 對照: FT-Transformer 在 NSL-KDD 二元上關閉 TF32 (一般 FP32) 跑 5 個種子，
# 與正式實驗 (開啟 TF32) 比較。超參數由調參紀錄重新挑選。背景執行。
cd "$(dirname "$0")"
unset NIDS_TF32
export KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cuda PYTHONIOENCODING=utf-8
mkdir -p logs
if pgrep -f "label cloud_rtx5090_fp32" >/dev/null; then echo "TF32 對照已經在跑了"; exit 0; fi
nohup env -u NIDS_TF32 .venv/bin/python device_check.py --label cloud_rtx5090_fp32 --models FTT \
    >> logs/tf32_check.log 2>&1 &
echo "已啟動 TF32 對照 (記錄檔 logs/tf32_check.log)"
