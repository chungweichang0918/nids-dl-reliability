#!/bin/bash
# 量測這台雲端 GPU 的速度 (約 3-5 分鐘)，把最後的表格貼給 Claude 估算費用。
cd "$(dirname "$0")"
export KERAS_BACKEND=torch
.venv/bin/python bench_devices.py --cuda
