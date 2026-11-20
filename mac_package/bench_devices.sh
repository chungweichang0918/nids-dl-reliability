#!/bin/bash
# 量測六個模型在這台 Mac 的 CPU 與 Apple GPU 上的速度 (約 5-15 分鐘)
cd "$(dirname "$0")"
source .venv/bin/activate
python bench_devices.py "$@"
