#!/bin/bash
# CSE-CIC-IDS2018 第二份獨立抽樣 (cic18b，種子 2019): DNN 與兩個樹模型，完整的調參 + 正式實驗 + 時序切分。
# 用來確認結論不依賴抽樣。與 run_cloud.sh 的四條線同時在背景執行 (樹模型只用 CPU)。
cd "$(dirname "$0")"
export KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cuda PYTHONIOENCODING=utf-8
mkdir -p logs
if pgrep -f "datasets cic18b" >/dev/null; then echo "cic18b 已經在跑了"; exit 0; fi
nohup bash -c '
.venv/bin/python tune.py --datasets cic18b --models DNN &&
.venv/bin/python run_v2.py --datasets cic18b --models DNN --groups main conditions &&
.venv/bin/python baselines_v2.py --datasets cic18b --jobs 8 &&
echo "cic18b 全部完成"
' >> logs/cic18b.log 2>&1 &
echo "已啟動 cic18b (記錄檔 logs/cic18b.log)"
