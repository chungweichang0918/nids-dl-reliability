#!/bin/bash
# 一致性威脅模型的對抗訓練 (CSE-CIC-IDS2018，CNN、LSTM、CNN-LSTM、GRU，各 3 個種子)。
# 先等 FT-Transformer 正式實驗與 2018 一致性攻擊結束，再四個模型同時開始。背景執行。
cd "$(dirname "$0")"
export KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cuda PYTHONIOENCODING=utf-8
mkdir -p logs
nohup bash -c '
while pgrep -f "[r]un_v2.py --datasets nsl unsw cic cic18 --models FTT" >/dev/null \
   || pgrep -f "[s]tage consistent --datasets" >/dev/null; do sleep 120; done
echo "[$(date "+%H:%M")] 開始一致性對抗訓練"
for m in CNN LSTM CNN-LSTM GRU; do
  .venv/bin/python adversarial_v2.py --stage advtrain_consistent --datasets cic18 --models $m >> logs/at_consistent_$m.log 2>&1 &
done
wait
echo "[$(date "+%H:%M")] 一致性對抗訓練全部結束"
' >> logs/at_consistent.log 2>&1 &
echo "已排程: FT-Transformer 與一致性攻擊結束後自動開始 (記錄檔 logs/at_consistent*.log)"
