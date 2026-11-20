#!/bin/bash
# CSE-CIC-IDS2018: GRU、LSTM、CNN-LSTM、CNN 四個模型同時在這張顯示卡上跑完全部階段
# (調參 -> 正式實驗 -> 對抗攻擊 -> 對抗訓練)。在背景執行，關掉 SSH 也不會中斷。
# 可隨時重新執行本腳本: 已完成的工作會自動略過，正在跑的不會重複啟動。
cd "$(dirname "$0")"
export KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cuda PYTHONIOENCODING=utf-8
mkdir -p logs
for m in GRU LSTM CNN-LSTM CNN; do
  if pgrep -f "pipeline_v2.py --models $m " >/dev/null; then
    echo "$m 已經在跑了"
    continue
  fi
  nohup .venv/bin/python pipeline_v2.py --models $m --datasets cic18 >> logs/$m.log 2>&1 &
  echo "已啟動 $m (記錄檔 logs/$m.log)"
done
echo "查看進度: bash status_cloud.sh"
