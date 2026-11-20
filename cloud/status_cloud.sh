#!/bin/bash
# 顯示三條線的最新進度與已完成的結果數
cd "$(dirname "$0")"
for m in GRU LSTM CNN-LSTM CNN; do
  state=$(pgrep -f "pipeline_v2.py --models $m " >/dev/null && echo "執行中" || echo "已停止")
  echo "== $m ($state)"
  tail -c 600 logs/$m.log 2>/dev/null | tr '\r' '\n' | grep -v '^\s*$' | tail -n 3
done
if [ -f logs/FTT.log ]; then
  echo "== FT-Transformer ($(pgrep -f 'models FTT' >/dev/null && echo 執行中 || echo 已停止))"
  tail -c 600 logs/FTT.log | tr '\r' '\n' | grep -v '^\s*$' | tail -n 3
  echo "   FTT 調參: $(ls results/v2/tune/*_FTT_t*.json 2>/dev/null | wc -l)/160   FTT 正式實驗: $(ls results/v2/runs/*_FTT_*.json 2>/dev/null | wc -l)/38"
fi
if [ -f logs/cic18b.log ]; then
  echo "== cic18b 第二份抽樣 ($(pgrep -f 'datasets cic18b' >/dev/null && echo 執行中 || echo 已停止))"
  tail -c 600 logs/cic18b.log | tr '\r' '\n' | grep -v '^\s*$' | tail -n 3
fi
echo
echo "調參試驗: $(ls results/v2/tune/cic18_*.json 2>/dev/null | grep -v FTT | wc -l)/160   正式實驗: $(ls results/v2/runs/cic18_*.json 2>/dev/null | grep -v FTT | wc -l)/60   對抗攻擊: $(ls results/v2/adversarial/attack/cic18_*.json 2>/dev/null | wc -l)/12   對抗訓練: $(ls results/v2/adversarial/advtrain/cic18_*.json 2>/dev/null | wc -l)/12"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
