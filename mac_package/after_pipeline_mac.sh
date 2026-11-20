#!/bin/bash
# Mac mini: pipeline_mac.sh 全部跑完後執行。可隨時 Ctrl+C，再執行一次會接續。
#   1) CIC-IDS2017 的樹模型基準線 (隨機森林、XGBoost)
#   2) Yin et al. 2017 的「122 個特徵當序列」解讀 (SimpleRNN 在 Mac CPU 上最快)
cd "$(dirname "$0")"
source .venv/bin/activate
export KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cpu

python -c "import xgboost" 2>/dev/null || pip install xgboost || exit 1
caffeinate -i bash -c '
python baselines_v2.py --datasets cic --jobs 8 &&
python yin_repro.py --interp featseq
' && ./collect_v2.sh
