#!/bin/bash
# 測量 SimpleRNN 在這台機器上每個 epoch 要多久 (Windows 筆電約 40 秒/epoch)
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo '找不到可執行的 .venv/bin/python。請先在這台 Mac 執行 bash setup_mac.sh。' >&2
    exit 1
fi
export KERAS_BACKEND=torch
export KERAS_TORCH_DEVICE="${KERAS_TORCH_DEVICE:-cpu}"

bench_dir=$(mktemp -d "${TMPDIR:-/tmp}/nids-bench.XXXXXXXX")
cleanup() {
    rm -f -- "$bench_dir"/seeds/binary_RNN_s99.{json,npz,lock} \
        "$bench_dir/models/binary_RNN_s99.weights.h5"
    rmdir -- "$bench_dir/seeds" "$bench_dir/models" "$bench_dir" 2>/dev/null || true
}
trap cleanup EXIT

NIDS_RESULTS_DIR="$bench_dir" .venv/bin/python multiseed.py \
    --seeds 99 --models RNN --tasks binary --epochs 3

.venv/bin/python - "$bench_dir/seeds/binary_RNN_s99.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
print(f"平均每個 epoch {result['seconds'] / result['epochs']:.1f} 秒（含驗證）")
PY
