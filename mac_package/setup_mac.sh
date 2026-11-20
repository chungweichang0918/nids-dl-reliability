#!/bin/bash
# 在 Mac 上建立虛擬環境並安裝套件 (只需執行一次)
set -e
cd "$(dirname "$0")"

if ! PY=$(command -v python3.12 || command -v python3.11 || command -v python3); then
    echo '找不到 Python 3。請先在這台 Mac 安裝 Python 3.12，再重跑此腳本。' >&2
    exit 1
fi
echo "使用 Python: $PY ($("$PY" --version))"

if { [ -e .venv ] || [ -L .venv ]; } && ! .venv/bin/python --version >/dev/null 2>&1; then
    backup=$(mktemp -d .venv-backup.XXXXXXXX)
    mv .venv "$backup/.venv"
    echo "已將舊環境移至 $backup/.venv"
fi
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip

# 優先安裝與 Windows 筆電相同的版本；若 Mac 找不到該版本就改裝最新版
.venv/bin/python -m pip install -r requirements.txt || \
    .venv/bin/python -m pip install keras torch numpy pandas scikit-learn h5py

echo
KERAS_BACKEND=torch .venv/bin/python - <<'PYEOF'
import keras, torch
print("Keras", keras.__version__, "/ torch", torch.__version__)
print("MPS (Apple GPU) 可用:", torch.backends.mps.is_available())
PYEOF
echo "安裝完成。下一步: ./bench.sh 測速"
