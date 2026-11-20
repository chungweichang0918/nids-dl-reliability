#!/bin/bash
# 雲端 GPU 主機 (Linux) 的環境安裝: Python 3.12.9，其餘套件版本與筆電相同。
# torch: RTX 30/40 系列 (Ampere/Ada) 裝與筆電相同的 2.6.0+cu124；
#        RTX 50 系列 / Blackwell (compute capability >= 10) 需要 CUDA 12.8，改裝 2.7.1+cu128。
# 用法: bash setup_cloud.sh          (約 5-10 分鐘，只需執行一次)
set -e
cd "$(dirname "$0")"
command -v uv >/dev/null || python3 -m pip install --user -q uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
major=${cap%%.*}
if [ "$major" -ge 10 ]; then
  TORCH="torch==2.7.1+cu128"; INDEX="https://download.pytorch.org/whl/cu128"
else
  TORCH="torch==2.6.0+cu124"; INDEX="https://download.pytorch.org/whl/cu124"
fi
echo "GPU compute capability $cap -> 安裝 $TORCH"

# sympy 由 torch 決定版本 (torch 2.6 需要 1.13.1，torch 2.7 需要 >= 1.13.3)
grep -v -e '^torch==' -e '^sympy==' requirements_laptop.txt > requirements_cloud.txt
echo "$TORCH" >> requirements_cloud.txt
uv venv .venv --python 3.12.9
uv pip install --python .venv/bin/python -r requirements_cloud.txt \
    --extra-index-url "$INDEX" --index-strategy unsafe-best-match
.venv/bin/python - <<'EOF'
import os
os.environ["KERAS_BACKEND"] = "torch"
import torch, keras
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0), "| keras", keras.__version__)
x = torch.randn(1000, 1000, device="cuda"); print("GPU 運算測試:", float((x @ x).sum()) == float((x @ x).sum()))
print("CPU 核心:", os.cpu_count())
EOF
echo "環境安裝完成。下一步: bash bench_cloud.sh"
