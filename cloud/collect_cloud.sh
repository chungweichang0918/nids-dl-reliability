#!/bin/bash
# 全部跑完後打包結果 (含模型權重)，下載回筆電後交給 Claude 合併
cd "$(dirname "$0")"
rm -f cloud_results.zip
.venv/bin/python - <<'EOF'
import pathlib, zipfile
with zipfile.ZipFile("cloud_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    files = list(pathlib.Path("results/v2").rglob("*")) + list(pathlib.Path("results/device_check").rglob("*"))
    for p in files:
        if p.is_file() and p.suffix not in (".lock", ".tmp"):
            z.write(p)
print("cloud_results.zip", round(pathlib.Path("cloud_results.zip").stat().st_size / 1e6, 1), "MB")
EOF
