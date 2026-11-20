#!/bin/bash
# 把 v2 實驗結果打包成 mac_v2_results.zip，複製回筆電專案資料夾解壓縮 (合併進 results/)
cd "$(dirname "$0")"
rm -f mac_v2_results.zip
zip -qr mac_v2_results.zip results/v2 -x "*.lock" "*.tmp"
ls -lh mac_v2_results.zip
