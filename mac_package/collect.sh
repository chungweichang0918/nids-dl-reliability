#!/bin/bash
# 把結果打包成 mac_results.zip，複製回 Windows 筆電專案資料夾解壓縮 (合併進 results/)
cd "$(dirname "$0")"
zip -r mac_results.zip results/seeds results/models results/unsw -x "*.lock"
ls -lh mac_results.zip
