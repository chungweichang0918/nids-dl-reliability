#!/bin/bash
# 看守: 等 FT-Transformer 正式實驗與所有對照實驗結束 -> 打包 -> 複製到 /vault 並驗證 -> gputw-shutdown 關機。
# 任何一步失敗都不關機 (寧可多付幾小時，也不讓結果消失)。在背景執行，記錄在 logs/finish.log。
cd "$(dirname "$0")"
log() { echo "[$(date '+%m/%d %H:%M:%S')] $*"; }
log "開始看守"
while pgrep -f "run_v2.py --datasets nsl unsw cic cic18 --models FTT" >/dev/null \
   || pgrep -f "tune.py --datasets nsl unsw cic cic18 --models FTT" >/dev/null \
   || pgrep -f "device_check.py" >/dev/null; do
  sleep 120
done
n=$(ls results/v2/runs/*_FTT_*.json 2>/dev/null | wc -l)
log "所有工作已結束；FTT 正式實驗 $n/38"
bash collect_cloud.sh || { log "打包失敗，不關機"; exit 1; }
[ -d /vault ] || { log "找不到 /vault，不關機"; exit 1; }
mkdir -p /vault/nids || { log "無法建立 /vault/nids，不關機"; exit 1; }
cp cloud_results.zip /vault/nids/cloud_results.zip && cp -r logs /vault/nids/ \
  || { log "複製到 /vault 失敗，不關機"; exit 1; }
a=$(sha256sum cloud_results.zip | cut -d' ' -f1)
b=$(sha256sum /vault/nids/cloud_results.zip | cut -d' ' -f1)
[ "$a" = "$b" ] || { log "SHA-256 不一致 ($a vs $b)，不關機"; exit 1; }
echo "$a  cloud_results.zip  (FTT 正式實驗 $n/38)" > /vault/nids/SHA256.txt
log "已複製到 /vault/nids 並驗證 SHA-256 一致: $a"
if [ "$n" -lt 38 ]; then
  log "FTT 正式實驗只有 $n/38 (可能有實驗中斷)。已備份到 /vault，但不關機，保留環境以便補跑。"
  exit 0
fi
command -v gputw-shutdown >/dev/null || { log "找不到 gputw-shutdown，不關機"; exit 1; }
log "執行 gputw-shutdown"
gputw-shutdown
