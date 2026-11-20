"""監看筆電的實驗流程，中斷時自動重新執行，直到全部完成。

1. 先等目前正在跑的 pipeline_v2.py (例如在終端機手動啟動的那一個) 結束
2. 接著反覆執行 pipeline_v2.py --from run (已完成的工作會自動略過)
   - 結束碼 0: 全部完成，監看結束
   - 失敗: 60 秒後重跑；若連續 3 次都沒有任何新結果 (代表是程式錯誤而非暫時性問題)，就停止並記錄原因
所有輸出寫在 results/v2/pipeline_laptop.log，監看紀錄寫在 results/v2/supervisor.log。

用法: python supervise_v2.py --models CNN LSTM CNN-LSTM GRU
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "results" / "v2"
LOG = V2 / "supervisor.log"
OUT = V2 / "pipeline_laptop.log"
MAX_STRIKES = 3


def log(msg):
    line = f"[{time.strftime('%m/%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def progress():
    """已完成的結果數 (正式實驗 + 對抗攻擊 + 對抗訓練)。"""
    return sum(len(list(p.glob("*.json"))) for p in
               (V2 / "runs", V2 / "adversarial" / "attack", V2 / "adversarial" / "advtrain") if p.exists())


def other_pipeline_running():
    cmd = ("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
           "Where-Object { $_.CommandLine -like '*pipeline_v2.py*' }).Count")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    try:
        return int(out.stdout.strip() or 0) > 0
    except ValueError:
        return False


def clear_stale_locks():
    """沒有任何 pipeline 在執行時，殘留的 .lock 一定是被強制中斷的工作留下的，刪除後才能補跑。"""
    if other_pipeline_running():
        return 0
    locks = [p for d in (V2 / "runs", V2 / "adversarial" / "attack", V2 / "adversarial" / "advtrain",
                         V2 / "adversarial" / "transfer") if d.exists() for p in d.glob("*.lock")]
    for p in locks:
        p.unlink(missing_ok=True)
    return len(locks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    if other_pipeline_running():
        log(f"偵測到正在執行的 pipeline_v2.py，等它結束 (目前已完成 {progress()} 個結果)")
        while other_pipeline_running():
            time.sleep(60)
        log(f"原本的 pipeline_v2.py 已結束 (已完成 {progress()} 個結果)")

    strikes, attempt = 0, 0
    while True:
        attempt += 1
        stale = clear_stale_locks()
        if stale:
            log(f"清除 {stale} 個殘留的鎖定檔 (沒有任何流程在執行，代表是被中斷的工作留下的)")
        before = progress()
        log(f"第 {attempt} 次執行 pipeline_v2.py --from run (已完成 {before} 個結果)")
        with OUT.open("a", encoding="utf-8") as f:
            f.write(f"\n===== 監看程式第 {attempt} 次啟動 {time.strftime('%m/%d %H:%M')} =====\n")
            f.flush()
            code = subprocess.call([sys.executable, "pipeline_v2.py", "--models", *args.models, "--from", "run"],
                                   cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        after = progress()
        if code == 0 and after == before:
            # 成功又沒有新結果: 沒有遺漏的工作 (被中斷的工作會在下一輪補跑)，才算真的完成
            log(f"全部完成 (共 {after} 個結果)，監看結束")
            return 0
        if code == 0:
            log(f"這一輪完成，新增 {after - before} 個結果；再跑一輪確認沒有遺漏的工作")
            strikes = 0
            continue
        strikes = 0 if after > before else strikes + 1
        log(f"pipeline_v2.py 失敗 (結束碼 {code})，本次新增 {after - before} 個結果；"
            f"連續沒有進展 {strikes}/{MAX_STRIKES} 次")
        if strikes >= MAX_STRIKES:
            log(f"連續 {MAX_STRIKES} 次沒有任何進展，判斷為程式錯誤，停止監看。請查看 {OUT.name} 的最後幾行。")
            return 2
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
