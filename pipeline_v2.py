"""依序執行 v2 的所有實驗階段。每個階段都可中斷後接續，已完成的工作會自動略過。

階段: 調參 -> 正式實驗 (多種子、評估條件、裁剪、特徵排列) -> 捷徑分析 (只有負責 DNN 的機器)
      -> 對抗攻擊 -> 對抗訓練

用法:
    python pipeline_v2.py --models CNN LSTM CNN-LSTM GRU     # 筆電 (顯示卡)
    python pipeline_v2.py --models RNN DNN                   # Mac mini (pipeline_mac.sh)
    python pipeline_v2.py --models RNN DNN --from attack     # 從某個階段開始

全部結束後，在筆電合併兩台的結果，再執行: python adversarial_v2.py --stage transfer
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("tune", "公平調參", ["tune.py"]),
    ("run", "正式實驗", ["run_v2.py"]),
    ("shortcut", "捷徑分析", ["shortcut_v2.py"]),
    ("attack", "對抗攻擊", ["adversarial_v2.py", "--stage", "attack"]),
    ("advtrain", "對抗訓練", ["adversarial_v2.py", "--stage", "advtrain"]),
]


RETRIES = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--from", dest="start", default="tune", choices=[s[0] for s in STAGES])
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="只跑指定資料集，例如 --datasets cic18 (預設: 各階段自己的預設值)")
    args = parser.parse_args()
    names = [s[0] for s in STAGES]
    extra = ["--datasets", *args.datasets] if args.datasets else []
    for key, title, cmd in STAGES[names.index(args.start):]:
        if key == "shortcut":
            if "DNN" not in args.models:
                continue
            argv = cmd + extra                           # 捷徑分析只用 DNN
        else:
            argv = cmd + ["--models", *args.models] + extra
        print(f"\n{'=' * 60}\n階段: {title}  ({time.strftime('%m/%d %H:%M')})\n{'=' * 60}", flush=True)
        # 顯示卡記憶體不足等暫時性錯誤: 每次都開新的行程重試 (已完成的工作會自動略過)
        for attempt in range(1, RETRIES + 1):
            code = subprocess.call([sys.executable, *argv], cwd=ROOT)
            if code == 0:
                break
            if attempt < RETRIES:
                print(f"\n階段「{title}」第 {attempt} 次失敗 (結束碼 {code})，60 秒後自動重試 "
                      f"({attempt + 1}/{RETRIES})。", flush=True)
                time.sleep(60)
        if code != 0:
            print(f"\n階段「{title}」失敗 (結束碼 {code})，請把上面的錯誤訊息貼給 Claude。"
                  f"修正後執行同一行指令即可接續。", flush=True)
            sys.exit(code)
    print(f"\n全部階段完成 ({time.strftime('%m/%d %H:%M')})。", flush=True)


if __name__ == "__main__":
    main()
