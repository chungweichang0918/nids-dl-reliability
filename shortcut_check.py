"""驗證 CIC-IDS2017 時間切分上的捷徑發現 (RQ4)。

shortcut_v2.py 發現: 拿掉 SHAP 排名前 3 的特徵 (Packet Length Variance、Destination Port、RST Flag Count)
後，3 個種子中有 2 個的時間切分準確率由約 82% 升到約 98%。這裡用 10 個種子確認，並找出是哪個特徵造成的。

每種設定都用調參後的二元 DNN、同樣的資料與訓練流程，只拿掉指定的特徵後重新訓練。
另外記錄各攻擊類型的偵測率，看拿掉特徵後是哪些新攻擊變得偵測得到。

用法: python shortcut_check.py [--seeds 10] [--device cpu|cuda]
可中斷後重跑接續；結果在 results/v2/shortcut/check/，彙整在 results/v2/shortcut/check_summary.json
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=10)
parser.add_argument("--device", default="cpu")
parser.add_argument("--epochs", type=int, default=None, help="只用於測試")
args = parser.parse_args()
os.environ.setdefault("KERAS_TORCH_DEVICE", args.device)

import json
import time

import numpy as np

import nids_core as core
import nids_data
import tune

PLV, PORT, RST = "Packet Length Variance", "Destination Port", "RST Flag Count"
SETTINGS = {
    "all": [],
    "drop_plv": [PLV],
    "drop_port": [PORT],
    "drop_rst": [RST],
    "drop_plv_port": [PLV, PORT],
    "drop_plv_rst": [PLV, RST],
    "drop_port_rst": [PORT, RST],
    "drop_top3": [PLV, PORT, RST],
}
DIR = core.OUT / "shortcut" / "check"


def subset(bundle, drop):
    keep = [i for i, f in enumerate(bundle["features"]) if f not in drop]
    assert len(keep) == bundle["n_features"] - len(drop), drop
    out = dict(bundle, n_features=len(keep), features=[bundle["features"][i] for i in keep])
    out["train"] = dict(bundle["train"], X=bundle["train"]["X"][:, keep])
    out["tests"] = {k: dict(t, X=t["X"][:, keep]) for k, t in bundle["tests"].items()}
    return out


def summarize():
    rows = {}
    for f in sorted(DIR.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        rows.setdefault(r["setting"], []).append(r)
    summary = {}
    print(f"{'設定':16s} {'n':>3s}  {'準確率 (平均 ± 標準差, 最小–最大)':38s} {'偵測率':>14s}  {'誤報率':>8s}  高於 95% 的種子")
    for name in SETTINGS:
        rs = rows.get(name, [])
        if not rs:
            continue
        acc = np.array([r["accuracy"] for r in rs]) * 100
        det = np.array([r["detection_rate"] for r in rs]) * 100
        fa = np.array([r["false_alarm_rate"] for r in rs]) * 100
        attacks = {}
        for r in rs:
            for a, (rate, n) in r["detection_by_attack"].items():
                attacks.setdefault(a, []).append(rate * 100)
        summary[name] = dict(n=len(rs), dropped=SETTINGS[name],
                             accuracy=dict(mean=float(acc.mean()), sd=float(acc.std(ddof=1)) if len(rs) > 1 else 0.0,
                                           min=float(acc.min()), max=float(acc.max()), values=acc.round(2).tolist()),
                             detection=dict(mean=float(det.mean()), sd=float(det.std(ddof=1)) if len(rs) > 1 else 0.0),
                             false_alarm=float(fa.mean()), n_above_95=int((acc > 95).sum()),
                             by_attack={a: float(np.mean(v)) for a, v in attacks.items()})
        sd = f"{acc.std(ddof=1):.2f}" if len(rs) > 1 else "—"
        print(f"{name:16s} {len(rs):3d}  {acc.mean():6.2f} ± {sd:5s} ({acc.min():.2f}–{acc.max():.2f})"
              f"{'':12s} {det.mean():6.1f} ± {det.std(ddof=1) if len(rs) > 1 else 0:4.1f}  {fa.mean():6.2f}"
              f"   {int((acc > 95).sum())}/{len(rs)}")
    if summary:
        attacks = sorted({a for s in summary.values() for a in s["by_attack"]})
        print("\n各攻擊類型的偵測率 (%):")
        print(f"{'設定':16s} " + " ".join(f"{a[:14]:>15s}" for a in attacks))
        for name, s in summary.items():
            print(f"{name:16s} " + " ".join(f"{s['by_attack'].get(a, float('nan')):15.1f}" for a in attacks))
        core.write_json(core.OUT / "shortcut" / "check_summary.json", summary)
    return summary


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    bundle = nids_data.load("cic", "temporal")
    hp = tune.best_hp("cic", "binary", "DNN")
    print(nids_data.describe(bundle), "\nDNN 超參數:", hp, flush=True)
    jobs = [(name, seed) for seed in range(1, args.seeds + 1) for name in SETTINGS]
    for n, (name, seed) in enumerate(jobs, 1):
        path = DIR / f"{name}_s{seed}.json"
        lock = core.claim(path)
        if lock is None:
            continue
        try:
            t0 = time.time()
            sub = subset(bundle, SETTINGS[name])
            model, info, _ = core.train(sub, "DNN", "binary", seed, hp=hp, epochs=args.epochs, verbose=False,
                                        keep_best=False, grad_norm=False)
            m = core.evaluate(sub, model, "DNN", "binary")["temporal"]
            core.write_json(path, dict(setting=name, dropped=SETTINGS[name], seed=seed, accuracy=m["accuracy"],
                                       detection_rate=m["detection_rate"], false_alarm_rate=m["false_alarm_rate"],
                                       macro_f1=m["macro_f1"], detection_by_attack=m["detection_by_attack"],
                                       collapses=info["collapses"], seconds=round(time.time() - t0)))
            print(f"[{n}/{len(jobs)}] {name:14s} s{seed:<2d} 準確率 {m['accuracy'] * 100:6.2f}%  "
                  f"偵測率 {m['detection_rate'] * 100:5.1f}%  ({time.time() - t0:.0f}s)", flush=True)
        finally:
            lock.unlink(missing_ok=True)
            core.cleanup()
    print()
    summarize()


if __name__ == "__main__":
    main()
