"""前處理洩漏的對照實驗: 隨機切分條件下，改成只用切分後的訓練部分擬合標準化，
以相同的調參結果與種子重新訓練，和正式實驗 (標準化用整個訓練檔擬合) 比較。

只對標準化過的欄位重新擬合 (one-hot 欄位不變)。因為正式實驗的資料已經標準化 z = (x - m) / s，
在 z 的訓練部分重新擬合 (m', s') 再轉換，等於直接在原始數值上只用訓練部分擬合:
(z - m') / s' = (x - (m + s m')) / (s s')。樹模型對每個特徵的線性縮放不變，不需要重跑。

用法: python scaler_control.py --models DNN CNN
      python scaler_control.py --compare
輸出: results/v2/scaler_control/{dataset}_{model}_s{seed}.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import nids_core as core
import nids_data

DIR = core.OUT / "scaler_control"
SEEDS = {"nsl": 5, "unsw": 5, "cic": 3, "cic18": 3}
COND = {"nsl": "default", "unsw": "default", "cic": "improved", "cic18": "improved"}


def refit_on_train(bundle):
    """只用訓練部分 (含驗證集，與正式實驗的 80% 相同) 重新擬合標準化過的欄位。"""
    numeric = np.array([g == f for g, f in zip(bundle["groups"], bundle["features"])])
    Xtr = bundle["train"]["X"]
    m = Xtr[:, numeric].astype("float64").mean(0)
    s = Xtr[:, numeric].astype("float64").std(0)
    s[s == 0] = 1.0
    out = dict(bundle)

    def fix(part):
        X = part["X"].copy()
        X[:, numeric] = ((X[:, numeric] - m) / s).astype("float32")
        return dict(part, X=X)

    out["train"] = fix(bundle["train"])
    out["tests"] = {k: fix(t) for k, t in bundle["tests"].items()}
    return out, int(numeric.sum())


def run(models, datasets):
    import tune
    tune.summarize(verbose=False)
    hp_all = json.loads((core.OUT / "tuned_hp.json").read_text(encoding="utf-8"))
    DIR.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        bundle, n_num = refit_on_train(nids_data.load(ds, COND[ds]))
        for name in models:
            hp = hp_all[f"{ds}/binary/{name}"]["hp"]
            for seed in range(1, SEEDS[ds] + 1):
                path = DIR / f"{ds}_{name}_s{seed}.json"
                if path.exists():
                    continue
                model, info, _ = core.train(bundle, name, "binary", seed, hp=hp,
                                            title=f"[train-only scaler] {ds} {name} 種子 {seed}")
                res = dict(dataset=ds, scaler="train_only", n_rescaled=n_num, **info,
                           tests_last=core.evaluate(bundle, model, name, "binary"))
                core.write_json(path, res)
                print("  完成: " + "  ".join(f"{k}={m['accuracy'] * 100:.2f}" for k, m in res["tests_last"].items()),
                      flush=True)
                core.cleanup()
        del bundle
        core.cleanup()


def compare():
    from scipy import stats
    out = {}
    for f in sorted(DIR.glob("*.json")):
        if f.name == "summary.json":
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        ref = core.OUT / "runs" / f"{r['dataset']}_{COND[r['dataset']]}_tuned_binary_{r['model']}_s{r['seed']}.json"
        orig = json.loads(ref.read_text(encoding="utf-8"))
        for test, m in r["tests_last"].items():
            d = out.setdefault(f"{r['dataset']}/{r['model']}/{test}", dict(full=[], train_only=[]))
            d["full"].append(orig["tests_last"][test]["accuracy"] * 100)
            d["train_only"].append(m["accuracy"] * 100)
    for k, d in out.items():
        a, b = np.array(d["full"]), np.array(d["train_only"])
        d.update(diff=float(b.mean() - a.mean()), sd_full=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                 p=float(stats.ttest_ind(a, b, equal_var=False).pvalue) if len(a) > 1 else None)
        print(f"{k:28s} 整個檔案 {a.mean():7.3f} ± {d['sd_full']:.3f}   只用訓練部分 {b.mean():7.3f} ± "
              f"{b.std(ddof=1) if len(b) > 1 else 0:.3f}   差 {d['diff']:+.3f}   p={d['p']}")
    (DIR / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["DNN", "CNN"])
    parser.add_argument("--datasets", nargs="+", default=list(SEEDS))
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        return compare()
    run(args.models, args.datasets)


if __name__ == "__main__":
    main()
