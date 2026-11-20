"""跨硬體對照實驗: 同一組超參數、同樣的種子，在不同機器上重跑 NSL-KDD 二元任務，
檢查硬體造成的差異是否小於種子之間的差異。訓練流程與 run_v2.py 的正式實驗完全相同。

用法: python device_check.py --label cloud_rtx5090 --models GRU DNN
      python device_check.py --label laptop_rtx4050 --models DNN
      python device_check.py --compare                       (彙整並做統計檢定)
輸出: results/device_check/{label}/{model}_s{seed}.json
"""
import argparse
import glob
import json
import platform
from pathlib import Path

import numpy as np

import nids_core as core
import nids_data

DIR = core.OUT.parent / "device_check"
HP_FILE = core.OUT / "tuned_hp.json"
REF = core.OUT / "runs"                     # 正式實驗的結果 (原本的機器)


def run(label, models, seeds, hp_file):
    import torch
    import tune
    if Path(hp_file) == HP_FILE:
        tune.summarize(verbose=False)                # 由調參紀錄重新挑選，確保是最終結果
    hp_all = json.loads(Path(hp_file).read_text(encoding="utf-8"))
    for name in models:
        n = hp_all[f"nsl/binary/{name}"]["n"]
        assert n >= tune.TRIALS, f"nsl/binary/{name} 調參未完成 ({n}/{tune.TRIALS})"
    out = DIR / label
    out.mkdir(parents=True, exist_ok=True)
    bundle = nids_data.load("nsl")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
    for name in models:
        hp = hp_all[f"nsl/binary/{name}"]["hp"]
        for seed in seeds:
            path = out / f"{name}_s{seed}.json"
            if path.exists():
                continue
            model, info, _ = core.train(bundle, name, "binary", seed, hp=hp, title=f"[{label}] {name} 種子 {seed}")
            res = dict(label=label, hardware=gpu, torch=torch.__version__,
                       tf32=bool(torch.backends.cuda.matmul.allow_tf32), **info,
                       tests_last=core.evaluate(bundle, model, name, "binary"))
            core.write_json(path, res)
            print("  完成: " + "  ".join(f"{k}={m['accuracy'] * 100:.2f}" for k, m in res["tests_last"].items()), flush=True)
            core.cleanup()


def compare():
    from scipy import stats
    groups = {}
    for f in glob.glob(str(DIR / "*" / "*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        groups.setdefault((r["model"], r["label"]), []).append(r)
    for name in sorted({m for m, _ in groups}):
        ref = [json.loads((REF / f"nsl_default_tuned_binary_{name}_s{s}.json").read_text(encoding="utf-8"))
               for s in range(1, 6)]
        tag = "TF32" if name == "FTT" else ref[0]["device"]       # FTT 的正式實驗在雲端以 TF32 執行
        sets = {f"original ({tag})": ref}
        sets.update({lab: rs for (m, lab), rs in groups.items() if m == name})
        print(f"\n== {name} ==")
        for test in ("random", "official"):
            vals = {lab: np.array([r["tests_last"][test]["accuracy"] * 100 for r in rs]) for lab, rs in sets.items()}
            base_lab = next(iter(vals))
            for lab, v in vals.items():
                line = f"  {test:8s} {lab:28s} {v.mean():7.3f} ± {v.std(ddof=1):.3f}  (n={len(v)})"
                if lab != base_lab:
                    p = stats.ttest_ind(vals[base_lab], v, equal_var=False).pvalue
                    line += f"  差 {v.mean() - vals[base_lab].mean():+.3f}  Welch p={p:.3f}"
                print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label")
    parser.add_argument("--models", nargs="+", default=["GRU", "DNN"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--hp-file", default=str(HP_FILE))
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        return compare()
    run(args.label, args.models, args.seeds, args.hp_file)


if __name__ == "__main__":
    main()
