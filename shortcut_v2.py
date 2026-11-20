"""RQ4 捷徑學習: 模型是否依賴少數資料集特有的特徵？

對每個資料集 (NSL-KDD、UNSW-NB15、CIC-IDS2017 修正版與時間切分) 的調參後二元 DNN:
  1. SHAP 全域重要度 (Permutation explainer)，分別在各測試集上計算，one-hot 欄位合併回原始特徵
  2. 分布外測試集 (官方 / 時間切分) 上被漏報與被偵測的攻擊，各特徵把預測推向哪一邊
  3. 特徵消融 (重新訓練): 只保留 SHAP 排名前 k 個特徵、或拿掉前 1 / 前 3 個特徵，每種 3 個種子

DNN 很小，全部用 CPU 即可。需要先跑完 run_v2.py 中 DNN 的 main 與 conditions 群組。

用法: python shortcut_v2.py [--datasets nsl unsw cic] [--samples 500]
"""
import argparse

import numpy as np
import torch

import nids_core as core
import nids_data
import run_v2
import tune

DIR = core.OUT / "shortcut"
TARGETS = [("nsl", "default", "main"), ("unsw", "default", "main"),
           ("cic", "improved", "main"), ("cic", "temporal", "conditions"),
           ("cic18", "improved", "main"), ("cic18", "temporal", "conditions")]
SEEDS = [1, 2, 3]
TOP_K = [1, 2, 3, 5, 10]
DROP_K = [1, 3]


def predictor(model):
    """numpy -> P(normal)。"""
    dev = next(model.parameters()).device

    def f(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 100_000):
                x = torch.as_tensor(np.asarray(X[i:i + 100_000], "float32"), device=dev)
                out.append(model(x, training=False).reshape(-1).cpu().numpy())
        return np.concatenate(out)
    return f


def group_index(groups):
    names = list(dict.fromkeys(groups))
    return names, np.array([names.index(g) for g in groups])


def grouped(values, index, n):
    out = np.zeros((values.shape[0], n))
    np.add.at(out, (slice(None), index), values)
    return out


def shap_values(f, background, X, seed):
    import shap
    explainer = shap.PermutationExplainer(f, shap.maskers.Independent(background, max_samples=100),
                                          seed=seed)
    return explainer(X, max_evals=2 * X.shape[1] + 1, silent=True).values


def explain(bundle, group, n_samples):
    """回傳各測試集的全域重要度，以及分布外測試集上漏報 / 偵測到的攻擊的平均 SHAP。"""
    names, index = group_index(bundle["groups"])
    rng = np.random.default_rng(0)
    background = bundle["train"]["X"][rng.choice(len(bundle["train"]["X"]), 100, replace=False)]
    per_seed = []
    for seed in SEEDS:
        job = dict(group=group, dataset=bundle["name"], condition=bundle["condition"], hpset="tuned",
                   task="binary", model="DNN", seed=seed, perm=None)
        model, _ = run_v2.load_model(job, bundle)
        f = predictor(model)
        out = {}
        for key, t in bundle["tests"].items():
            pick = rng.choice(len(t["X"]), min(n_samples, len(t["X"])), replace=False)
            sv = grouped(shap_values(f, background, t["X"][pick], seed), index, len(names))
            out[f"importance/{key}"] = np.abs(sv).mean(axis=0)
        # 最後一個測試集視為分布外 (官方測試集 / 時間切分)；只有 random 時就用它
        key, t = list(bundle["tests"].items())[-1]
        prob = f(t["X"])
        attack = t["binary"] == 0
        for label, idx in (("missed", np.flatnonzero(attack & (prob > 0.5))),
                           ("caught", np.flatnonzero(attack & (prob <= 0.5)))):
            if len(idx) == 0:
                out[label] = np.zeros(len(names))
                continue
            pick = rng.choice(idx, min(n_samples, len(idx)), replace=False)
            out[label] = grouped(shap_values(f, background, t["X"][pick], seed),
                                 index, len(names)).mean(axis=0)
        out["n_missed"] = int((attack & (prob > 0.5)).sum())
        out["n_attack"] = int(attack.sum())
        per_seed.append(out)
        top = np.argsort(out[f"importance/{list(bundle['tests'])[0]}"])[::-1][:5]
        print(f"  [s{seed}] 前 5 名: {[names[i] for i in top]}  "
              f"{key} 漏報 {out['n_missed']:,}/{out['n_attack']:,}", flush=True)
    result = dict(features=names, ood_test=key)
    for k in per_seed[0]:
        stack = np.array([o[k] for o in per_seed], dtype="float64")
        if stack.ndim == 1:
            result[k] = stack.tolist()
        else:
            result[k] = dict(mean=stack.mean(axis=0).tolist(), std=stack.std(axis=0).tolist())
    return result


def subset(bundle, cols):
    out = dict(bundle, n_features=len(cols))
    out["train"] = dict(bundle["train"], X=bundle["train"]["X"][:, cols])
    out["tests"] = {k: dict(t, X=t["X"][:, cols]) for k, t in bundle["tests"].items()}
    return out


def ablation(bundle, ranking, epochs=None):
    """依 SHAP 排名保留或移除特徵後重新訓練 DNN (調參後的超參數)。"""
    names, index = group_index(bundle["groups"])
    hp = tune.best_hp(bundle["name"], "binary", "DNN")
    settings = [(f"top{k}", ranking[:k]) for k in TOP_K if k < len(names)]
    settings += [(f"drop{k}", ranking[k:]) for k in DROP_K]
    settings.append(("all", ranking))
    rows = []
    for label, keep in settings:
        cols = np.flatnonzero(np.isin(index, keep))
        sub = subset(bundle, cols)
        accs = {key: [] for key in bundle["tests"]}
        det = {key: [] for key in bundle["tests"]}
        for seed in SEEDS:
            model, _, _ = core.train(sub, "DNN", "binary", seed, hp=hp, epochs=epochs, verbose=False,
                                     keep_best=False, grad_norm=False)
            ev = core.evaluate(sub, model, "DNN", "binary")
            for key, m in ev.items():
                accs[key].append(m["accuracy"])
                det[key].append(m["detection_rate"])
            core.cleanup()
        rows.append(dict(setting=label, n_features=int(len(keep)), n_columns=int(len(cols)),
                         features=[names[i] for i in keep[:10]], accuracy=accs, detection_rate=det))
        print(f"  {label:6s} ({len(keep):3d} 個特徵): " + "  ".join(
            f"{key}={np.mean(v) * 100:.2f}±{np.std(v, ddof=1) * 100:.2f}" for key, v in accs.items()),
            flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["nsl", "unsw", "cic"])
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=None, help="只用於測試")
    args = parser.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    for dataset, condition, group in TARGETS:
        path = DIR / f"{dataset}_{condition}.json"
        if dataset not in args.datasets or path.exists():
            continue
        bundle = nids_data.load(dataset, condition)
        print(nids_data.describe(bundle), flush=True)
        result = explain(bundle, group, args.samples)
        first = f"importance/{list(bundle['tests'])[0]}"
        ranking = np.argsort(result[first]["mean"])[::-1]
        result["ranking"] = [result["features"][i] for i in ranking]
        result["ablation"] = ablation(bundle, ranking, args.epochs)
        core.write_json(path, dict(dataset=dataset, condition=condition, samples=args.samples,
                                   seeds=SEEDS, **result))
        del bundle
    print("捷徑分析完成。")


if __name__ == "__main__":
    main()
