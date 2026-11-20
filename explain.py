"""SHAP 可解釋性分析: DNN 靠哪些特徵判斷攻擊？是否學到資料集的捷徑？

1. 全域重要度: 內部測試集與 KDDTest+ 上各特徵的平均 |SHAP| (one-hot 欄位合併回原始 41 個特徵)
2. 漏報分析: KDDTest+ 上被誤判為 normal 的攻擊，是哪些特徵把預測推向 normal
3. 捷徑檢驗: 只用 SHAP 排名前 k 個特徵重新訓練 DNN，看準確率剩多少

用法: .venv/Scripts/python.exe explain.py            (需先跑完 multiseed.py 的 DNN)
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import json

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

import multiseed as ms
import nids_repro as nr

OUTDIR = nr.OUT / "shap"
FIG = nr.OUT / "figures"
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 10})
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8f98"
TOP_K = [1, 2, 3, 5, 10, 20, 41]


def predictor(model, name):
    """numpy -> numpy 的機率函式 (直接走 torch，比 model.predict 快很多)。"""
    device = next(model.parameters()).device

    def f(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 50000):
                x = torch.as_tensor(ms.shape_input(name, np.asarray(X[i:i + 50000], "float32")),
                                    device=device)
                out.append(model(x, training=False).cpu().numpy())
        return np.concatenate(out)
    return f


def groups(features):
    """每個 one-hot 欄位對應回原始特徵名稱，回傳 (原始特徵清單, 欄位 -> 群組索引)。"""
    def original(col):
        for c in nr.CATEGORICAL:
            if col.startswith(c + "_"):
                return c
        return col
    names = list(dict.fromkeys(original(c) for c in features))
    return names, np.array([names.index(original(c)) for c in features])


def grouped(values, index, n):
    """把 (樣本, 122) 的 SHAP 值依原始特徵加總成 (樣本, 41)；SHAP 具可加性所以可以直接相加。"""
    out = np.zeros((values.shape[0], n))
    np.add.at(out, (slice(None), index), values)
    return out


def shap_values(f, background, X, seed=0):
    explainer = shap.PermutationExplainer(f, shap.maskers.Independent(background, max_samples=100),
                                          seed=seed)
    return explainer(X, max_evals=2 * X.shape[1] + 1, silent=True).values


def analyse(data, seed, n_explain, rng):
    names, index = groups(data["features"])
    X_train, X_test, y_train, y_test = ms.split(data, "binary")
    background = X_train[rng.choice(len(X_train), 100, replace=False)]
    kdd_X, kdd_y = data["test"]["X"], data["test"]["binary"]

    model = ms.load_model("binary", "DNN", seed)
    f = lambda X: predictor(model, "DNN")(X).ravel()     # P(normal)
    out = {}
    for key, X in (("split", X_test), ("kddtest", kdd_X)):
        pick = rng.choice(len(X), n_explain, replace=False)
        sv = grouped(shap_values(f, background, X[pick]), index, len(names))
        out[key] = np.abs(sv).mean(axis=0)

    # 漏報: KDDTest+ 中實際為攻擊、卻被判成 normal
    prob = f(kdd_X)
    missed = np.where((kdd_y == 0) & (prob > 0.5))[0]
    caught = np.where((kdd_y == 0) & (prob <= 0.5))[0]
    for key, idx in (("missed", missed), ("caught", caught)):
        pick = rng.choice(idx, min(n_explain, len(idx)), replace=False)
        out[key] = grouped(shap_values(f, background, kdd_X[pick]), index, len(names)).mean(axis=0)
    out["n_missed"], out["n_attack"] = int(len(missed)), int((kdd_y == 0).sum())

    # 多元分類: 每個類別的重要度
    model = ms.load_model("multi", "DNN", seed)
    f_multi = predictor(model, "DNN")
    X_tr_m, X_te_m, _, _ = ms.split(data, "multi")
    pick = rng.choice(len(X_te_m), n_explain // 2, replace=False)
    sv = shap_values(f_multi, X_tr_m[rng.choice(len(X_tr_m), 100, replace=False)], X_te_m[pick])
    out["multi"] = np.stack([np.abs(grouped(sv[:, :, c], index, len(names))).mean(axis=0)
                             for c in range(sv.shape[2])])
    return names, out


def topk_retrain(data, names, ranking, seed=1):
    """只保留 SHAP 排名前 k 個原始特徵，重新訓練二元 DNN。"""
    _, index = groups(data["features"])
    X_train, X_test, y_train, y_test = ms.split(data, "binary")
    rows = []
    for k in TOP_K:
        cols = np.where(np.isin(index, ranking[:k]))[0]
        keras.utils.set_random_seed(seed)
        model = nr.build_model("DNN", "binary", n_features=len(cols))
        model.fit(X_train[:, cols], y_train, epochs=nr.EPOCHS, batch_size=nr.BATCH_SIZE,
                  validation_split=0.2, verbose=0)
        acc = {}
        for key, (X, y) in (("split", (X_test, y_test)),
                            ("kddtest", (data["test"]["X"], data["test"]["binary"]))):
            pred = model.predict(X[:, cols], batch_size=nr.BATCH_SIZE, verbose=0).ravel() > 0.5
            acc[key] = float((pred == y).mean())
        rows.append(dict(k=k, n_columns=int(len(cols)),
                         features=[names[i] for i in ranking[:k]], **acc))
        print(f"top-{k:2d} ({len(cols):3d} 欄): split={acc['split']*100:.2f} "
              f"kddtest={acc['kddtest']*100:.2f}", flush=True)
    return rows


def plot(names, result):
    imp = np.array(result["split_mean"])
    kdd = np.array(result["kddtest_mean"])
    order = np.argsort(imp)[::-1][:15][::-1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), gridspec_kw=dict(width_ratios=[1.1, 1.1, 0.9]))
    y = np.arange(len(order))
    axes[0].barh(y - 0.2, imp[order], 0.4, color=BLUE, label="內部測試集",
                 xerr=np.array(result["split_std"])[order], error_kw=dict(lw=0.8))
    axes[0].barh(y + 0.2, kdd[order], 0.4, color=ORANGE, label="KDDTest+",
                 xerr=np.array(result["kddtest_std"])[order], error_kw=dict(lw=0.8))
    axes[0].set_yticks(y, [names[i] for i in order])
    axes[0].set_xlabel("平均 |SHAP|（對 P(normal) 的影響）")
    axes[0].set_title("(a) 二元 DNN 的全域特徵重要度")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    missed = np.array(result["missed_mean"])
    order = np.argsort(np.abs(missed))[::-1][:12][::-1]
    caught = np.array(result["caught_mean"])
    y = np.arange(len(order))
    axes[1].barh(y - 0.2, missed[order], 0.4, color=ORANGE, label="漏報的攻擊")
    axes[1].barh(y + 0.2, caught[order], 0.4, color=GRAY, label="成功偵測的攻擊")
    axes[1].axvline(0, color="#444", lw=0.8)
    axes[1].set_yticks(y, [names[i] for i in order])
    axes[1].set_xlabel("平均 SHAP（正值 = 推向 normal）")
    axes[1].set_title("(b) KDDTest+ 漏報攻擊的成因")
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    ks = [r["k"] for r in result["topk"]]
    axes[2].plot(ks, [r["split"] * 100 for r in result["topk"]], "o-", color=BLUE, label="內部測試集")
    axes[2].plot(ks, [r["kddtest"] * 100 for r in result["topk"]], "s-", color=ORANGE, label="KDDTest+")
    axes[2].set_xscale("log")
    axes[2].set_xticks(ks, [str(k) for k in ks])
    axes[2].set_xlabel("只保留 SHAP 排名前 k 個特徵")
    axes[2].set_ylabel("準確率 (%)")
    axes[2].set_title("(c) 只用少數特徵重新訓練 DNN")
    axes[2].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    axes[2].grid(alpha=0.3)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "shap.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    seeds = args.seeds or [s for s in ms.SEEDS if ms.weights_path("multi", "DNN", s).exists()
                           and ms.weights_path("binary", "DNN", s).exists()]
    assert seeds, "找不到 DNN 權重，請先執行 multiseed.py"

    data = ms.load_both()
    rng = np.random.default_rng(0)
    per_seed = []
    for seed in seeds:
        names, out = analyse(data, seed, args.samples, rng)
        per_seed.append(out)
        top = np.argsort(out["split"])[::-1][:5]
        print(f"[s{seed}] top-5: {[names[i] for i in top]}  "
              f"KDDTest+ 漏報 {out['n_missed']}/{out['n_attack']}", flush=True)

    result = dict(seeds=seeds, samples=args.samples, features=names,
                  multi_classes=nr.MULTI_CLASSES)
    for key in ("split", "kddtest", "missed", "caught", "multi"):
        stack = np.stack([o[key] for o in per_seed])
        result[f"{key}_mean"] = stack.mean(axis=0).tolist()
        result[f"{key}_std"] = stack.std(axis=0).tolist()
    result["n_missed"] = [o["n_missed"] for o in per_seed]
    result["n_attack"] = per_seed[0]["n_attack"]
    ranking = list(np.argsort(result["split_mean"])[::-1])
    result["ranking"] = [names[i] for i in ranking]
    result["topk"] = topk_retrain(data, names, ranking)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "shap.json").write_text(json.dumps(result, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    plot(names, result)
    print("已輸出", OUTDIR / "shap.json", "與", FIG / "shap.png")


if __name__ == "__main__":
    main()
