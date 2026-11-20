"""產生 Word 報告需要的額外圖表 (流程圖、類別分布、診斷實驗)。

用法: .venv/Scripts/python.exe report_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import nids_repro as nr

FIG = nr.OUT / "figures"
ABL = nr.OUT / "ablation"
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 10})
BLUE, ORANGE, GREEN, GRAY, RED = "#2a78d6", "#eb6834", "#2e9e6b", "#8a8f98", "#c9353b"

VARIANT_LABEL = {
    "main": "主實驗 (種子 42)",
    "base_s1": "換種子 (種子 1)",
    "clip": "梯度裁剪 clipnorm=1",
    "lr3e-4": "學習率 3e-4",
    "plain5000": "不做梯度累積",
    "tf": "TensorFlow 2.14",
}
VARIANT_COLOR = {"main": GRAY, "base_s1": BLUE, "clip": GREEN, "lr3e-4": ORANGE,
                 "plain5000": "#7b4fc9", "tf": RED}
JOBS = [("binary", "LSTM"), ("binary", "GRU"), ("multi", "CNN-LSTM")]


def pipeline():
    steps = [("NSL-KDD\nKDDTrain+\n125,973 筆", GRAY),
             ("標準化\n38 個數值特徵", BLUE),
             ("One-hot 編碼\n3 個類別特徵\n→ 122 維", BLUE),
             ("隨機切分\n訓練 80% / 測試 20%\n(驗證 = 訓練的 20%)", BLUE),
             ("六種模型\nDNN · RNN · CNN\nLSTM · GRU · CNN-LSTM", ORANGE),
             ("評估\nAccuracy · Precision\nRecall · F1 · 混淆矩陣", GREEN)]
    fig, ax = plt.subplots(figsize=(12, 1.7))
    ax.set_xlim(0, len(steps) * 2)
    ax.set_ylim(0.2, 1.8)
    ax.axis("off")
    for i, (text, color) in enumerate(steps):
        x = i * 2 + 0.1
        ax.add_patch(FancyBboxPatch((x, 0.25), 1.6, 1.5, boxstyle="round,pad=0.03",
                                    fc=color, ec="none", alpha=0.15))
        ax.add_patch(FancyBboxPatch((x, 0.25), 1.6, 1.5, boxstyle="round,pad=0.03",
                                    fc="none", ec=color, lw=1.5))
        ax.text(x + 0.8, 1.0, text, ha="center", va="center", fontsize=9.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.95, 1.0), xytext=(x + 1.68, 1.0),
                        arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.3))
    fig.tight_layout()
    fig.savefig(FIG / "pipeline.png", dpi=200)
    plt.close(fig)


def class_distribution():
    df = pd.read_csv(nr.DATA, header=None, names=nr.COLUMNS)
    counts = df["label"].map(nr.ATTACK_CATEGORY).value_counts()
    counts = counts.reindex(["Normal", "DoS", "Probe", "R2L", "U2R"])
    fig, ax = plt.subplots(figsize=(6, 2.8))
    bars = ax.bar(counts.index, counts.values,
                  color=[GREEN, BLUE, BLUE, ORANGE, RED])
    ax.set_yscale("log")
    ax.set_ylabel("筆數 (對數刻度)")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15,
                f"{v:,}\n({v / counts.sum() * 100:.2f}%)", ha="center", fontsize=8.5)
    ax.set_ylim(10, 4e5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "class_distribution.png", dpi=200)
    plt.close(fig)
    return counts


def load_ablation():
    """回傳 {(task, model): {variant: record}}，main 來自主實驗。"""
    data = {}
    for task, name in JOBS:
        d = {}
        main = json.loads((nr.OUT / "runs" / f"{task}_{name}.json").read_text())
        val = main["history"]["val_accuracy"]
        d["main"] = dict(last=dict(accuracy=main["accuracy"]), val_accuracy=val,
                         best_epoch=int(np.argmax(val)) + 1, best=None,
                         collapses=int(sum(1 for i in range(1, len(val))
                                           if val[i] < max(val[:i]) - 0.05)))
        for f in sorted(ABL.glob(f"{task}_{name}_*.json")):
            r = json.loads(f.read_text())
            if len(r["val_accuracy"]) < 100:
                continue  # 未完成或測試用的短跑
            key = "tf" if f.stem.endswith("_tf") else r["variant"]
            d[key] = r
        data[(task, name)] = d
    return data


def ablation_curves(data):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
    titles = {("binary", "LSTM"): "二元分類 LSTM", ("binary", "GRU"): "二元分類 GRU",
              ("multi", "CNN-LSTM"): "多元分類 CNN-LSTM"}
    for ax, key in zip(axes, JOBS):
        for v, r in data[key].items():
            ax.plot(np.array(r["val_accuracy"]) * 100, color=VARIANT_COLOR[v],
                    lw=1.8 if v == "main" else 1.2, label=VARIANT_LABEL[v],
                    ls="--" if v == "main" else "-")
        ax.set_title(titles[key])
        ax.set_xlabel("epoch")
        ax.set_ylim(88, 100)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("驗證準確率 (%)")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "ablation_curves.png", dpi=200)
    plt.close(fig)


def ablation_table(data):
    rows = []
    for (task, name), d in data.items():
        for v in VARIANT_LABEL:
            if v not in d:
                continue
            r = d[v]
            rows.append(dict(task=task, model=name, variant=v,
                             last=r["last"]["accuracy"] * 100,
                             best=None if r["best"] is None else r["best"]["accuracy"] * 100,
                             best_epoch=r["best_epoch"], collapses=r["collapses"]))
    df = pd.DataFrame(rows)
    df.to_csv(FIG / "ablation_table.csv", index=False)
    return df


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    pipeline()
    print(class_distribution())
    data = load_ablation()
    ablation_curves(data)
    print(ablation_table(data).to_string(index=False))


if __name__ == "__main__":
    main()
