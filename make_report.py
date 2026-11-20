"""把 results/runs/*.json 彙整成表格與圖 (對應論文 Fig.2-5、Table 2-8)。

用法: .venv/Scripts/python.exe make_report.py
"""
import json
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nids_repro import BINARY_CLASSES, HP, MULTI_CLASSES, OUT, PAPER

MODELS = list(HP["binary"])
METRICS = ["accuracy", "precision", "recall", "f1"]
OURS, THEIRS = "#2a78d6", "#eb6834"


def _width(text):
    """顯示寬度：中日韓全形字佔 2 格。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def align_tables(lines):
    """把連續的 markdown 表格列補空白，讓每一欄在編輯器裡對齊。"""
    out, i = [], 0
    while i < len(lines):
        if not lines[i].startswith("|"):
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].startswith("|"):
            j += 1
        rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[i:j]]
        is_sep = [all(c and set(c) <= set("-:") for c in r) for r in rows]
        widths = [max(3, *(_width(r[k]) for r, s in zip(rows, is_sep) if not s))
                  for k in range(len(rows[0]))]
        for r, s in zip(rows, is_sep):
            if s:
                out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
            else:
                cells = [c + " " * (w - _width(c)) for c, w in zip(r, widths)]
                out.append("| " + " | ".join(cells) + " |")
        i = j
    return out


def load(task):
    runs = {}
    for name in MODELS:
        path = OUT / "runs" / f"{task}_{name}.json"
        if path.exists():
            runs[name] = json.loads(path.read_text())
    return runs


def summary_table(task, runs):
    rows = []
    for name, r in runs.items():
        row = {"model": name}
        for i, m in enumerate(METRICS):
            row[f"{m}_ours"] = round(r[m] * 100, 3)
            row[f"{m}_paper"] = PAPER[task][name][i]
        row["acc_diff"] = round(row["accuracy_ours"] - row["accuracy_paper"], 3)
        row["train_seconds"] = r["train_seconds"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(task, table):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    x, w = np.arange(len(table)), 0.38
    low = min(table[[f"{m}_{s}" for m in METRICS for s in ("ours", "paper")]].min()) - 0.4
    for ax, m in zip(axes, METRICS):
        ax.bar(x - w / 2, table[f"{m}_ours"], w, color=OURS, label="Reproduction")
        ax.bar(x + w / 2, table[f"{m}_paper"], w, color=THEIRS, label="Paper")
        ax.set_title(m.capitalize())
        ax.set_xticks(x, table["model"], rotation=30)
        ax.set_ylim(low, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("%")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", ncol=2)
    fig.suptitle(f"{task.capitalize()} classification: reproduction vs. paper", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / f"{task}_comparison.png", dpi=150)
    plt.close(fig)


def plot_curves(task, runs):
    fig, axes = plt.subplots(2, len(runs), figsize=(3.6 * len(runs), 6), squeeze=False)
    for col, (name, r) in enumerate(runs.items()):
        h = r["history"]
        for row, key in enumerate(["accuracy", "loss"]):
            ax = axes[row][col]
            ax.plot(h[key], color=OURS, label="train")
            ax.plot(h[f"val_{key}"], color=THEIRS, label="validation")
            ax.set_title(f"{name} {key}")
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0][col].set_ylim(0.9, 1.0)
        axes[1][col].set_yscale("log")
    axes[0][0].legend(frameon=False)
    fig.suptitle(f"{task.capitalize()} classification: accuracy / loss vs. epoch")
    fig.tight_layout()
    fig.savefig(OUT / f"{task}_curves.png", dpi=130)
    plt.close(fig)


def plot_confusion(task, runs):
    names = BINARY_CLASSES if task == "binary" else MULTI_CLASSES
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, r) in zip(axes.ravel(), runs.items()):
        ax.axis("on")
        cm = np.array(r["confusion_matrix"])
        ax.imshow(np.log1p(cm), cmap="Blues")
        for i in range(len(names)):
            for j in range(len(names)):
                dark = cm[i, j] > cm.max() ** 0.5
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if dark else "black", fontsize=9)
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(name)
    fig.suptitle(f"{task.capitalize()} classification: confusion matrices")
    fig.tight_layout()
    fig.savefig(OUT / f"{task}_confusion.png", dpi=130)
    plt.close(fig)


def per_class_table(runs):
    rows = []
    for name, r in runs.items():
        for cls in MULTI_CLASSES:
            c = r["report"][cls]
            rows.append(dict(model=name, **{"class": cls},
                             precision=round(c["precision"], 4),
                             recall=round(c["recall"], 4),
                             f1=round(c["f1-score"], 4), support=int(c["support"])))
    return pd.DataFrame(rows)


def main():
    lines = ["# 重現結果\n"]
    for task in ["binary", "multi"]:
        runs = load(task)
        if not runs:
            continue
        table = summary_table(task, runs)
        table.to_csv(OUT / f"{task}_summary.csv", index=False)
        plot_comparison(task, table)
        plot_curves(task, runs)
        plot_confusion(task, runs)

        title = "二元分類" if task == "binary" else "多元分類"
        lines += [f"## {title}（重現 / 論文，單位 %）\n",
                  "| 模型 | Accuracy | Precision | Recall | F1 | Accuracy 差距 | 訓練秒數 |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in table.iterrows():
            cells = [f"{r[f'{m}_ours']:.2f} / {r[f'{m}_paper']:.2f}" for m in METRICS]
            lines.append(f"| {r['model']} | " + " | ".join(cells) +
                         f" | {r['acc_diff']:+.2f} | {r['train_seconds']:.0f} |")
        lines.append("")
        if task == "binary":
            lines += ["混淆矩陣（positive = normal，與論文 Table 2 相同定義）：\n",
                      "| 模型 | TN | FP | FN | TP |", "|---|---|---|---|---|"]
            for name, r in runs.items():
                (tn, fp), (fn, tp) = r["confusion_matrix"]
                lines.append(f"| {name} | {tn} | {fp} | {fn} | {tp} |")
            lines.append("")
        else:
            per_class = per_class_table(runs)
            per_class.to_csv(OUT / "multi_per_class.csv", index=False)
            pivot = per_class.pivot(index="model", columns="class", values="f1")
            pivot = pivot.loc[list(runs), MULTI_CLASSES]
            support = per_class.drop_duplicates("class").set_index("class")["support"]
            lines += ["各類別 F1-score（對應論文 Table 3-8）：\n",
                      "| 模型 | " + " | ".join(f"{c} (n={support[c]})"
                                                for c in MULTI_CLASSES) + " |",
                      "|---|" + "---|" * len(MULTI_CLASSES)]
            for name, row in pivot.iterrows():
                lines.append(f"| {name} | " +
                             " | ".join(f"{v:.4f}" for v in row) + " |")
            lines.append("")
    text = "\n".join(align_tables(lines))
    (OUT / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
