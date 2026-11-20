"""彙整延伸實驗: 多種子統計、顯著性檢定、KDDTest+ 泛化、UNSW-NB15。

讀取 results/seeds、results/unsw，輸出 results/analysis.json、results/EXTENSION.md 與圖。
尚未跑完的模型會自動略過，可以隨時執行查看目前進度。

用法: .venv/Scripts/python.exe analyze.py
"""
import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import multiseed as ms
import nids_repro as nr
from make_report import align_tables

FIG = nr.OUT / "figures"
MODELS = list(nr.HP["binary"])
TASKS = ["binary", "multi"]
TASK_ZH = {"binary": "二元", "multi": "五類"}
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 10})
BLUE, ORANGE, GREEN, GRAY, RED = "#2a78d6", "#eb6834", "#2e9e6b", "#8a8f98", "#c9353b"


def load_runs(folder):
    runs = {}
    for f in sorted((nr.OUT / folder).glob("*_s*.json")):
        r = json.loads(f.read_text())
        if r["epochs"] < nr.EPOCHS:
            continue
        runs.setdefault((r["task"], r["model"]), {})[r["seed"]] = r
    return runs


def ms_fmt(x):
    return f"{np.mean(x):.2f} ± {np.std(x, ddof=1):.2f}" if len(x) > 1 else f"{x[0]:.2f}"


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def seed_stats(runs):
    """每個 (任務, 模型) 的多種子統計，數值單位為 %。"""
    out = {}
    for (task, model), by_seed in runs.items():
        seeds = sorted(by_seed)
        get = lambda key, metric="accuracy": [by_seed[s][key][metric] * 100 for s in seeds]
        acc = get("split_last")
        paper = nr.PAPER[task][model][0]
        row = dict(seeds=seeds, split=acc, split_best=get("split_best"),
                   kddtest=get("kddtest_last"), kddtest_best=get("kddtest_best"),
                   kddtest_macro_f1=get("kddtest_last", "macro_f1"),
                   split_macro_f1=get("split_last", "macro_f1"),
                   collapses=[by_seed[s]["collapses"] for s in seeds],
                   seconds=[by_seed[s]["seconds"] for s in seeds], paper=paper)
        if len(seeds) > 1:
            # 原論文數值是否落在重現分布之內 (單一樣本 t 檢定)
            t, p = stats.ttest_1samp(acc, paper)
            half = stats.t.ppf(0.975, len(acc) - 1) * stats.sem(acc)
            row.update(ci95=[np.mean(acc) - half, np.mean(acc) + half], paper_p=float(p))
        out[f"{task}/{model}"] = row
    return out


def pairwise(seed_table, task, key="split"):
    """模型兩兩比較 (Welch t 檢定 + Holm 校正)。"""
    models = [m for m in MODELS if f"{task}/{m}" in seed_table
              and len(seed_table[f"{task}/{m}"][key]) > 1]
    pairs = list(itertools.combinations(models, 2))
    if not pairs:
        return []
    p = [stats.ttest_ind(seed_table[f"{task}/{a}"][key], seed_table[f"{task}/{b}"][key],
                         equal_var=False).pvalue for a, b in pairs]
    adj = holm(np.array(p))
    return [dict(a=a, b=b, diff=float(np.mean(seed_table[f"{task}/{a}"][key]) -
                                      np.mean(seed_table[f"{task}/{b}"][key])),
                 p=float(pi), p_holm=float(ai)) for (a, b), pi, ai in zip(pairs, p, adj)]


def best_counts(seed_table, task, key="split"):
    """每個種子裡誰是最佳模型 (只統計所有模型都跑完的種子)。"""
    models = [m for m in MODELS if f"{task}/{m}" in seed_table]
    if not models:
        return {}, []
    common = set.intersection(*(set(seed_table[f"{task}/{m}"]["seeds"]) for m in models))
    wins = {m: 0 for m in models}
    for s in sorted(common):
        score = {m: seed_table[f"{task}/{m}"][key][seed_table[f"{task}/{m}"]["seeds"].index(s)]
                 for m in models}
        wins[max(score, key=score.get)] += 1
    return wins, sorted(common)


def kddtest_detail(runs, data):
    """KDDTest+: 已知 / 未知攻擊類型的偵測率 (二元)，與各類別 recall (多元)。"""
    test = data["test"]
    is_attack = test["binary"] == 0
    unseen = is_attack & ~np.isin(test["label"], data["train_attacks"])
    seen = is_attack & ~unseen
    out = dict(n_seen=int(seen.sum()), n_unseen=int(unseen.sum()),
               n_normal=int((~is_attack).sum()), binary={}, multi={})
    for (task, model), by_seed in runs.items():
        rows = []
        for seed, r in sorted(by_seed.items()):
            if task == "binary":
                prob = np.load(nr.OUT / "seeds" / f"{task}_{model}_s{seed}.npz")["kddtest"].ravel()
                flagged = prob <= 0.5
                rows.append([flagged[seen].mean() * 100, flagged[unseen].mean() * 100,
                             flagged[~is_attack].mean() * 100])
            else:
                rows.append([r["kddtest_last"]["per_class"][c]["recall"] * 100
                             for c in nr.MULTI_CLASSES])
        rows = np.array(rows)
        out[task][model] = dict(mean=rows.mean(axis=0).tolist(),
                                std=(rows.std(axis=0, ddof=1) if len(rows) > 1
                                     else np.zeros(rows.shape[1])).tolist())
    return out


def unsw_stats(runs):
    out = {}
    for (task, model), by_seed in runs.items():
        seeds = sorted(by_seed)
        out[f"{task}/{model}"] = dict(
            seeds=seeds,
            test=[by_seed[s]["last"]["accuracy"] * 100 for s in seeds],
            test_best=[by_seed[s]["best"]["accuracy"] * 100 for s in seeds],
            macro_f1=[by_seed[s]["last"]["macro_f1"] * 100 for s in seeds],
            val=[by_seed[s]["val_accuracy"][-1] * 100 for s in seeds],
            collapses=[by_seed[s]["collapses"] for s in seeds],
            seconds=[by_seed[s]["seconds"] for s in seeds],
            per_class={c: float(np.mean([by_seed[s]["last"]["per_class"][c]["recall"]
                                         for s in seeds]) * 100)
                       for c in by_seed[seeds[0]]["last"]["per_class"]})
    return out


# ---------------------------------------------------------------------------- 圖
def plot_seeds(seed_table):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, task in zip(axes, TASKS):
        for i, m in enumerate(MODELS):
            row = seed_table.get(f"{task}/{m}")
            if row is None:
                continue
            acc = np.array(row["split"])
            ax.scatter(np.full(len(acc), i) + np.linspace(-0.12, 0.12, len(acc)), acc,
                       color=BLUE, s=22, zorder=3, label="各種子" if i == 0 else None)
            ax.hlines(acc.mean(), i - 0.25, i + 0.25, color="#12365f", lw=2,
                      label="平均" if i == 0 else None)
            ax.scatter([i], [row["paper"]], marker="D", color=ORANGE, s=42, zorder=4,
                       label="原論文" if i == 0 else None)
        ax.set_xticks(range(len(MODELS)), MODELS)
        ax.set_title(f"{TASK_ZH[task]}分類")
        ax.set_ylabel("內部測試集準確率 (%)")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "seeds.png", dpi=200)
    plt.close(fig)


def plot_kddtest(seed_table, detail):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    x, w = np.arange(len(MODELS)), 0.38
    for ax, task in zip(axes[:2], TASKS):
        have = [m for m in MODELS if f"{task}/{m}" in seed_table]
        xi = np.array([MODELS.index(m) for m in have])
        for off, key, color, label in ((-w / 2, "split", BLUE, "內部測試集"),
                                       (w / 2, "kddtest", ORANGE, "KDDTest+")):
            vals = [seed_table[f"{task}/{m}"][key] for m in have]
            ax.bar(xi + off, [np.mean(v) for v in vals], w, color=color, label=label,
                   yerr=[np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals],
                   error_kw=dict(lw=0.9, capsize=2))
        ax.set_xticks(x, MODELS, rotation=25, fontsize=8.5)
        ax.set_ylim(50, 103)
        ax.set_ylabel("準確率 (%)")
        ax.set_title(f"({'ab'[TASKS.index(task)]}) {TASK_ZH[task]}分類")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    axes[1].legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    ax = axes[2]
    have = [m for m in MODELS if m in detail["binary"]]
    xi = np.array([MODELS.index(m) for m in have])
    for off, col, color, label in ((-w / 2, 0, GREEN, f"訓練時見過的攻擊類型 (n={detail['n_seen']:,})"),
                                   (w / 2, 1, RED, f"未見過的攻擊類型 (n={detail['n_unseen']:,})")):
        ax.bar(xi + off, [detail["binary"][m]["mean"][col] for m in have], w, color=color,
               label=label, yerr=[detail["binary"][m]["std"][col] for m in have],
               error_kw=dict(lw=0.9, capsize=2))
    ax.set_xticks(x, MODELS, rotation=25, fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("攻擊偵測率 (%)")
    ax.set_title("(c) KDDTest+ 上的攻擊偵測率")
    ax.legend(["見過的攻擊類型", "未見過的攻擊類型"], frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "kddtest.png", dpi=200)
    plt.close(fig)


def plot_unsw(unsw):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    x, w = np.arange(len(MODELS)), 0.38
    for ax, task in zip(axes, TASKS):
        have = [m for m in MODELS if f"{task}/{m}" in unsw]
        xi = np.array([MODELS.index(m) for m in have])
        for off, key, color, label in ((-w / 2, "val", BLUE, "驗證集（取自訓練集）"),
                                       (w / 2, "test", ORANGE, "官方測試集")):
            vals = [unsw[f"{task}/{m}"][key] for m in have]
            ax.bar(xi + off, [np.mean(v) for v in vals], w, color=color, label=label, alpha=0.85)
            for xx, v in zip(xi + off, vals):
                ax.scatter(np.full(len(v), xx), v, s=12, color="#222", zorder=3,
                           label="各種子" if (key == "test" and xx == xi[0] + off) else None)
        ax.set_xticks(x, MODELS, rotation=25, fontsize=8.5)
        ax.set_ylim(50, 100)
        ax.set_ylabel("準確率 (%)")
        ax.set_title(f"UNSW-NB15 {'二元' if task == 'binary' else '十類'}分類")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "unsw.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------- 文字
def markdown(a):
    L = ["# 延伸實驗結果\n"]
    st = a["seeds"]
    for task in TASKS:
        rows = [m for m in MODELS if f"{task}/{m}" in st]
        if not rows:
            continue
        L += [f"## {TASK_ZH[task]}分類：多種子統計（%）\n",
              "| 模型 | 種子數 | 內部測試集 平均 ± 標準差 | 最小–最大 | 原論文 | 原論文是否在分布內 (p) | 崩落次數 | KDDTest+ 平均 ± 標準差 | KDDTest+ macro-F1 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m in rows:
            r = st[f"{task}/{m}"]
            p = f"{r['paper_p']:.3f}" if "paper_p" in r else "—"
            L.append(f"| {m} | {len(r['seeds'])} | {ms_fmt(r['split'])} | "
                     f"{min(r['split']):.2f}–{max(r['split']):.2f} | {r['paper']:.2f} | {p} | "
                     f"{sum(r['collapses'])} | {ms_fmt(r['kddtest'])} | {ms_fmt(r['kddtest_macro_f1'])} |")
        wins, common = a["best_counts"][task]
        if common:
            L += ["", f"各種子的最佳模型（種子 {common}）：" +
                  "、".join(f"{m} {n} 次" for m, n in wins.items() if n)]
        sig = [r for r in a["pairwise"][task] if r["p_holm"] < 0.05]
        L += ["", f"模型兩兩比較（Welch t 檢定，Holm 校正）：{len(a['pairwise'][task])} 組中 "
              f"{len(sig)} 組達顯著（p < 0.05）" +
              ("：" + "、".join(f"{r['a']} vs {r['b']} ({r['diff']:+.2f})" for r in sig) if sig else ""), ""]
    d = a["kddtest_detail"]
    if d["binary"]:
        L += ["## KDDTest+：已知與未知攻擊類型的偵測率（二元，%）\n",
              f"| 模型 | 見過的攻擊 (n={d['n_seen']}) | 未見過的攻擊 (n={d['n_unseen']}) | 正常流量誤報率 (n={d['n_normal']}) |",
              "|---|---|---|---|"]
        for m in MODELS:
            if m in d["binary"]:
                r = d["binary"][m]
                L.append(f"| {m} | " + " | ".join(f"{u:.2f} ± {s:.2f}"
                                                   for u, s in zip(r["mean"], r["std"])) + " |")
        L.append("")
    if d["multi"]:
        L += ["## KDDTest+：各類別 recall（五類，%）\n",
              "| 模型 | " + " | ".join(nr.MULTI_CLASSES) + " |", "|---|" + "---|" * 5]
        for m in MODELS:
            if m in d["multi"]:
                L.append(f"| {m} | " + " | ".join(f"{u:.2f}" for u in d["multi"][m]["mean"]) + " |")
        L.append("")
    u = a["unsw"]
    for task in TASKS:
        rows = [m for m in MODELS if f"{task}/{m}" in u]
        if not rows:
            continue
        L += [f"## UNSW-NB15 {'二元' if task == 'binary' else '十類'}分類（%）\n",
              "| 模型 | 種子數 | 驗證集 | 官方測試集 | 測試集 macro-F1 | 崩落次數 | 訓練秒數 |",
              "|---|---|---|---|---|---|---|"]
        for m in rows:
            r = u[f"{task}/{m}"]
            L.append(f"| {m} | {len(r['seeds'])} | {ms_fmt(r['val'])} | {ms_fmt(r['test'])} | "
                     f"{ms_fmt(r['macro_f1'])} | {sum(r['collapses'])} | {np.mean(r['seconds']):.0f} |")
        L.append("")
    return "\n".join(align_tables(L))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs("seeds")
    seed_table = seed_stats(runs)
    analysis = dict(seeds=seed_table,
                    pairwise={t: pairwise(seed_table, t) for t in TASKS},
                    pairwise_kddtest={t: pairwise(seed_table, t, "kddtest") for t in TASKS},
                    best_counts={t: best_counts(seed_table, t) for t in TASKS},
                    kddtest_detail=kddtest_detail(runs, ms.load_both()) if runs else
                    dict(binary={}, multi={}),
                    unsw=unsw_stats(load_runs("unsw")) if (nr.OUT / "unsw").exists() else {})
    (nr.OUT / "analysis.json").write_text(json.dumps(analysis, indent=1))
    if seed_table:
        plot_seeds(seed_table)
        plot_kddtest(seed_table, analysis["kddtest_detail"])
    if analysis["unsw"]:
        plot_unsw(analysis["unsw"])
    text = markdown(analysis)
    (nr.OUT / "EXTENSION.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
