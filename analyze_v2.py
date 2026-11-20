"""v2 實驗的彙整、統計檢定與圖表 (對應論文 RQ1–RQ5)。

讀取 results/v2 底下所有已完成的結果 (筆電與 Mac 合併後即為完整版)，輸出:
  results/v2/analysis/summary.json    所有彙整數字 (論文表格的來源)
  results/v2/analysis/runs.csv        每次訓練 x 每個測試集一列的原始資料
  results/v2/SUMMARY_v2.md            人可讀的摘要表格
  results/v2/figures/*.png            論文圖 (英文標示)

尚未完成的部分會自動略過，可隨時執行查看進度。

用法: python analyze_v2.py
"""
import glob
import itertools
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
V2 = Path(os.environ.get("NIDS_OUT") or ROOT / "results" / "v2")
OUT, FIG = V2 / "analysis", V2 / "figures"
DL = ["DNN", "CNN", "RNN", "LSTM", "CNN-LSTM", "GRU"]
TREES = ["RF", "XGB"]
ALL = TREES + DL + ["FTT"]                                          # FTT: FT-Transformer (第 9 個模型)
GATED = ["LSTM", "CNN-LSTM", "GRU"]
MAIN = {"nsl": "default", "unsw": "default", "cic": "improved", "cic18": "improved", "cic18b": "improved"}
ID_TEST = {"nsl": "random", "unsw": "random", "cic": "random", "cic18": "random", "cic18b": "random"}
OOD_TEST = {"nsl": "official", "unsw": "official"}                   # 分布外測試集
TEMPORAL = ("cic", "cic18")                                          # 有時間切分的資料集
DS_NAME = {"nsl": "NSL-KDD", "unsw": "UNSW-NB15", "cic": "CIC-IDS2017", "cic18": "CSE-CIC-IDS2018",
           "cic18b": "CSE-CIC-IDS2018 (second subsample)"}
COLORS = {"RF": "#6b8e23", "XGB": "#2e9e6b", "DNN": "#2a78d6", "CNN": "#7b4fc9", "RNN": "#8a8f98",
          "LSTM": "#eb6834", "CNN-LSTM": "#c9353b", "GRU": "#d4a017", "FTT": "#1a9fb0"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------------------- 載入
def load_runs():
    rows = []
    for f in glob.glob(str(V2 / "runs" / "*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        base = dict(source="dl", dataset=r["dataset"], condition=r["condition"], group=r["group"],
                    hpset=r["hpset"], task=r["task"], model=r["model"], seed=r["seed"],
                    perm=r.get("perm") or 0, collapses=r["collapses"], nan_loss=r["nan_loss"],
                    grad_max=max(r["grad_norm_max"]) if r["grad_norm_max"] else np.nan,
                    seconds=r["seconds"], device=r.get("device", ""))
        for test, m in r["tests_last"].items():
            best = r.get("tests_best", {}).get(test, {})
            rows.append(dict(base, test=test, **metric_cols(m), best_accuracy=best.get("accuracy", np.nan)))
    for f in glob.glob(str(V2 / "baselines" / "runs" / "*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        group = "main" if r["condition"] == MAIN[r["dataset"]] else "conditions"
        base = dict(source="tree", dataset=r["dataset"], condition=r["condition"], group=group,
                    hpset="tuned", task=r["task"], model=r["model"], seed=r["seed"], perm=0,
                    collapses=0, nan_loss=False, grad_max=np.nan, seconds=r["seconds"], device="cpu")
        for test, m in r["tests"].items():
            rows.append(dict(base, test=test, **metric_cols(m), best_accuracy=np.nan))
    return pd.DataFrame(rows)


def metric_cols(m):
    du = m.get("detection_unseen")
    return dict(accuracy=m["accuracy"] * 100, macro_f1=m["macro_f1"] * 100,
                detection=m.get("detection_rate", np.nan) * 100 if "detection_rate" in m else np.nan,
                false_alarm=m.get("false_alarm_rate", np.nan) * 100 if "false_alarm_rate" in m else np.nan,
                detection_unseen=du * 100 if du is not None and du == du else np.nan)


# ---------------------------------------------------------------------------- 統計工具
def ms(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "—"
    return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v[0]:.2f}"


def holm(p):
    p = np.asarray(p, float)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def pairwise(values):
    """values: {model: [各種子的分數]}。回傳 Welch t 檢定 + Holm 校正，以及單次比較與平均排名矛盾的機率。"""
    models = [m for m in values if len(values[m]) >= 2]
    pairs = list(itertools.combinations(models, 2))
    if not pairs:
        return []
    out = []
    for a, b in pairs:
        va, vb = np.asarray(values[a]), np.asarray(values[b])
        t = stats.ttest_ind(va, vb, equal_var=False)
        sign = np.sign(va.mean() - vb.mean())
        # 任取 A 的一個種子對上 B 的一個種子，結論和平均相反 (或相同) 的比例
        diffs = (va[:, None] - vb[None, :]).ravel()
        flip = float(np.mean(np.sign(diffs) != sign)) if sign != 0 else 0.5
        out.append(dict(a=a, b=b, diff=float(va.mean() - vb.mean()), p=float(t.pvalue), flip=flip))
    for r, adj in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = float(adj)
    return out


# ---------------------------------------------------------------------------- RQ1: 模型排名
def rq1(df):
    """主要條件 (調參後) 的多種子統計、兩兩比較，以及單次實驗排名的可靠度。"""
    res = {}
    main = df[(df.hpset == "tuned") & (df.group.isin(["main", "conditions"])) & (df.perm == 0)]
    for (ds, cond, task, test), g in main.groupby(["dataset", "condition", "task", "test"]):
        values = {m: g[g.model == m].sort_values("seed").accuracy.tolist() for m in ALL if (g.model == m).any()}
        f1 = {m: g[g.model == m].macro_f1.tolist() for m in values}
        pw = pairwise(values)
        seeds = sorted(set.intersection(*[set(g[g.model == m].seed) for m in values])) if values else []
        wins = {}
        for s in seeds:
            sc = {m: float(g[(g.model == m) & (g.seed == s)].accuracy.iloc[0]) for m in values}
            w = max(sc, key=sc.get)
            wins[w] = wins.get(w, 0) + 1
        res[f"{ds}/{cond}/{task}/{test}"] = dict(
            accuracy={m: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                              n=len(v), values=v) for m, v in values.items()},
            macro_f1={m: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                      for m, v in f1.items()},
            pairwise=pw, wins=wins, common_seeds=seeds,
            n_significant=sum(r["p_holm"] < 0.05 for r in pw),
            mean_flip=float(np.mean([r["flip"] for r in pw])) if pw else None)
    return res


def rank_shift(rq1_res):
    """同一批模型在同分布與分布外測試集上的平均排名是否一致 (Kendall tau)。"""
    out = {}
    for ds, ood in OOD_TEST.items():
        for task in ("binary", "multi"):
            a = rq1_res.get(f"{ds}/{MAIN[ds]}/{task}/{ID_TEST[ds]}")
            b = rq1_res.get(f"{ds}/{MAIN[ds]}/{task}/{ood}")
            if not a or not b:
                continue
            models = [m for m in a["accuracy"] if m in b["accuracy"]]
            if len(models) < 3:
                continue
            x = [a["accuracy"][m]["mean"] for m in models]
            y = [b["accuracy"][m]["mean"] for m in models]
            tau = stats.kendalltau(x, y)
            out[f"{ds}/{task}"] = dict(models=models, tau=float(tau.statistic), p=float(tau.pvalue),
                                       rank_id=list(np.array(models)[np.argsort(x)[::-1]]),
                                       rank_ood=list(np.array(models)[np.argsort(y)[::-1]]))
    # CIC: 隨機切分 (修正標記版) 與時間切分 (不同訓練集，同一批模型設定)
    for ds in TEMPORAL:
        a, b = rq1_res.get(f"{ds}/improved/binary/random"), rq1_res.get(f"{ds}/temporal/binary/temporal")
        if not a or not b:
            continue
        models = [m for m in a["accuracy"] if m in b["accuracy"]]
        if len(models) >= 3:
            x = [a["accuracy"][m]["mean"] for m in models]
            y = [b["accuracy"][m]["mean"] for m in models]
            tau = stats.kendalltau(x, y)
            out[f"{ds}/binary"] = dict(models=models, tau=float(tau.statistic), p=float(tau.pvalue),
                                       rank_id=list(np.array(models)[np.argsort(x)[::-1]]),
                                       rank_ood=list(np.array(models)[np.argsort(y)[::-1]]))
    return out


def subsample_check(df):
    """CSE-CIC-IDS2018 兩份獨立樣本 (cic18 與 cic18b) 上相同模型的結果是否一致。"""
    out = {}
    tuned = df[(df.hpset == "tuned") & (df.perm == 0)]
    for cond, task, test in (("improved", "binary", "random"), ("improved", "multi", "random"),
                             ("temporal", "binary", "temporal")):
        for m in ALL:
            v = {}
            for ds in ("cic18", "cic18b"):
                g = tuned[(tuned.dataset == ds) & (tuned.condition == cond) & (tuned.task == task) &
                          (tuned.model == m) & (tuned.test == test)]
                if len(g):
                    v[ds] = dict(acc=g.accuracy.tolist(), f1=g.macro_f1.tolist(),
                                 detection=g.detection.tolist() if task == "binary" else None)
            if len(v) == 2:
                a, b = np.array(v["cic18"]["acc"]), np.array(v["cic18b"]["acc"])
                p = float(stats.ttest_ind(a, b, equal_var=False).pvalue) if len(a) > 1 and len(b) > 1 else None
                out[f"{cond}/{task}/{m}"] = dict(a=dict(mean=float(a.mean()), sd=float(a.std(ddof=1))),
                                                 b=dict(mean=float(b.mean()), sd=float(b.std(ddof=1))),
                                                 f1_a=float(np.mean(v["cic18"]["f1"])),
                                                 f1_b=float(np.mean(v["cic18b"]["f1"])), p=p)
    return out


# ---------------------------------------------------------------------------- RQ2: 訓練穩定性
def rq2(df):
    dl = df[(df.source == "dl") & (df.perm == 0)]
    first = dl.drop_duplicates(["dataset", "condition", "group", "hpset", "task", "model", "seed", "perm"])
    stab = {}
    for (ds, task, model, hpset), g in first[first.condition == first.dataset.map(MAIN)].groupby(
            ["dataset", "task", "model", "hpset"]):
        stab[f"{ds}/{task}/{model}/{hpset}"] = dict(
            n=int(len(g)), collapses_mean=float(g.collapses.mean()),
            runs_with_collapse=float((g.collapses > 0).mean()),
            grad_max_median=float(g.grad_max.median()), nan_runs=int(g.nan_loss.sum()))
    # 調參前後與裁剪的比較 (同分布測試集準確率)
    compare = {}
    for (ds, task, model), g in dl[(dl.condition == dl.dataset.map(MAIN)) &
                                   (dl.test == dl.dataset.map(ID_TEST))].groupby(["dataset", "task", "model"]):
        row = {}
        for hpset in ("paper", "paper_clip", "tuned"):
            v = g[g.hpset == hpset]
            if len(v):
                row[hpset] = dict(mean=float(v.accuracy.mean()),
                                  sd=float(v.accuracy.std(ddof=1)) if len(v) > 1 else 0.0,
                                  collapses=int(v.collapses.sum()), n=int(len(v)))
        compare[f"{ds}/{task}/{model}"] = row
    return dict(stability=stab, hp_compare=compare)


def rq2_permutation(df):
    """特徵順序打亂: 5 種排列 (種子 1) 的變異是否大於 5 個種子 (原始順序) 的變異。DNN 是對照組。"""
    out = {}
    dl = df[(df.source == "dl") & (df.hpset == "tuned") & (df.task == "binary") & (df.group.isin(["main", "perm"]))]
    for (ds, model), g in dl.groupby(["dataset", "model"]):
        if ds not in OOD_TEST:
            continue
        for test in (ID_TEST[ds], OOD_TEST[ds]):
            seeds = g[(g.perm == 0) & (g.group == "main") & (g.test == test)].accuracy.values
            perms = g[(g.perm > 0) & (g.test == test)].accuracy.values
            if len(seeds) < 2 or len(perms) < 2:
                continue
            t = stats.ttest_ind(perms, seeds, equal_var=False)
            lev = stats.levene(perms, seeds)
            out[f"{ds}/{model}/{test}"] = dict(
                seeds=dict(mean=float(seeds.mean()), sd=float(seeds.std(ddof=1)), n=len(seeds)),
                perms=dict(mean=float(perms.mean()), sd=float(perms.std(ddof=1)), n=len(perms)),
                diff=float(perms.mean() - seeds.mean()), p_mean=float(t.pvalue), p_var=float(lev.pvalue))
    return out


# ---------------------------------------------------------------------------- RQ3: 評估陷阱
def rq3(df):
    out = {}
    tuned = df[(df.hpset == "tuned") & (df.perm == 0)]
    # NSL / UNSW: 同分布 vs 分布外
    for ds, ood in OOD_TEST.items():
        for task in ("binary", "multi"):
            g = tuned[(tuned.dataset == ds) & (tuned.task == task) & (tuned.group == "main")]
            row = {}
            for m in ALL:
                a = g[(g.model == m) & (g.test == ID_TEST[ds])]
                b = g[(g.model == m) & (g.test == ood)]
                if len(a) and len(b):
                    row[m] = dict(id_acc=float(a.accuracy.mean()), ood_acc=float(b.accuracy.mean()),
                                  gap=float(a.accuracy.mean() - b.accuracy.mean()),
                                  id_f1=float(a.macro_f1.mean()), ood_f1=float(b.macro_f1.mean()),
                                  unseen=float(b.detection_unseen.mean()) if task == "binary" else None)
            out[f"{ds}/{task}"] = row
    # CIC: 原始版 -> 去重 -> 修正標記 -> 時間切分 (CSE-CIC-IDS2018 只有後兩步)
    steps = [("orig", "random"), ("dedup", "random"), ("improved", "random"), ("temporal", "temporal")]
    for ds in TEMPORAL:
        for task in ("binary", "multi"):
            row = {}
            for m in ALL:
                cell = {}
                for cond, test in steps:
                    v = tuned[(tuned.dataset == ds) & (tuned.condition == cond) & (tuned.task == task) &
                              (tuned.model == m) & (tuned.test == test)]
                    if len(v):
                        cell[cond] = dict(acc=float(v.accuracy.mean()), f1=float(v.macro_f1.mean()),
                                          detection=float(v.detection.mean()) if task == "binary" else None,
                                          false_alarm=float(v.false_alarm.mean()) if task == "binary" else None,
                                          n=int(len(v)))
                if cell:
                    row[m] = cell
            out[f"{ds}/{task}"] = row
    return out


def temporal_by_attack(ds="cic"):
    """時間切分下各攻擊類型的偵測率 (調參後二元模型，種子平均)。"""
    acc = {}
    for f in glob.glob(str(V2 / "runs" / f"{ds}_temporal_tuned_binary_*.json")) + \
            glob.glob(str(V2 / "baselines" / "runs" / f"{ds}_temporal_binary_*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        tests = r.get("tests_last") or r.get("tests")
        for attack, (rate, n) in tests["temporal"].get("detection_by_attack", {}).items():
            acc.setdefault(r["model"], {}).setdefault(attack, []).append(rate * 100)
            acc.setdefault("_n", {})[attack] = n
    n = acc.pop("_n", {})
    return dict(n=n, rates={m: {a: float(np.mean(v)) for a, v in d.items()} for m, d in acc.items()})


# ---------------------------------------------------------------------------- RQ4 / RQ5
def rq4():
    out = {}
    for f in glob.glob(str(V2 / "shortcut" / "*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        if "ranking" not in r:          # check_summary.json (10 種子消融) 另外處理
            continue
        imp_key = [k for k in r if k.startswith("importance/")][0]
        imp = np.array(r[imp_key]["mean"])
        out[f"{r['dataset']}/{r['condition']}"] = dict(
            top5=r["ranking"][:5], top5_share=float(np.sort(imp)[::-1][:5].sum() / imp.sum()),
            ablation=[dict(setting=a["setting"], n=a["n_features"],
                           **{k: float(np.mean(v)) * 100 for k, v in a["accuracy"].items()})
                      for a in r["ablation"]])
    return out


def rq5():
    out = {}
    for stage in ("attack", "advtrain", "advtrain_consistent"):
        rows = []
        for f in glob.glob(str(V2 / "adversarial" / stage / "*.json")):
            r = json.loads(Path(f).read_text(encoding="utf-8"))
            rob = r if stage == "attack" else r["robustness"]
            if not rob.get("pgd"):
                continue
            for threat, rates in rob["pgd"].items():
                for e, p, q in zip(rob["eps"], rates, rob["rs"][threat]):
                    rows.append(dict(dataset=r["dataset"], model=r["model"], seed=r["seed"], threat=threat,
                                     eps=e, pgd=p * 100, rs=q * 100, best=max(p, q) * 100))
        if rows:
            d = pd.DataFrame(rows)
            out[stage] = (d.groupby(["dataset", "threat", "model", "eps"])[["pgd", "rs", "best"]]
                          .mean().round(2).reset_index().to_dict("records"))
    trans = []
    for f in glob.glob(str(V2 / "adversarial" / "transfer" / "*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        for threat, models in r["rates"].items():
            for m, rates in models.items():
                for e, v in zip(r["eps"], rates):
                    trans.append(dict(dataset=r["dataset"], seed=r["seed"], threat=threat, model=m, eps=e, rate=v * 100))
    if trans:
        out["transfer"] = (pd.DataFrame(trans).groupby(["dataset", "threat", "model", "eps"]).rate.mean()
                           .round(2).reset_index().to_dict("records"))
    return out


# ---------------------------------------------------------------------------- 圖
def fig_seeds(df):
    panels = [(ds, MAIN[ds], ID_TEST[ds]) for ds in ("nsl", "unsw", "cic", "cic18")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))
    axes = axes.ravel()
    for ax, (ds, cond, test) in zip(axes, panels):
        g = df[(df.dataset == ds) & (df.condition == cond) & (df.task == "binary") & (df.test == test) & (df.perm == 0)]
        for i, m in enumerate(ALL):
            for hpset, dx, mk, col in (("paper", -0.15, "x", "#555555"), ("tuned", 0.15, "o", COLORS[m])):
                v = g[(g.model == m) & (g.hpset == hpset)].accuracy.values
                if len(v):
                    ax.scatter(np.full(len(v), i + dx), v, marker=mk, s=18, color=col, zorder=3,
                               label=None)
        ax.set_xticks(range(len(ALL)), [{"FTT": "FT-T"}.get(m, m) for m in ALL], rotation=30)
        ax.set_title(f"{DS_NAME[ds]} (binary, {test} split)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].scatter([], [], marker="x", color="#555555", label="original hyperparameters")
    axes[0].scatter([], [], marker="o", color="#2a78d6", label="tuned (equal budget)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v2_seeds.png", dpi=200)
    plt.close(fig)


def fig_gradnorm():
    """NSL-KDD 二元: 閘門模型在原論文設定、加裁剪、調參後的每 epoch 最大梯度範數。"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    styles = {"paper": ("#eb6834", "original"), "paper_clip": ("#2e9e6b", "original + clipnorm=1"),
              "tuned": ("#2a78d6", "tuned")}
    drawn = False
    for ax, m in zip(axes, GATED):
        for hpset, (col, label) in styles.items():
            for i, f in enumerate(sorted(glob.glob(str(V2 / "runs" / f"nsl_default_{hpset}_binary_{m}_s*.json")))):
                r = json.loads(Path(f).read_text(encoding="utf-8"))
                if r.get("perm"):
                    continue
                if r["grad_norm_max"]:
                    ax.plot(r["grad_norm_max"], color=col, lw=0.8, alpha=0.7, label=label if i == 0 else None)
                    drawn = True
        ax.set_yscale("log")
        ax.set_title(f"{m} (NSL-KDD, binary)")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("max gradient norm per epoch")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if drawn:
        fig.savefig(FIG / "v2_gradnorm.png", dpi=200)
    plt.close(fig)


def fig_conditions(rq3_res):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw=dict(width_ratios=[1.35, 1.25, 0.9]))
    # (a) NSL / UNSW: in-distribution vs out-of-distribution
    ax = axes[0]
    labels, gaps, xs, groups = [], [], [], []
    for gi, ds in enumerate(OOD_TEST):
        row = rq3_res.get(f"{ds}/binary", {})
        start = len(xs)
        for m in ALL:
            if m in row:
                labels.append("C-LSTM" if m == "CNN-LSTM" else m)
                gaps.append((row[m]["id_acc"], row[m]["ood_acc"]))
                xs.append(len(xs) + gi)              # 兩個資料集之間空一格
        if len(xs) > start:
            groups.append(((xs[start] + xs[-1]) / 2, DS_NAME[ds]))
    if gaps:
        x = np.array(xs, float)
        ax.bar(x - 0.2, [g[0] for g in gaps], 0.4, color="#2a78d6", label="random split (in-distribution)")
        ax.bar(x + 0.2, [g[1] for g in gaps], 0.4, color="#eb6834", label="official test set")
        ax.set_xticks(x, labels, fontsize=7, rotation=45, ha="right")
        for cx, name in groups:
            ax.text(cx, 101, name, ha="center", va="bottom", fontsize=8)
        ax.set_ylim(50, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
        ax.set_title("(a) NSL-KDD and UNSW-NB15, binary", pad=14)
    # (b) CIC: orig -> dedup -> improved -> temporal
    ax = axes[1]
    row = rq3_res.get("cic/binary", {})
    conds = [("orig", "original"), ("dedup", "deduplicated"), ("improved", "relabelled"),
             ("temporal", "temporal split")]
    models = [m for m in ALL if m in row]
    w = 0.8 / len(conds)
    for k, (c, label) in enumerate(conds):
        vals = [row[m][c]["f1"] if c in row[m] else np.nan for m in models]
        ax.bar(np.arange(len(models)) + (k - 1.5) * w, vals, w, label=label)
    lab = lambda ms: [{"FTT": "FT-T", "CNN-LSTM": "C-LSTM"}.get(m, m) for m in ms]
    ax.set_xticks(range(len(models)), lab(models), fontsize=7, rotation=45, ha="right")
    ax.set_ylim(40, 100)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("(b) CIC-IDS2017, binary: evaluation conditions")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    # (c) CSE-CIC-IDS2018: corrected random split -> temporal split
    ax = axes[2]
    row = rq3_res.get("cic18/binary", {})
    models = [m for m in ALL if m in row]
    for k, (c, label, col) in enumerate((("improved", "random split", "#2ca02c"),
                                         ("temporal", "temporal split", "#d62728"))):
        vals = [row[m][c]["f1"] if c in row[m] else np.nan for m in models]
        ax.bar(np.arange(len(models)) + (k - 0.5) * 0.4, vals, 0.4, label=label, color=col)
    ax.set_xticks(range(len(models)), lab(models), fontsize=7, rotation=45, ha="right")
    ax.set_ylim(40, 100)
    ax.set_title("(c) CSE-CIC-IDS2018, binary")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(FIG / "v2_conditions.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------- 摘要
def markdown(df, res):
    L = ["# v2 實驗摘要（自動產生）\n",
         f"已讀取 {df[['source', 'dataset', 'condition', 'group', 'hpset', 'task', 'model', 'seed', 'perm']].drop_duplicates().shape[0]} 次訓練"
         f"（深度模型 {df[df.source == 'dl'][['dataset', 'condition', 'group', 'hpset', 'task', 'model', 'seed', 'perm']].drop_duplicates().shape[0]}、"
         f"樹模型 {df[df.source == 'tree'][['dataset', 'condition', 'task', 'model', 'seed']].drop_duplicates().shape[0]}）。\n"]

    L.append("## RQ1 模型排名（調參後，準確率 %，平均 ± 標準差）\n")
    for key, r in res["rq1"].items():
        ds, cond, task, test = key.split("/")
        if not r["accuracy"]:
            continue
        L.append(f"**{DS_NAME[ds]} / {cond} / {task} / 測試集 {test}**　顯著差異 {r['n_significant']}/{len(r['pairwise'])} 組；"
                 + (f"單次比較與平均排名矛盾的平均機率 {r['mean_flip'] * 100:.1f}%；" if r["mean_flip"] is not None else "")
                 + f"各種子最佳: {r['wins']}\n")
        L.append("| 模型 | 種子 | 準確率 | macro-F1 |\n|---|---|---|---|")
        for m in ALL:
            if m in r["accuracy"]:
                a = r["accuracy"][m]
                f = r["macro_f1"][m]
                L.append(f"| {m} | {a['n']} | {a['mean']:.2f} ± {a['sd']:.2f} | {f['mean']:.2f} ± {f['sd']:.2f} |")
        L.append("")

    if res["rank_shift"]:
        L.append("## RQ1 排名在分布外測試集是否改變（Kendall τ）\n")
        L.append("| 設定 | τ | p | 同分布排名 | 分布外排名 |\n|---|---|---|---|---|")
        for k, r in res["rank_shift"].items():
            L.append(f"| {k} | {r['tau']:.2f} | {r['p']:.3f} | {' > '.join(r['rank_id'])} | {' > '.join(r['rank_ood'])} |")
        L.append("")

    L.append("## RQ2 原論文超參數 / 加裁剪 / 調參後（同分布測試集準確率 %，崩落次數合計）\n")
    L.append("| 設定 | 原論文 | 原論文 + 裁剪 | 調參後 |\n|---|---|---|---|")
    for k, r in sorted(res["rq2"]["hp_compare"].items()):
        cell = lambda h: (f"{r[h]['mean']:.2f} ± {r[h]['sd']:.2f}（崩落 {r[h]['collapses']}，n={r[h]['n']}）"
                          if h in r else "—")
        L.append(f"| {k} | {cell('paper')} | {cell('paper_clip')} | {cell('tuned')} |")
    L.append("")

    if res["rq2_perm"]:
        L.append("## RQ2 打亂特徵順序（5 種排列 vs 5 個種子，二元）\n")
        L.append("| 設定 | 原始順序（5 種子） | 打亂（5 排列） | 差 | p（平均） | p（變異） |\n|---|---|---|---|---|---|")
        for k, r in sorted(res["rq2_perm"].items()):
            L.append(f"| {k} | {r['seeds']['mean']:.2f} ± {r['seeds']['sd']:.2f} | {r['perms']['mean']:.2f} ± {r['perms']['sd']:.2f} "
                     f"| {r['diff']:+.2f} | {r['p_mean']:.3f} | {r['p_var']:.3f} |")
        L.append("")

    L.append("## RQ3 評估陷阱\n")
    for ds in OOD_TEST:
        for task in ("binary", "multi"):
            row = res["rq3"].get(f"{ds}/{task}", {})
            if not row:
                continue
            L.append(f"**{DS_NAME[ds]} {task}：隨機切分 vs 官方測試集**\n")
            L.append("| 模型 | 隨機切分 | 官方測試集 | 落差 | 未見過攻擊偵測率 |\n|---|---|---|---|---|")
            for m, r in row.items():
                un = f"{r['unseen']:.1f}" if r.get("unseen") == r.get("unseen") and r.get("unseen") is not None else "—"
                L.append(f"| {m} | {r['id_acc']:.2f} | {r['ood_acc']:.2f} | {r['gap']:.2f} | {un} |")
            L.append("")
    for task in ("binary", "multi"):
        row = res["rq3"].get(f"cic/{task}", {})
        if not row:
            continue
        L.append(f"**CIC-IDS2017 {task}：準確率 / macro-F1（原始版 → 去重 → 修正標記 → 時間切分）**\n")
        L.append("| 模型 | 原始版 | 去重 | 修正標記 | 時間切分 |\n|---|---|---|---|---|")
        for m, cell in row.items():
            c = lambda k: f"{cell[k]['acc']:.2f} / {cell[k]['f1']:.2f}" if k in cell else "—"
            L.append(f"| {m} | {c('orig')} | {c('dedup')} | {c('improved')} | {c('temporal')} |")
        L.append("")

    if res["temporal_attacks"]["rates"]:
        t = res["temporal_attacks"]
        attacks = sorted(t["n"], key=lambda a: -t["n"][a])
        L.append("**CIC-IDS2017 時間切分：各攻擊類型偵測率 %（全部是訓練時沒見過的類型）**\n")
        L.append("| 模型 | " + " | ".join(f"{a} (n={t['n'][a]:,})" for a in attacks) + " |")
        L.append("|---|" + "---|" * len(attacks))
        for m in ALL:
            if m in t["rates"]:
                L.append(f"| {m} | " + " | ".join(f"{t['rates'][m].get(a, float('nan')):.1f}" for a in attacks) + " |")
        L.append("")

    if res["rq4"]:
        L.append("## RQ4 捷徑分析\n")
        for k, r in res["rq4"].items():
            L.append(f"**{k}**：前 5 名特徵 {r['top5']}，佔重要度 {r['top5_share'] * 100:.0f}%\n")
            tests = [x for x in r["ablation"][0] if x not in ("setting", "n")]
            L.append("| 設定 | 特徵數 | " + " | ".join(tests) + " |\n|---|---|" + "---|" * len(tests))
            for a in r["ablation"]:
                L.append(f"| {a['setting']} | {a['n']} | " + " | ".join(f"{a[t]:.2f}" for t in tests) + " |")
            L.append("")

    if res["rq5"]:
        L.append("## RQ5 對抗穩健性（逃避偵測成功率 %，PGD / 隨機搜尋）\n")
        for stage, rows in res["rq5"].items():
            if stage == "transfer":
                continue
            d = pd.DataFrame(rows)
            L.append(f"**{'標準訓練' if stage == 'attack' else '對抗訓練後'}**\n")
            for (ds, threat), g in d.groupby(["dataset", "threat"]):
                eps = sorted(g.eps.unique())
                L.append(f"{DS_NAME[ds]}，{threat}\n")
                L.append("| 模型 | " + " | ".join(f"ε={e:g}" for e in eps) + " |\n|---|" + "---|" * len(eps))
                for m in ALL:
                    gm = g[g.model == m]
                    if len(gm):
                        L.append(f"| {m} | " + " | ".join(
                            f"{gm[gm.eps == e].pgd.iloc[0]:.1f} / {gm[gm.eps == e].rs.iloc[0]:.1f}" for e in eps) + " |")
                L.append("")
    try:
        from make_report import align_tables
        L = align_tables(L)
    except Exception:
        pass
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_runs()
    if df.empty:
        print("還沒有任何結果。")
        return
    df.to_csv(OUT / "runs.csv", index=False)
    res = dict(rq1=rq1(df))
    res["rank_shift"] = rank_shift(res["rq1"])
    res["rq2"] = rq2(df)
    res["rq2_perm"] = rq2_permutation(df)
    res["rq3"] = rq3(df)
    res["temporal_attacks"] = temporal_by_attack("cic")
    res["temporal_attacks18"] = temporal_by_attack("cic18")
    res["subsample"] = subsample_check(df)
    res["rq4"] = rq4()
    res["rq5"] = rq5()
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    fig_seeds(df)
    fig_gradnorm()
    fig_conditions(res["rq3"])
    text = markdown(df, res)
    (V2 / "SUMMARY_v2.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n已輸出 {OUT / 'summary.json'}、{V2 / 'SUMMARY_v2.md'}、{FIG}")


if __name__ == "__main__":
    main()
