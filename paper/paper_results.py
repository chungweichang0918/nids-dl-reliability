"""論文第 V 節 (結果) 的 RQ1–RQ4。

所有數字都從 results/v2/analysis/summary.json (analyze_v2.py 產生) 與
results/v2/shortcut/check_summary.json 即時計算；實驗跑完後重新執行
analyze_v2.py 與 build_paper.py 即可更新，不需要手動改數字。
尚未跑完的部分 (例如 CIC-IDS2017 的樹模型) 會以黃色 TODO 標示。
"""
import json
from pathlib import Path

import numpy as np

from paper_docx import ROMAN

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
FIG = V2 / "figures"
DEEP = ["DNN", "CNN", "RNN", "LSTM", "CNN-LSTM", "GRU"]
TREES = ["RF", "XGB"]
ALL = TREES + DEEP + ["FTT"]                     # DEEP: [@elsayed2024] 的六個架構; FTT: FT-Transformer
GATED = ["LSTM", "CNN-LSTM", "GRU"]
DISP = {"RNN": "SimpleRNN", "RF": "Random forest", "XGB": "XGBoost", "FTT": "FT-Transformer"}
SHORT = {"RNN": "SimpleRNN", "RF": "RF", "XGB": "XGB", "FTT": "FT-Transformer"}
DSN = {"nsl": "NSL-KDD", "unsw": "UNSW-NB15", "cic": "CIC-IDS2017", "cic18": "CSE-CIC-IDS2018"}


WORD = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
        9: "nine", 10: "ten"}
word = lambda n: WORD.get(n, str(n))
ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
           9: "ninth"}


def disp(m):
    return DISP.get(m, m)


def short(m):
    return SHORT.get(m, m)


def load():
    S = json.loads((V2 / "analysis" / "summary.json").read_text(encoding="utf-8"))
    # 第二份 CSE-CIC-IDS2018 樣本只含 DNN 與樹模型，是穩健性檢查，不列入排名統計
    S["rq1_b"] = {k: v for k, v in S["rq1"].items() if k.startswith("cic18b/")}
    S["rq1"] = {k: v for k, v in S["rq1"].items() if not k.startswith("cic18b/")}
    check = V2 / "shortcut" / "check_summary.json"
    S["check"] = json.loads(check.read_text(encoding="utf-8")) if check.exists() else {}
    S["tuned_hp"] = json.loads((V2 / "tuned_hp.json").read_text(encoding="utf-8"))
    return S


def mean_sd(d, digits=2):
    return f"{d['mean']:.{digits}f} ± {d['sd']:.{digits}f}"


def ranking(entry):
    acc = entry["accuracy"]
    return sorted(acc, key=lambda m: -acc[m]["mean"])


def mean_flip(entry):
    return 100 * float(np.mean([p["flip"] for p in entry["pairwise"]]))


def max_flip(entry):
    return 100 * float(max(p["flip"] for p in entry["pairwise"]))


def rng1(vals):
    lo, hi = f"{min(vals):.1f}", f"{max(vals):.1f}"
    return f"about {lo}" if lo == hi else f"{lo}–{hi}"


# ---------------------------------------------------------------------------- 摘要用的句子
def abstract_sentences(S):
    r = S["rq1"]
    id_keys = [k for k in r if k.endswith("/random")]
    flips = [mean_flip(r[k]) for k in r]
    imp = r["cic/improved/binary/random"]
    nsl = r["nsl/default/binary/random"]
    gru_rank = ranking(nsl).index("GRU") + 1
    rq3 = S["rq3"]
    nsl_gap = [v["gap"] for v in rq3["nsl/binary"].values()]
    cic = rq3["cic/binary"]
    t_f1 = [cic[m]["temporal"]["f1"] for m in cic if "temporal" in cic[m]]
    i_f1 = [cic[m]["improved"]["f1"] for m in cic if "improved" in cic[m]]
    chk = S["check"]
    s = (f"Across {len(r)} dataset–task–test combinations, a comparison based on one run per model "
         f"contradicts the multi-seed ordering of a model pair in {np.mean(flips):.0f}% of cases on average, and "
         f"on the corrected CIC-IDS2017 none of the {len(imp['pairwise'])} pairwise differences is significant. "
         f"Under tuning with equal trial budgets, GRU, reported as the best binary model, ranks {gru_rank}th of eight on "
         "NSL-KDD, and tree ensembles rank first on the in-distribution tests of NSL-KDD and UNSW-NB15. The gated "
         "recurrent models collapse with the "
         "original hyperparameters; gradient clipping removes most collapses, but tuning with shortened "
         "training favors the largest learning rate and can reintroduce them. Moving from a random split to "
         f"the official test set costs {min(nsl_gap):.0f}–{max(nsl_gap):.0f} points on NSL-KDD, and a temporal "
         f"split of the corrected CIC-IDS2017 lowers macro-F1 from {rng1(i_f1)}% to "
         f"{min(t_f1):.1f}–{max(t_f1):.1f}%, because almost only DDoS is detected among the unseen attacks.")
    if "drop_port_rst" in chk:
        a, b = chk["all"], chk["drop_port_rst"]
        s += (f" On this split, removing two features (destination port and RST flag count) raises accuracy from "
              f"{a['accuracy']['mean']:.1f}% to {b['accuracy']['mean']:.1f}%, exposing a shortcut learned from the "
              "training days.")
    return s


# ---------------------------------------------------------------------------- RQ1
RQ1_COLS = [("nsl/default/binary/random", "NSL random"), ("nsl/default/binary/official", "NSL KDDTest+"),
            ("unsw/default/binary/random", "UNSW random"), ("unsw/default/binary/official", "UNSW official"),
            ("cic/improved/binary/random", "CIC17 random"), ("cic/temporal/binary/temporal", "CIC17 temporal"),
            ("cic18/improved/binary/random", "CIC18 random"), ("cic18/temporal/binary/temporal", "CIC18 temporal")]


def rq1(P, S):
    r = S["rq1"]
    P.subsection("RQ1: Reliability of Model Rankings")
    nsl = r["nsl/default/binary/random"]
    rank = ranking(nsl)
    gru_rank = rank.index("GRU") + 1
    acc = nsl["accuracy"]
    hpc = S["rq2"]["hp_compare"]["nsl/binary/GRU"]["paper"]
    deep_no_lstm = [acc[m]["mean"] for m in DEEP if m != "LSTM"]
    P.para(f"Table {P.next_table()} lists the binary accuracy of all tuned models on the in-distribution "
           "(random-split) and out-of-distribution test sets, and Fig. 1 shows every individual run. "
           f"**The central claim of the case study does not hold.** With the original hyperparameters, GRU "
           f"reaches {mean_sd(hpc)}% on the NSL-KDD random split (five seeds), and after tuning with the same number of trials as every other model it "
           f"ranks {ORDINAL[gru_rank]} of the {len(acc)} models ({mean_sd(acc['GRU'])}%). The two tree ensembles rank first "
           f"({mean_sd(acc['XGB'])}% for XGBoost, {mean_sd(acc['RF'])}% for random forest), and the six deep models "
           f"of [@elsayed2024] other than LSTM lie within {max(deep_no_lstm) - min(deep_no_lstm):.2f} points of "
           "each other.")

    cic_has_trees = all(m in r["cic/improved/binary/random"]["accuracy"] for m in TREES)
    rows = []
    for m in ALL:
        row = [disp(m)]
        for key, _ in RQ1_COLS:
            a = r.get(key, {}).get("accuracy", {})
            if m not in a:
                row.append("—")
                continue
            best = max(a, key=lambda k: a[k]["mean"])
            cell = mean_sd(a[m])
            row.append(f"**{cell}**" if m == best else cell)
        rows.append(row)
    rows.append(["Significant pairs"] + [f"{r[k]['n_significant']}/{len(r[k]['pairwise'])}" for k, _ in RQ1_COLS])
    rows.append(["Mean flip prob. (%)"] + [f"{mean_flip(r[k]):.1f}" for k, _ in RQ1_COLS])
    P.columns(1)
    P.table(["Model"] + [c for _, c in RQ1_COLS], rows, [2.6] + [1.8] * 8,
            "Binary accuracy (%) of the tuned models, mean ± standard deviation over five seeds (NSL-KDD, "
            "UNSW-NB15) or three seeds (CIC-IDS2017, CSE-CIC-IDS2018). Bold: best mean per column. Significant "
            "pairs: pairwise Welch tests with Holm correction at α = 0.05. Flip probability: Eq. (1), averaged "
            "over all pairs.", left_cols=(0,))
    S["tab_rq1"] = ROMAN[P.tab_no - 1]
    P.figure(FIG / "v2_seeds.png", 17.0,
             "Accuracy of every run in the binary task on the random-split test sets. Crosses: original "
             "hyperparameters of [@elsayed2024]; dots: models tuned with equal trial budgets. Note the different y-axis "
             "ranges.")
    P.columns(2)
    if not cic_has_trees:
        P.todo("Add the random-forest and XGBoost results on CIC-IDS2017 (running on the second machine).")

    # FT-Transformer: 在每個測試集的名次，以及在 [@elsayed2024] 的六個深度模型之中是否最好
    pos = {c: ranking(r[k]).index("FTT") + 1 for k, c in RQ1_COLS if "FTT" in r[k]["accuracy"]}
    best_deep = [c for k, c in RQ1_COLS if "FTT" in r[k]["accuracy"]
                 and all(r[k]["accuracy"]["FTT"]["mean"] >= r[k]["accuracy"][m]["mean"] for m in DEEP)]
    rand_cols = [c for k, c in RQ1_COLS if k.endswith("/random")]
    ct = r["cic/temporal/binary/temporal"]["accuracy"]
    P.para("**A modern tabular architecture does not change the picture.** The FT-Transformer is the strongest "
           + (f"deep model on all {word(len(rand_cols))} random splits"
              if all(c in best_deep for c in rand_cols) else
              f"deep model on {word(sum(c in best_deep for c in rand_cols))} of the {word(len(rand_cols))} random splits")
           + (f", where it ranks third of {word(len(ct))} every time, directly behind the tree ensembles"
              if all(pos[c] == 3 for c in rand_cols) else
              f", where it ranks {', '.join(ORDINAL[pos[c]] for c in rand_cols)} of {word(len(ct))} (NSL-KDD, "
              "UNSW-NB15, CIC-IDS2017, CSE-CIC-IDS2018)")
           + ". Out of distribution it is not consistently "
           f"better: it ranks {ORDINAL[pos['NSL KDDTest+']]} on KDDTest+, {ORDINAL[pos['UNSW official']]} on the "
           f"UNSW-NB15 official test set, and last on the CIC-IDS2017 temporal split "
           f"({mean_sd(ct['FTT'])}%, against {mean_sd(ct['CNN'])}% for CNN). The conclusions of this section "
           "therefore do not depend on the age of the architectures in [@elsayed2024].")

    flips = {k: mean_flip(v) for k, v in r.items()}
    ood_flips = [flips[k] for k in flips if not k.endswith("/random")]
    id_flips = [flips[k] for k in flips if k.endswith("/random") and k.startswith(("nsl", "unsw"))]
    imp = r["cic/improved/binary/random"]
    imp_acc = [v["mean"] for v in imp["accuracy"].values()]
    imp18 = r["cic18/improved/binary/random"]
    t18 = r["cic18/temporal/binary/temporal"]
    t18_acc = [v["mean"] for v in t18["accuracy"].values()]
    unsw_off = r["unsw/default/binary/official"]
    P.para(f"**Single runs frequently reverse the order of two models.** On the NSL-KDD and UNSW-NB15 random "
           f"splits, where tree ensembles separate clearly from the deep models, the mean flip probability is "
           f"{min(id_flips):.1f}–{max(id_flips):.1f}%. On the out-of-distribution test sets it rises to "
           f"{min(ood_flips):.1f}–{max(ood_flips):.1f}%. On the corrected CIC-IDS2017 random split, all "
           f"{word(len(imp_acc))} models lie within {max(imp_acc) - min(imp_acc):.2f} points, none of the "
           f"{len(imp['pairwise'])} pairwise differences is significant, and a randomly chosen pair of single "
           f"runs contradicts the order of the means in {flips['cic/improved/binary/random']:.1f}% of cases "
           f"(up to {max_flip(imp):.1f}% for the closest pair); on the CSE-CIC-IDS2018 random split, "
           + (f"again none of the {len(imp18['pairwise'])} differences is significant" if imp18["n_significant"] == 0
              else f"{imp18['n_significant']} of {len(imp18['pairwise'])} differences are significant")
           + ". The opposite "
           f"failure occurs on the CSE-CIC-IDS2018 temporal split, where {word(t18['n_significant'])} pairwise "
           f"differences are significant although all models lie between {min(t18_acc):.2f}% and "
           f"{max(t18_acc):.2f}%, which is the share of benign flows in that test set (RQ3): with three seeds "
           "and almost no seed-to-seed variance, a statistically significant difference can be practically "
           "meaningless. Even the identity of the best model depends on the "
           f"seed: on the UNSW-NB15 official test set, the per-seed winner is "
           + ", ".join(f"{short(m)} ({n}×)" for m, n in sorted(unsw_off['wins'].items(), key=lambda x: -x[1]))
           + " across five seeds.")

    shift = S["rank_shift"]
    taus = {k: v for k, v in shift.items() if k.startswith(("nsl", "unsw"))}
    sig = [k for k, v in taus.items() if v["p"] < 0.05]
    nsl_s = shift["nsl/binary"]
    ood_nsl = [m for m in nsl_s["rank_ood"]]
    text = (f"**Rankings change with the test distribution.** Kendall's τ between the in-distribution and the "
            f"out-of-distribution ranking of the {word(len(nsl_s['models']))} models is "
            f"{min(v['tau'] for v in taus.values()):.2f}–{max(v['tau'] for v in taus.values()):.2f} on NSL-KDD and "
            f"UNSW-NB15 and is significantly different from zero only for "
            f"{' and '.join(k.replace('/', ' ').replace('nsl', 'NSL-KDD').replace('unsw', 'UNSW-NB15').replace('multi', 'five-class') for k in sig) or 'none of the settings'}. "
            f"On NSL-KDD binary, for example, GRU moves from position "
            f"{nsl_s['rank_id'].index('GRU') + 1} to position {ood_nsl.index('GRU') + 1} when the official KDDTest+ "
            "set replaces the random split.")
    if "cic/binary" in shift:
        c = shift["cic/binary"]
        w = r["cic/temporal/binary/temporal"]["wins"]
        n_seeds = sum(w.values())
        text += (f" On CIC-IDS2017, the ranking on the temporal split is unrelated to, and if anything the "
                 f"reverse of, the one on the corrected random split (τ = {c['tau']:.2f}, p = {c['p']:.2f}); CNN, "
                 f"position {c['rank_id'].index('CNN') + 1} in distribution, is the best model under temporal shift "
                 + ("in every seed" if w.get("CNN") == n_seeds else f"in {w.get('CNN', 0)} of {n_seeds} seeds")
                 + ", and the FT-Transformer moves from position "
                 f"{c['rank_id'].index('FTT') + 1} to position {c['rank_ood'].index('FTT') + 1}.")
    P.para(text + " A ranking established on a random split therefore says little about which model generalizes "
                  "best to new traffic.")


# ---------------------------------------------------------------------------- RQ2
def rq2(P, S):
    P.subsection("RQ2: Training Stability of Recurrent Models")
    hpc, stab = S["rq2"]["hp_compare"], S["rq2"]["stability"]

    def cell(d):
        if not d:
            return "—"
        n = f", n={d['n']}" if d["n"] not in (3, 5) else ""
        return f"{mean_sd(d)} ({d['collapses']}{n})"

    rows = []
    for ds, task in (("nsl", "binary"), ("nsl", "multi")):
        for m in ["DNN", "RNN"] + GATED:
            k = f"{ds}/{task}/{m}"
            if k not in hpc:
                continue
            h = hpc[k]
            rows.append([f"NSL-KDD {task}" if m == "DNN" else "", disp(m), cell(h.get("paper")),
                         cell(h.get("paper_clip")), cell(h.get("tuned"))])
    tot = {g: sum(hpc[f"nsl/{t}/{m}"].get(g, {}).get("collapses", 0) for t in ("binary", "multi") for m in GATED)
           for g in ("paper", "paper_clip", "tuned")}
    g = lambda m, h: stab.get(f"nsl/binary/{m}/{h}", {}).get("grad_max_median", float("nan"))
    P.para(f"Table {P.next_table()} compares the original hyperparameters, the original hyperparameters with "
           "gradient clipping, and the tuned configurations on NSL-KDD, where the reproduction had failed. "
           f"**With the original hyperparameters, only the gated models collapse**: across the binary and "
           f"five-class tasks, LSTM, CNN-LSTM, and GRU accumulate {tot['paper']} collapsed epochs, whereas DNN, "
           "CNN, and SimpleRNN have none or almost none. The median of the per-run maximum gradient norm in the binary "
           f"task is {g('DNN', 'paper'):.1f} for DNN and {g('RNN', 'paper'):.1f} for SimpleRNN, but "
           f"{g('GRU', 'paper'):.0f}, {g('LSTM', 'paper'):.0f}, and {g('CNN-LSTM', 'paper'):.0f} for GRU, LSTM, and "
           "CNN-LSTM (Fig. 2). Clipping the global gradient norm at 1, with nothing else changed, reduces the "
           f"collapses of the gated models from {tot['paper']} to {tot['paper_clip']} and raises the mean accuracy "
           "of GRU and CNN-LSTM while shrinking their seed-to-seed spread, confirming exploding gradients as the "
           "main cause of the failed reproduction.")
    P.columns(1)
    P.table(["Task", "Model", "Original", "Original + clipping", "Tuned"], rows, [2.6, 2.4, 4.0, 4.0, 4.0],
            "Accuracy (%) on the NSL-KDD random split under three configurations: mean ± standard deviation over "
            "five seeds, with the total number of collapsed epochs in parentheses (n: number of finished seeds "
            "when fewer than five). Clipping was applied to the gated models only.", left_cols=(0, 1))
    P.figure(FIG / "v2_gradnorm.png", 17.0,
             "Maximum raw (unclipped) gradient norm per epoch for the gated models on NSL-KDD (binary), one line "
             "per seed, under the original, clipped, and tuned configurations. Logarithmic scale.")
    P.columns(2)
    if any(h.get("paper_clip", {}).get("n", 5) < 5 for k, h in hpc.items() if k.startswith("nsl/")):
        P.todo("Clipping runs still in progress; numbers will be updated.")

    hp = S["tuned_hp"]
    deep = {k: v for k, v in hp.items() if k.split("/")[-1] in DEEP}
    gated = {k: v for k, v in deep.items() if k.split("/")[-1] in GATED}
    top_lr = sum(v["hp"].get("lr") == 3e-3 for v in gated.values())
    clip_n = sum(bool(v["hp"].get("clipnorm")) for v in gated.values())
    lstm = S["rq1"]["nsl/default/binary/random"]["accuracy"]["LSTM"]
    vals = lstm["values"]
    bad = sorted(v for v in vals if v < 95)
    lt = hp["nsl/binary/LSTM"]
    other = [k for k in hpc if k.split("/")[0] in ("unsw", "cic") and k.split("/")[-1] in GATED]
    # 每個種子的平均崩落次數 (原論文設定在 UNSW 只跑 3 個種子，調參後 5 個)
    rate = lambda grp: (sum(hpc[k][grp]["collapses"] for k in other if grp in hpc[k])
                        / max(1, sum(hpc[k][grp]["n"] for k in other if grp in hpc[k])))
    still = [k for k in other if hpc[k].get("tuned", {}).get("collapses", 0) > 0]
    nsl_still = {k: hpc[k]["tuned"]["collapses"] for k in hpc
                 if k.startswith("nsl/") and k.split("/")[-1] in ("GRU", "CNN-LSTM")
                 and hpc[k].get("tuned", {}).get("collapses", 0) >= 5}
    P.para(f"**Tuning with equal trial budgets removes the instability in most, but not all, cases.** Tuning "
           "selected the "
           f"largest learning rate in the grid (3×10⁻³) for {top_lr} of the {len(gated)} gated "
           f"configurations and gradient clipping for {clip_n}. On UNSW-NB15 and CIC-IDS2017 this reduced the "
           f"collapses of the gated models from {rate('paper'):.1f} to {rate('tuned'):.1f} per run"
           + (f" (remaining in {', '.join(k.split('/')[0].upper().replace('CIC', 'CIC-IDS2017').replace('UNSW', 'UNSW-NB15') + ' ' + k.split('/')[1] + ' ' + short(k.split('/')[2]) for k in still)})"
              if still else "")
           + ", and on NSL-KDD tuned GRU and CNN-LSTM are more accurate than with the original settings"
           + (", although " + " and ".join(f"{short(k.split('/')[2])} ({'five-class' if k.split('/')[1] == 'multi' else 'binary'}) "
                                           f"still collapses in {v} epochs over five seeds" for k, v in nsl_still.items())
              if nsl_still else "")
           + ". The clearest failure is the binary LSTM on NSL-KDD. Its selected "
           f"configuration (learning rate 3×10⁻³, clipping at 1) reached {lt['score'] * 100:.2f}% validation "
           "accuracy after the 25 epochs used during tuning, compared with "
           f"{lt['paper_score'] * 100:.2f}% for the original configuration, but over 100 epochs "
           f"{len(bad)} of {len(vals)} seeds collapsed and ended at "
           + " and ".join(f"{v:.1f}%" for v in bad)
           + f", giving {mean_sd(lstm)}% in Table {S['tab_rq1']}. The median maximum gradient "
           f"norm of these runs is {g('LSTM', 'tuned'):.0f}, an order of magnitude above the original "
           "configuration, even though clipping bounds the size of each update. We report this failure rather "
           "than re-tuning, because it illustrates a general risk: shortening training to make tuning affordable "
           "biases the search toward aggressive learning rates whose instability only appears later.")
    h18 = {k.split("/", 1)[1]: v for k, v in hpc.items() if k.startswith("cic18/")}
    if h18:
        l18 = S["rq1"]["cic18/improved/binary/random"]["accuracy"]["LSTM"]
        hp18 = hp["cic18/binary/LSTM"]["hp"]
        lr = {3e-3: "3×10⁻³", 1e-3: "10⁻³", 3e-4: "3×10⁻⁴"}.get(hp18["lr"], f"{hp18['lr']:g}")
        p_coll = sum(h18[f"{t}/{m}"].get("paper", {}).get("collapses", 0) for t in ("binary", "multi") for m in GATED
                     if f"{t}/{m}" in h18)
        t_coll = sum(h18[f"{t}/{m}"].get("tuned", {}).get("collapses", 0) for t in ("binary", "multi") for m in GATED
                     if f"{t}/{m}" in h18)
        low = min(l18["values"])
        n_low = sum(v < 99 for v in l18["values"])
        P.para("**The same pattern recurs on CSE-CIC-IDS2018.** With the original hyperparameters the gated models "
               f"accumulate {p_coll} collapsed epochs on the random split and the other models none; after tuning, "
               f"{word(t_coll)} remain, all in the binary LSTM, whose selected configuration again uses the largest "
               f"learning rate ({lr}, {'clipping at 1' if hp18.get('clipnorm') else 'no clipping'}). "
               + (f"One of its {word(len(l18['values']))} seeds ends" if n_low == 1 else
                  f"{word(n_low).capitalize()} of its {word(len(l18['values']))} seeds end")
               + f" at {low:.1f}%, giving {mean_sd(l18)}%, while every "
               "other tuned binary model on this dataset reaches at least "
               f"{min(v['mean'] for m, v in S['rq1']['cic18/improved/binary/random']['accuracy'].items() if m != 'LSTM'):.2f}%. "
               "A failure that we observed on a benchmark derived from 1998 traffic therefore reappears, with the "
               "same cause, on the most recent benchmark in this study.")
        nf = [k for k in hpc if k.endswith("/FTT") and not k.startswith("cic18b")]
        f_coll = sum(hpc[k]["tuned"]["collapses"] for k in nf)
        P.para(f"The FT-Transformer, trained with AdamW and a tuned learning rate, collapses in "
               f"{'only one epoch' if f_coll == 1 else f'{f_coll} epochs'} over all "
               f"{sum(hpc[k]['tuned']['n'] for k in nf)} runs on the "
               "random splits of the four datasets. Its stability is consistent with the view that the collapses "
               "of the gated models stem from exploding gradients in the recurrence, not from training on "
               "tabular flow data as such.")
    sensitivity(P, S, vals)

    perm = S.get("rq2_perm", {})
    if perm:
        models = sorted({k.split("/")[1] for k in perm}, key=DEEP.index)
        diffs = [abs(v["diff"]) for v in perm.values()]
        ps = sorted(v["p_mean"] for v in perm.values())
        pv = [v["p_var"] for v in perm.values()]
        n = len(ps)
        holm = max(min(1.0, (n - i) * p) for i, p in enumerate(ps[:1]))      # Holm: 最小的 p 乘以檢定數
        n05 = sum(p < 0.05 for p in ps)
        big = max(perm, key=lambda k: abs(perm[k]["diff"]))
        bd, bm = big.split("/")[0], big.split("/")[1]
        P.para("**No significant effect of feature order was detected.** The convolutional and recurrent models of [@elsayed2024] read "
               "the 122 (NSL-KDD) or 67 (UNSW-NB15) features as a sequence, although their order is arbitrary. "
               "We trained the tuned binary models on five random permutations of the feature order "
               f"({', '.join(short(m) for m in models)}; {n} model–dataset–test combinations) and compared them "
               f"with the five seeds of the original order. The mean accuracy differs by less than one point in "
               f"{sum(d < 1 for d in diffs)} of the {n} combinations; the largest difference, "
               f"{max(diffs):.2f} points for {short(bm)} on the {'UNSW-NB15' if bd == 'unsw' else 'NSL-KDD'} "
               f"{'official test set' if big.endswith('official') else 'random split'}, favors the permuted order. "
               f"Of the {n} uncorrected t-tests, {n05} fall below 0.05, but none remains significant after Holm "
               f"correction (smallest adjusted p = {holm:.2f}), and the variances do not differ (p ≥ {min(pv):.2f}). "
               "We thus find no evidence that the sequence models exploit the original ordering, and the variability caused by an "
               "arbitrary permutation is comparable to that caused by the random seed.")
        missing = [m for m in DEEP if m not in models]
        if missing:
            P.todo(f"Permutation runs for {', '.join(short(m) for m in missing)} still in progress.")


def sensitivity(P, S, rank1_vals):
    """調參敏感度: 排名第 2、3 的設定以 5 個種子、完整 epoch 重跑 (results/v2/sensitivity)。"""
    runs = {}
    for f in sorted((V2 / "sensitivity").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        runs.setdefault((d["task"], d["model"], d["rank"]), []).append(d)
    if not runs:
        P.todo("Tuning-sensitivity runs (ranks 2–3) still in progress.")
        return
    acc = lambda rs, t: [r["tests_last"][t]["accuracy"] * 100 for r in rs]
    lr = lambda v: {3e-3: "3×10⁻³", 1e-3: "10⁻³", 3e-4: "3×10⁻⁴"}.get(v, f"{v:g}")
    desc = lambda h: f"learning rate {lr(h['lr'])}, {'clipping at 1' if h.get('clipnorm') else 'no clipping'}"
    parts = []
    for rank in (2, 3):
        rs = runs.get(("binary", "LSTM", rank))
        if not rs:
            continue
        a = acc(rs, "random")
        bad = sorted(v for v in a if v < 95)
        ok = [v for v in a if v >= 95]
        o = acc([r for r, v in zip(rs, a) if v >= 95], "official")
        parts.append(f"the rank-{rank} configuration ({desc(rs[0]['hp'])}) "
                     + (f"collapses in {len(bad)} of {len(rs)} seeds ("
                        + " and ".join(f"{v:.1f}%" for v in bad) + ")" if bad else
                        f"collapses in none of {len(rs)} seeds ({min(ok):.2f}–{max(ok):.2f}% on the random split, "
                        f"{min(o):.1f}–{max(o):.1f}% on KDDTest+)"))
    g = [runs.get(("multi", "GRU", r), []) for r in (2, 3)]
    ga = [v for rs in g for v in acc(rs, "random")]
    n_bad1 = sum(v < 95 for v in rank1_vals)
    P.para("**The failure depends on the selected configuration, not on LSTM as such.** To test how sensitive "
           "this conclusion is to the tuning outcome, we retrained the second- and third-ranked configurations of "
           f"the binary LSTM search with five seeds and the full 100 epochs. The selected configuration "
           f"collapses in {n_bad1} of {len(rank1_vals)} seeds; " + "; ".join(parts) + ". "
           "The configurations that keep the largest learning rate inherit its instability regardless of "
           "clipping, while the smaller learning rate is stable and only marginally below the selected one in "
           "validation accuracy during tuning. "
           + (f"As a control, the second- and third-ranked configurations of the five-class GRU, which did not "
              f"fail, are stable as well ({min(ga):.2f}–{max(ga):.2f}% over {len(ga)} runs). " if ga else "")
           + "A tuning protocol that ranks configurations by a single shortened run therefore cannot distinguish a "
           "fast configuration from a fragile one; checking the top few configurations over the full schedule, "
           "or ranking by the worst of several seeds, would have avoided the failure.")


# ---------------------------------------------------------------------------- RQ3
def rq3(P, S):
    P.subsection("RQ3: Evaluation Pitfalls")
    r3 = S["rq3"]
    nb, ub = r3["nsl/binary"], r3["unsw/binary"]
    rng = lambda d, k: (min(v[k] for v in d.values()), max(v[k] for v in d.values()))
    ng, ug = rng(nb, "gap"), rng(ub, "gap")
    nm, um = rng(r3["nsl/multi"], "gap"), rng(r3["unsw/multi"], "gap")
    nu = rng({m: v for m, v in nb.items() if v.get("unseen") is not None}, "unseen")
    nsl_id = [v["id_acc"] for m, v in nb.items() if m != "LSTM"]
    P.para("**Random splits overstate performance far more than the models differ.** Moving from the random "
           f"split to the official test set lowers binary accuracy by {ng[0]:.1f}–{ng[1]:.1f} points on NSL-KDD "
           f"and by {ug[0]:.1f}–{ug[1]:.1f} points on UNSW-NB15 (five-class: {nm[0]:.1f}–{nm[1]:.1f} and "
           f"{um[0]:.1f}–{um[1]:.1f} points; Fig. 3a). On NSL-KDD this drop is about "
           f"{ng[0] / (max(nsl_id) - min(nsl_id)):.0f} times larger than the spread of the in-distribution accuracies "
           f"of the models other than LSTM ({max(nsl_id) - min(nsl_id):.2f} points). Attack types that are absent "
           f"from KDDTrain+ are detected at only {nu[0]:.0f}–{nu[1]:.0f}%, which explains most of the drop: the "
           "random split measures recognition of known attacks, the official test set measures generalization.")

    cic = r3["cic/binary"]
    models = [m for m in ALL if m in cic]
    conds = ["orig", "dedup", "improved", "temporal"]
    rows = []
    for m in models:
        c = cic[m]
        f = lambda k: f"{c[k]['f1']:.2f}" if k in c else "—"
        t = c.get("temporal", {})
        rows.append([disp(m)] + [f(k) for k in conds]
                    + ([f"{t['detection']:.1f}", f"{t['false_alarm']:.3f}"] if t else ["—", "—"]))
    P.columns(1)
    P.table(["Model", "Original", "Deduplicated", "Corrected", "Temporal", "Temporal DR", "Temporal FAR"], rows,
            [2.7, 2.2, 2.4, 2.2, 2.2, 2.4, 2.4],
            "CIC-IDS2017, binary task: macro-F1 (%) under progressively stricter evaluation conditions (mean over "
            "three seeds), and the "
            "attack detection rate (DR, %) and false-alarm rate (FAR, %) in the temporal condition.",
            left_cols=(0,))
    P.figure(FIG / "v2_conditions.png", 17.0,
             "Evaluation conditions, binary task: (a) random split vs. official test set on NSL-KDD and "
             "UNSW-NB15 (accuracy); (b) CIC-IDS2017 from the original release to the temporal split (macro-F1).")
    tab_cic = P.tab_no
    S["tab_cic"] = ROMAN[tab_cic - 1]
    ta = S["temporal_attacks"]
    attacks = sorted(ta["n"], key=lambda a: -ta["n"][a])
    main_att = [a for a in attacks if ta["n"][a] >= 500]
    rest = [a for a in attacks if a not in main_att]
    n_rest = sum(ta["n"][a] for a in rest)
    rows = []
    for m in [x for x in ALL if x in ta["rates"]]:
        rt = ta["rates"][m]
        other = sum(rt[a] * ta["n"][a] for a in rest) / n_rest if n_rest else float("nan")
        rows.append([disp(m)] + [f"{rt[a]:.1f}" for a in main_att] + [f"{other:.1f}"])
    P.table(["Model"] + [f"{a} (n={ta['n'][a]:,})" for a in main_att] + [f"Others (n={n_rest:,})"], rows,
            [2.7] + [2.9] * len(main_att) + [2.9],
            "CIC-IDS2017 temporal condition: detection rate (%) per attack type, mean over three seeds. None of "
            "these types occurs in the training days. Others: "
            + ", ".join(rest) + ".", left_cols=(0,))
    tab_att = P.tab_no
    P.columns(2)

    dd = [cic[m]["dedup"]["f1"] - cic[m]["orig"]["f1"] for m in models if "dedup" in cic[m]]
    imp = [cic[m]["improved"]["f1"] for m in models if "improved" in cic[m]]
    tf = [cic[m]["temporal"]["f1"] for m in models if "temporal" in cic[m]]
    tdr = [cic[m]["temporal"]["detection"] for m in models if "temporal" in cic[m]]
    tfa = [cic[m]["temporal"]["false_alarm"] for m in models if "temporal" in cic[m]]
    cm = r3.get("cic/multi", {})
    mo = [cm[m]["orig"]["f1"] for m in cm if "orig" in cm[m]]
    mi = [cm[m]["improved"]["f1"] for m in cm if "improved" in cm[m]]
    best_t = max(models, key=lambda m: cic[m].get("temporal", {}).get("f1", -1))
    ddos = [ta["rates"][m]["DDoS"] for m in ta["rates"]]
    ps = {m: ta["rates"][m]["PortScan"] for m in ta["rates"]}
    P.para(f"Table {ROMAN[tab_cic - 1]} follows "
           "CIC-IDS2017 as the evaluation protocol becomes progressively stricter. **Removing duplicates changes "
           f"little** here: macro-F1 changes by {min(dd):+.2f} to {max(dd):+.2f} points. **Scores increase on the "
           f"corrected release**: every model reaches {min(imp):.2f}–{max(imp):.2f}% macro-F1, so the "
           "benchmark is saturated, and with three seeds our tests cannot separate the models; in the five-class "
           f"task macro-F1 rises from {min(mo):.0f}–{max(mo):.0f}% to {min(mi):.0f}–{max(mi):.0f}%. Since the "
           "corrected release also re-extracts the flows, this increase cannot be attributed to the relabeling "
           "alone, but it is consistent with label errors concentrated in the small attack classes [@liu2022]. "
           "**The temporal split reveals what the random split "
           f"hides**: macro-F1 falls to {min(tf):.1f}–{max(tf):.1f}% and the attack detection rate to "
           f"{min(tdr):.0f}–{max(tdr):.0f}%, while the false-alarm rate stays at "
           f"{min(tfa):.3f}–{max(tfa):.3f}%. Table "
           f"{ROMAN[tab_att - 1]} shows why. "
           f"Every model detects DDoS ({min(ddos):.0f}–{max(ddos):.0f}%), which resembles the DoS attacks of the "
           f"training days, but the port scans are almost entirely missed ({min(ps.values()):.1f}–"
           f"{max(v for m, v in ps.items() if m != best_t):.1f}%), with the exception of {short(best_t)} "
           f"({ps[best_t]:.1f}%), which explains its first place under temporal shift. Botnet, web attacks, and "
           "infiltration are missed by every model. The models are thus conservative detectors of attacks similar "
           "to the training attacks, a property that a random split cannot reveal.")

    c18 = r3.get("cic18/binary", {})
    if c18:
        ms18 = [m for m in ALL if m in c18 and "temporal" in c18[m]]
        rf = [c18[m]["improved"]["f1"] for m in ms18]
        tf = [c18[m]["temporal"]["f1"] for m in ms18]
        td = [c18[m]["temporal"]["detection"] for m in ms18]
        ta = [c18[m]["temporal"]["acc"] for m in ms18]
        t18 = S["temporal_attacks18"]
        n_att = sum(t18["n"].values())
        best = max(((v, m, a) for m, d in t18["rates"].items() for a, v in d.items()), key=lambda x: x[0])
        benign = 100 * (1 - n_att / 1_108_978)
        P.para("**On CSE-CIC-IDS2018 the temporal split removes detection altogether.** On the corrected random "
               f"split every model reaches {min(rf):.1f}–{max(rf):.1f}% macro-F1. Trained on the first five days "
               "(brute force, DoS, and DDoS) and tested on the last five (web attacks, infiltration, and botnet "
               f"traffic, {n_att:,} attack flows), the same models detect {min(td):.3f}–{max(td):.2f}% of the "
               f"attacks; their accuracy of {min(ta):.2f}–{max(ta):.2f}% equals the share of benign flows in the "
               f"test set ({benign:.2f}%), and macro-F1 drops to {min(tf):.1f}–{max(tf):.1f}%. "
               + ("No model detects a single botnet flow"
                  if max(d.get("Botnet Ares", 0) for d in t18["rates"].values()) == 0 else
                  f"No model detects more than {max(d.get('Botnet Ares', 0) for d in t18['rates'].values()):.2f}% "
                  "of the botnet flows")
               + ", and the highest per-type detection rate of any model is "
               f"{best[0]:.1f}% ({short(best[1])}, {best[2]}, n = {t18['n'][best[2]]}). Unlike on CIC-IDS2017, "
               "where the unseen DDoS attacks resemble the DoS attacks of the training days, none of the test "
               "attacks resembles a training attack, and all nine models, trees and FT-Transformer included, "
               "behave as a constant benign classifier. An accuracy above 82%, reported without the class "
               "distribution, would hide this completely.")


# ---------------------------------------------------------------------------- RQ4
FEAT = {"min_seg_size_forward": "min. fwd. segment size", "Packet Length Variance": "pkt. length variance",
        "Packet Length Std": "pkt. length std.", "RST Flag Count": "RST flag count",
        "SYN Flag Count": "SYN flag count", "Destination Port": "destination port",
        "Bwd Packet Length Std": "bwd. pkt. length std.", "Init_Win_bytes_forward": "initial fwd. window",
        "CWE Flag Count": "CWE flag count", "Bwd Packet Length Max": "bwd. pkt. length max."}


def rq4(P, S):
    P.subsection("RQ4: Shortcut Features")
    r4 = S["rq4"]
    targets = [("nsl/default", "NSL-KDD", "official", "KDDTest+"), ("unsw/default", "UNSW-NB15", "official", "official"),
               ("cic/improved", "CIC17 corrected", "random", "random"), ("cic/temporal", "CIC17 temporal", "temporal", "temporal"),
               ("cic18/improved", "CIC18", "random", "random"), ("cic18/temporal", "CIC18 temporal", "temporal", "temporal")]
    rows = []
    for key, name, test, tname in targets:
        if key not in r4:
            continue
        d = r4[key]
        ab = {a["setting"]: a for a in d["ablation"]}
        get = lambda s: f"{ab[s][test]:.2f}" if s in ab and test in ab[s] else "—"
        rows.append([f"{name} ({tname})", ", ".join(FEAT.get(f, f) for f in d["top5"]), f"{d['top5_share'] * 100:.0f}",
                     get("all"), get("top1"), get("top5"), get("drop3")])
    P.para(f"Table {P.next_table()} lists the five features with the largest mean absolute SHAP value for the "
           "tuned binary DNN and the accuracy obtained after retraining on subsets of the features (three seeds "
           "each).")
    P.columns(1)
    P.table(["Dataset (test set)", "Top-5 features by SHAP", "Share (%)", "All", "Top-1 only", "Top-5 only",
             "Without top-3"], rows, [3.0, 6.6, 1.5, 1.5, 1.6, 1.6, 1.9],
            "Shortcut analysis of the tuned binary DNN: five most important features, their share of the total "
            "importance, and accuracy (%) on the indicated test set after retraining on subsets of the features.",
            left_cols=(0, 1))
    S["tab_rq4"] = ROMAN[P.tab_no - 1]
    P.columns(2)
    nsl = {a["setting"]: a for a in r4["nsl/default"]["ablation"]}
    unsw = {a["setting"]: a for a in r4["unsw/default"]["ablation"]}
    P.para("**A single feature carries most of the out-of-distribution score on NSL-KDD and UNSW-NB15.** On "
           f"NSL-KDD, the model trained on *service* alone reaches {nsl['top1']['random']:.1f}% on the random split "
           f"and {nsl['top1']['official']:.1f}% on KDDTest+, compared with {nsl['all']['official']:.1f}% with all "
           f"41 features. On UNSW-NB15, three of the top five features are derived from the IP time-to-live "
           f"(*dttl*, *sttl*, *ct_state_ttl*); *dttl* alone gives {unsw['top1']['random']:.1f}% on the random "
           f"split, and removing the top three features does not hurt but slightly improves accuracy on the "
           f"official test set ({unsw['all']['official']:.2f}% → {unsw['drop3']['official']:.2f}%). Time-to-live "
           "values depend on the operating systems and the network path in the testbed rather than on the attack "
           "itself, so a detector that relies on them is unlikely to transfer to another network.")
    if "cic18/improved" in r4:
        a18 = {a["setting"]: a for a in r4["cic18/improved"]["ablation"]}
        t18 = {a["setting"]: a for a in r4["cic18/temporal"]["ablation"]} if "cic18/temporal" in r4 else {}
        top = r4["cic18/improved"]["top5"]
        P.para(f"On the CSE-CIC-IDS2018 random split, {FEAT.get(top[0], top[0])} alone gives "
               f"{a18['top1']['random']:.1f}% and the top three features, which include the "
               f"{FEAT.get(top[1], top[1])} and the {FEAT.get(top[2], top[2])}, give {a18['top3']['random']:.1f}%, "
               f"against {a18['all']['random']:.2f}% with all 77; both are properties of the hosts' TCP stacks "
               "rather than of the attacks. "
               + (f"On the temporal split, every subset gives "
                  f"{min(v['temporal'] for v in t18.values()):.1f}–{max(v['temporal'] for v in t18.values()):.1f}%, "
                  f"never meaningfully above the constant benign prediction ({t18['all']['temporal']:.2f}% with "
                  "all features): with no transferable signal, there is no shortcut to remove either."
                  if t18 else ""))

    chk = S["check"]
    if not chk:
        P.todo("10-seed ablation on the CIC-IDS2017 temporal split.")
        return
    order = ["all", "drop_plv", "drop_port", "drop_rst", "drop_plv_port", "drop_plv_rst", "drop_port_rst", "drop_top3"]
    label = lambda k: "none (all 77)" if k == "all" else ", ".join(FEAT.get(f, f) for f in chk[k]["dropped"])
    rows = []
    for k in [k for k in order if k in chk]:
        v = chk[k]
        rows.append([label(k), mean_sd(v["accuracy"]), f"{v['detection']['mean']:.1f}",
                     f"{v['by_attack'].get('PortScan', float('nan')):.1f}",
                     f"{v['by_attack'].get('Infiltration Portscan', float('nan')):.1f}",
                     f"{v['false_alarm']:.3f}", f"{v['n_above_95']}/{v['n']}"])
    n = chk["all"]["n"]
    a, b = chk["all"], chk["drop_port_rst"]
    P.para(f"**On the CIC-IDS2017 temporal split, two features suppress the detection of new attacks.** Removing "
           f"the top three features raised accuracy on this split, so we repeated the ablation with {n} seeds for "
           f"every subset of these features (Table {P.next_table()}). Removing the destination port and the RST "
           f"flag count together raises accuracy from {mean_sd(a['accuracy'])}% to {mean_sd(b['accuracy'])}% and "
           f"the attack detection rate from {a['detection']['mean']:.1f}% to {b['detection']['mean']:.1f}%, almost "
           f"entirely through the port scans ({a['by_attack']['PortScan']:.1f}% → "
           f"{b['by_attack']['PortScan']:.1f}%), at the cost of a false-alarm rate of {b['false_alarm']:.3f}% "
           f"instead of {a['false_alarm']:.3f}%. Removing either feature alone has a much smaller effect, and "
           "packet length variance plays no role. A plausible explanation, which we have not verified at the packet "
           "level, is that the training attacks (FTP and SSH brute force on Tuesday, denial of service against a "
           "web server on Wednesday) target a few service ports and terminate their connections in a "
           "characteristic way, so that the model learns to treat traffic to other ports, or without this "
           "pattern, as benign; port scans contradict both rules.")
    P.columns(1)
    P.table(["Features removed", "Accuracy", "DR", "PortScan", "Infiltr. PortScan", "FAR", "Seeds > 95%"], rows,
            [5.0, 2.6, 1.5, 1.8, 2.3, 1.6, 2.0],
            f"CIC-IDS2017 temporal condition: ablation of the three most important features of the tuned binary "
            f"DNN over {n} seeds. Accuracy as mean ± standard deviation; detection rates (DR, PortScan, "
            "Infiltration–PortScan) and false-alarm rate (FAR) in %.", left_cols=(0,))
    P.columns(2)
    P.para(f"This result is diagnostic, not a better detector: the three features were selected with SHAP values "
           "computed on the temporal test set, so the improved accuracy is itself obtained by looking at the test "
           f"data and must not be compared with Table {S['tab_cic']}. What it shows is that the poor generalization in "
           "RQ3 is not an inevitable consequence of unseen attack types; it is at least partly caused by a "
           "shortcut that the training days make available and that an in-distribution evaluation rewards.")


# ---------------------------------------------------------------------------- RQ5
ADV_DS = {"nsl": ("nsl_default", "official", "KDDTest+"), "cic": ("cic_improved", "random", "random split"),
          "cic18": ("cic18_improved", "random", "random split")}
ADV_TAG = {"nsl": "NSL", "cic": "CIC17", "cic18": "CIC18"}


def rq5_stats(S):
    """RQ5 的所有數字: 攻擊 (PGD 與隨機搜尋中較強者)、轉移、對抗訓練與乾淨準確率代價。"""
    r5 = S["rq5"]
    idx = lambda rows, key: {(r["dataset"], r["threat"], r["model"], r["eps"]): r[key] for r in rows}
    out = dict(best=idx(r5["attack"], "best"), pgd=idx(r5["attack"], "pgd"), rs=idx(r5["attack"], "rs"),
               at_best=idx(r5["advtrain"], "best"), at_pgd=idx(r5["advtrain"], "pgd"),
               at_rs=idx(r5["advtrain"], "rs"), transfer=idx(r5["transfer"], "rate"))
    clean = {}
    for ds, (tag, test, _) in ADV_DS.items():
        for m in DEEP:
            std, adv = [], []
            for s in (1, 2, 3):
                a = V2 / "adversarial" / "advtrain" / f"{ds}_{m}_s{s}.json"
                p = V2 / "runs" / f"{tag}_tuned_binary_{m}_s{s}.json"
                if a.exists() and p.exists():
                    adv.append(json.loads(a.read_text(encoding="utf-8"))["tests"][test]["accuracy"] * 100)
                    std.append(json.loads(p.read_text(encoding="utf-8"))["tests_last"][test]["accuracy"] * 100)
            if adv:
                clean[(ds, m)] = (float(np.mean(std)), float(np.mean(adv)), len(adv))
    out["clean"] = clean
    return out


def rq5(P, S):
    P.subsection("RQ5: Adversarial Robustness")
    R = rq5_stats(S)
    b, pgd, rs = R["best"], R["pgd"], R["rs"]
    rngm = lambda d, ds, th, e, ms=DEEP: (min(d[(ds, th, m, e)] for m in ms), max(d[(ds, th, m, e)] for m in ms))
    eps_cols = [("unconstrained", 1.0), ("realistic", 0.5), ("realistic", 1.0), ("realistic", 2.0)]
    rows = []
    fmt = lambda d, k: f"{d[k]:.1f}" if k in d else "—"
    for m in DEEP:
        rows.append([short(m)] + [fmt(b, (ds, th, m, e)) for ds in ADV_DS for th, e in eps_cols])
    head = ["Model"] + [f"{ADV_TAG[ds]}: {'unc.' if th == 'unconstrained' else 'real.'} ε={e:g}"
                        for ds in ADV_DS for th, e in eps_cols]
    nu, cu = rngm(b, "nsl", "unconstrained", 1.0), rngm(b, "cic", "unconstrained", 1.0)
    nr, cr = rngm(b, "nsl", "realistic", 1.0), rngm(b, "cic", "realistic", 1.0)
    nr_wo = rngm(b, "nsl", "realistic", 1.0, [m for m in DEEP if m != "GRU"])
    cr_wo = rngm(b, "cic", "realistic", 1.0, [m for m in DEEP if m != "CNN"])
    tab_att = P.next_table()
    P.para(f"Table {tab_att} reports the evasion rate, the fraction of correctly detected attack records that an "
           "attack turns into a *normal* prediction, for the tuned binary models. Each entry is the stronger of PGD "
           "and random search, averaged over three seeds. **Unconstrained attacks overstate the threat.** At "
           f"ε = 1 they evade {nu[0]:.0f}–{nu[1]:.0f}% of the detections on NSL-KDD and at least {cu[0]:.1f}% on "
           f"CIC-IDS2017. Restricting the attacker to controllable features that can only grow lowers the rates on "
           f"NSL-KDD to {nr_wo[0]:.1f}–{nr_wo[1]:.1f}% for five of the six models; GRU is the exception "
           f"({b[('nsl', 'realistic', 'GRU', 1.0)]:.1f}%). **Under realistic constraints, the two datasets behave "
           "very differently.** On "
           f"CIC-IDS2017 the same budget evades {cr_wo[0]:.0f}–{cr_wo[1]:.0f}% of the detections of every model "
           f"except CNN ({b[('cic', 'realistic', 'CNN', 1.0)]:.0f}%), and ε = 2 evades "
           f"{rngm(b, 'cic', 'realistic', 2.0)[0]:.0f}–{rngm(b, 'cic', 'realistic', 2.0)[1]:.0f}%. On CIC-IDS2017 "
           "the controllable forward-direction features, such as packet counts, byte counts, and inter-arrival "
           "times, are informative enough that inflating them moves attacks across the decision boundary; on "
           f"NSL-KDD the most important features (Table {S['tab_rq4']}), such as *service* and the error rates, are outside "
           "the attacker's control.")
    if all(("cic18", "realistic", m, 1.0) in b for m in DEEP):
        c18 = {m: b[("cic18", "realistic", m, 1.0)] for m in DEEP}
        lo = min(c18, key=c18.get)
        rest18 = [v for m, v in c18.items() if m != lo]
        P.para(f"CSE-CIC-IDS2018, which uses the same flow features, confirms this explanation: at ε = 1 the "
               f"realistic attack evades {min(rest18):.0f}–{max(rest18):.0f}% of the detections of five of the six "
               f"models and {c18[lo]:.0f}% for {short(lo)}, and ε = 2 evades "
               f"{rngm(b, 'cic18', 'realistic', 2.0)[0]:.0f}–{rngm(b, 'cic18', 'realistic', 2.0)[1]:.0f}%. The "
               "model that resists best is not the same on the two CIC datasets "
               f"({short(min(DEEP, key=lambda m: b[('cic', 'realistic', m, 1.0)]))} on CIC-IDS2017, {short(lo)} on "
               "CSE-CIC-IDS2018), so robustness rankings are as dataset-specific as accuracy rankings.")
    P.columns(1)
    P.table(head, rows, [1.7] + [1.275] * 12,
            "Evasion rate (%) of the tuned binary models under unconstrained (unc.) and realistic (real.) attacks, "
            "the stronger of PGD and random search, mean over three seeds and 2,000 detected attack records per "
            "seed. ε is the L∞ budget in standard deviations.", left_cols=(0,))
    P.columns(2)

    gaps = {(ds, m): rs[(ds, "realistic", m, 1.0)] - pgd[(ds, "realistic", m, 1.0)] for ds in ADV_DS for m in DEEP
            if (ds, "realistic", m, 1.0) in rs}
    big = max(gaps, key=gaps.get)
    n_rs = sum(rs[k] > pgd[k] + 1 for k in rs if k[1] == "realistic")
    n_all = sum(1 for k in rs if k[1] == "realistic")
    unc_small = [(pgd[k], rs[k]) for k in rs if k[1] == "unconstrained" and k[3] == 0.1]
    P.para("**Gradient-based attacks alone would underestimate the realistic threat.** Under the realistic "
           f"threat model, random search is more than one point stronger than PGD in {n_rs} of {n_all} "
           "model–dataset–budget combinations, and the largest gap occurs for "
           f"{short(big[1])} on {DSN[big[0]]} at ε = 1 "
           f"({pgd[('%s' % big[0], 'realistic', big[1], 1.0)]:.1f}% with PGD, "
           f"{rs[(big[0], 'realistic', big[1], 1.0)]:.1f}% with random search). Without constraints PGD is the "
           f"stronger attack at small budgets (ε = 0.1: PGD better in {sum(p > q for p, q in unc_small)} of "
           f"{len(unc_small)} cases). The rounding of count features and the restriction to increases make the "
           "projected gradient a poor search direction, so a robustness evaluation that relies on PGD under "
           "domain constraints should always be complemented by a gradient-free attack [@athalye2018].")

    tr = R["transfer"]
    tr18 = {m: tr[("cic18", "realistic", m, 1.0)] for m in DEEP if m != "DNN" and ("cic18", "realistic", m, 1.0) in tr}
    if not tr18:
        P.todo("CSE-CIC-IDS2018 transfer stage not run yet (laptop: adversarial_v2.py --stage transfer "
               "--datasets cic18); describe it here once available.")
    nt = {m: tr[("nsl", "realistic", m, 1.0)] for m in DEEP}
    ct = {m: tr[("cic", "realistic", m, 1.0)] for m in DEEP}
    others = [m for m in DEEP if m != "DNN"]
    at, atp, atr, clean = R["at_best"], R["at_pgd"], R["at_rs"], R["clean"]
    rows = []
    for m in DEEP:
        row = [short(m)]
        for ds in ADV_DS:
            s, a, _ = clean.get((ds, m), (np.nan, np.nan, 0))
            row += ["—" if m == "DNN" else fmt(tr, (ds, "realistic", m, 1.0)),
                    fmt(at, (ds, "realistic", m, 1.0)), fmt(at, (ds, "realistic", m, 2.0)),
                    "—" if np.isnan(a - s) else f"{a - s:+.2f}"]
        rows.append(row)
    tab_def = ROMAN[P.tab_no]                                      # 這一節的第二個表
    P.para(f"**Black-box transfer is effective on the CIC datasets.** The first column of each dataset in Table "
           f"{tab_def} applies realistic examples crafted on the DNN (ε = 1) to the other models without "
           f"querying them. On CIC-IDS2017 they evade {min(ct[m] for m in others):.1f}–"
           f"{max(ct[m] for m in others):.1f}% of the detections, "
           f"{ct['LSTM']:.0f}% for LSTM against {b[('cic', 'realistic', 'LSTM', 1.0)]:.0f}% in the white-box "
           "setting; on NSL-KDD transfer stays at or below "
           f"{max(nt[m] for m in others):.1f}%. On both of these datasets CNN is the least affected "
           f"({nt['CNN']:.1f}% and {ct['CNN']:.1f}%)"
           + ("" if not tr18 else
              f". On CSE-CIC-IDS2018 transfer is similarly effective, {min(tr18.values()):.1f}–"
              f"{max(tr18.values()):.1f}% ({', '.join(f'{short(m)} {v:.0f}%' for m, v in sorted(tr18.items(), key=lambda x: -x[1]))}), "
              f"so that the least affected models differ between the datasets as well")
           + ". An attacker therefore does not need access to the "
           "deployed model to evade most detectors on the CIC datasets; a surrogate trained on the same data "
           "suffices.")
    P.columns(1)
    P.table(["Model"] + [f"{ADV_TAG[ds]}: {c}" for ds in ADV_DS
                         for c in ("transfer", "AT ε=1", "AT ε=2",
                                   "Δ acc. (KDDTest+)" if ds == "nsl" else "Δ acc. (random)")], rows,
            [1.7] + [1.275] * 12,
            "Realistic threat model. Transfer: evasion rate (%) of examples crafted on the DNN at ε = 1. AT: "
            "evasion rate (%) after adversarial training at ε = 1, evaluated at ε = 1 and ε = 2 (stronger of PGD "
            "and random search). Δ acc.: change of clean accuracy (percentage points) caused by adversarial "
            "training, seeds 1–3.", left_cols=(0,))
    P.columns(2)

    at1 = {ds: max(at[(ds, "realistic", m, 1.0)] for m in DEEP) for ds in ADV_DS}
    at2 = sorted(((at[("cic", "realistic", m, 2.0)], m) for m in DEEP), reverse=True)
    at2_18 = sorted(((at[("cic18", "realistic", m, 2.0)], m) for m in DEEP), reverse=True)
    dn = {m: clean[("nsl", m)][1] - clean[("nsl", m)][0] for m in DEEP}
    dc = [clean[("cic", m)][1] - clean[("cic", m)][0] for m in DEEP]
    dc18 = {m: clean[("cic18", m)][1] - clean[("cic18", m)][0] for m in DEEP}
    w18 = min(dc18, key=dc18.get)
    worst = min(dn, key=dn.get)
    rest = [v for m, v in dn.items() if m != worst]
    mask = max(((atr[k] - atp[k], k) for k in atr if k[1] == "realistic"))
    P.para("**Adversarial training removes most of the realistic threat at a small cost.** After three-step PGD "
           "training at ε = 1, the evasion rate at the training budget drops to at most "
           f"{at1['nsl']:.1f}% on NSL-KDD, {at1['cic']:.1f}% on CIC-IDS2017, and {at1['cic18']:.1f}% on "
           "CSE-CIC-IDS2018. Robustness degrades beyond the "
           f"training budget: at ε = 2 the strongest attack still evades {at2[0][0]:.1f}% of the detections of "
           f"{short(at2[0][1])}, {at2[1][0]:.1f}% for {short(at2[1][1])}, and {at2[2][0]:.1f}% for "
           f"{short(at2[2][1])} on CIC-IDS2017, and {at2_18[0][0]:.1f}% for {short(at2_18[0][1])} on "
           "CSE-CIC-IDS2018. The clean accuracy on the CIC-IDS2017 random split changes by "
           f"{min(dc):+.2f} to {max(dc):+.2f} points and on the CSE-CIC-IDS2018 random split by "
           f"{min(v for m, v in dc18.items() if m != w18):+.2f} to {max(dc18.values()):+.2f} points, except for "
           f"{short(w18)} ({dc18[w18]:+.2f} points), one of whose three adversarially trained runs ends at "
           + f"{min(json.loads(f.read_text(encoding='utf-8'))['tests']['random']['accuracy'] * 100 for f in (V2 / 'adversarial' / 'advtrain').glob(f'cic18_{w18}_s*.json')):.2f}% "
           "while the other two stay above 99.5%"
           + (f"; the largest gain ({short(max(dc18, key=dc18.get))}) reflects a seed that collapsed in standard "
              "training, not a benefit of adversarial training"
              if max(dc18.values()) > 1 and max(dc18, key=dc18.get) == "LSTM" else "")
           + ". "
           f"On KDDTest+, the change ranges from {min(rest):+.1f} to "
           f"{max(rest):+.1f} points for five of the six models, but GRU loses {abs(dn[worst]):.1f} points "
           f"({clean[('nsl', worst)][0]:.1f}% → {clean[('nsl', worst)][1]:.1f}%), so the cost of robustness is "
           "paid mainly on the out-of-distribution test set. The largest gap between random search and PGD after "
           f"adversarial training is {mask[0]:.1f} points ({short(mask[1][2])}, "
           f"{DSN[mask[1][0]]}, ε = {mask[1][3]:g}), which is consistent with some "
           "gradient masking. As the next experiment shows, these low evasion rates hold only against the threat "
           "model used in training.")
    S["rq5_stats"] = R
    rq5_consistent(P, S)


def consistent_stats():
    """一致性威脅模型的攻擊結果: {(ds, group, variant, model): [每個 ε 的平均逃逸率 (%)]} 與種子數。"""
    eps, out, n = None, {}, {}
    for f in sorted((V2 / "adversarial" / "consistent").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        eps = d["eps"]
        for grp in ("standard", "advtrained"):
            if grp not in d:
                continue
            x = d[grp]
            for var, suf in (("consistent", ""), ("notime", "_notime")):
                best = np.maximum(x["pgd" + suf], x["rs" + suf]) * 100
                out.setdefault((d["dataset"], grp, var, d["model"]), []).append(best)
        n[(d["dataset"], d["model"])] = n.get((d["dataset"], d["model"]), 0) + 1
    return eps, {k: np.mean(v, axis=0) for k, v in out.items()}, n


def rq5_consistent(P, S):
    eps, C, n = consistent_stats()
    if not C:
        P.todo("Consistent-threat-model attacks still in progress.")
        return
    e = {v: eps.index(v) for v in eps}
    dsets = [ds for ds in ("cic", "cic18") if any(k[0] == ds for k in C)]
    name = {"cic": "CIC-IDS2017", "cic18": "CSE-CIC-IDS2018"}
    get = lambda ds, grp, var, m, ep: C.get((ds, grp, var, m), [np.nan] * len(eps))[e[ep]]
    b = S["rq5_stats"]["best"]
    at = S["rq5_stats"]["at_best"]
    rows = []
    for m in DEEP:
        row = [short(m)]
        for ds in ("cic", "cic18"):
            vals = [get(ds, "standard", "consistent", m, 0.1), get(ds, "standard", "consistent", m, 1.0),
                    get(ds, "standard", "notime", m, 0.1), get(ds, "standard", "notime", m, 1.0),
                    get(ds, "advtrained", "consistent", m, 1.0), get(ds, "advtrained", "notime", m, 1.0)]
            row += ["—" if np.isnan(v) else f"{v:.1f}" for v in vals]
        rows.append(row)
    tab = P.next_table()
    tab_at = ROMAN[P.tab_no - 1]                      # RQ5 的對抗訓練表 (已輸出的最後一個表)
    ds0 = "cic"
    rng = lambda grp, var, ep, ms=DEEP: (min(get(ds0, grp, var, m, ep) for m in ms),
                                        max(get(ds0, grp, var, m, ep) for m in ms))
    s1, s01 = rng("standard", "consistent", 1.0), rng("standard", "consistent", 0.1)
    t1, t01 = rng("standard", "notime", 1.0), rng("standard", "notime", 0.1)
    a1, at1 = rng("advtrained", "consistent", 1.0), rng("advtrained", "notime", 1.0)
    real_at = max(at[("cic", "realistic", m, 1.0)] for m in DEEP)
    real_std = rngm_real = (min(b[("cic", "realistic", m, 1.0)] for m in DEEP),
                            max(b[("cic", "realistic", m, 1.0)] for m in DEEP))
    nseed = min(n[(ds0, m)] for m in DEEP if (ds0, m) in n)
    P.para("**Recomputing the derived features does not make the attack harder.** Table "
           f"{tab} repeats the attacks under the consistent threat model, in which every perturbation of a "
           "primary feature is propagated to the means, rates, and flag counts that depend on it, and only "
           "feature vectors that satisfy the implemented consistency and feasibility constraints are produced. "
           "On CIC-IDS2017 the consistent attack at ε = 1 evades "
           f"{s1[0]:.1f}–{s1[1]:.1f}% of the detections of the standard models, at least as much as the realistic "
           f"attack at the same nominal budget ({real_std[0]:.0f}–{real_std[1]:.0f}%), although the two budgets "
           "do not describe the same perturbation set (Section IV). Consistency removes impossible feature vectors but adds an effect that the "
           "realistic model lacks: increasing the number of forward packets or the duration also changes the "
           "rates and mean packet sizes, so a single primary change moves many inputs at once without counting "
           "against the budget of the derived features. Most of the evasion at small budgets "
           "comes from delaying packets: with the three timing features frozen, the evasion rate at ε = 0.1 "
           f"falls from {s01[0]:.0f}–{s01[1]:.0f}% to {t01[0]:.0f}–{t01[1]:.0f}%, but at ε = 1 it is still "
           f"{t1[0]:.0f}–{t1[1]:.0f}%, so an attacker who must not slow down can evade by adding packets and "
           "bytes alone.")
    P.columns(1)
    head = ["Model"]
    for ds in ("cic", "cic18"):
        tag = "CIC17" if ds == "cic" else "CIC18"
        head += [f"{tag}: ε=0.1", f"{tag}: ε=1", f"{tag}: no time ε=0.1", f"{tag}: no time ε=1",
                 f"{tag}: AT ε=1", f"{tag}: AT no time ε=1"]
    P.table(head, rows, [1.7] + [1.275] * 12,
            "Evasion rate (%) under the consistent threat model, stronger of PGD and random search, mean over "
            f"{word(nseed)} seeds and 2,000 detected attack records per seed. No time: the three timing features are "
            "frozen. AT: models adversarially trained under the realistic threat model (Table "
            f"{tab_at}).", left_cols=(0,))
    P.columns(2)
    missing18 = [short(m) for m in DEEP if ("cic18", m) not in n]
    ms18 = [m for m in DEEP if ("cic18", m) in n]
    r18 = lambda grp, var, ep: (min(get("cic18", grp, var, m, ep) for m in ms18),
                                max(get("cic18", grp, var, m, ep) for m in ms18))
    if ms18:
        u1, u01 = r18("standard", "consistent", 1.0), r18("standard", "consistent", 0.1)
        v1, v01 = r18("standard", "notime", 1.0), r18("standard", "notime", 0.1)
        strong = [m for m in ms18 if get("cic18", "standard", "notime", m, 1.0) >= 95]
        P.para(f"On CSE-CIC-IDS2018 the consistent attack is even more effective: at ε = 1 it evades "
               f"{u1[0]:.1f}–{u1[1]:.1f}% of the detections of the {word(len(ms18))} models, and "
               f"{v1[0]:.1f}–{v1[1]:.1f}% with the timing features frozen; at ε = 0.1 freezing them lowers the "
               f"evasion rate only from {u01[0]:.0f}–{u01[1]:.0f}% to {v01[0]:.0f}–{v01[1]:.0f}%. For "
               f"{word(len(strong))} of the {word(len(ms18))} models ({', '.join(short(m) for m in strong)}), adding forward "
               "packets and bytes alone, without slowing the attack down, evades at least 95% of the detections.")
    a18 = r18("advtrained", "consistent", 1.0) if ms18 else None
    P.para("**Adversarial training does not transfer across threat models.** The models adversarially trained "
           f"under the realistic threat model, which the realistic attack evades in at most {real_at:.1f}% of the "
           f"cases at ε = 1 on CIC-IDS2017, are evaded in {a1[0]:.0f}–{a1[1]:.0f}% of the cases by the consistent "
           f"attack at the same budget, and in {at1[0]:.0f}–{at1[1]:.0f}% even with the timing features frozen"
           + (f"; on CSE-CIC-IDS2018 the figures are "
              f"{max(at[('cic18', 'realistic', m, 1.0)] for m in DEEP if ('cic18', 'realistic', m, 1.0) in at):.1f}% and "
              f"{a18[0]:.0f}–{a18[1]:.0f}%" if a18 else "")
           + ". The defense "
           "has thus learned to resist the specific perturbations it was trained on, inflated primary features "
           "with unchanged derived features, which are exactly the perturbations that no real flow can "
           "produce. Robustness measured against the threat model used for training says little about "
           "robustness against a more faithful one, and the conclusion drawn from Table "
           f"{tab_at} alone would have been wrong.")
    rq5_consistent_training(P, S)
    if missing18:
        P.todo(f"Consistent attacks on CSE-CIC-IDS2018 for {', '.join(missing18)} still running on the laptop; "
               "rebuild when finished.")


def rq5_consistent_training(P, S):
    """CSE-CIC-IDS2018: 以 consistent 威脅模型做對抗訓練，再以三種威脅模型評估。"""
    rows = [r for r in S["rq5"].get("advtrain_consistent", []) if r["dataset"] == "cic18"]
    if not rows:
        P.todo("Consistent adversarial training on CSE-CIC-IDS2018 still in progress.")
        return
    best = {(r["threat"], r["model"], r["eps"]): r["best"] for r in rows}
    pgd = {(r["threat"], r["model"], r["eps"]): r["pgd"] for r in rows}
    rs = {(r["threat"], r["model"], r["eps"]): r["rs"] for r in rows}
    models = [m for m in DEEP if ("consistent", m, 1.0) in best]
    clean = {}
    for m in models:
        std, adv = [], []
        for s in (1, 2, 3):
            a = V2 / "adversarial" / "advtrain_consistent" / f"cic18_{m}_s{s}.json"
            p = V2 / "runs" / f"cic18_improved_tuned_binary_{m}_s{s}.json"
            if a.exists() and p.exists():
                adv.append(json.loads(a.read_text(encoding="utf-8"))["tests"]["random"]["accuracy"] * 100)
                std.append(json.loads(p.read_text(encoding="utf-8"))["tests_last"]["random"]["accuracy"] * 100)
        clean[m] = float(np.mean(adv) - np.mean(std)) if adv else np.nan
    cols = [("consistent", 1.0), ("consistent", 2.0), ("consistent_notime", 1.0), ("realistic", 1.0),
            ("realistic", 2.0)]
    tab = P.next_table()
    t_rows = [[short(m)] + [f"{best[(th, m, e)]:.1f}" for th, e in cols] + [f"{clean[m]:+.2f}"] for m in models]
    rng = lambda th, e: (min(best[(th, m, e)] for m in models), max(best[(th, m, e)] for m in models))
    c1, c2, r1, r2 = rng("consistent", 1.0), rng("consistent", 2.0), rng("realistic", 1.0), rng("realistic", 2.0)
    gap = max(((rs[k] - pgd[k], k) for k in rs if k[0] == "consistent"))
    P.para("**Training under the consistent threat model closes that gap but not the other one.** We adversarially "
           f"trained {word(len(models))} models ({', '.join(short(m) for m in models)}) on CSE-CIC-IDS2018 with "
           f"consistent PGD at ε = 1 (Table {tab}). Against "
           f"the threat model used in training they become robust: the consistent attack evades "
           f"{c1[0]:.1f}–{c1[1]:.1f}% of their detections at ε = 1. The realistic attack, however, which the "
           f"realistic-trained models resisted, now evades {r1[0]:.0f}–{r1[1]:.0f}% at ε = 1 and "
           f"{r2[0]:.0f}–{r2[1]:.0f}% at ε = 2. Each defense is thus robust to its own threat model and "
           "vulnerable to the other. The realistic attack may produce feature vectors that violate the "
           "consistency constraints, so this vulnerability exists in feature space and need not correspond to an "
           "evasion that is realizable in traffic; it does show that robustness does not carry over between "
           "threat models in either direction. Beyond the training budget the consistent attack also recovers "
           f"({c2[0]:.1f}–{c2[1]:.1f}% at ε = 2), and here random search is far stronger than PGD: for "
           f"{short(gap[1][1])} at ε = {gap[1][2]:g}, PGD evades {pgd[gap[1]]:.1f}% and random search "
           f"{rs[gap[1]]:.1f}%. This pattern is consistent with gradient masking [@athalye2018], which we have "
           "not diagnosed further, so the low evasion "
           "rates at the training budget, too, should be read as an upper bound on robustness. The clean accuracy "
           f"on the random split changes by {min(clean.values()):+.2f} to {max(clean.values()):+.2f} points.")
    P.columns(1)
    P.table(["Model", "Consistent ε=1", "Consistent ε=2", "Consistent, no time ε=1", "Realistic ε=1",
             "Realistic ε=2", "Δ acc. (random)"], t_rows, [2.6, 2.4, 2.4, 2.6, 2.4, 2.3, 2.3],
            "CSE-CIC-IDS2018: evasion rate (%) after adversarial training under the consistent threat model "
            "(ε = 1, three-step PGD), stronger of PGD and random search, mean over three seeds. Δ acc.: change "
            "of clean accuracy on the random split relative to standard training (percentage points).",
            left_cols=(0,))
    P.columns(2)


def results(P):
    S = load()
    P.section("Results")
    rq1(P, S)
    rq2(P, S)
    rq3(P, S)
    rq4(P, S)
    rq5(P, S)
    return S
