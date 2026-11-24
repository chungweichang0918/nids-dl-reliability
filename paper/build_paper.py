"""英文期刊論文草稿 (IEEE Access 格式)。

全文 (緒論到結論、RQ1–RQ5) 已寫完；數字全部由結果檔即時計算。
仍以黃色標示的只有程式庫網址 (尚未公開)。
所有資料集筆數都是 nids_data.describe() 實測的數字。

用法: python paper/build_paper.py   ->  paper/NIDS_paper_draft.docx
"""
import glob
import json
from pathlib import Path

import numpy as np
from docx.enum.text import WD_ALIGN_PARAGRAPH
from scipy import stats
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

from paper_docx import Paper, set_font
from paper_refs import REFS
from paper_results import load as load_results
from paper_results import results, rq5_stats, consistent_stats, DEEP

HERE = Path(__file__).resolve().parent
OUTFILE = HERE / "NIDS_paper_draft.docx"
YIN_DIR = HERE.parent / "results" / "v2" / "yin"
YIN_PAPER = {"binary": (83.28, 68.55), "multi": (81.29, 64.67)}
YIN_CFG = {"binary": (80, 0.1), "multi": (80, 0.5)}


def yin_stats():
    """從 results/v2/yin 的原始結果計算第二個案例的所有數字 (每個種子各自挑選)。"""
    runs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in glob.glob(str(YIN_DIR / "record_*.json"))]
    trees = [json.loads(Path(f).read_text(encoding="utf-8")) for f in glob.glob(str(YIN_DIR / "trees_*.json"))]
    out = {}
    for task in ("binary", "multi"):
        rs = [r for r in runs if r["task"] == task]
        seeds = sorted({r["seed"] for r in rs})
        paper_cfg = [r for r in rs if (r["hidden"], r["lr"]) == YIN_CFG[task]]
        by_test, by_val, val_cfgs = [], [], []
        for s in seeds:
            g = [r for r in rs if r["seed"] == s]
            t = max(g, key=lambda r: r["KDDTest+"])          # 原論文的做法: 在 KDDTest+ 上挑
            v = max(g, key=lambda r: r["validation"])        # 正確做法: 在驗證集上挑
            by_test.append((t["KDDTest+"], t["KDDTest-21"]))
            by_val.append((v["KDDTest+"], v["KDDTest-21"]))
            val_cfgs.append((v["hidden"], v["lr"]))
        by_test, by_val = np.array(by_test), np.array(by_val)
        gap = by_test[:, 0] - by_val[:, 0]
        lr_means = {lr: float(np.mean([r["KDDTest+"] for r in rs if r["lr"] == lr])) for lr in (0.01, 0.1, 0.5)}
        tr = {m: np.array([[r["KDDTest+"], r["KDDTest-21"]] for r in trees if r["task"] == task and r["model"] == m])
              for m in ("RF", "XGB")}
        out[task] = dict(
            n_seeds=len(seeds), n_runs=len(rs),
            paper_cfg=np.array([[r["KDDTest+"], r["KDDTest-21"]] for r in paper_cfg]),
            by_test=by_test, by_val=by_val, gap=gap, val_cfgs=val_cfgs,
            p_gap=float(stats.ttest_rel(by_test[:, 0], by_val[:, 0]).pvalue),
            best_run=float(max(r["KDDTest+"] for r in rs)),
            best_run_21=float(max(r["KDDTest-21"] for r in rs)), lr_means=lr_means, trees=tr)
    return out


def featseq_paragraph(Y):
    """另一種解讀: 把每筆紀錄的特徵當成長度為特徵數的序列 (Keras SimpleRNN，原論文設定，5 個種子)。"""
    runs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in glob.glob(str(YIN_DIR / "featseq_*.json"))]
    if not runs:
        return "{{TODO: alternative interpretation (features as a sequence) not available.}}"
    g = {t: sorted([r for r in runs if r["task"] == t], key=lambda r: r["seed"]) for t in ("binary", "multi")}
    b = np.array([r["KDDTest+"] for r in g["binary"]])
    m = np.array([r["KDDTest+"] for r in g["multi"]])
    bad = [r for r in g["multi"] if r["validation"] < 80]
    ok = np.array([r["KDDTest+"] for r in g["multi"] if r["validation"] >= 80])
    text = ("As an alternative reading of the unspecified input format, we treated the encoded features of a "
            "record as a sequence with one feature per time step, as in the first case study, and trained a "
            f"Keras SimpleRNN with sigmoid activation and the configurations reported in [@yin2017] ({len(b)} seeds). "
            f"This reading is clearly worse: {ms(b)}% on KDDTest+ in the binary task and "
            f"{ms(ok) if len(ok) > 1 else f'{ok[0]:.2f}'}% in the five-class task")
    if bad:
        text += (f" (mean over the {len(ok)} of {len(m)} five-class runs that trained normally; the other "
                 f"{len(bad)} stalled at " + " and ".join(f"{r['validation']:.1f}%" for r in bad) +
                 " validation accuracy, giving " + " and ".join(f"{r['KDDTest+']:.1f}%" for r in bad) +
                 " on KDDTest+)")
    text += (", against the published 83.28% and 81.29%. Neither reading reproduces the five-class result, and "
             "only the sequence-length-one reading, in which the network is a plain sigmoid MLP, approaches the "
             "published binary accuracy.")
    return text


def ms(a, d=2):
    a = np.asarray(a, float)
    return f"{a.mean():.{d}f} ± {a.std(ddof=1):.{d}f}" if len(a) > 1 else f"{a[0]:.{d}f}"

V2 = HERE.parent / "results" / "v2"


def count_runs():
    """論文中所有訓練過的模型數 (調參試驗、正式執行、控制組、消融、案例 2、對抗訓練)。"""
    n = lambda pattern: len(glob.glob(str(V2 / pattern)))
    S = load_results()
    ablation = sum(len(v["ablation"]) for v in S["rq4"].values()) * 3
    parts = dict(deep_runs=n("runs/*.json"), deep_trials=n("tune/*.json"), tree_trials=n("baselines/tune/*.json"),
                 tree_runs=n("baselines/runs/*.json"), yin=n("yin/record_*.json") + n("yin/featseq_*.json")
                 + n("yin/trees_*.json"), ablation=ablation + n("shortcut/check/*.json"),
                 advtrain=n("adversarial/advtrain/*.json") + n("adversarial/advtrain_consistent/*.json"),
                 sensitivity=n("sensitivity/*.json"),
                 device=len(glob.glob(str(V2.parent / "device_check" / "*" / "*.json"))))
    return sum(parts.values()), parts


def headline():
    """緒論與摘要共用的關鍵數字。"""
    S = load_results()
    r, rq3 = S["rq1"], S["rq3"]
    flips = [100 * np.mean([p["flip"] for p in v["pairwise"]]) for v in r.values()]
    nsl = r["nsl/default/binary/random"]
    order = sorted(nsl["accuracy"], key=lambda m: -nsl["accuracy"][m]["mean"])
    nsl_gap = [v["gap"] for v in rq3["nsl/binary"].values()]
    cic = rq3["cic/binary"]
    t_f1 = [cic[m]["temporal"]["f1"] for m in cic if "temporal" in cic[m]]
    i_f1 = [cic[m]["improved"]["f1"] for m in cic if "improved" in cic[m]]
    c18 = rq3["cic18/binary"]
    t18 = [c18[m]["temporal"]["f1"] for m in c18 if "temporal" in c18[m]]
    d18 = [c18[m]["temporal"]["detection"] for m in c18 if "temporal" in c18[m]]
    R = rq5_stats(S)
    deep = ["DNN", "CNN", "RNN", "LSTM", "CNN-LSTM", "GRU"]
    cic_r = [R["best"][("cic", "realistic", m, 1.0)] for m in deep]
    nsl_r = [R["best"][("nsl", "realistic", m, 1.0)] for m in deep if m != "GRU"]     # GRU 另外說明
    at = max(R["at_best"][(ds, "realistic", m, 1.0)] for ds in ("nsl", "cic") for m in deep)
    chk = S["check"]
    return dict(flip=float(np.mean(flips)), n_combos=len(r), gru_rank=order.index("GRU") + 1, n_models=len(order),
                t18_f1=(min(t18), max(t18)), t18_det=max(d18),
                r18_f1=max(c18[m]["improved"]["f1"] for m in c18 if "improved" in c18[m]),
                n_pairs=len(r["cic/improved/binary/random"]["pairwise"]),
                nsl_gap=(min(nsl_gap), max(nsl_gap)), i_f1=(min(i_f1), max(i_f1)), t_f1=(min(t_f1), max(t_f1)),
                cic_r=(min(cic_r), max(cic_r)), nsl_r=(min(nsl_r), max(nsl_r)), at=at,
                short_acc=(chk["all"]["accuracy"]["mean"], chk["drop_port_rst"]["accuracy"]["mean"]))


ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
       9: "ninth"}
NUM = {8: "eight", 9: "nine"}


def abstract_text(Y):
    """IEEE Access: 單一段落、150–250 字、不含引用、公式與不必要的縮寫。"""
    H = headline()
    eps, C, _ = consistent_stats()
    a1 = [C[("cic", "advtrained", "consistent", m)][eps.index(1.0)] for m in DEEP]
    return (
        "Comparative studies of deep-learning network intrusion detectors often report accuracies above 99% and "
        "rank architectures by fractions of a percentage point. We examine how far such rankings can be trusted. "
        "We reproduce two published studies and re-evaluate six deep architectures, the FT-Transformer, and two "
        "tree ensembles on NSL-KDD, UNSW-NB15, two releases of CIC-IDS2017, and CSE-CIC-IDS2018, with "
        "equal trial budgets for tuning, three to five seeds, and evaluation conditions from random splits to "
        "temporal splits with unseen attacks. Under our reproduction assumptions, one study's test-set "
        f"hyperparameter selection yields {Y['binary']['gap'].mean():.1f} points more binary accuracy "
        f"than validation-based selection. Averaged over {H['n_combos']} dataset–task–test combinations, a single "
        f"run per model reverses the order of a model pair in {H['flip']:.0f}% of cases, and on the random splits "
        "of the corrected CIC datasets no pairwise difference is significant. The gated recurrent unit, reported "
        f"as the best binary model, ranks {ORD[H['gru_rank']]} of {NUM[H['n_models']]} after tuning; tree "
        "ensembles rank first in distribution, and exploding gradients explain the failed reproduction. "
        "Replacing the random split by "
        f"the official test set costs {H['nsl_gap'][0]:.0f}–{H['nsl_gap'][1]:.0f} accuracy points on NSL-KDD. "
        f"Temporal splits lower macro-F1 to {H['t_f1'][0]:.0f}–{H['t_f1'][1]:.0f}% on CIC-IDS2017 and to "
        f"{H['t18_f1'][0]:.0f}% on CSE-CIC-IDS2018, where no model detects more than "
        f"{np.ceil(H['t18_det']):.0f}% of the unseen attacks. Under realistic constraints, attacks evade "
        f"{H['cic_r'][0]:.0f}–{H['cic_r'][1]:.0f}% of detections on CIC-IDS2017. Adversarial training lowers "
        f"evasion to at most {H['at']:.1f}%, but attacks that keep derived flow features consistent evade "
        f"{min(a1):.0f}–{max(a1):.0f}% of the hardened models' detections, and training against them leaves "
        "the original attack effective. We derive recommendations and release all code and results.")


TITLE = ("How Reliable Are Deep-Learning Comparisons for Network Intrusion Detection? "
         "A Reproducibility and Evaluation-Pitfall Study on Four Benchmarks")


def front_matter(P, Y):
    d = P.doc
    p = d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    set_font(p.add_run(TITLE), 18)
    p = d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("Chun-Wei Chang"), 11)
    p = d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    set_font(p.add_run("Department of Computer Science and Information Engineering, National Taichung University of Science and Technology, Taichung, Taiwan; "
                       "corresponding author: s1111232077@nutc.edu.tw"), 9, italic=True)

    p = d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(4)
    set_font(p.add_run("ABSTRACT "), 9, bold=True)
    text = abstract_text(Y)
    n_words = len(text.split())
    print(f"摘要 {n_words} 字 (IEEE Access: 150–250 字)")
    if not 150 <= n_words <= 250:
        print("  警告: 摘要字數不符合規定")
    P._runs(p, text, 9)
    p = d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(10)
    set_font(p.add_run("INDEX TERMS "), 9, bold=True)
    set_font(p.add_run("Adversarial robustness, benchmark datasets, deep learning, evaluation methodology, "
                       "network intrusion detection, reproducibility, shortcut learning."), 9)
    # 分隔線
    pPr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for k, v in (("val", "single"), ("sz", "6"), ("space", "4"), ("color", "000000")):
        bottom.set(qn(f"w:{k}"), v)
    bdr.append(bottom)
    pPr.append(bdr)


def introduction(P):
    H = headline()
    n_runs, _ = count_runs()
    P.section("Introduction")
    P.para("Machine learning has become the dominant approach in the network intrusion detection (NIDS) "
           "literature. Deep architectures such as multilayer perceptrons, convolutional networks, and recurrent "
           "networks are regularly compared on public benchmarks, and the resulting rankings are used to argue "
           "that one architecture is better suited to intrusion detection than another [@yin2017, @vinayakumar2019, "
           "@ferrag2020]. Many of these comparisons report accuracies above 99% and separate the best and the "
           "worst model by less than one percentage point.")
    P.para("Two lines of prior work give reasons to doubt such conclusions. In machine learning, results obtained "
           "from a single training run can change with nothing more than a different random seed, and published "
           "comparisons rarely report this variability [@bouthillier2019, @pineau2021, @reimers2017]. In security, "
           "evaluation choices such as randomly splitting a single capture into training and test data, ignoring "
           "duplicates and label errors, and testing only on attack types seen during training are known to "
           "inflate performance [@sommer2010, @arp2022, @engelen2021]. Yet few studies have measured how much "
           "these factors change the conclusions of a concrete, recent comparison, and how large each effect is "
           "relative to the differences being reported.")
    P.para("This paper performs such a measurement. We take a recent comparative study published in this "
           "journal [@elsayed2024], which compares six deep architectures on NSL-KDD and identifies the gated "
           "recurrent unit (GRU) as the best binary classifier, and use it as our main case study. A second, "
           "widely cited study [@yin2017] serves as a complementary case: it evaluates on the official test set but "
           "selects its hyperparameters on that same test set. We first reproduce both studies. We then "
           "re-evaluate the six architectures of the first study, together with the FT-Transformer "
           "[@gorishniy2021], a deep architecture designed for tabular data, and random-forest [@breiman2001] and "
           "XGBoost [@chen2016] baselines, on four benchmarks: NSL-KDD "
           "[@tavallaee2009], UNSW-NB15 [@moustafa2015], CIC-IDS2017 [@sharafaldin2018] in both its original "
           "release and the corrected release of Liu *et al.* [@liu2022], and the corrected release of "
           "CSE-CIC-IDS2018. Every model receives the same "
           "number of hyperparameter-tuning trials, every configuration is trained with several random seeds, and every "
           "model is evaluated under up to eight evaluation conditions. We organize the study around five research "
           "questions:")
    P.bullets([
        "**RQ1 (ranking reliability).** Do model rankings obtained from single runs survive repeated runs, "
        "statistical testing, and a change of the test distribution?",
        "**RQ2 (training stability).** What causes the training instability of recurrent models, and does it "
        "persist when every model is tuned with the same number of trials?",
        "**RQ3 (evaluation pitfalls).** How much does the reported performance change when random splits, "
        "duplicate records, uncorrected labels, and the absence of temporal shift are replaced by a stricter "
        "evaluation protocol?",
        "**RQ4 (shortcut learning).** Do the models rely on a few dataset-specific features rather than on "
        "behavior that transfers to new traffic?",
        "**RQ5 (adversarial robustness).** How easily can the models be evaded when the attacker is restricted "
        "to realistic traffic modifications, and does adversarial training help against attacks other than "
        "the one it was trained on?",
    ])
    P.para("Our contributions are as follows:", indent=False)
    P.bullets([
        "Reproductions of two published studies [@elsayed2024, @yin2017], including diagnostic experiments that "
        "explain why part of their results could not be reproduced and a measurement, under our reproduction "
        "assumptions, of the optimism caused by selecting hyperparameters on the test set.",
        f"A controlled multi-dataset protocol with equal trial budgets for tuning {NUM[H['n_models']]} model families, "
        "three to five random seeds per configuration, and eight evaluation conditions on four benchmarks, "
        f"amounting to {n_runs:,} trained models including tuning trials and control groups, with a second, "
        "independent subsample of CSE-CIC-IDS2018 and cross-hardware replications as robustness checks.",
        "A quantification of how often single-run comparisons contradict multi-seed results "
        f"({H['flip']:.0f}% of model pairs on average) and of how much the reported performance changes between "
        f"evaluation protocols (up to {H['nsl_gap'][1]:.0f} accuracy points on NSL-KDD and "
        f"{H['r18_f1'] - H['t18_f1'][0]:.0f} macro-F1 points on CSE-CIC-IDS2018).",
        "A mechanistic analysis of recurrent-model instability based on per-epoch gradient norms, gradient "
        "clipping, feature-order permutation, and the retraining of next-ranked tuning configurations.",
        "A shortcut-feature analysis with SHAP and feature ablation, and an adversarial evaluation that "
        "distinguishes unconstrained, realistic, and feature-consistent attacks, combines gradient-based and "
        "gradient-free attacks, and covers black-box transfer and adversarial training under two threat models.",
        "An open artifact with all code, configurations, seeds, and per-run results.",
    ], numbered=True)
    P.para("The rest of the paper is organized as follows. Section II reviews related work. Section III "
           "presents the two case studies and our reproductions. Section IV describes the datasets, "
           "evaluation conditions, models, and experimental protocol. Section V presents the results for each "
           "research question, Section VI discusses implications and threats to validity, and Section VII "
           "concludes.")


def related_work(P):
    P.section("Related Work")
    P.subsection("Deep Learning for Network Intrusion Detection")
    P.para("Yin *et al.* [@yin2017] trained a recurrent network on NSL-KDD and evaluated it on the official "
           "KDDTest+ set, reporting 83.28% binary and 81.29% five-class accuracy. Vinayakumar *et al.* "
           "[@vinayakumar2019] selected a deep network on KDD Cup 99 and transferred it to NSL-KDD, UNSW-NB15, "
           "CIC-IDS2017, and other datasets. Ferrag *et al.* [@ferrag2020] surveyed 35 security datasets and "
           "compared seven deep models on CSE-CIC-IDS2018 and Bot-IoT. Elsayed *et al.* [@elsayed2024] compared "
           "six architectures on NSL-KDD using a random split of KDDTrain+ and reported accuracies between 97.96% "
           "and 99.54%. The gap of more than fifteen percentage points between studies that test on KDDTest+ and "
           "studies that test on a random split of the training file illustrates how strongly the evaluation "
           "protocol shapes the reported numbers.")
    P.para("Outside the security domain, systematic benchmarks have repeatedly found that tree ensembles match or "
           "outperform deep networks on tabular data [@grinsztajn2022, @shwartz2022]. Flow and connection "
           "records are tabular, yet deep NIDS comparisons often omit strong tree-based baselines.")
    P.subsection("Dataset Quality and Evaluation Pitfalls")
    P.para("The quality of NIDS benchmarks has been questioned since the DARPA 1998 evaluation on which KDD Cup 99 "
           "is based [@mchugh2000]. Tavallaee *et al.* [@tavallaee2009] removed the massive redundancy of KDD Cup 99 "
           "to create NSL-KDD. Moustafa and Slay [@moustafa2016] argued that UNSW-NB15 is more complex and more "
           "representative of modern traffic than KDD99. Newer datasets are not free of problems: Engelen *et al.* "
           "[@engelen2021] and Liu *et al.* [@liu2022] found flaws in the flow construction and labeling of "
           "CIC-IDS2017 and reconstructed or relabeled a substantial fraction of its flows, and Lanvin *et al.* "
           "[@lanvin2022] found misordered and duplicated packets and an unlabeled port scan, whose correction "
           "lowered the recall of four classical classifiers by about 20 percentage points. Flood "
           "*et al.* [@flood2024] identified recurring design smells in seven benchmark datasets, and Goldschmidt "
           "and Chudá [@goldschmidt2025] surveyed 89 datasets and their limitations. D’hooge *et al.* "
           "[@dhooge2022] demonstrated that metadata such as the destination port alone can reach high balanced "
           "accuracy, an early instance of the shortcut problem we study in RQ4.")
    P.para("Several studies question how well detectors generalize beyond the capture they were trained on. "
           "Cantone *et al.* [@cantone2024] found near-perfect within-dataset but near-chance cross-dataset "
           "performance on CIC-IDS2017 and CSE-CIC-IDS2018, Apruzzese *et al.* [@apruzzese2022xeval] proposed a "
           "framework for cross-evaluating detectors across datasets, and Sarhan *et al.* [@sarhan2022] compared "
           "flow feature sets across datasets and used SHAP to explain the differences. Kus *et al.* [@kus2022] "
           "re-evaluated published industrial intrusion detectors on attack types held out of training and "
           "observed a sharp drop in detection.")
    P.para("On the methodological side, Sommer and Paxson [@sommer2010] explained why intrusion detection is a "
           "difficult setting for machine learning, and Arp *et al.* [@arp2022] cataloged ten common pitfalls of "
           "machine learning in security, such as sampling bias, data snooping, spurious correlations, and "
           "lab-only evaluation, and found them to be widespread in top-tier venues. Apruzzese *et al.* "
           "[@apruzzese2023sok] argued for a pragmatic assessment of learning-based detectors that covers many "
           "configurations, adversarial scenarios, and deployment platforms. Our study measures the size of "
           "several of these pitfalls on a single, concrete comparison.")
    P.subsection("Reproducibility of Machine-Learning Results")
    P.para("Bouthillier *et al.* [@bouthillier2019] distinguish methods reproducibility from inferential "
           "reproducibility and show that conclusions can fail to hold even when code is available. Pineau "
           "*et al.* [@pineau2021] describe the NeurIPS reproducibility program and recommend reporting seeds, "
           "number of runs, and variability. Reimers and Gurevych [@reimers2017] trained 50,000 LSTM networks and "
           "showed that the random seed alone can change the F1 score of a sequence tagger by about one "
           "percentage point, and argued for comparing score distributions rather than single scores. In NIDS, "
           "Schrötter *et al.* [@schrotter2024] replicated a published deep detector for IoT networks and found "
           "that its accuracy dropped sharply on new traffic from the same environment. We apply the "
           "multi-seed, distribution-based perspective to published deep NIDS comparisons.")
    P.subsection("Explainability and Adversarial Robustness")
    P.para("SHAP [@lundberg2017] attributes a prediction to individual features using Shapley values. Warnecke "
           "*et al.* [@warnecke2020] showed that explanation methods differ widely in quality on security tasks "
           "and can reveal artifacts learned by a model. Geirhos *et al.* [@geirhos2020] describe such artifacts "
           "as shortcut learning: decision rules that work on standard benchmarks but fail under more challenging "
           "test conditions.")
    P.para("Projected gradient descent (PGD) [@madry2018] is the standard white-box attack for evaluating "
           "robustness. Athalye *et al.* [@athalye2018] showed that apparent robustness often results from "
           "obfuscated gradients rather than from a robust model. For NIDS, Apruzzese *et al.* [@apruzzese2022] "
           "argued that many published attacks assume unrealistic attacker capabilities and proposed more "
           "realistic threat models. Sheatsley *et al.* [@sheatsley2022] crafted adversarial examples that respect "
           "domain constraints, Vitorino *et al.* [@vitorino2023] and Debicha *et al.* [@debicha2023] generated "
           "realistic evasion attacks and evaluated adversarial training or other defenses, and Apruzzese *et al.* "
           "[@apruzzese2024drift] studied how realizable perturbations interact with concept drift. Each of these "
           "studies examines one attack family on one model class; we compare unconstrained, constrained, and "
           "feature-consistent attacks, gradient-based and gradient-free, on all models of the comparison.")
    P.subsection("Positioning of This Work")
    P.para("Each of the above lines of work addresses one aspect in isolation. Table "
           f"{P.next_table()} compares the closest studies with ours along seven dimensions. The closest in "
           "spirit is the pragmatic assessment of Apruzzese *et al.* [@apruzzese2023sok], which also repeats "
           "every experiment many times, compares methods with significance tests, and starts from conflicting "
           "published claims about neural networks and decision trees; it evaluates mainly tree-based learners in "
           "its own pipeline, however, and does not reproduce published deep-learning comparisons or examine "
           "training instability, label errors, duplicates, temporal shift, and shortcut features separately. "
           "No study evaluates the same models under all of these evaluation pitfalls, shortcut features, and "
           "constrained attacks at once. Some individual observations, such as "
           "the optimism of random splits on NSL-KDD, are known [@tavallaee2009, @yin2017]; our contribution is "
           "to measure their impact on the conclusions of specific published comparisons and to compare their "
           "magnitude across datasets and model families.")
    Y, P_, N = "●", "◐", "–"
    rows = [
        ["Arp *et al.* [@arp2022]", N, N, N, P_, P_, N, N],
        ["Engelen *et al.* [@engelen2021]; Liu *et al.* [@liu2022]", N, N, N, Y, N, N, Y],
        ["Lanvin *et al.* [@lanvin2022]", N, N, N, Y, N, N, N],
        ["D’hooge *et al.* [@dhooge2022]", N, P_, N, P_, Y, N, N],
        ["Flood *et al.* [@flood2024]", N, N, N, P_, P_, N, N],
        ["Cantone *et al.* [@cantone2024]", N, N, N, P_, N, N, N],
        ["Apruzzese *et al.* [@apruzzese2022xeval]", N, N, N, P_, N, N, N],
        ["Apruzzese *et al.* [@apruzzese2023sok]", P_, Y, P_, P_, N, P_, Y],
        ["Kus *et al.* [@kus2022]", P_, N, N, P_, N, N, N],
        ["Schrötter *et al.* [@schrotter2024]", Y, P_, N, P_, N, N, N],
        ["Sheatsley *et al.* [@sheatsley2022]", N, N, N, N, N, Y, N],
        ["Apruzzese *et al.* [@apruzzese2024drift]", N, N, N, P_, N, Y, N],
        ["**This work**", Y, Y, Y, Y, Y, Y, Y],
    ]
    P.columns(1)
    P.table(["Study", "Reproduction of published results", "Multiple seeds and tests",
             "Equal tuning incl. tree baselines", "Pitfalls measured separately", "Shortcut features",
             "Constrained adversarial evaluation", "Open artifact"], rows,
            [5.0, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.5],
            "Comparison with the closest prior studies (● addressed, ◐ partially addressed, – not addressed). "
            "Multiple seeds and tests: ● repeated runs with significance tests between models; ◐ repeated runs "
            "reported with dispersion only.", left_cols=(0,))
    P.columns(2)


def case_study(P, Y):
    P.section("Case Studies: Two Published Studies")
    P.subsection("Case 1: A Comparison of Six Deep Architectures")
    P.para("Elsayed *et al.* [@elsayed2024] standardized the 38 numeric features of KDDTrain+, one-hot encoded the "
           "three categorical features (122 input dimensions in total), and split the data randomly into 80% "
           "training and 20% test data, after standardization. Six architectures were tuned with KerasTuner "
           "[@omalley2019]: a dense network (DNN), a SimpleRNN, a one-dimensional CNN, an LSTM [@hochreiter1997], a "
           "GRU [@cho2014], and a hybrid CNN-LSTM. Recurrent and convolutional models treat the 122 features as a "
           "sequence of length 122. All models were trained with Adam [@kingma2015] for 100 epochs with a batch "
           "size of 5,000. The study reports accuracies between 97.96% and 99.54%, identifies GRU as the best "
           "binary classifier (99.54%), and DNN as the best five-class classifier (99.34%).")
    P.subsection("Case 1: Reproduction and Diagnosis")
    P.para("We re-implemented all twelve models (six architectures, binary and five-class) in Keras 3 "
           "[@chollet2015] with the PyTorch backend [@paszke2019], using the published best hyperparameters and a "
           "single seed. DNN, CNN, and SimpleRNN were reproduced within 0.4 percentage points in both tasks. "
           "LSTM and GRU reached only 96.41% and 96.06% in binary classification, about three points below the "
           "published values, and five-class CNN-LSTM reached 93.42%. Training curves showed abrupt collapses of "
           "validation accuracy accompanied by loss spikes, a typical symptom of exploding gradients "
           "[@bengio1994, @pascanu2013].")
    P.para("Five diagnostic experiments located the cause. Changing only the random seed moved the accuracy of "
           "the affected models by 1.4–4.5 points. Gradient clipping removed all collapses and raised binary GRU "
           "to 98.78%. A lower learning rate made convergence too slow for the 100-epoch budget. Training LSTM with "
           "a true batch of 5,000 instead of gradient accumulation (97.72%), and training it in the original "
           "TensorFlow 2.14 framework (98.21%), produced results within the seed-to-seed range, which reduces "
           "the concern that these implementation differences explain the gap. Even after these corrections, the recurrent models remained 0.8–1.5 points "
           "below the published numbers, a gap compatible with seed selection. These findings motivate the "
           "systematic re-evaluation in the rest of the paper.")
    yin_case(P, Y)


def yin_case(P, Y):
    b, m = Y["binary"], Y["multi"]
    P.subsection("Case 2: Model Selection on the Test Set")
    P.para("The second study, RNN-IDS by Yin *et al.* [@yin2017], follows a different and in principle sounder "
           "protocol: it trains on the full KDDTrain+ file and evaluates on the official KDDTest+ set and its harder "
           "subset KDDTest-21. Three heavy-tailed features (duration, source bytes, destination bytes) are "
           "log-scaled and all features are min-max normalized. The model is a single-layer Elman network with a "
           "sigmoid hidden layer and a softmax output, trained with gradient descent in Theano for 100 epochs "
           "(binary) or 80 epochs (five-class). The study evaluates a grid of five hidden-layer sizes "
           "(20, 60, 80, 120, 240) and three learning rates (0.01, 0.1, 0.5) and reports the configuration with "
           "the highest accuracy *on KDDTest+*: 80 units with a learning rate of 0.1 for binary classification "
           f"({YIN_PAPER['binary'][0]}% on KDDTest+, {YIN_PAPER['binary'][1]}% on KDDTest-21) and 80 units with "
           f"0.5 for five-class classification ({YIN_PAPER['multi'][0]}% and {YIN_PAPER['multi'][1]}%). No separate "
           "validation set is described, so the reported numbers are selected on the test data, a form of data "
           "snooping [@arp2022]. The study concludes that RNN-IDS outperforms J48, random forest, SVM, and other "
           "classical methods.")
    P.para("We reproduced the study in PyTorch. Several details are not specified. The paper does not state how a "
           "record is turned into an input sequence; we treat each record as a sequence of length one, in which "
           "case the Elman cell reduces to a sigmoid hidden layer. We use plain stochastic gradient descent "
           "without momentum, a batch size of 64, default initialization, and compute the min-max ranges on "
           "training data only. To separate the effect of test-set selection from everything else, we hold out a "
           "fixed 20% of KDDTrain+ as validation data and train all 15 configurations with "
           f"{b['n_seeds']} seeds on the remaining 80% ({b['n_runs']} models per task). For each seed we then "
           "select one configuration twice, once by KDDTest+ accuracy as in the original study and once by "
           "validation accuracy, and evaluate both choices on KDDTest+ and KDDTest-21. Random forest and XGBoost "
           f"with default parameters, trained on the same data, serve as reference. Table {P.next_table()} "
           "summarizes the results.")

    def cell(task_stats, key):
        return [ms(task_stats[key][:, 0]), ms(task_stats[key][:, 1])]

    rows = [
        ["Reported in [@yin2017]", f"{YIN_PAPER['binary'][0]:.2f}", f"{YIN_PAPER['binary'][1]:.2f}",
         f"{YIN_PAPER['multi'][0]:.2f}", f"{YIN_PAPER['multi'][1]:.2f}"],
        ["Reported configuration (ours)"] + cell(b, "paper_cfg") + cell(m, "paper_cfg"),
        ["Best of grid, selected on KDDTest+"] + cell(b, "by_test") + cell(m, "by_test"),
        ["Best of grid, selected on validation"] + cell(b, "by_val") + cell(m, "by_val"),
        ["Selection bias (KDDTest+)", f"+{b['gap'].mean():.2f}", "", f"+{m['gap'].mean():.2f}", ""],
        ["Random forest (default)"] + [ms(b["trees"]["RF"][:, 0]), ms(b["trees"]["RF"][:, 1])]
        + [ms(m["trees"]["RF"][:, 0]), ms(m["trees"]["RF"][:, 1])],
        ["XGBoost (default)"] + [ms(b["trees"]["XGB"][:, 0]), ms(b["trees"]["XGB"][:, 1])]
        + [ms(m["trees"]["XGB"][:, 0]), ms(m["trees"]["XGB"][:, 1])],
    ]
    P.columns(1)
    P.table(["Setting", "Binary: KDDTest+", "Binary: KDDTest-21", "Five-class: KDDTest+", "Five-class: KDDTest-21"],
            rows, [4.6, 3.1, 3.1, 3.1, 3.1],
            f"Reproduction of RNN-IDS [@yin2017]: accuracy (%) as mean ± standard deviation over {b['n_seeds']} "
            "seeds. Configurations are selected per seed from the 15-point grid of the original study.",
            left_cols=(0,))
    P.columns(2)

    lo, hi = b["by_test"][:, 0].min(), b["by_test"][:, 0].max()
    P.para(f"**The reported binary accuracy is reproduced only under test-set selection.** With the configuration "
           f"reported in the paper, all {b['n_seeds']} seeds stay below the published value "
           f"({ms(b['paper_cfg'][:, 0])}% vs. {YIN_PAPER['binary'][0]}%). Choosing, for each seed, the best of "
           f"the 15 configurations on KDDTest+ yields {lo:.2f}–{hi:.2f}%, a range that contains the published "
           f"83.28%. Selecting on validation data instead gives {ms(b['by_val'][:, 0])}%: test-set selection "
           f"inflates the binary result by {b['gap'].mean():.2f} points on average "
           f"({b['gap'].min():.2f}–{b['gap'].max():.2f} across seeds; paired t-test p = {b['p_gap']:.3f}) and "
           f"the five-class result by {m['gap'].mean():.2f} points (p = {m['p_gap']:.3f}).")
    P.para(f"**The five-class result is not reproduced at all.** Even the best of all {m['n_runs']} five-class runs "
           f"reaches only {m['best_run']:.2f}% on KDDTest+, {YIN_PAPER['multi'][0] - m['best_run']:.1f} points below "
           f"the published {YIN_PAPER['multi'][0]}%, and the published KDDTest-21 accuracy "
           f"({YIN_PAPER['multi'][1]}%) is far above the best KDDTest-21 accuracy of any run "
           f"({m['best_run_21']:.2f}%). A default "
           f"XGBoost model ({ms(m['trees']['XGB'][:, 0])}%) outperforms every RNN-IDS configuration in this task, "
           "whereas in the binary task the validation-selected RNN-IDS remains ahead of XGBoost "
           f"({ms(b['by_val'][:, 0])}% vs. {ms(b['trees']['XGB'][:, 0])}%). XGBoost was not among the baselines of "
           "the original study.")
    sizes = sorted({h for h, _ in b["val_cfgs"]})
    lrs = {lr for _, lr in b["val_cfgs"]}
    lr_text = (f"always chose a learning rate of {lrs.pop()}" if len(lrs) == 1
               else f"chose learning rates {sorted(lrs)}")
    P.para(f"**The selected hyperparameters are not stable.** Across seeds, validation-based selection {lr_text} "
           f"but different hidden-layer sizes ({', '.join(str(s) for s in sizes)} units "
           "in the binary task), indicating that the layer size reported as optimal is within seed noise. "
           "Averaged over all hidden sizes and seeds, a learning rate of 0.5 reached "
           f"{b['lr_means'][0.5]:.2f}% on KDDTest+ in the binary task versus {b['lr_means'][0.1]:.2f}% for the "
           "published 0.1; the published optimum may depend on unreported details such as the batch size.")
    P.para(featseq_paragraph(Y))
    P.para("These results depend on our interpretation of the input sequence and on the unreported settings above, "
           "and should be read as a reproduction under stated assumptions. They nevertheless show that a "
           "protocol with an official test set does not protect against optimistic reporting when the same test "
           "set is used for model selection.")


def methodology(P):
    P.section("Methodology")
    P.subsection("Datasets")
    P.para("We use four benchmarks of different age and construction. NSL-KDD [@tavallaee2009] is derived from "
           "the DARPA 1998 traffic; its official test set, KDDTest+, contains 17 attack types that do not occur in "
           "the training file. UNSW-NB15 [@moustafa2015] was generated in 2015 and is distributed with an official "
           "training and test split. CIC-IDS2017 [@sharafaldin2018] contains five days of traffic captured in 2017 "
           "with a different set of attacks on each day. For CIC-IDS2017 we use both the original release "
           "(MachineLearningCSV) and the release corrected by Liu *et al.* [@liu2022], in which flows were "
           "re-extracted with a fixed flow meter and relabeled; following the authors' recommendation, flows "
           "labeled as *attempted* attacks are treated as benign. CSE-CIC-IDS2018 was produced by the same group "
           "with the same flow meter on a larger testbed and contains ten days of traffic captured in 2018; we use "
           "only the corrected release of [@liu2022], with the same treatment of attempted attacks. "
           f"Table {P.next_table()} summarizes the data.")
    P.columns(1)
    P.table(["Dataset", "Release", "Input dims", "Records used", "Duplicates", "Attack classes (multi-class)"],
            [["NSL-KDD", "KDDTrain+ / KDDTest+", "122", "125,973 / 22,544", "—", "4 (DoS, Probe, R2L, U2R)"],
             ["UNSW-NB15", "official train / test", "67", "175,341 / 82,332", "—", "9"],
             ["CIC-IDS2017", "original [@sharafaldin2018]", "77", "2,827,876", "329,896 (11.7%)", "8"],
             ["CIC-IDS2017", "corrected [@liu2022]", "77", "2,099,971", "260,257 (12.4%)", "8"],
             ["CSE-CIC-IDS2018", "corrected [@liu2022], subsample", "77", "3,449,968", "465,294 (13.5%)",
              "6 (DoS, DDoS, brute force, web, botnet, infiltration)"]],
            [2.4, 3.8, 1.8, 3.2, 2.8, 4.0],
            "Datasets. Counts exclude records with non-finite feature values (2,867 in the original CIC-IDS2017 "
            "release, 5 in the corrected one, 2 in the CSE-CIC-IDS2018 subsample).")
    P.columns(2)
    P.para("Categorical features are one-hot encoded and numeric features standardized. For UNSW-NB15, the "
           "*proto* feature has 133 values; we keep the five most frequent and map the rest to *other*. CIC-IDS2017 "
           "and CSE-CIC-IDS2018 provide the same 77 numeric flow features after removing a duplicated column.")
    P.para("The corrected CSE-CIC-IDS2018 release contains 63.2 million flows, 59.7 million of them benign, which "
           "exceeds what we can train with repeated seeds and nine models. We therefore draw a stratified "
           "subsample before any other processing: every fine-grained attack label is kept in full up to a cap "
           "of 300,000 flows (DoS Hulk and DDoS HOIC are capped; the other 13 attack labels, including all web "
           "attacks and all infiltration flows, are complete), and the benign flows, including the attempted "
           "attacks, are sampled at a uniform rate of 3.69% to 2.2 million. Sampling is done independently on each "
           "day, so the day-wise composition needed for the temporal split is preserved. Because the cap and the "
           "benign rate change the class ratio, absolute accuracies on this dataset are not comparable with those "
           "on the full release; the comparisons between models and between evaluation conditions, which are the "
           "subject of this study, are. To check that our conclusions do not depend on the particular subsample, "
           "we draw a second, independent subsample with a different seed and repeat the binary experiments on it "
           "(Section VI). CIC-IDS2017 is used in full.")

    P.subsection("Evaluation Conditions")
    P.para(f"Table {P.next_table()} lists the eight evaluation conditions. The *random* conditions reproduce the protocol of "
           "[@elsayed2024]: the scaler is fitted on the whole training file and the data are then split randomly "
           "80/20, so the random test set contributes to the standardization statistics; we quantify the effect "
           "of this choice in Section VI. They are compared with "
           "the official test sets of NSL-KDD and UNSW-NB15 and, for CIC-IDS2017, with a sequence of conditions "
           "that progressively changes the evaluation protocol: removing exact duplicates, switching to the "
           "corrected release, and finally replacing the random split by a temporal one. Each step can change "
           "more than one property of the data (the corrected release, for example, re-extracts the flows as "
           "well as relabeling them, and the temporal split changes the amount and composition of the training "
           "data), so the differences between conditions are effects of the protocol, not of a single isolated "
           "factor. In the "
           "CIC-IDS2017 temporal condition the models are trained on Monday to Wednesday and tested on Thursday and "
           "Friday; all 227,819 attack flows in this test set belong to attack types that do not occur in the "
           "training days. CSE-CIC-IDS2018 is evaluated on the corrected, deduplicated data with a random split and "
           "with a temporal split that trains on the first five days (14–21 February: brute force, DoS, and DDoS) "
           "and tests on the last five (22 February–2 March: web attacks, infiltration, and botnet); again every "
           "one of the 195,038 attack flows in the test set belongs to an unseen attack type. In the temporal "
           "conditions the scaler is fitted on the training days only. In every condition, the last 20% of the "
           "training data serve as validation data for tuning and monitoring; test sets are never used for any "
           "decision.")
    P.columns(1)
    P.table(["Condition", "Training data", "Test data", "Change"],
            [["NSL-KDD", "80% of KDDTrain+ (100,778)", "random 20% (25,195); KDDTest+ (22,544)", "—"],
             ["UNSW-NB15", "80% of official train (140,273)", "random 20% (35,068); official test (82,332)", "—"],
             ["CIC orig", "80% of original release (2,262,301)", "random 20% (565,575)", "—"],
             ["CIC dedup", "80% after removing duplicates (1,998,384)", "random 20% (499,596)", "deduplicated"],
             ["CIC improved", "80% of corrected, deduplicated (1,471,771)", "random 20% (367,943)",
              "+ corrected release"],
             ["CIC temporal", "Mon–Wed, corrected, dedup. (1,107,009)", "Thu–Fri (732,705)", "+ temporal split"],
             ["CIC18 random", "80% of corrected, dedup. subsample (2,387,739)", "random 20% (596,935)",
              "dedup., corrected"],
             ["CIC18 temporal", "14–21 Feb., corrected, dedup. (1,875,696)", "22 Feb.–2 Mar. (1,108,978)",
              "+ temporal split"]],
            [2.5, 5.6, 6.2, 2.7],
            "Evaluation conditions. Numbers of records in parentheses.", left_cols=(0, 1, 2))
    P.columns(2)

    P.subsection("Models")
    P.para("We use the six architectures of [@elsayed2024] with the same layer structure; recurrent and "
           "convolutional models receive the features as a univariate sequence. To fit the GRU into the 6-GB "
           "memory of our GPU, recurrent models accumulate gradients over five micro-batches of 1,000 samples, "
           "which is mathematically equivalent to a batch of 5,000; our diagnostic experiments confirmed that this "
           "does not change the results (Section III-B). As baselines we add a random forest [@breiman2001] and "
           "XGBoost [@chen2016], which are standard choices for tabular data [@grinsztajn2022, @shwartz2022].")
    P.para("The six architectures of [@elsayed2024] predate the deep models designed for tabular data. To test "
           "whether our conclusions merely reflect outdated deep baselines, we add the FT-Transformer "
           "[@gorishniy2021] as a ninth model. It embeds every numeric feature as a learned token, prepends a "
           "classification token, and applies pre-normalization Transformer blocks with ReGLU feed-forward "
           "layers; categorical features enter through their one-hot columns, which the tokenizer treats like "
           "numeric ones. Trial 0 of its search is the default configuration of [@gorishniy2021] (three blocks, "
           "token dimension 192, eight heads, attention dropout 0.2, feed-forward dropout 0.1, feed-forward factor "
           "4/3, AdamW with learning rate 10⁻⁴ and weight decay 10⁻⁵, no weight decay on the tokenizer, the "
           "normalization layers, and the biases).")

    P.subsection("Hyperparameter Tuning with Equal Trial Budgets")
    P.para("Every model receives the same number of tuning trials on every dataset and task: 20 trials of random "
           "search [@bergstra2012]. An equal trial budget is simple and reproducible, but it is not an equal "
           "computational budget, since a trial of a recurrent model or the FT-Transformer costs far more than a "
           "trial of a DNN or a tree ensemble, and it does not guarantee that every model is equally well "
           "optimized. The rankings we report are therefore rankings of models under this tuning protocol, not "
           "statements about the intrinsic quality of the architectures. For the deep models, trial 0 is the configuration published in [@elsayed2024], so "
           "tuning can only improve on the original setting as measured on validation data. The search space "
           "covers the original structural hyperparameters and, for all deep models, the learning rate "
           "({3×10⁻⁴, 10⁻³, 3×10⁻³}) and gradient clipping (none or a global norm of 1); the FT-Transformer has "
           "its own space around its published defaults. To keep the budget "
           "affordable, each trial is trained for one quarter of the final number of epochs, and the configuration "
           "with the best validation accuracy at the last epoch is selected. For the tree models, trial 0 is the "
           f"library default. Table {P.next_table()} lists the search spaces.")
    P.columns(1)
    P.table(["Model", "Search space"],
            [["DNN", "units per layer ∈ {8, 16, 32, 50, 64, 128} (two layers)"],
             ["SimpleRNN / LSTM / GRU", "three recurrent layers with {8, …, 128} units; dropout ∈ {0.1, …, 0.5}; "
                                        "dense units ∈ {16, 25, 32, 50, 64}"],
             ["CNN", "two Conv1D layers with {8, 16, 32, 64} filters and kernel size ∈ {3, 5, 7}; "
                     "dropout ∈ {0.1, …, 0.5}; dense units ∈ {16, …, 64}"],
             ["CNN-LSTM", "Conv1D with {8, 16, 32, 64} filters, kernel ∈ {3, 5, 7}; LSTM units ∈ {32, 50, 64, 128}; "
                          "dropout ∈ {0.1, …, 0.5}"],
             ["All six models of [@elsayed2024]", "learning rate ∈ {3×10⁻⁴, 10⁻³, 3×10⁻³}; clipnorm ∈ {none, 1}"],
             ["FT-Transformer", "blocks ∈ {1, 2, 3}; token dimension ∈ {64, 128, 192, 256}; attention and "
                                "feed-forward dropout ∈ {0, 0.1, 0.2, 0.3}; residual dropout ∈ {0, 0.1}; "
                                "feed-forward factor ∈ {2/3, 4/3, 2}; weight decay ∈ {10⁻⁶, 10⁻⁵, 10⁻⁴}; "
                                "learning rate ∈ {3×10⁻⁵, 10⁻⁴, 3×10⁻⁴, 10⁻³}; clipnorm ∈ {none, 1}"],
             ["Random forest", "trees ∈ {100, 200, 300}; depth ∈ {none, 10, 20, 40}; min. leaf ∈ {1, 2, 5}; "
                               "max. features ∈ {√d, log₂ d, 0.5d}; sample fraction ∈ {0.3, 0.6, 1}"],
             ["XGBoost", "trees ∈ {100, 200, 400}; depth ∈ {3, 6, 9}; learning rate ∈ {0.03, 0.1, 0.3}; "
                         "subsample ∈ {0.7, 1}; column sample ∈ {0.5, 0.8, 1}; min. child weight ∈ {1, 5}"]],
            [3.4, 13.6], "Hyperparameter search spaces (20 random-search trials each).", left_cols=(0, 1))
    P.columns(2)

    P.subsection("Training Protocol and Repeated Runs")
    P.para("The six models of [@elsayed2024] are trained with Adam and the FT-Transformer with AdamW, all with a "
           "batch size of 5,000. NSL-KDD and UNSW-NB15 models are "
           "trained for 100 epochs as in [@elsayed2024]. A CIC-IDS2017 epoch contains about fifteen times as many "
           "weight updates as an NSL-KDD epoch, so we train CIC-IDS2017 models for 20 epochs, which still gives "
           "about 4,700 updates compared with about 1,700 on NSL-KDD; CSE-CIC-IDS2018 models are trained for 12 "
           "epochs (about 4,600 updates on the random split). Each configuration is trained with five "
           "seeds on NSL-KDD and UNSW-NB15 and with three seeds on CIC-IDS2017 and CSE-CIC-IDS2018; the data split "
           "is fixed, so the seed only affects initialization and data order. Besides the tuned models, we train "
           "three control groups: the original hyperparameters; the original hyperparameters with gradient "
           "clipping (gated recurrent models only); and the tuned models with the input features randomly permuted "
           "(five permutations, binary task).")
    P.para("The experiments were run on three kinds of hardware. On NSL-KDD, UNSW-NB15, and CIC-IDS2017, "
           "convolutional and gated recurrent models were trained on an NVIDIA RTX 4050 laptop GPU, and DNN and "
           "SimpleRNN on an Apple M6 CPU, which was six times faster for SimpleRNN. On CSE-CIC-IDS2018, the "
           "convolutional and gated recurrent models, and the FT-Transformer on all four datasets, were trained on "
           "a rented NVIDIA RTX 5090 GPU; DNN, SimpleRNN, and the tree models remained on the CPU. Within a "
           "dataset, all runs of a given model use the same device. The FT-Transformer used TensorFloat-32 matrix "
           "multiplication on the RTX 5090, which we checked against full FP32 precision; all other GPU runs used "
           "FP32. We quantify the effect of the device in Section VI. Software versions are listed in the "
           "artifact.")

    P.subsection("Metrics and Statistical Analysis")
    P.para("We report accuracy, macro-averaged F1, the attack detection rate, the false-alarm rate, and, where "
           "applicable, the detection rate on attack types unseen during training. For each pair of models we "
           "apply Welch's t-test [@welch1947] with Holm's correction [@holm1979]. To quantify how misleading a "
           "single-run comparison can be, we define the flip probability of models *A* and *B* with seed scores "
           "a₁,…,aₙ and b₁,…,bₘ as the fraction of single-run pairs whose order contradicts the order of the means:")
    P.equation("p_flip(A, B) = (1 / nm) Σᵢ Σⱼ 𝟙[sign(aᵢ − bⱼ) ≠ sign(ā − b̄)]", 1)
    P.para("We compare model rankings across test distributions with Kendall's τ [@kendall1938]. A training run is "
           "said to *collapse* in an epoch when its validation accuracy falls more than five points below the "
           "highest value reached so far. During training we record the L2 norm of the raw (unclipped) gradient "
           "of every weight update and keep its maximum and mean per epoch.")

    P.subsection("Shortcut Analysis")
    P.para("For the tuned binary DNN of each dataset we compute SHAP values [@lundberg2017] with the permutation "
           "explainer on 500 records per test set, using 100 training records as background, and sum the values "
           "of one-hot columns back to the original features. We then retrain the model on the top-*k* "
           "features (k ∈ {1, 2, 3, 5, 10}) and without the top one or three features, with three seeds each, "
           "to test whether the features that dominate the in-distribution score also carry the "
           "out-of-distribution performance [@geirhos2020].")

    P.subsection("Adversarial Evaluation")
    P.para("The attacker modifies attack records so that they are classified as normal. We first consider two "
           "threat models. In the *unconstrained* model every numeric feature may change. In the *realistic* model "
           "[@apruzzese2022] only features under the attacker's control may change (13 features for NSL-KDD, "
           "such as duration, source bytes, and connection counts; 13 forward-direction features for "
           "the two CIC datasets, such as forward packets, forward bytes, inter-arrival times, and the initial "
           "TCP window); "
           "all of them except the initial TCP window can only increase, because traffic that has been sent "
           "cannot be withdrawn, whereas the window is chosen by the sender and may be raised or lowered; count "
           "features must remain non-negative integers, and all values must stay within the range observed in "
           "training (for the TCP window, also within 0–65,535). "
           "Perturbations are bounded in the L∞ norm, measured in standard deviations, with "
           "ε ∈ {0.1, 0.25, 0.5, 1, 2}. We run PGD [@madry2018] with 20 steps and a random start, and a "
           "gradient-free random search with 300 queries; a gradient-free attack that is much stronger than PGD "
           "indicates gradient masking [@athalye2018]; we report the stronger of the two attacks. Attacks are "
           "evaluated on 2,000 attack records per model and seed that the model detects correctly, with three "
           "seeds, for the tuned binary models of the six architectures of [@elsayed2024] on NSL-KDD and on the "
           "corrected random splits of CIC-IDS2017 and CSE-CIC-IDS2018 (the FT-Transformer and the tree ensembles "
           "are not attacked; trees are not differentiable, and the gradient-free attack alone would not be "
           "comparable); the "
           "evasion rate is the fraction of these records that the attack turns into a normal prediction. To "
           "measure black-box transfer, examples crafted with PGD on the DNN are applied to the other five "
           "models. Finally, we adversarially train every model with three-step PGD under the realistic threat "
           "model at ε = 1, using the tuned hyperparameters and the same number of epochs as standard training, "
           "and evaluate it with both attacks at ε = 1 and ε = 2.")
    P.para("The realistic threat model leaves derived features, such as means and rates, unchanged when the "
           "primary features they are computed from are perturbed, so it admits feature vectors that no flow "
           "could produce. For the two CIC datasets we therefore add a *consistent* threat model. The attacker "
           "controls 11 primary forward-direction quantities (forward packets, forward bytes, maximum forward "
           "packet length, forward header length, flow duration, forward inter-arrival time total and maximum, "
           "the forward PSH and URG flags, the number of forward packets with payload, and the initial TCP "
           "window); the subflow features, which the corrected releases report as per-subflow averages that "
           "cannot be derived from the totals, are frozen. After every "
           "attack step, 14 derived features (mean packet lengths and segment sizes, the four rate features, the "
           "down/up ratio, the mean inter-arrival times, the maximum packet length, and the flag counts) are "
           "recomputed from the perturbed primaries. To avoid the loss of precision of very small durations in "
           "standardized space, the rates are updated multiplicatively, as the original rate times the ratio of "
           "the new to the old numerator and denominator. Simple feasibility constraints are enforced as well: every added "
           "packet carries at least 8 bytes of header, the duration and forward inter-arrival times can grow only "
           "for flows with at least two packets (two forward packets for the inter-arrival times), inter-arrival "
           "times never exceed the duration, means never exceed maxima, and flag and payload-packet counts never "
           "exceed the number of forward packets. We verified the recomputation formulas on the raw flows of both datasets, where "
           "they reproduce the released derived features for all records, and every attack run checks that "
           "recomputing the derived features of an unperturbed record returns it unchanged (maximum deviation "
           "below 10⁻⁵ standard deviations, the rounding error of single-precision arithmetic). The budget ε "
           "bounds only the primary features; the derived features follow from them and are not bounded "
           "separately. The same ε therefore describes different perturbation sets under the realistic and the "
           "consistent threat model, and evasion rates at equal ε are comparable within a threat model but not "
           "directly across threat models. As a diagnostic, a *consistent, timing frozen* variant additionally "
           "fixes the three timing features, since delaying packets is the cheapest change for an attacker but "
           "also slows the attack down. We attack the standard and the adversarially trained models under both "
           "variants and, on CSE-CIC-IDS2018, also adversarially train the convolutional and gated recurrent "
           "models under the consistent threat model.")
    P.para("Even the consistent model operates on flow features rather than on packets: it does not model the "
           "backward direction, which the attacker influences only indirectly, nor protocol semantics such as "
           "valid TCP state sequences. Its evasion rates are therefore still an upper bound on what is "
           "achievable by modifying packets, but a much tighter one than that of the realistic model.")

    P.subsection("Reproducibility Artifact")
    P.para("We release the complete code, the search spaces and selected hyperparameters, all seeds, the "
           "preprocessing scripts with download instructions for the datasets, and the raw per-run results "
           "(training curves, gradient norms, and all metrics) at "
           "https://github.com/chungweichang0918/nids-dl-reliability, archived on Zenodo under DOI "
           "10.5281/zenodo.24391814. The trained model weights are archived separately under DOI "
           "10.5281/zenodo.24391892.")


def scaler_sentence():
    """隨機切分條件的標準化以整個訓練檔擬合 (沿用 [@elsayed2024])，對照實驗只用訓練部分擬合。"""
    leak = V2 / "analysis" / "scaler_leak.json"
    ctrl = V2 / "scaler_control" / "summary.json"
    s = ("In the random conditions the scaler is fitted on the whole training file before the 80/20 split, as in "
         "[@elsayed2024], so the random test set contributes to the standardization statistics; the official "
         "test sets and the temporal conditions are not affected, and the tree ensembles are invariant to this "
         "per-feature scaling. ")
    if leak.exists():
        L = json.loads(leak.read_text(encoding="utf-8"))
        med = [v["test_shift"]["median"] for v in L.values()]
        dstd = [v["max_dstd"] for v in L.values()]
        s += ("Fitting the scaler on the training part only would change the standardized test values by a "
              f"median of {min(med):.4f}–{max(med):.4f} standard deviations, but individual heavy-tailed features "
              f"change their standard deviation by up to {max(dstd) * 100:.0f}%. ")
    if ctrl.exists():
        C = json.loads(ctrl.read_text(encoding="utf-8"))
        d_rnd = [abs(v["diff"]) for k, v in C.items() if k.endswith("/random")]
        d_off = [abs(v["diff"]) for k, v in C.items() if not k.endswith("/random")]
        sd_off = [v["sd_full"] for k, v in C.items() if not k.endswith("/random")]
        ps = [v["p"] for v in C.values() if v["p"] is not None]
        models = sorted({k.split("/")[1] for k in C})
        s += (f"Retraining {' and '.join(models)} with a scaler fitted on the training part only, with the same "
              f"hyperparameters and seeds, changes the mean accuracy on the random test sets of the four datasets "
              f"by at most {max(d_rnd):.2f} points")
        if d_off:
            s += (f" and on the official NSL-KDD and UNSW-NB15 test sets by at most {max(d_off):.2f} points, "
                  f"within the seed-to-seed standard deviation there ({min(sd_off):.2f}–{max(sd_off):.2f})")
        s += (f"; no difference is significant (Welch p ≥ {min(ps):.2f}), so the leak does not account for the "
              "comparisons reported here, although with three to five seeds small effects cannot be excluded.")
    else:
        s += "{{Train-only scaler control (scaler_control.py, DNN and CNN): pending}}"
    return s


def subsample_sentence():
    """第二份 CSE-CIC-IDS2018 樣本 (不同隨機種子) 與主要樣本的比較。"""
    sub = load_results().get("subsample", {})
    if not sub:
        return "{{cic18b: second-subsample comparison pending}}"
    rnd = [v for k, v in sub.items() if k.startswith("improved/")]
    tmp = [v for k, v in sub.items() if k.startswith("temporal/")]
    d = lambda vs: max(abs(v["a"]["mean"] - v["b"]["mean"]) for v in vs)
    models = sorted({k.split("/")[2] for k in sub}, key=["RF", "XGB", "DNN"].index)
    names = {"RF": "random forest", "XGB": "XGBoost", "DNN": "DNN"}
    return ("To check that the conclusions do not depend on the particular subsample, we drew a second one with "
            f"a different seed and repeated the experiments for {', '.join(names[m] for m in models[:-1])} and "
            f"{names[models[-1]]} (tuning included). The mean accuracies of the two subsamples differ by at most "
            f"{d(rnd):.3f} points on the random split and {d(tmp):.3f} points on the temporal split, where "
            "every model again behaves as a constant benign classifier; with seed-to-seed standard deviations "
            "close to zero, some of these differences are formally significant, but none changes any "
            "conclusion.")


def discussion(P, Y):
    H = headline()
    S = load_results()
    acc = S["rq1"]["nsl/default/binary/random"]["accuracy"]
    deep_spread = max(acc[m]["mean"] for m in acc if m not in ("LSTM", "RF", "XGB")) - \
        min(acc[m]["mean"] for m in acc if m not in ("LSTM", "RF", "XGB"))
    P.section("Discussion")
    P.subsection("Implications for NIDS Research")
    P.para("**Evaluation choices matter more than architectures.** The largest effects in this study are not "
           "differences between models but differences between evaluation protocols. On NSL-KDD the deep models "
           f"other than LSTM differ by {deep_spread:.2f} points on the random split, while replacing that split "
           f"by the official test set costs {H['nsl_gap'][0]:.0f}–{H['nsl_gap'][1]:.0f} points; on CIC-IDS2017 "
           f"the corrected random split saturates every model at {H['i_f1'][0]:.1f}% macro-F1 or more, while the "
           f"temporal split lowers macro-F1 to {H['t_f1'][0]:.0f}–{H['t_f1'][1]:.0f}%, and on CSE-CIC-IDS2018 it "
           "turns all nine models into constant benign classifiers. In our reproduction of the second case "
           f"study, selecting hyperparameters on the test set adds {Y['binary']['gap'].mean():.1f} points of "
           "optimism, more than the gap between most pairs of models. A claim that one architecture is better suited to intrusion detection must "
           "therefore be supported under the evaluation condition that matches its intended use, and the size of "
           "the claimed difference must be compared with the seed-to-seed variability and with the effect of the "
           "protocol.")
    P.para("**No architecture is best across conditions.** The tree ensembles are the most accurate models on "
           "every in-distribution test and remain competitive out of distribution, at a small fraction of the "
           "training cost. Where a deep model is ahead, the advantage is specific to one dataset and test set, "
           "such as CNN under the temporal shift of CIC-IDS2017, and it does not coincide with the ranking on the "
           "random split. A modern tabular architecture does not change this: the FT-Transformer is the best deep "
           "model in distribution but not out of distribution. The gated recurrent models, which the first case "
           "study ranked highest, are the least "
           "stable: their reproduction failed because of exploding gradients, and an affordable tuning protocol "
           "can reintroduce the same failure. Part of what single-run comparisons report as architectural "
           "differences is therefore a difference in optimization luck.")
    P.para("**High benchmark accuracy does not imply that the attack has been learned.** The shortcut analysis "
           "shows that a single feature such as *service* on NSL-KDD or the time-to-live on UNSW-NB15 carries most "
           "of the out-of-distribution score, and that on CIC-IDS2017 two features learned from the training days "
           "suppress the detection of port scans. These are properties of the capture, not of the attacks, and an "
           "in-distribution evaluation rewards them. Robustness, too, is a property of the dataset as much as of "
           f"the model: under realistic constraints, attacks evade {H['nsl_r'][0]:.1f}–{H['nsl_r'][1]:.1f}% of "
           "the detections of five of the six architectures on NSL-KDD but "
           f"{H['cic_r'][0]:.0f}–{H['cic_r'][1]:.0f}% of those of all six on CIC-IDS2017, depending on whether "
           "the informative features are under the attacker's control. "
           "Robustness claims based only on unconstrained attacks or only on PGD would be misleading in both "
           "directions. The same holds for defenses: adversarial training appeared to remove the realistic "
           "threat, but a threat model that keeps the flow features consistent evades the adversarially trained "
           "models almost as easily as the standard ones, and training under the consistent threat model in turn "
           "leaves the models vulnerable to the realistic attack. Because the realistic attack can produce "
           "inconsistent feature vectors, the latter is a vulnerability in feature space rather than evidence of an "
           "attack that is realizable in traffic; what both results show is that robustness does not carry over "
           "between threat models. A defense must be evaluated under threat models other than the one it "
           "was trained on, and preferably under the most faithful one available.")
    P.subsection("Recommendations")
    P.para("Based on our experiments we recommend the following practices for comparative NIDS studies:",
           indent=False)
    P.bullets([
        "Train every configuration with several random seeds and report the mean, the dispersion, and the "
        "outcome of a statistical test before ranking models.",
        "Give every model, including strong tree-based baselines, the same tuning budget, and report both the "
        "number of trials and their computational cost.",
        "Fit all preprocessing, including standardization, on the training data only.",
        "Evaluate on an independent or temporally later test set in addition to, or instead of, a random split of "
        "the training capture.",
        "Select hyperparameters and models on validation data only; touch the test set once, for the final "
        "evaluation.",
        "Remove exact duplicates before splitting, and prefer corrected dataset releases where available.",
        "Report per-class and macro-averaged metrics, and the detection rate on attack types unseen in training.",
        "For recurrent models, report learning rate and gradient clipping, and monitor gradient norms; when "
        "tuning with shortened training, check the top few configurations over the full schedule and with "
        "several seeds before selecting one.",
        "Check which features drive the predictions, and test whether the model still works without "
        "features that describe the capture environment rather than the attack.",
        "Evaluate robustness under domain constraints that keep derived features consistent, with both a "
        "gradient-based and a gradient-free attack; report black-box transfer, evaluate defenses under threat "
        "models other than the one used for training, and report their clean-accuracy cost on an "
        "out-of-distribution test set.",
        "Release code, configurations, seeds, and per-run results.",
    ], numbered=True)
    P.subsection("Threats to Validity")
    P.para("*Internal validity.* Tuning trials used one quarter of the final epochs; configurations that are "
           "stable early may become unstable later, which we observed for one model and report as a finding; "
           "retraining the next-ranked configurations shows that the failure is tied to the selected learning "
           "rate. Deep models were trained on different devices; all runs of a model on a dataset share one "
           "device, so comparisons across seeds of the same model are unaffected, but cross-model comparisons "
           "include a hardware difference. To bound it, we retrained the tuned binary NSL-KDD models on other "
           "devices with the same hyperparameters and five seeds. The DNN, originally trained on the Apple CPU "
           "(99.56% on the random split, 78.22% on KDDTest+), reaches 99.56% and 77.46% on the laptop GPU and "
           "99.53% and 77.47% on the RTX 5090; the GRU, originally trained on the laptop GPU (99.18% and 79.06%), "
           "reaches 99.17% and 77.79% on the RTX 5090; and the FT-Transformer trained in full FP32 precision "
           "instead of TensorFloat-32 reaches 99.61% and 78.07% instead of 99.57% and 77.56%. None of these "
           "differences is significant (Welch p ≥ 0.22). The device effects on KDDTest+, up to 1.3 points, are "
           "of the same size as the seed-to-seed standard deviation on that test set (0.8–2.0 points), which "
           "is a further reason to compare models only over several seeds. Recurrent models used gradient "
           "accumulation, which we verified to be equivalent within seed variability. CIC-IDS2017 and "
           "CSE-CIC-IDS2018 were run with three seeds because of their size, which "
           "limits the power of the pairwise tests on these datasets; the absence of significant differences on the "
           "corrected release should be read together with the very small differences between the means. The "
           "adversarial evaluation covers the binary task, the six architectures of [@elsayed2024], and the random "
           "splits only, and the clean-accuracy cost "
           "of adversarial training on the CIC datasets was measured on the random split, not under temporal "
           "shift.")
    P.para("*Preprocessing.* " + scaler_sentence())
    P.para("*Sampling.* CSE-CIC-IDS2018 was subsampled, CIC-IDS2017 was not. The subsample keeps every attack "
           "label with fewer than 300,000 flows complete and samples the benign traffic uniformly within each "
           "day, so the rare attack types and the day structure are preserved, but the class ratio differs from "
           "that of the full release. " + subsample_sentence())
    P.para("*External validity.* The datasets were captured between 1998 and 2018 in laboratory settings and may "
           "not represent current traffic. We re-examined two published studies, one in depth; they exhibit "
           "different pitfalls, but whether these pitfalls are as prevalent in the wider literature requires a "
           "larger survey. The second reproduction depends on our interpretation of unreported details "
           "(Section III-C).")
    P.para("*Construct validity.* Our adversarial attacks operate on flow features. The consistent threat model "
           "recomputes the derived forward-direction features and enforces physical feasibility, but it does not "
           "model the reaction of the victim, which determines the backward-direction features, so its evasion "
           "rates still over-approximate what an attacker can achieve by modifying packets. Accuracy is a "
           "poor summary under class imbalance, which is why we also report macro-F1 and per-attack detection.")


def conclusion(P, Y):
    H = headline()
    P.section("Conclusion")
    P.para("We reproduced two published comparisons of deep intrusion detectors and re-evaluated their models, "
           "together with the FT-Transformer and tree ensembles, on four benchmarks with equal trial budgets for "
           "tuning, repeated runs, and eight evaluation conditions. The ranking that motivated the first study does "
           f"not survive this treatment: single runs reverse the order of a model pair in {H['flip']:.0f}% of cases "
           f"on average, the GRU ranks {ORD[H['gru_rank']]} of {NUM[H['n_models']]} once every model is tuned with "
           "the same number of trials, and the instability "
           "behind the failed reproduction is caused by exploding gradients rather than by the architecture. Under "
           "our reproduction assumptions, selecting hyperparameters on the test set, as the second study did, "
           f"yields {Y['binary']['gap'].mean():.1f} points higher binary accuracy than validation-based selection. "
           "Changes of the evaluation protocol move accuracy and "
           "macro-F1 by tens of points, far more than any architectural choice; on CSE-CIC-IDS2018 a temporal "
           f"split leaves every model detecting less than {np.ceil(H['t18_det']):.0f}% of the unseen attacks. Part "
           "of the remaining out-of-distribution error stems from shortcut features of the capture. Realistic "
           "adversarial "
           f"attacks evade {H['cic_r'][0]:.0f}–{H['cic_r'][1]:.0f}% of the detections on CIC-IDS2017 and "
           "transfer between models. Adversarial training appears to remove most of this threat at a small cost "
           "in clean accuracy, but an attack that keeps the derived flow features consistent still evades the "
           "adversarially trained models in most cases, and training against that attack does not protect "
           "against the original one, so robustness measured under a defense's own threat model does not carry "
           "over to another. "
           "We hope that the recommendations in Section VI and the released "
           "artifact make future comparisons easier to trust. Validating the adversarial results at the packet "
           "level, including the victim's responses, is the most important next step.")


def acknowledgment(P):
    """IEEE 規定: 使用 AI 產生的內容須在致謝中揭露系統、用途與程度。內容經作者確認。"""
    P.unnumbered_section("Acknowledgment")
    P.para("Use of generative AI. The authors designed the study, the research questions, and the evaluation "
           "conditions. They used Claude (Anthropic; Claude Opus 5.5, accessed through Claude Code) as a coding "
           "and writing assistant. Under the authors' direction, the AI system wrote most of the experiment and "
           "analysis code (data preprocessing, model training, hyperparameter tuning, adversarial attacks, and "
           "statistical tests) and the scripts that generate the figures and tables, and drafted the English text "
           "of all sections from the results produced by these scripts. The authors reviewed and verified all "
           "code and text produced by the AI system, decided the direction of every revision, which the AI system "
           "then implemented, and ran the experiments. All numerical results come from the released code and raw "
           "per-run results, not from the AI system. The authors take full responsibility for the content of "
           "this article.")


def build():
    P = Paper(REFS)
    Y = yin_stats()
    front_matter(P, Y)
    P.columns(2)
    introduction(P)
    related_work(P)
    case_study(P, Y)
    methodology(P)
    results(P)
    discussion(P, Y)
    conclusion(P, Y)
    acknowledgment(P)
    unused = P.references()
    P.save(OUTFILE)
    print(f"已輸出 {OUTFILE}  (圖 {P.fig_no}、表 {P.tab_no}、引用 {len(P.order)} 篇)")
    if unused:
        print("未引用的參考文獻:", unused)


if __name__ == "__main__":
    build()


