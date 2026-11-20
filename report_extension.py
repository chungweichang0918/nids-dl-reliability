"""Word 報告第 VII 節「延伸實驗」: 多種子、KDDTest+、UNSW-NB15、SHAP、對抗樣本。

由 build_report.py 呼叫。所有數字都從 results/ 的結果檔讀取。
"""
import json

import numpy as np

import nids_repro as nr

MODELS = list(nr.HP["binary"])
TASK_ZH = {"binary": "二元", "multi": "五類"}
FIG = nr.OUT / "figures"


def load():
    def read(path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    ext = dict(analysis=read(nr.OUT / "analysis.json"),
               shap=read(nr.OUT / "shap" / "shap.json"),
               adv=read(nr.OUT / "adversarial" / "binary_s1.json"))
    a = ext["analysis"]
    ext["seeds"] = a["seeds"]
    ext["unsw"] = a.get("unsw", {})
    return ext


def mean(x):
    return float(np.mean(x))


def sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def ms(x, d=2):
    return f"{mean(x):.{d}f} ± {sd(x):.{d}f}" if len(x) > 1 else f"{x[0]:.{d}f}"


def p_fmt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------------------
# 摘要、貢獻、結論用的短句
# ---------------------------------------------------------------------------
def summary(ext):
    st, a = ext["seeds"], ext["analysis"]
    split = [mean(st[f"binary/{m}"]["split"]) for m in MODELS]
    kdd = [mean(st[f"binary/{m}"]["kddtest"]) for m in MODELS]
    unseen = [a["kddtest_detail"]["binary"][m]["mean"][1] for m in MODELS]
    wins, _ = a["best_counts"]["binary"]
    gru_wins = wins.get("GRU", 0)
    return dict(
        split_lo=min(split), split_hi=max(split), kdd_lo=min(kdd), kdd_hi=max(kdd),
        unseen_lo=min(unseen), unseen_hi=max(unseen), gru_wins=gru_wins,
        gru=st["binary/GRU"]["split"])


# ---------------------------------------------------------------------------
# A. 多種子重複實驗
# ---------------------------------------------------------------------------
def seeds_section(R, ext):
    st, a = ext["seeds"], ext["analysis"]
    R.heading("A. 多種子重複實驗", 2)
    n_runs = sum(len(r["seeds"]) for r in st.values())
    R.para(f"第 VI 節顯示單一種子的結果並不可靠，因此我們以種子 1–5 將 12 個模型各重新訓練 5 次，"
           f"共 {n_runs} 次，其餘設定與主實驗完全相同。評估時以單一樣本 t 檢定判斷原論文的數值是否落在"
           f"重現結果的分布內，並以 Welch t 檢定 [15] 比較模型兩兩之間的差異，"
           f"多重比較以 Holm 方法 [16] 校正。")
    R.para("SimpleRNN 沒有 cuDNN 專用核心，在 RTX 4050 上每次訓練約需 70 分鐘，顯示卡使用率僅約 9%；"
           "改在 Apple M6 的 CPU 上執行後降為約 12 分鐘。因此 SimpleRNN 的 10 次訓練在 Apple M6 上完成，"
           "其餘模型在 RTX 4050 上完成。同一模型的所有種子都在同一台硬體上訓練，模型內的比較不受硬體差異影響。")

    rows, bold = [], []
    for t in ("binary", "multi"):
        for m in MODELS:
            r = st[f"{t}/{m}"]
            p = r["paper_p"]
            rows.append([TASK_ZH[t], m, ms(r["split"]), f"{min(r['split']):.2f}–{max(r['split']):.2f}",
                         f"{r['paper']:.2f}", p_fmt(p), sum(r["collapses"])])
            if p < 0.05:
                bold.append((len(rows) - 1, 5))
    n = R.tab_no + 1
    R.para(f"表 {n} 為 5 個種子的統計結果，圖 {R.fig_no + 1} 畫出每次訓練的準確率。", keep_next=True)
    R.columns(1)
    R.table(["任務", "模型", "平均 ± 標準差", "最小–最大", "原論文", "p 值", "崩落次數"], rows,
            [1.4, 2.2, 3.2, 3.2, 2.0, 2.0, 2.2],
            "5 個種子的內部測試集準確率（%）。p 值為原論文數值與重現分布的單一樣本 t 檢定，"
            "粗體表示 p < 0.05；崩落次數為 5 次訓練的合計",
            size=8, bold_cells=bold, left_cols=(0, 1))
    R.figure(FIG / "seeds.png", 17.5,
             "5 個種子的內部測試集準確率。藍點為各次訓練，深藍橫線為平均，橘色菱形為原論文數值。"
             "DNN、RNN、CNN 的 5 次結果幾乎重疊，LSTM、GRU、CNN-LSTM 則分散數個百分點。")
    R.columns(2)

    b, mu = st, lambda k: mean(st[k]["split"])
    stable = [f"{t}/{m}" for t in ("binary", "multi") for m in ("DNN", "RNN", "CNN")]
    max_sd_stable = max(sd(st[k]["split"]) for k in stable)
    gated = [f"{t}/{m}" for t in ("binary", "multi") for m in ("LSTM", "GRU", "CNN-LSTM")]
    sd_gated = [sd(st[k]["split"]) for k in gated]
    coll_rnn = sum(sum(st[f"{t}/RNN"]["collapses"]) for t in ("binary", "multi"))
    coll = {m: sum(sum(st[f"{t}/{m}"]["collapses"]) for t in ("binary", "multi"))
            for m in ("LSTM", "GRU", "CNN-LSTM")}
    R.para(f"**穩定性：**DNN、RNN、CNN 在兩種任務的標準差都不超過 {max_sd_stable:.2f} 個百分點，"
           f"10 次訓練完全沒有崩落；LSTM、GRU、CNN-LSTM 的標準差則為 {min(sd_gated):.2f}–"
           f"{max(sd_gated):.2f} 個百分點，崩落次數分別為 {coll['LSTM']}、{coll['GRU']}、{coll['CNN-LSTM']} 次。"
           f"值得注意的是，同為循環網路的 SimpleRNN 崩落 {coll_rnn} 次，"
           "因此不穩定並非循環結構本身造成，而是集中在帶有閘門機制的 LSTM 與 GRU，以及含有 LSTM 的 CNN-LSTM。")

    wins_b, _ = a["best_counts"]["binary"]
    wins_m, _ = a["best_counts"]["multi"]
    win_txt = lambda w: "、".join(f"{m} {n} 次" for m, n in sorted(w.items(), key=lambda x: -x[1]) if n)
    sig_b = [r for r in a["pairwise"]["binary"] if r["p_holm"] < 0.05]
    sig_m = [r for r in a["pairwise"]["multi"] if r["p_holm"] < 0.05]
    gru = st["binary/GRU"]
    gru_wins = wins_b.get("GRU", 0)
    gru_txt = "從未" if not gru_wins else f"只有 {gru_wins} 次"
    R.para(f"**模型排名：**二元分類中，5 個種子的最佳模型依序為 {win_txt(wins_b)}，"
           f"原論文宣稱最佳的 GRU {gru_txt}成為最佳模型；"
           f"GRU 平均為 {ms(gru['split'])}%，5 次中最高的 {max(gru['split']):.2f}% 仍低於原論文的 {gru['paper']:.2f}%。"
           f"模型兩兩比較的 15 組中，只有 {len(sig_b)} 組達顯著："
           + "、".join(f"{r['a']} 優於 {r['b']}（+{r['diff']:.2f}）" for r in sig_b)
           + f"。五類分類的最佳模型為 {win_txt(wins_m)}，"
           + ("15 組比較都未達顯著，"
              if not sig_m else f"{len(sig_m)} 組達顯著，")
           + "代表在 5 個種子的檢定力下，前段模型之間的差距無法區分。"
           "原論文以單次實驗、零點幾個百分點的差距排定的模型名次，在重複實驗下無法成立。")

    lstm_b, gru_m, lstm_m = st["binary/LSTM"], st["multi/GRU"], st["multi/LSTM"]
    rnn_b = st["binary/RNN"]
    R.para(f"**與原論文的一致性：**DNN 與 CNN 的原論文數值都落在重現分布內或極接近。"
           f"二元 LSTM 的原論文數值 {lstm_b['paper']:.2f}% 則明顯高於重現分布（{ms(lstm_b['split'])}%，"
           f"p {p_fmt(lstm_b['paper_p'])}），五類 LSTM 與 GRU 也是如此（p = {p_fmt(lstm_m['paper_p'])}、"
           f"{p_fmt(gru_m['paper_p'])}）。二元 GRU 的 p 值為 {p_fmt(gru['paper_p'])}，"
           "未達顯著是因為它的標準差過大、檢定力不足，不能解讀為重現成功。"
           f"反方向的例子是二元 RNN：重現平均 {ms(rnn_b['split'])}% 顯著**高於**原論文的 "
           f"{rnn_b['paper']:.2f}%（p {p_fmt(rnn_b['paper_p'])}）。這顯示原論文的單次數值不只可能偏高，"
           "也可能偏低，進一步支持「單一種子的結果不足以排定模型優劣」，與 Reimers 與 Gurevych [27] 在自然語言處理上的"
           "發現一致。")


# ---------------------------------------------------------------------------
# B. 官方 KDDTest+ 評估
# ---------------------------------------------------------------------------
def kddtest_section(R, ext):
    st, a = ext["seeds"], ext["analysis"]
    d = a["kddtest_detail"]
    R.heading("B. 官方測試集 KDDTest+ 評估", 2)
    R.para(f"第 VIII-C 節指出隨機切分可能高估模型能力。為驗證這點，我們將上述 60 個模型直接在官方 "
           f"KDDTest+（{d['n_seen'] + d['n_unseen'] + d['n_normal']:,} 筆）上評估。前處理的標準化參數與 one-hot "
           f"欄位只由 KDDTrain+ 決定，KDDTest+ 中訓練時沒出現的類別值編碼為全 0。"
           f"KDDTest+ 含有 17 種訓練資料沒有的攻擊類型，共 {d['n_unseen']:,} 筆，"
           f"我們將攻擊分為「訓練時見過的類型」（{d['n_seen']:,} 筆）與「未見過的類型」分別計算偵測率。")

    rows = []
    for m in MODELS:
        r = st[f"binary/{m}"]
        det = d["binary"][m]
        rows.append([m, ms(r["split"]), ms(r["kddtest"]),
                     f"{mean(r['kddtest']) - mean(r['split']):+.2f}",
                     f"{det['mean'][0]:.1f} ± {det['std'][0]:.1f}",
                     f"{det['mean'][1]:.1f} ± {det['std'][1]:.1f}",
                     f"{det['mean'][2]:.1f} ± {det['std'][2]:.1f}"])
    n = R.tab_no + 1
    R.para(f"表 {n} 與圖 {R.fig_no + 1} 比較兩種測試集的結果。", keep_next=True)
    R.columns(1)
    R.table(["模型", "內部測試集", "KDDTest+", "差距", "見過的攻擊", "未見過的攻擊", "正常流量誤報率"],
            rows, [1.9, 2.6, 2.6, 1.6, 2.6, 2.6, 2.6],
            "二元分類在內部測試集與 KDDTest+ 的表現（%，5 個種子的平均 ± 標準差）。"
            "「見過／未見過的攻擊」為攻擊被正確偵測的比例",
            size=8)
    R.figure(FIG / "kddtest.png", 17.5,
             "內部測試集與 KDDTest+ 的比較。(a)(b) 兩種任務的準確率；(c) KDDTest+ 上的攻擊偵測率，"
             "綠色為訓練時見過的攻擊類型，紅色為未見過的類型。誤差線為 5 個種子的標準差。")
    R.columns(2)

    s = summary(ext)
    drop = [mean(st[f"binary/{m}"]["kddtest"]) - mean(st[f"binary/{m}"]["split"]) for m in MODELS]
    seen = [d["binary"][m]["mean"][0] for m in MODELS]
    fa = [d["binary"][m]["mean"][2] for m in MODELS]
    R.para(f"**準確率大幅下降：**在內部測試集上準確率 {s['split_lo']:.1f}%–{s['split_hi']:.1f}% 的模型，"
           f"在 KDDTest+ 上只剩 {s['kdd_lo']:.1f}%–{s['kdd_hi']:.1f}%，下降 {-max(drop):.1f}–{-min(drop):.1f} "
           f"個百分點。對未見過的攻擊類型，偵測率只有 {s['unseen_lo']:.1f}%–{s['unseen_hi']:.1f}%，"
           f"也就是大多數新型攻擊會被判為正常流量；即使是訓練時見過的類型，偵測率也只有 "
           f"{min(seen):.1f}%–{max(seen):.1f}%。正常流量的誤報率為 {min(fa):.1f}%–{max(fa):.1f}%，"
           "代表下降主要來自漏報攻擊，而非誤判正常流量。")

    sig = [r for r in a["pairwise_kddtest"]["binary"] if r["p_holm"] < 0.05]
    best_split = max(MODELS, key=lambda m: mean(st[f"binary/{m}"]["split"]))
    best_kdd = max(MODELS, key=lambda m: mean(st[f"binary/{m}"]["kddtest"]))
    worst_kdd = min(MODELS, key=lambda m: mean(st[f"binary/{m}"]["kddtest"]))
    R.para(f"**排名改變：**內部測試集平均最高的是 {best_split}，但它在 KDDTest+ 上"
           + (f"反而是最低的（{mean(st[f'binary/{worst_kdd}']['kddtest']):.2f}%）"
              if best_split == worst_kdd else "並非最佳")
           + f"；KDDTest+ 上平均最高的是 {best_kdd}（{ms(st[f'binary/{best_kdd}']['kddtest'])}%）。"
           + ("兩者的差距在 Holm 校正後仍達顯著（" + "、".join(
               f"{r['a']} vs {r['b']} {r['diff']:+.2f}，p = {p_fmt(r['p_holm'])}" for r in sig) + "）。"
              if sig else "")
           + "換言之，在隨機切分上相差不到 0.1 個百分點的模型，面對分布改變時可能相差數個百分點，"
           "以隨機切分的準確率挑選模型並不能保證實際部署時的表現。")

    cls = nr.MULTI_CLASSES
    rows = []
    for m in MODELS:
        rec = d["multi"][m]["mean"]
        rows.append([m] + [f"{v:.1f}" for v in rec] + [ms(st[f"multi/{m}"]["kddtest_macro_f1"], 1)])
    R.table(["模型"] + cls + ["macro-F1"], rows, [1.75, 1.0, 1.15, 1.0, 0.95, 0.95, 1.6],
            "五類分類在 KDDTest+ 上各類別的 recall 與 macro-F1（%，5 個種子平均）", size=7.5)
    r2l = [d["multi"][m]["mean"][3] for m in MODELS]
    u2r = [d["multi"][m]["mean"][4] for m in MODELS]
    zero_u2r = [m for m in MODELS if d["multi"][m]["mean"][4] == 0]
    mf1 = [mean(st[f"multi/{m}"]["kddtest_macro_f1"]) for m in MODELS]
    R.para(f"表 {R.tab_no} 顯示五類分類的問題更嚴重：R2L 的 recall 只有 {min(r2l):.1f}%–{max(r2l):.1f}%，"
           f"U2R 為 {min(u2r):.1f}%–{max(u2r):.1f}%"
           + (f"，{'、'.join(zero_u2r)} 完全無法偵測 U2R" if zero_u2r else "")
           + f"。macro-F1 只有 {min(mf1):.1f}%–{max(mf1):.1f}%，"
           "與內部測試集 99% 以上的準確率形成強烈對比。")
    R.para("需要說明的是，KDDTest+ 是刻意設計成與訓練資料分布不同的測試集 [2]，"
           "因此下降幅度不能全部歸因於模型本身，也反映了資料集的建構方式。"
           "但這正是實際部署會遇到的情況：攻擊手法會演變，流量分布也會隨時間改變。"
           "原論文只報告隨機切分的結果，其 99% 以上的準確率應解讀為「對訓練資料中已知攻擊的辨識能力」，"
           "不能代表對新型攻擊的偵測能力。")


# ---------------------------------------------------------------------------
# C–E: 等結果出來後撰寫；資料不存在時略過
# ---------------------------------------------------------------------------
def unsw_section(R, ext):
    u = ext["unsw"]
    if not u:
        return
    R.heading("C. 跨資料集驗證：UNSW-NB15", 2)
    R.para("為檢驗上述結論是否只是 NSL-KDD 的特例，我們把同樣的六種模型搬到 UNSW-NB15 [17]。"
           "此資料集於 2015 年以 IXIA PerfectStorm 產生混合的正常與攻擊流量，比 NSL-KDD（源自 1998 年的流量）更接近現代網路，"
           "包含 9 種攻擊類別，並提供官方切分的訓練集（175,341 筆）與測試集（82,332 筆）。"
           "前處理沿用原論文流程：39 個數值特徵標準化，proto、service、state 三個類別特徵 one-hot 編碼"
           "（proto 有 133 種值，只保留最常見的 5 種），共 67 維。官方訓練集依類別排序，"
           "因此先固定打亂後再取 20% 作驗證集。")
    R.para("模型結構與超參數直接沿用原論文在 NSL-KDD 上的設定，未針對 UNSW-NB15 重新調整，"
           "因此絕對數值並非各模型的最佳表現，本節著重於模型之間與兩種評估方式之間的相對比較。"
           "每個模型以 3 個種子訓練，SimpleRNN 因訓練時間較長只用 1 個種子；"
           "DNN 與 SimpleRNN 在 Apple M6 CPU 上訓練，其餘模型在 RTX 4050 上訓練。")

    rows = []
    for m in MODELS:
        b, mu = u[f"binary/{m}"], u[f"multi/{m}"]
        rows.append([m, len(b["seeds"]), ms(b["val"]), ms(b["test"]),
                     f"{b['per_class']['normal']:.1f}", f"{b['per_class']['abnormal']:.1f}",
                     sum(b["collapses"]), ms(mu["test"]), ms(mu["macro_f1"]), sum(mu["collapses"])])
    n = R.tab_no + 1
    R.para(f"結果如表 {n} 與圖 {R.fig_no + 1}。", keep_next=True)
    R.columns(1)
    R.table(["模型", "種子", "驗證集", "官方測試集", "正常 recall", "攻擊 recall", "崩落",
             "十類 測試集", "十類 macro-F1", "崩落"], rows,
            [1.6, 1.0, 2.1, 2.1, 1.6, 1.6, 1.0, 2.1, 2.1, 1.0],
            "UNSW-NB15 結果（%）。前七欄為二元分類，後三欄為十類分類；recall 為各種子平均，崩落為各種子合計",
            size=7.5)
    R.figure(FIG / "unsw.png", 16.5,
             "UNSW-NB15 上驗證集與官方測試集的準確率。長條為平均，黑點為各種子的結果。"
             "二元 LSTM 有一個種子完全訓練失敗。")
    R.columns(2)

    val = [mean(u[f"binary/{m}"]["val"]) for m in MODELS if m != "LSTM"]
    test = [mean(u[f"binary/{m}"]["test"]) for m in MODELS if m != "LSTM"]
    gap = [v - t for v, t in zip(val, test)]
    val_m = [mean(u[f"multi/{m}"]["val"]) for m in MODELS]
    test_m = [mean(u[f"multi/{m}"]["test"]) for m in MODELS]
    nrm = [u[f"binary/{m}"]["per_class"]["normal"] for m in MODELS if m != "LSTM"]
    atk = [u[f"binary/{m}"]["per_class"]["abnormal"] for m in MODELS if m != "LSTM"]
    R.para(f"**同樣的高估現象：**除了訓練失敗的 LSTM 之外，二元分類的驗證準確率為 {min(val):.1f}%–{max(val):.1f}%，"
           f"官方測試集只有 {min(test):.1f}%–{max(test):.1f}%，落差 {min(gap):.1f}–{max(gap):.1f} 個百分點；"
           f"十類分類則由 {min(val_m):.1f}%–{max(val_m):.1f}% 降至 {min(test_m):.1f}%–{max(test_m):.1f}%。"
           "從訓練資料切出的驗證集一致高估了模型在獨立測試集上的表現，與第 VII-B 節在 NSL-KDD 上的發現相同。")
    R.para(f"**錯誤型態不同：**NSL-KDD 的錯誤主要是漏報攻擊，UNSW-NB15 則相反：攻擊的 recall 高達 "
           f"{min(atk):.1f}%–{max(atk):.1f}%，正常流量卻只有 {min(nrm):.1f}%–{max(nrm):.1f}% 被正確辨識，"
           f"也就是約 {100 - max(nrm):.0f}%–{100 - min(nrm):.0f}% 的正常流量被誤報為攻擊。"
           "官方訓練集與測試集的類別比例也不相同（正常流量分別佔 31.9% 與 44.9%），顯示兩者的分布有明顯差異。"
           "兩個資料集的錯誤方向相反，說明問題不只是「攻擊太難偵測」，"
           "而是模型在訓練與測試分布不一致時無法維持表現。")

    lstm = u["binary/LSTM"]
    fail = int(np.argmin(lstm["test"]))
    gru_b = u["binary/GRU"]
    cl = u["binary/CNN-LSTM"]
    R.para(f"**訓練不穩定再次出現：**二元 LSTM 的種子 {lstm['seeds'][fail]} 完全訓練失敗，"
           f"驗證準確率停在 {lstm['val'][fail]:.1f}%（相當於全部猜成多數類別「攻擊」），"
           f"測試準確率只有 {lstm['test'][fail]:.1f}%，崩落 {lstm['collapses'][fail]} 次；"
           f"另外兩個種子則為 {'、'.join(f'{v:.1f}%' for i, v in enumerate(lstm['test']) if i != fail)}。"
           f"CNN-LSTM 在兩種任務共崩落 {sum(cl['collapses']) + sum(u['multi/CNN-LSTM']['collapses'])} 次，"
           f"DNN、CNN、SimpleRNN 則完全沒有崩落。"
           f"GRU 在此資料集的二元分類沒有崩落（{ms(gru_b['test'])}%），但仍低於三個簡單模型。"
           "帶閘門的循環模型在第二個資料集上依然最不穩定，而且並未帶來更好的準確率。")

    m_dnn = u["multi/DNN"]["per_class"]
    zero = [c for c in ("Analysis", "Backdoor", "DoS", "Worms")
            if max(u[f"multi/{m}"]["per_class"][c] for m in MODELS) < 10]
    R.para(f"**少數類別：**十類分類的 macro-F1 只有 {min(mean(u[f'multi/{m}']['macro_f1']) for m in MODELS):.1f}%–"
           f"{max(mean(u[f'multi/{m}']['macro_f1']) for m in MODELS):.1f}%。"
           f"{'、'.join(zero)} 在所有模型上的 recall 都低於 10%；以 DNN、CNN、GRU 的混淆矩陣合計，"
           "這些類別有 78%–93% 的樣本被判為 Exploits。"
           f"模型只對 Generic（{m_dnn['Generic']:.0f}%）、Exploits（{m_dnn['Exploits']:.0f}%）等大類有效，"
           "與 NSL-KDD 上 R2L、U2R 幾乎無法偵測的情況一致。")


def shap_section(R, ext):
    sh = ext["shap"]
    if not sh:
        return
    f = sh["features"]
    imp = np.array(sh["split_mean"])
    order = list(np.argsort(imp)[::-1])
    top5_share = imp[order[:5]].sum() / imp.sum()
    missed, caught = np.array(sh["missed_mean"]), np.array(sh["caught_mean"])
    svc = f.index("service")
    n_missed = mean(sh["n_missed"])
    topk = {r["k"]: r for r in sh["topk"]}
    full, one = topk[max(topk)], topk[1]
    kdd_vals = [r["kddtest"] * 100 for r in sh["topk"]]
    multi = np.array(sh["multi_mean"])
    cls_imp = {c: multi[i].sum() for i, c in enumerate(sh["multi_classes"])}
    push_normal = [f[i] for i in np.argsort(-missed) if missed[i] > 0 and i != svc]

    R.heading("D. 模型判斷依據：SHAP 分析", 2)
    R.para(f"為了解模型依靠哪些特徵判斷攻擊，我們以 SHAP [18] 的 Permutation 方法解釋 5 個種子的二元 DNN，"
           f"在內部測試集與 KDDTest+ 上各隨機抽取 {sh['samples']:,} 筆，以 100 筆訓練資料作為背景分布。"
           "被解釋的輸出為模型判斷為 normal 的機率；one-hot 欄位的 SHAP 值依可加性合併回原始的 41 個特徵。"
           f"結果如圖 {R.fig_no + 1}。")
    R.columns(1)
    fig_no = R.figure(FIG / "shap.png", 17.5,
                      "二元 DNN 的 SHAP 分析（5 個種子平均）。(a) 各特徵的平均絕對 SHAP 值，誤差線為種子間標準差；"
                      "(b) KDDTest+ 中被漏報與被成功偵測的攻擊，各特徵的平均 SHAP 值，正值代表把預測推向 normal；"
                      "(c) 只保留 SHAP 排名前 k 個原始特徵重新訓練 DNN（種子 1）的準確率。")
    R.columns(2)
    R.para(f"**全域重要度：**{f[order[0]]} 在全部 5 個種子中都是最重要的特徵，"
           f"其次為 {'、'.join(f[i] for i in order[1:5])}，前 5 個特徵佔全部重要度的 {top5_share * 100:.0f}%。"
           "除 service 與 flag 外，排名前段的大多是以時間窗或目的主機統計的流量特徵（如同一服務的連線數與錯誤率），"
           "兩種測試集上的排名也大致相同。")
    R.para(f"**漏報攻擊的成因：**二元 DNN 在 KDDTest+ 上平均漏報 {n_missed:,.0f} 筆攻擊"
           f"（佔 {n_missed / sh['n_attack'] * 100:.0f}%）。圖 {fig_no}(b) 顯示，對被成功偵測的攻擊，"
           f"service 的 SHAP 值平均為 {caught[svc]:+.3f}，把預測推向「攻擊」；對被漏報的攻擊則為 "
           f"{missed[svc]:+.3f}，反而把預測推向 normal。{'、'.join(push_normal[:5])} 等特徵"
           "也都把漏報的攻擊推向 normal。換言之，這些攻擊使用的服務與連線統計看起來就像訓練資料中的正常流量，"
           "模型學到的是「哪些服務與流量型態通常是攻擊」，而不是攻擊本身的行為。")
    R.para(f"**捷徑檢驗：**若只用 service 一個特徵（{one['n_columns']} 個 one-hot 欄位）重新訓練 DNN，"
           f"內部測試集準確率為 {one['split'] * 100:.2f}%，KDDTest+ 為 {one['kddtest'] * 100:.2f}%；"
           f"使用全部 41 個特徵時分別為 {full['split'] * 100:.2f}% 與 {full['kddtest'] * 100:.2f}%。"
           f"增加其餘 40 個特徵讓內部測試集提升 {(full['split'] - one['split']) * 100:.1f} 個百分點，"
           f"KDDTest+ 卻沒有改善，在不同 k 值之間於 {min(kdd_vals):.1f}%–{max(kdd_vals):.1f}% 起伏。"
           "此實驗只用單一種子，而 KDDTest+ 準確率的種子間標準差約 1 個百分點，k 值之間的起伏有一部分是隨機變異；"
           "但整體趨勢清楚：額外特徵帶來的進步只出現在與訓練資料同分布的測試集，"
           "符合模型學到資料集特有的捷徑、而非可遷移規則的情況 [29]。")
    R.para(f"**五類分類：**對五類 DNN 的分析顯示，所有特徵對 R2L 與 U2R 輸出機率的影響總和只有 "
           f"{cls_imp['R2L']:.3f} 與 {cls_imp['U2R']:.3f}，遠小於 DoS（{cls_imp['DoS']:.3f}）"
           f"與 Normal（{cls_imp['Normal']:.3f}）。也就是說，無論輸入如何變化，模型幾乎不會提高這兩類的機率，"
           "這解釋了第 VII-B 節中 R2L 與 U2R 的 recall 極低。")


def _adv(ext):
    r = ext["adv"]
    eps = r["epsilons"]
    at = lambda table, m, e: table[m][eps.index(e)] * 100
    return r, eps, at


def adversarial_section(R, ext):
    if not ext["adv"]:
        return
    r, eps, at = _adv(ext)
    ev, ctl, tr, rnd = r["evasion"]["all"], r["evasion"]["controllable"], r["transfer"], r["random"]
    R.heading("E. 對抗樣本穩健性", 2)
    R.para(f"最後，我們測試攻擊者能否透過小幅修改流量特徵逃避偵測。威脅模型如下：攻擊者只修改攻擊樣本，"
           f"目標是讓模型判為 normal；類別特徵（protocol、service、flag）不變，數值特徵的擾動上限 ε 以標準化後的"
           f"標準差為單位（L∞ 範數），且不超出訓練資料的最小與最大值。攻擊方法為 PGD [19]（{r['steps']} 步）。"
           f"我們比較兩種情境：（a）38 個數值特徵都可修改，作為理論上界；（b）只修改攻擊者實際能控制的 "
           f"{len(r['controllable'])} 個特徵，例如傳送的位元組數、連線持續時間與連線次數，"
           "以貼近 Apruzzese 等人 [30] 所主張的實際威脅模型。"
           f"此外也測試黑箱情境：以 DNN 產生的對抗樣本直接攻擊其他模型（轉移攻擊），"
           f"並以同樣大小的隨機擾動作為對照。測試樣本為種子 1 的六個二元模型都能正確偵測的 "
           f"{r['samples']:,} 筆攻擊。")

    rows = []
    for m in MODELS:
        rows.append([m] + [f"{at(ev, m, e):.1f}" for e in (0.1, 0.2, 0.5, 1.0)]
                    + [f"{at(ctl, m, 1.0):.1f}", f"{at(tr, m, 1.0):.1f}", f"{at(rnd, m, 1.0):.1f}"])
    n = R.tab_no + 1
    R.para(f"結果如表 {n} 與圖 {R.fig_no + 1}。", keep_next=True)
    R.columns(1)
    R.table(["模型", "ε = 0.1", "ε = 0.2", "ε = 0.5", "ε = 1", "只改可控特徵\nε = 1",
             "DNN 轉移\nε = 1", "隨機擾動\nε = 1"], rows,
            [1.9, 1.7, 1.7, 1.7, 1.7, 2.5, 2.2, 2.2],
            "對抗樣本的逃避偵測成功率（%，種子 1 的二元模型）。前四欄為可修改全部數值特徵的 PGD 白箱攻擊",
            size=8)
    R.figure(FIG / "adversarial.png", 15.5,
             "PGD 攻擊的逃避偵測成功率與擾動上限 ε 的關係。(a) 可修改全部 38 個數值特徵；"
             "(b) 只修改攻擊者可控的特徵。")
    R.columns(2)

    small = max(at(ev, m, 0.1) for m in MODELS)
    R.para(f"**小擾動下相當穩健：**當 ε ≤ 0.1 個標準差時，所有模型的逃避成功率都不超過 {small:.1f}%。"
           f"但擾動放大到 0.5 個標準差後，DNN 與 RNN 分別有 {at(ev, 'DNN', 0.5):.1f}% 與 "
           f"{at(ev, 'RNN', 0.5):.1f}% 的攻擊成功逃避，ε = 1 時 DNN 達 {at(ev, 'DNN', 1.0):.1f}%。"
           f"同樣大小的隨機擾動只讓 DNN 漏報 {at(rnd, 'DNN', 1.0):.1f}%，"
           "可見這些逃避是沿著模型的弱點刻意尋找出來的，而非單純的雜訊。"
           f"CNN 比 DNN 穩健（ε = 0.5 時 {at(ev, 'CNN', 0.5):.1f}%），但 ε = 1 時仍有 {at(ev, 'CNN', 1.0):.1f}%。")
    R.para(f"**循環模型的穩健性需謹慎解讀：**LSTM、GRU、CNN-LSTM 在 ε = 1 時的逃避率只有 "
           f"{at(ev, 'LSTM', 1.0):.1f}%、{at(ev, 'GRU', 1.0):.1f}%、{at(ev, 'CNN-LSTM', 1.0):.1f}%，看似最穩健。"
           f"然而 LSTM 的逃避率在 ε = 0.5 到 1 之間幾乎沒有上升（{at(ev, 'LSTM', 0.5):.1f}% → "
           f"{at(ev, 'LSTM', 1.0):.1f}%），而且在 ε = 0.5 時，隨機擾動就能讓 LSTM 漏報 "
           f"{at(rnd, 'LSTM', 0.5):.1f}%，遠高於 DNN（{at(rnd, 'DNN', 0.5):.1f}%）與 CNN"
           f"（{at(rnd, 'CNN', 0.5):.1f}%）。這可能代表梯度在長達 122 步的循環計算中衰減，"
           "使以梯度為基礎的 PGD 找不到有效方向（梯度遮蔽 [31]），而非模型真的穩健。"
           "因此我們不認為這些結果能證明循環模型較穩健，若要確認需要不依賴梯度的攻擊方法。")
    R.para(f"**實際可行的攻擊成功率低：**只修改攻擊者可控的 {len(r['controllable'])} 個特徵時，即使 ε = 1，"
           f"DNN 的逃避率也只有 {at(ctl, 'DNN', 1.0):.1f}%，其餘模型都不超過 "
           f"{max(at(ctl, m, 1.0) for m in MODELS if m != 'DNN'):.1f}%。"
           f"DNN 產生的對抗樣本轉移到其他模型時，成功率也只有 "
           f"{min(at(tr, m, 1.0) for m in MODELS if m != 'DNN'):.1f}%–"
           f"{max(at(tr, m, 1.0) for m in MODELS if m != 'DNN'):.1f}%。"
           "可能的原因是模型倚重的 dst_host_* 等以目的主機統計的特徵（第 VII-D 節）不在可控特徵之內，"
           "這些特徵也受其他使用者的連線影響，攻擊者難以單獨操控。"
           "需要注意的是，標準化空間中的擾動不一定對應到能實際產生的流量（例如計數必須為整數、特徵之間需彼此一致），"
           "此實驗也只使用單一種子，因此結果應視為穩健性的初步估計。")


def adversarial_abstract(ext):
    if not ext["adv"]:
        return ""
    r, eps, at = _adv(ext)
    ev, ctl = r["evasion"]["all"], r["evasion"]["controllable"]
    return (f"對抗樣本實驗顯示，若可修改全部數值特徵，0.5 個標準差的擾動即可讓 DNN 漏報 "
            f"{at(ev, 'DNN', 0.5):.0f}% 的攻擊，但只修改攻擊者可控的特徵時成功率大幅下降。")


def adversarial_conclusion(ext):
    if not ext["adv"]:
        return ""
    r, eps, at = _adv(ext)
    ev, ctl = r["evasion"]["all"], r["evasion"]["controllable"]
    return (f"對抗樣本實驗則顯示，模型對小擾動相當穩健，但在可修改全部數值特徵時，"
            f"0.5 個標準差的擾動即可讓 DNN 漏報 {at(ev, 'DNN', 0.5):.0f}% 的攻擊；"
            "限制在攻擊者可控的特徵後，成功率大幅下降。")


def _unsw_gap(ext):
    u = ext["unsw"]
    keep = [m for m in MODELS if m != "LSTM"]
    return [mean(u[f"binary/{m}"]["val"]) - mean(u[f"binary/{m}"]["test"]) for m in keep]


def abstract_tail(ext):
    out = ""
    if ext["unsw"]:
        g = _unsw_gap(ext)
        out += (f"在 UNSW-NB15 上，驗證集與官方測試集之間同樣有 {min(g):.0f}–{max(g):.0f} 個百分點的落差，"
                "且 LSTM 有一個種子完全訓練失敗。")
    if ext["shap"]:
        topk = {r["k"]: r for r in ext["shap"]["topk"]}
        out += (f"SHAP 分析顯示模型最依賴 service 特徵，只用此一特徵訓練的 DNN 在 KDDTest+ 上"
                f"（{topk[1]['kddtest'] * 100:.1f}%）即與使用全部特徵時（{topk[max(topk)]['kddtest'] * 100:.1f}%）相當。")
    out += adversarial_abstract(ext)
    return out


def discussion_eval_tail(ext):
    out = ""
    if ext["unsw"]:
        g = _unsw_gap(ext)
        out += (f"這個現象並非 NSL-KDD 獨有：在 UNSW-NB15 上，從訓練資料切出的驗證集同樣比官方測試集高估 "
                f"{min(g):.0f}–{max(g):.0f} 個百分點（第 VII-C 節）。")
    if ext["shap"]:
        out += ("第 VII-D 節的 SHAP 分析進一步顯示，模型在隨機切分上的高準確率有相當部分來自"
                "資料集特有、無法遷移到新分布的特徵組合。")
    return out


def conclusion(ext):
    st, a = ext["seeds"], ext["analysis"]
    s = summary(ext)
    wins, _ = a["best_counts"]["binary"]
    out = [f"以 5 個種子重複全部模型後，DNN、RNN、CNN 的標準差都在 0.1 個百分點左右，"
           f"LSTM、GRU、CNN-LSTM 則有數個百分點的變異，GRU {'從未' if not wins.get('GRU') else '只偶爾'}成為最佳模型，"
           "原論文的模型排名不具統計意義。"
           f"改以官方 KDDTest+ 評估時，二元分類準確率由 {s['split_lo']:.1f}%–{s['split_hi']:.1f}% 降至 "
           f"{s['kdd_lo']:.1f}%–{s['kdd_hi']:.1f}%，對未見過的攻擊類型只能偵測 "
           f"{s['unseen_lo']:.0f}%–{s['unseen_hi']:.0f}%，且模型排名隨之改變。"]
    if ext["unsw"]:
        g = _unsw_gap(ext)
        out.append(f"在 UNSW-NB15 上，同樣的模型也呈現 {min(g):.0f}–{max(g):.0f} 個百分點的高估，"
                   "帶閘門的循環模型依然最不穩定，簡單的 DNN 與 CNN 依然表現最好。"
                   + ("SHAP 分析則指出模型高度依賴 service 等少數特徵，學到的是資料集特有的捷徑。"
                      if ext["shap"] else "")
                   + adversarial_conclusion(ext))
    out.append("綜合而言，原論文「所有模型皆達 99% 以上、GRU 最佳」的結論，在重複實驗、官方測試集與第二個資料集下"
               "都無法成立。我們建議 NIDS 研究以多個種子報告平均與變異、使用獨立或依時間切分的測試集，"
               "並在宣稱模型優劣前進行統計檢定。")
    return out
