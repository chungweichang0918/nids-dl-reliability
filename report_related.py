"""Word 報告第 II 節「相關研究」。

引用編號使用 build_report.REFERENCES 中的位置 (1 起算)，產生報告時會依首次出現順序自動重新編號。
"""
import report_extension as rx

# 新增參考文獻 (附加在 build_report.REFERENCES 之後，編號 20 起)
NEW_REFERENCES = [
    # [20]
    "C. Yin, Y. Zhu, J. Fei, and X. He, “A deep learning approach for intrusion detection using "
    "recurrent neural networks,” IEEE Access, vol. 5, pp. 21954–21961, 2017, "
    "doi: 10.1109/ACCESS.2017.2762418.",
    # [21]
    "R. Vinayakumar, M. Alazab, K. P. Soman, P. Poornachandran, A. Al-Nemrat, and S. Venkatraman, "
    "“Deep learning approach for intelligent intrusion detection system,” IEEE Access, vol. 7, "
    "pp. 41525–41550, 2019, doi: 10.1109/ACCESS.2019.2895334.",
    # [22]
    "M. A. Ferrag, L. Maglaras, S. Moschoyiannis, and H. Janicke, “Deep learning for cyber security "
    "intrusion detection: Approaches, datasets, and comparative study,” J. Inf. Secur. Appl., "
    "vol. 50, Art. no. 102419, 2020.",
    # [23]
    "J. McHugh, “Testing intrusion detection systems: A critique of the 1998 and 1999 DARPA intrusion "
    "detection system evaluations as performed by Lincoln Laboratory,” ACM Trans. Inf. Syst. Secur., "
    "vol. 3, no. 4, pp. 262–294, 2000.",
    # [24]
    "N. Moustafa and J. Slay, “The evaluation of network anomaly detection systems: Statistical "
    "analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set,” Inf. Secur. J.: "
    "A Global Perspective, vol. 25, no. 1–3, pp. 18–31, 2016.",
    # [25]
    "R. Sommer and V. Paxson, “Outside the closed world: On using machine learning for network "
    "intrusion detection,” in Proc. IEEE Symp. Secur. Privacy, 2010, pp. 305–316.",
    # [26]
    "G. Engelen, V. Rimmer, and W. Joosen, “Troubleshooting an intrusion detection dataset: The "
    "CICIDS2017 case study,” in Proc. IEEE Secur. Privacy Workshops (SPW), 2021, pp. 7–12.",
    # [27]
    "N. Reimers and I. Gurevych, “Reporting score distributions makes a difference: Performance study "
    "of LSTM-networks for sequence tagging,” in Proc. EMNLP, 2017, pp. 338–348.",
    # [28]
    "A. Warnecke, D. Arp, C. Wressnegger, and K. Rieck, “Evaluating explanation methods for deep "
    "learning in security,” in Proc. IEEE Eur. Symp. Secur. Privacy (EuroS&P), 2020, pp. 158–174.",
    # [29]
    "R. Geirhos et al., “Shortcut learning in deep neural networks,” Nat. Mach. Intell., vol. 2, "
    "no. 11, pp. 665–673, 2020.",
    # [30]
    "G. Apruzzese, M. Andreolini, L. Ferretti, M. Marchetti, and M. Colajanni, “Modeling realistic "
    "adversarial attacks against network intrusion detection systems,” Digit. Threats: Res. Pract., "
    "vol. 3, no. 3, Art. no. 31, 2022.",
    # [31]
    "A. Athalye, N. Carlini, and D. Wagner, “Obfuscated gradients give a false sense of security: "
    "Circumventing defenses to adversarial examples,” in Proc. ICML, 2018, pp. 274–283.",
]


def related_work_section(R, ext):
    s = rx.summary(ext)
    R.heading("II. 相關研究")

    R.heading("A. 深度學習於網路入侵偵測", 2)
    R.para("深度學習應用於網路入侵偵測的研究已相當多。Yin 等人 [20] 以 RNN 在 NSL-KDD 上進行二元與五類分類，"
           "以 KDDTrain+ 訓練、官方 KDDTest+ 測試，報告的準確率為 83.28% 與 81.29%，並優於 J48、隨機森林與 SVM 等"
           "傳統方法。Vinayakumar 等人 [21] 先在 KDD Cup 99 上決定 DNN 的超參數，再將模型套用到 NSL-KDD、"
           "UNSW-NB15、CICIDS2017 等多個資料集，與傳統機器學習方法比較。Ferrag 等人 [22] 整理了 35 個常見的資安"
           "資料集，並在 CSE-CIC-IDS2018 與 Bot-IoT 上比較七種深度學習模型。本報告重現的 Elsayed 等人 [1] 則在"
           "NSL-KDD 上比較六種架構，但只以 KDDTrain+ 內部的隨機切分評估，所有模型的準確率都在 98% 以上。")
    R.para("同樣使用 NSL-KDD，以官方測試集評估的研究（如 [20]）與以隨機切分評估的研究（如 [1]）所報告的準確率"
           "相差十幾個百分點，使不同論文之間的數字難以直接比較。這也是本研究在第 VII-B 節以兩種方式評估同一批模型的原因："
           f"我們的模型在隨機切分上達到 {s['split_lo']:.1f}%–{s['split_hi']:.1f}%，在 KDDTest+ 上則是 "
           f"{s['kdd_lo']:.1f}%–{s['kdd_hi']:.1f}%，與上述兩類文獻的數值範圍各自相符。")

    R.heading("B. 資料集與評估方法的問題", 2)
    R.para("NIDS 基準資料集的品質一直受到質疑。McHugh [23] 批評 KDD Cup 99 所源自的 1998 年 DARPA 評估資料，"
           "指出其背景流量的產生方式與真實網路不符。Tavallaee 等人 [2] 發現 KDD Cup 99 含有大量重複紀錄，"
           "因而提出 NSL-KDD。Moustafa 與 Slay [24] 以統計分析比較 UNSW-NB15 與 KDD99，認為前者更複雜、"
           "更適合作為新的基準。然而較新的資料集同樣有問題：Engelen 等人 [26] 重新檢查 CICIDS2017 的資料產生流程，"
           "重建或重新標記了超過 20% 的流量紀錄。")
    R.para("在評估方法上，Sommer 與 Paxson [25] 指出機器學習用於入侵偵測的特殊困難：機器學習較擅長辨識相似性而非"
           "找出離群值、錯誤的代價很高、網路流量本身的變化性很大，而且難以取得具代表性的評估資料。"
           "Arp 等人 [13] 系統性地整理了資安機器學習的十種常見陷阱，例如抽樣偏差、資料窺探、虛假相關與只在實驗室"
           "環境評估，並發現這些問題在頂尖資安會議的論文中相當普遍。本研究在 NSL-KDD 與 UNSW-NB15 上觀察到的高估現象，"
           "正是這類陷阱的具體案例。")

    R.heading("C. 機器學習的可重現性", 2)
    R.para("機器學習研究的可重現性近年受到廣泛關注。Bouthillier 等人 [11] 區分「方法的可重現」與「結論的可重現」，"
           "指出即使程式碼公開，結論仍可能因實驗中的隨機性而不成立。Pineau 等人 [12] 介紹 NeurIPS 2019 的可重現性計畫"
           "與檢查清單，建議論文報告隨機種子、重複次數與結果的變異。Reimers 與 Gurevych [27] 訓練了五萬個 LSTM 模型，"
           "發現僅改變隨機種子就可使命名實體辨識的 F1 相差約一個百分點，因此主張比較多次訓練的分數分布，而非單次結果。"
           "本研究將這個觀點應用到 NIDS：第 VII-A 節顯示，帶閘門的循環模型因種子造成的變異可達數個百分點，"
           "大於原論文中模型之間的差距。")

    R.heading("D. 可解釋性與對抗穩健性", 2)
    R.para("SHAP [18] 以 Shapley 值為基礎，把一筆預測分配給各個特徵，是目前最常用的模型解釋方法之一。"
           "Warnecke 等人 [28] 在多個資安任務上比較解釋方法，發現不同方法的品質差異很大，"
           "並展示解釋結果能幫助找出模型學到的問題。Geirhos 等人 [29] 將深度學習的許多失敗歸納為「捷徑學習」："
           "模型學到在標準基準上有效、卻無法轉移到更困難測試條件的決策規則。本研究在第 VII-D 節以 SHAP 檢驗"
           "NIDS 模型是否也有這種現象。")
    R.para("在對抗穩健性方面，Madry 等人 [19] 提出的 PGD 是評估模型穩健性的標準攻擊方法。Athalye 等人 [31] 指出，"
           "許多看似穩健的防禦其實是梯度遭到遮蔽，使以梯度為基礎的攻擊失效，而非模型真正穩健。"
           "針對 NIDS，Apruzzese 等人 [30] 指出現有研究常假設攻擊者完全了解偵測器或能自由與其互動，"
           "這在實際情境中並不合理，並提出更貼近實際的威脅模型。因此本研究的對抗實驗同時比較「可修改全部特徵」與"
           "「只修改攻擊者可控的特徵」兩種情境，並以隨機擾動作為對照。")

    R.heading("E. 本研究的定位", 2)
    R.para("上述研究分別探討資料集品質、評估方式、可重現性、可解釋性與對抗穩健性，多數只針對單一面向。"
           "本研究以一篇近期的比較研究 [1] 為對象，在相同的模型與超參數下，把重現、診斷實驗、多種子統計、官方測試集、"
           "第二個資料集、SHAP 與對抗樣本整合在同一個實驗框架中，逐一檢驗原論文的結論。"
           "我們的部分發現在文獻中已有相關報告，例如 NSL-KDD 隨機切分會高估準確率 [2], [20]；"
           "本研究的貢獻在於量化這些問題對一篇特定論文的結論造成多大的影響，"
           "並指出帶閘門的循環模型在兩個資料集上都系統性地不穩定，而 SimpleRNN 則沒有這個問題。")
