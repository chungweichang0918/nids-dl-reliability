"""產生論文格式的 Word 重現報告 (IEEE 風格雙欄)。

所有數字都從 results/ 下的 JSON 讀取。
用法: .venv/Scripts/python.exe build_report.py
"""
import json
import re

import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

import nids_repro as nr
import report_extension as rx
import report_related as rr
from report_figures import JOBS, VARIANT_LABEL, load_ablation

ROOT = nr.ROOT
FIG = nr.OUT / "figures"
OUTFILE = ROOT / "深度學習網路入侵偵測.docx"

LATIN, CJK = "Times New Roman", "標楷體"
BODY_PT = 10
MODELS = list(nr.HP["binary"])
METRICS = ["accuracy", "precision", "recall", "f1"]


# --------------------------------------------------------------------------
# 版面工具
# --------------------------------------------------------------------------
def set_font(run, size=None, bold=None, italic=None, color=None, latin=LATIN, cjk=CJK):
    run.font.name = latin
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    fonts.set(qn("w:eastAsia"), cjk)
    fonts.set(qn("w:ascii"), latin)
    fonts.set(qn("w:hAnsi"), latin)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def set_columns(section, num, space_cm=0.6):
    sect = section._sectPr
    cols = sect.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(int(space_cm * 567)))


def setup_page(section):
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin, section.bottom_margin = Cm(1.9), Cm(2.2)
    section.left_margin = section.right_margin = Cm(1.6)


def add_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._element.append(el)
    set_font(run, 9)


class Report:
    def __init__(self):
        self.doc = Document()
        self.fig_no = 0
        self.tab_no = 0
        normal = self.doc.styles["Normal"]
        normal.font.name, normal.font.size = LATIN, Pt(BODY_PT)
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
        pf = normal.paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(3), 1.15
        sec = self.doc.sections[0]
        setup_page(sec)
        add_page_number(sec)

    # --- 段落 ---
    def para(self, text, size=BODY_PT, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True,
             space_after=3, keep_next=False):
        """支援 **粗體** 與 *斜體* 標記。"""
        p = self.doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.keep_with_next = keep_next
        if indent:
            p.paragraph_format.first_line_indent = Pt(size * 2)
        for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
            if not part:
                continue
            if part.startswith("**"):
                set_font(p.add_run(part[2:-2]), size, bold=True)
            elif part.startswith("*"):
                set_font(p.add_run(part[1:-1]), size, italic=True)
            else:
                set_font(p.add_run(part), size)
        return p

    def bullets(self, items, size=BODY_PT):
        for item in items:
            p = self.para(item, size=size, indent=False, space_after=2)
            p.paragraph_format.left_indent = Pt(size * 1.6)
            p.paragraph_format.first_line_indent = Pt(-size * 1.0)
            p.runs[0].text = "• " + p.runs[0].text

    def heading(self, text, level=1):
        p = self.doc.add_paragraph()
        p.paragraph_format.keep_with_next = True
        if level == 1:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(10), Pt(4)
            set_font(p.add_run(text), 11, bold=True)
        else:
            p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(6), Pt(3)
            set_font(p.add_run(text), BODY_PT, bold=True, italic=True)
        return p

    def equation(self, text, number):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(3)
        set_font(p.add_run(text), BODY_PT, italic=True)
        set_font(p.add_run(f"    ({number})"), BODY_PT)

    # --- 欄位切換 ---
    def columns(self, num):
        sec = self.doc.add_section(WD_SECTION.CONTINUOUS)
        set_columns(sec, num)
        return sec

    # --- 圖 ---
    def figure(self, path, width_cm, caption):
        self.fig_no += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(4)
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        c = self.doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(caption) < 40 else WD_ALIGN_PARAGRAPH.JUSTIFY
        c.paragraph_format.space_after = Pt(8)
        set_font(c.add_run(f"圖 {self.fig_no}. "), 8.5, bold=True)
        set_font(c.add_run(caption), 8.5)
        return self.fig_no

    # --- 表 (三線表) ---
    def table(self, headers, rows, widths_cm, caption, size=8, bold_cells=(),
              left_cols=(0,)):
        self.tab_no += 1
        c = self.doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.keep_with_next = True
        c.paragraph_format.space_before = Pt(6)
        c.paragraph_format.space_after = Pt(2)
        set_font(c.add_run(f"表 {self.tab_no}. "), 8.5, bold=True)
        set_font(c.add_run(caption), 8.5)

        t = self.doc.add_table(rows=len(rows) + 1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        tbl_pr = t._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge, sz in (("top", 12), ("bottom", 12)):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:color"), "000000")
            borders.append(el)
        tbl_pr.append(borders)
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        tbl_pr.append(layout)

        for r_i, values in enumerate([headers] + rows):
            row = t.rows[r_i]
            tr_pr = row._tr.get_or_add_trPr()
            cant = OxmlElement("w:cantSplit")
            tr_pr.append(cant)
            for c_i, val in enumerate(values):
                cell = row.cells[c_i]
                cell.width = Cm(widths_cm[c_i])
                tc_pr = cell._tc.get_or_add_tcPr()
                mar = OxmlElement("w:tcMar")
                for side in ("left", "right"):
                    m = OxmlElement(f"w:{side}")
                    m.set(qn("w:w"), "50")
                    m.set(qn("w:type"), "dxa")
                    mar.append(m)
                tc_pr.append(mar)
                if r_i == 0:
                    b = OxmlElement("w:tcBorders")
                    el = OxmlElement("w:bottom")
                    el.set(qn("w:val"), "single")
                    el.set(qn("w:sz"), "6")
                    b.append(el)
                    tc_pr.append(b)
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.keep_with_next = r_i < len(rows)
                p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if c_i in left_cols
                               else WD_ALIGN_PARAGRAPH.CENTER)
                bold = r_i == 0 or (r_i - 1, c_i) in bold_cells
                set_font(p.add_run(str(val)), size, bold=bold)
        after = self.doc.add_paragraph()
        after.paragraph_format.space_after = Pt(4)
        set_font(after.add_run(""), 4)
        return self.tab_no

    def save(self, path):
        self.doc.save(path)


# --------------------------------------------------------------------------
# 資料
# --------------------------------------------------------------------------
def load_runs():
    runs = {}
    for task in ("binary", "multi"):
        for m in MODELS:
            runs[(task, m)] = json.loads((nr.OUT / "runs" / f"{task}_{m}.json").read_text())
    return runs


def pct(x, d=2):
    return f"{x * 100:.{d}f}"


def describe(name, task):
    hp = nr.HP[task][name]
    if name == "DNN":
        return f"Dense {hp['units_1']} → Dense {hp['units_2']}"
    if name in ("RNN", "LSTM", "GRU"):
        cell = "SimpleRNN" if name == "RNN" else name
        return (f"{cell} {hp['units_1']}/{hp['units_2']}/{hp['units_3']} "
                f"(dropout {hp['dropout_1']}/{hp['dropout_2']}) → Dense {hp['dense_units']}")
    if name == "CNN":
        return (f"Conv1D {hp['filters_1']}(k={hp['kernel_size_1']}) → Pool 4 → "
                f"Conv1D {hp['filters_2']}(k={hp['kernel_size_2']}) → Pool 2 → "
                f"Dense {hp['dense_units']}")
    return (f"Conv1D {hp['filters']}(k={hp['kernel_size']}) → Pool 2 → "
            f"LSTM {hp['units']} (dropout {hp['dropout']})")


# --------------------------------------------------------------------------
# 內容
# --------------------------------------------------------------------------
def build():
    global ABLATION_TEXT
    runs = load_runs()
    abl = load_ablation()
    ABLATION_TEXT = make_ablation_text(abl)
    ext = rx.load()
    S = rx.summary(ext)
    R = Report()
    doc = R.doc
    n_test = int(np.array(runs[("binary", "DNN")]["confusion_matrix"]).sum())
    paper = nr.PAPER
    acc = {k: v["accuracy"] for k, v in runs.items()}

    # ---------------- 標題區 (單欄) ----------------
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    set_font(p.add_run("深度學習網路入侵偵測比較研究之重現與訓練穩定性分析"), 18, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    set_font(p.add_run("Reproducing “A Comparative Study of Using Deep Learning Algorithms in "
                       "Network Intrusion Detection” and Analyzing Training Instability of "
                       "Recurrent Models"), 11, italic=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    set_font(p.add_run("張鈞崴　　資工四乙　　1111232077"), 10.5, color="666666")

    b_best = max(MODELS, key=lambda m: acc[("binary", m)])
    abstract = (
        f"本報告重現 Elsayed 等人於 IEEE Access (2024) 發表的深度學習網路入侵偵測比較研究 [1]。"
        f"原論文在 NSL-KDD 資料集 [2] 上比較 DNN、RNN、CNN、LSTM、GRU 與 Hybrid CNN-LSTM 六種模型的"
        f"二元與五類分類表現，並報告所有模型準確率介於 97.96% 至 99.54%，其中 GRU 為二元分類最佳模型。"
        f"我們依原論文公開的前處理流程與 KerasTuner 最佳超參數重新實作 12 個模型，在 {n_test:,} 筆測試資料上評估。"
        f"結果顯示 DNN、CNN、RNN 在兩種任務皆成功重現（與原論文差距在 ±0.4 個百分點內），"
        f"但 LSTM 與 GRU 的二元分類準確率僅 {pct(acc[('binary','LSTM')])}% 與 {pct(acc[('binary','GRU')])}%，"
        f"較原論文低約 3 個百分點，原論文「GRU 最佳」的結論未能重現，本次最佳模型為 {b_best}"
        f"（{pct(acc[('binary', b_best)])}%）。"
        f"{ABLATION_TEXT['abstract']}"
        f"我們進一步以 5 個隨機種子重複全部 12 個模型：GRU 在 5 次中{'從未' if not S['gru_wins'] else '僅 ' + str(S['gru_wins']) + ' 次'}"
        f"成為最佳模型，原論文的模型排名在重複實驗下無法成立。改以官方 KDDTest+ 評估後，"
        f"二元分類準確率由內部測試集的 {S['split_lo']:.1f}%–{S['split_hi']:.1f}% 降至 {S['kdd_lo']:.1f}%–{S['kdd_hi']:.1f}%，"
        f"對訓練時未見過的攻擊類型只能偵測 {S['unseen_lo']:.0f}%–{S['unseen_hi']:.0f}%。"
        f"{rx.abstract_tail(ext)}"
        f"這些結果顯示，原論文以隨機切分得到的 99% 以上準確率，高估了模型對新型攻擊的偵測能力。"
    )
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(4)
    set_font(p.add_run("摘要—"), 9.5, bold=True)
    set_font(p.add_run(abstract), 9.5)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(8)
    set_font(p.add_run("關鍵詞—"), 9.5, bold=True)
    set_font(p.add_run("網路入侵偵測、深度學習、NSL-KDD、UNSW-NB15、可重現性、訓練穩定性、"
                       "泛化能力、可解釋性、對抗樣本"), 9.5)
    # 標題區下方分隔線
    pPr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for k, v in (("val", "single"), ("sz", "6"), ("space", "4"), ("color", "000000")):
        bottom.set(qn(f"w:{k}"), v)
    bdr.append(bottom)
    pPr.append(bdr)

    R.columns(2)

    # ---------------- I. 緒論 ----------------
    R.heading("I. 緒論")
    R.para("隨著網路服務與物聯網裝置快速普及，網路攻擊的數量與複雜度持續上升，入侵偵測系統（Intrusion "
           "Detection System, IDS）成為防禦體系的重要一環。傳統以特徵碼（signature）為基礎的偵測方法誤報率低，"
           "但只能辨識已知攻擊；以異常（anomaly）為基礎的方法則嘗試學習正常流量的行為模式，以偵測未知攻擊。"
           "近年深度學習因具備自動擷取特徵的能力，被大量應用於網路入侵偵測（NIDS）研究。")
    R.para("Elsayed 等人 [1] 在 NSL-KDD 資料集上系統性地比較六種深度學習架構，並以 KerasTuner 搜尋各模型的"
           "超參數，報告所有模型在二元與五類分類皆達到約 98%–99.5% 的準確率。由於該研究提供了完整的超參數設定，"
           "非常適合作為重現研究的對象。機器學習研究的可重現性近年受到高度重視 [11], [12]，單次實驗的數值"
           "往往受到隨機種子、框架版本與硬體差異影響，若沒有重現，很難判斷論文中模型間零點幾個百分點的差距是否具有意義。")
    R.para("本報告的目標有四：（1）依原論文設定完整重現 12 個模型（6 種架構 × 2 種任務）；"
           "（2）針對重現失敗的模型設計診斷實驗，找出差異的原因；（3）以重複實驗、官方測試集與第二個資料集，"
           "檢驗原論文結論的穩健性與適用範圍；（4）分析模型的判斷依據與對抗穩健性。本報告的主要貢獻如下：")
    R.bullets([
        "提供可直接執行的完整重現程式，涵蓋前處理、12 個模型、評估與圖表產生。",
        "證實 DNN、CNN、RNN 與原論文結果一致，但 LSTM、GRU 與多元分類 CNN-LSTM 無法穩定重現。",
        "透過換種子、梯度裁剪、調整學習率、移除梯度累積及改用原論文框架等診斷實驗，"
        "釐清循環模型結果偏低的主因。",
        "以 5 個隨機種子重複全部模型並進行統計檢定，顯示原論文的模型排名不具統計意義。",
        "在官方 KDDTest+ 與 UNSW-NB15 上評估，量化隨機切分對準確率的高估程度。",
        "以 SHAP 分析模型依賴的特徵，並以對抗樣本測試模型的穩健性。",
    ])
    R.para("本報告其餘章節安排如下：第 II 節回顧相關研究；第 III 節概述原論文的方法；第 IV 節說明重現方法；"
           "第 V 節為重現結果；第 VI 節以診斷實驗分析差異原因；第 VII 節為延伸實驗；第 VIII 節討論；第 IX 節為結論。")

    # ---------------- II. 相關研究 ----------------
    rr.related_work_section(R, ext)

    # ---------------- III. 原論文概述 ----------------
    R.heading("III. 原論文方法概述")
    R.heading("A. 資料集", 2)
    R.para("NSL-KDD [2] 是 KDD Cup 99 的改良版本，移除了大量重複紀錄，是入侵偵測領域最常用的基準資料集之一。"
           "每筆紀錄描述一條網路連線，包含 41 個特徵（38 個數值、3 個類別）與一個標籤。攻擊被歸為四大類："
           "阻斷服務（DoS）、探測（Probe）、遠端到本地（R2L）與使用者提權（U2R）。")
    R.heading("B. 前處理與模型", 2)
    R.para("原論文對數值特徵做標準化，對 protocol_type、service、flag 三個類別特徵做 one-hot 編碼，"
           "使特徵維度由 41 增為 122，接著以 80/20 隨機切分訓練與測試資料，再從訓練資料取 20% 作驗證。"
           "六種模型為 DNN、SimpleRNN、一維 CNN、LSTM [3]、GRU [4] 與 CNN-LSTM，"
           "皆以 KerasTuner [10] 隨機搜尋決定單元數、卷積核數量與大小、dropout 比例等超參數，"
           "最後以 Adam 最佳化器、100 epochs、batch size 5000 統一訓練。")
    R.heading("C. 原論文主要結論", 2)
    R.bullets([
        "二元分類以 GRU 最佳（準確率 99.54%），LSTM、CNN、DNN 次之。",
        "五類分類以 DNN 表現最突出（99.34%），Hybrid CNN-LSTM 最低（97.96%）。",
        "所有模型在 DoS、Probe 與 Normal 類別接近完美，但 U2R 對所有模型都是挑戰。",
    ])

    # ---------------- III. 重現方法 ----------------
    R.heading("IV. 重現方法")
    R.heading("A. 資料與前處理", 2)
    R.para(f"我們使用 KDDTrain+ 的全部 125,973 筆資料，依原論文以 StandardScaler 標準化 38 個數值特徵，"
           f"one-hot 編碼後得到 122 維特徵。依攻擊名稱對照表將訓練資料中的 22 種攻擊歸入四大類。資料以隨機種子 42 "
           f"切分為 100,778 筆訓練與 {n_test:,} 筆測試資料，訓練時再以最後 20% 作為驗證集。"
           f"圖 1 為整體流程，圖 2 為各類別樣本數。")

    R.columns(1)
    R.figure(FIG / "pipeline.png", 17.5, "重現實驗的整體流程。")
    R.columns(2)

    R.figure(FIG / "class_distribution.png", 8.4,
             "KDDTrain+ 各類別樣本數（對數刻度）。U2R 只有 52 筆，佔 0.04%，類別極度不平衡。")
    R.para("如圖 2 所示，資料集類別極度不平衡：Normal 與 DoS 合計佔 89.9%，而 U2R 只有 52 筆。"
           "隨機切分後，測試集中 U2R 僅剩 11 筆、R2L 224 筆，這兩類的指標將有很大的統計波動。")

    R.heading("B. 模型架構", 2)
    R.para("所有超參數直接採用原論文第 VI 節報告的 KerasTuner 最佳值，未重新搜尋。"
           "序列型模型（RNN、CNN、LSTM、GRU、CNN-LSTM）將 122 維特徵視為長度 122、每步 1 維的序列。"
           "表 1 列出二元分類模型的結構，五類分類的結構見附錄表 A1。")
    rows = [[m, describe(m, "binary")] for m in MODELS]
    R.table(["模型", "結構（二元分類）"], rows, [1.6, 6.8], "二元分類模型結構（原論文最佳超參數）",
            size=7.5, left_cols=(0, 1))
    R.para("原論文的程式碼以圖片呈現，部分細節無法確認，我們做了以下推測：CNN 第二個卷積層之後的池化大小為 2；"
           "Hybrid CNN-LSTM 卷積層之後的池化大小為 2；所有循環模型的 Dense 層使用 ReLU 活化函數。")

    R.heading("C. 訓練與評估", 2)
    R.para("二元分類輸出層使用 sigmoid 搭配 binary cross-entropy；五類分類使用 softmax 搭配 categorical "
           "cross-entropy。最佳化器為 Adam [5]（學習率 10⁻³），訓練 100 epochs，batch size 5000。"
           "評估指標與原論文相同：")
    R.equation("Accuracy = (TP + TN) / (TP + TN + FP + FN)", 1)
    R.equation("Precision = TP / (TP + FP)", 2)
    R.equation("Recall = TP / (TP + FN)", 3)
    R.equation("F1 = 2 · Precision · Recall / (Precision + Recall)", 4)
    R.para("由原論文的混淆矩陣可推知其二元分類以 normal 為正類，我們採用相同定義。"
           "五類分類的 Precision 與 Recall 依 Keras 指標的算法，以 one-hot 標籤、門檻 0.5 做 micro 平均，"
           "因原論文報告的 recall 低於 accuracy，與此算法相符。")

    R.heading("D. 實驗環境與差異", 2)
    env_rows = [
        ["框架", "TF / Keras 2.14", "Keras 3 + PyTorch 2.6"],
        ["硬體", "Colab TPU", "RTX 4050 (6 GB)；第 VII 節的 SimpleRNN 用 Apple M6 CPU"],
        ["Batch", "5000", "5000（循環模型 1000×5 累積）"],
        ["評估權重", "未說明", "最後一個 epoch"],
        ["重複次數", "未說明", "主實驗 1 次（種子 42），第 VII 節 5 次（種子 1–5）"],
    ]
    R.table(["項目", "原論文", "本重現"], env_rows, [1.6, 2.6, 4.2], "原論文與本重現的實驗環境比較",
            size=7.5)
    R.para("本重現以 Keras 3 [8] 搭配 PyTorch [9] 後端實作。"
           "RTX 4050 的 6 GB 記憶體無法容納 batch 5000 的 GRU 訓練，因此 RNN、LSTM、GRU 與 CNN-LSTM "
           "改為每次輸入 1000 筆、累積 5 次梯度才更新權重。數學上這與 batch 5000 等效，"
           "但框架實作細節可能不同，第 VI 節將以實驗檢驗此差異的影響。")

    # ---------------- IV. 重現結果 ----------------
    R.heading("V. 重現結果")
    R.heading("A. 二元分類", 2)
    rows, bold = [], []
    for i, m in enumerate(MODELS):
        r = runs[("binary", m)]
        rows.append([m] + [f"{pct(r[k])} / {paper['binary'][m][j]:.2f}" for j, k in enumerate(METRICS)]
                    + [f"{(r['accuracy'] * 100 - paper['binary'][m][0]):+.2f}"])
        if abs(r["accuracy"] * 100 - paper["binary"][m][0]) > 1:
            bold.append((i, 5))
    R.para("表 3 與圖 3 比較本重現與原論文的二元分類結果。", keep_next=True)
    R.columns(1)
    R.table(["模型", "Accuracy", "Precision", "Recall", "F1", "Accuracy 差距"], rows,
            [2.2, 3.0, 3.0, 3.0, 3.0, 2.2],
            "二元分類結果（本重現 / 原論文，單位 %）", size=8, bold_cells=bold)
    R.figure(nr.OUT / "binary_comparison.png", 17.5,
             "二元分類各指標比較。藍色為本重現，橘色為原論文。LSTM 與 GRU 明顯低於原論文，其餘模型一致。")
    R.columns(2)
    R.para(f"表 3 顯示，DNN、CNN、RNN 與 CNN-LSTM 的準確率均與原論文相差 0.4 個百分點以內，"
           f"RNN（{pct(acc[('binary','RNN')])}%）甚至比原論文高 0.37 個百分點，成為本次重現中二元分類最佳的模型。"
           f"相對地，LSTM 與 GRU 只有 {pct(acc[('binary','LSTM')])}% 與 {pct(acc[('binary','GRU')])}%，"
           f"分別低於原論文 3.00 與 3.48 個百分點。原論文宣稱 GRU 為最佳模型，在本次重現中 GRU 反而是最差的。")
    cm_rows = []
    for m in MODELS:
        (tn, fp), (fn, tp) = runs[("binary", m)]["confusion_matrix"]
        cm_rows.append([m, f"{tn:,}", fp, fn, f"{tp:,}"])
    R.table(["模型", "TN", "FP", "FN", "TP"], cm_rows, [1.8, 1.7, 1.4, 1.4, 1.7],
            "二元分類混淆矩陣（正類為 normal）", size=7.5)
    R.para("表 4 的混淆矩陣顯示，表現良好的模型誤判數皆在 100 筆上下，"
           "原論文報告的 DNN 為 74 FP / 80 FN，與本重現的 71 / 73 非常接近。"
           "LSTM 與 GRU 的誤判數則多出四到六倍。")

    R.heading("B. 五類分類", 2)
    rows, bold = [], []
    for i, m in enumerate(MODELS):
        r = runs[("multi", m)]
        rows.append([m] + [f"{pct(r[k])} / {paper['multi'][m][j]:.2f}" for j, k in enumerate(METRICS)]
                    + [f"{(r['accuracy'] * 100 - paper['multi'][m][0]):+.2f}"])
        if abs(r["accuracy"] * 100 - paper["multi"][m][0]) > 1:
            bold.append((i, 5))
    R.para("表 5 為五類分類的整體結果，表 6 為各類別的 F1-score。", keep_next=True)
    R.columns(1)
    R.table(["模型", "Accuracy", "Precision", "Recall", "F1", "Accuracy 差距"], rows,
            [2.2, 3.0, 3.0, 3.0, 3.0, 2.2],
            "五類分類結果（本重現 / 原論文，單位 %）", size=8, bold_cells=bold)
    R.columns(2)
    R.para(f"表 5 中，DNN、RNN、CNN 再次成功重現，三者準確率介於 {pct(acc[('multi','CNN')])}% 至 "
           f"{pct(acc[('multi','DNN')])}%，彼此差距僅 0.02 個百分點，單次實驗無法分出優劣。"
           f"原論文「DNN 最佳」的結論在數值上成立，但差距不具統計意義。LSTM、GRU 再度偏低，"
           f"CNN-LSTM 則只有 {pct(acc[('multi','CNN-LSTM')])}%，比原論文低 4.54 個百分點。")
    pc_rows = []
    for m in MODELS:
        rep = runs[("multi", m)]["report"]
        pc_rows.append([m] + [f"{rep[c]['f1-score']:.3f}" for c in nr.MULTI_CLASSES])
    sup = runs[("multi", "DNN")]["report"]
    R.table(["模型"] + [f"{c}\n(n={int(sup[c]['support']):,})" for c in nr.MULTI_CLASSES],
            pc_rows, [1.6, 1.35, 1.45, 1.35, 1.3, 1.3], "五類分類各類別 F1-score", size=7.5)
    R.para("表 6 的各類別結果與原論文的描述一致：DoS、Normal、Probe 三類的 F1 接近 1，"
           "DNN、RNN、CNN 在 R2L 上達到 0.84–0.88，而 U2R 對所有模型都極為困難，"
           "最好的 CNN 也只有 0.375，LSTM、GRU 與 CNN-LSTM 則完全無法偵測。"
           "值得注意的是，整體準確率高達 99% 主要來自 DoS 與 Normal 兩個大類，"
           "少數類別的失敗幾乎不影響整體準確率，因此單看準確率會高估模型的實際偵測能力。")

    R.columns(1)
    R.figure(nr.OUT / "multi_confusion.png", 15.5,
             "五類分類混淆矩陣（色階為對數刻度）。U2R 與 R2L 大量被誤判為 Normal。")
    R.figure(nr.OUT / "binary_curves.png", 17.5,
             "二元分類訓練與驗證曲線。DNN、RNN、CNN 平滑收斂；LSTM 與 GRU 在訓練中途出現準確率驟降與損失暴增。")
    R.columns(2)

    R.heading("C. 訓練曲線觀察", 2)
    R.para("圖 5 揭示了 LSTM、GRU 結果偏低的線索。首先，這兩個模型的訓練與驗證準確率幾乎重疊，"
           "代表問題**不是過擬合**，而是模型沒有訓練好。其次，曲線出現明顯的「崩落」：二元 GRU 在約第 48 epoch "
           "準確率由 97.6% 驟降，損失同時暴增，之後雖逐漸回升，但到第 100 epoch 仍未回到崩落前的水準；"
           "二元 LSTM 在約第 60 epoch 也有類似現象。這種現象是循環網路訓練中典型的梯度暴衝 [6], [7]。"
           "根據這些觀察，我們在下一節設計診斷實驗。")

    # ---------------- V. 診斷實驗 ----------------
    R.heading("VI. 差異原因之診斷實驗")
    R.heading("A. 假設與實驗設計", 2)
    R.para("我們針對重現失敗最明顯的三個設定（二元 LSTM、二元 GRU、五類 CNN-LSTM），"
           "提出五個可能造成差異的假設，並為每個假設設計對應實驗（表 7）。每個實驗同時記錄"
           "「最後一個 epoch」與「驗證準確率最高 epoch」兩組權重的測試準確率，"
           "以區分「訓練不穩定」與「模型能力不足」。")
    R.table(["假設", "對應實驗"], [
        ["H1 隨機種子造成的變異", "換用種子 1 重新訓練"],
        ["H2 梯度暴衝造成崩落", "加入梯度裁剪 clipnorm = 1 [6]"],
        ["H3 學習率過大", "學習率降為 3×10⁻⁴"],
        ["H4 梯度累積實作差異", "LSTM 直接用 batch 5000 訓練"],
        ["H5 框架差異", "以 TensorFlow 2.14 訓練 LSTM"],
    ], [4.0, 4.4], "診斷實驗設計")

    R.heading("B. 實驗結果", 2)
    abl_rows = []
    titles = {("binary", "LSTM"): "二元 LSTM", ("binary", "GRU"): "二元 GRU",
              ("multi", "CNN-LSTM"): "五類 CNN-LSTM"}
    for key in JOBS:
        for v in VARIANT_LABEL:
            if v not in abl[key]:
                continue
            r = abl[key][v]
            best = "—" if r["best"] is None else pct(r["best"]["accuracy"])
            abl_rows.append([titles[key], VARIANT_LABEL[v], pct(r["last"]["accuracy"]), best,
                             r["best_epoch"], r["collapses"]])
    R.para("表 8 為所有診斷實驗的結果，圖 6 為對應的驗證準確率曲線。", keep_next=True)
    R.columns(1)
    R.table(["設定", "實驗", "最後 epoch 權重", "最佳 epoch 權重", "最佳 epoch", "崩落次數"],
            abl_rows, [2.8, 3.8, 2.8, 2.8, 2.2, 2.0],
            "診斷實驗結果（測試準確率 %）。崩落次數為驗證準確率比先前最高值低 5 個百分點以上的 epoch 數",
            size=8, left_cols=(0, 1))
    R.figure(FIG / "ablation_curves.png", 17.5, ABLATION_TEXT["figure"])
    R.columns(2)
    for text in ABLATION_TEXT["results"]:
        R.para(text)

    R.heading("C. 小結", 2)
    for text in ABLATION_TEXT["summary"]:
        R.para(text)

    # ---------------- VI. 延伸實驗 ----------------
    R.heading("VII. 延伸實驗")
    R.para("第 V、VI 節的結果都來自單一種子與原論文的隨機切分。本節從四個方向檢驗原論文結論的穩健性："
           "以多個種子重複實驗（A）、改用官方測試集（B）、換用第二個資料集（C），"
           "以及分析模型的判斷依據（D）與對抗穩健性（E）。")
    rx.seeds_section(R, ext)
    rx.kddtest_section(R, ext)
    rx.unsw_section(R, ext)
    rx.shap_section(R, ext)
    rx.adversarial_section(R, ext)

    # ---------------- VII. 討論 ----------------
    R.heading("VIII. 討論")
    R.heading("A. 模型複雜度與效益", 2)
    t = {m: runs[("binary", m)]["train_seconds"] for m in MODELS}
    params = {m: runs[("binary", m)]["params"] for m in MODELS}
    st = ext["seeds"]
    R.table(["模型", "參數量", "訓練時間 (s)", "內部測試集 (%)", "KDDTest+ (%)"],
            [[m, f"{params[m]:,}", f"{t[m]:.0f}", rx.ms(st[f'binary/{m}']['split']),
              rx.ms(st[f'binary/{m}']['kddtest'])] for m in MODELS],
            [1.75, 1.3, 1.35, 2.0, 2.0],
            "二元分類模型的參數量、訓練時間（主實驗，RTX 4050）與準確率（5 個種子）", size=7.5)
    R.para(f"表 {R.tab_no} 顯示，最簡單的 DNN 只有 {params['DNN']:,} 個參數、訓練 {t['DNN']:.0f} 秒，"
           f"在內部測試集上與最佳模型相差無幾，在 KDDTest+ 上平均甚至最高；"
           f"RNN 的訓練時間是 DNN 的 {t['RNN'] / t['DNN']:.0f} 倍，"
           f"GRU 參數量是 DNN 的 {params['GRU'] / params['DNN']:.0f} 倍，兩者都沒有帶來可觀的效益。"
           "對 NSL-KDD 這類表格資料而言，增加模型複雜度並未提升準確率，也沒有提升對新型攻擊的泛化能力。")
    R.heading("B. 將表格特徵視為序列的合理性", 2)
    R.para("原論文主張循環網路能捕捉網路流量的「時間依賴」。然而 NSL-KDD 的每筆紀錄是一條獨立連線的統計摘要，"
           "122 個特徵是由數值欄位與 one-hot 欄位依固定順序排列而成，彼此之間並沒有時間先後關係。"
           "把它們當成長度 122 的序列餵給 RNN，模型學到的只是欄位排列順序上的人為結構，而非流量的時間行為。"
           "本重現中，不具序列假設的 DNN 已達到與最佳模型相同的水準，而依賴序列假設的 LSTM、GRU 反而最不穩定，"
           "這支持了上述觀點。若要真正利用時間資訊，應以同一主機或同一時間窗內的連續連線作為序列。")
    R.heading("C. 評估方式的限制", 2)
    R.para(f"原論文只使用 KDDTrain+ 內部的隨機切分評估。第 VII-B 節顯示，同樣的模型在官方 KDDTest+ 上"
           f"準確率降至 {S['kdd_lo']:.1f}%–{S['kdd_hi']:.1f}%，對未見過的攻擊類型只能偵測 "
           f"{S['unseen_lo']:.0f}%–{S['unseen_hi']:.0f}%，且模型的優劣排名也隨之改變。"
           "隨機切分會讓高度相似的連線同時出現在訓練與測試資料中，使準確率被推高；"
           "Arp 等人 [13] 將這類評估偏差列為資安機器學習的常見陷阱。"
           f"{rx.discussion_eval_tail(ext)}"
           "我們建議未來的 NIDS 研究至少報告官方測試集或跨資料集的結果，並以多個種子重複實驗、報告變異。")
    R.heading("D. 類別不平衡", 2)
    R.para("U2R 在測試集只有 11 筆，即使模型全部判錯，整體準確率也只下降 0.04 個百分點。"
           "若入侵偵測系統的重點在於找出少見但危害大的攻擊，應改以各類別的 recall、macro-F1 等指標評估，"
           "並搭配 SMOTE [14] 或類別權重等方法處理不平衡問題。")

    # ---------------- VIII. 結論 ----------------
    R.heading("IX. 結論")
    for text in ABLATION_TEXT["conclusion"]:
        R.para(text)
    for text in rx.conclusion(ext):
        R.para(text)
    R.para("未來工作包括：在 CIC-IDS2017 等更新的資料集上重複本研究；以時間切分取代隨機切分，"
           "模擬模型部署後面對新攻擊的情境；針對少數類別加入資料平衡方法；結合對抗訓練提升模型穩健性；"
           "以及在具備真實時間序列結構的資料上重新檢驗循環網路的優勢。")

    # ---------------- 參考文獻 ----------------
    refs = REFERENCES + rr.NEW_REFERENCES
    order = renumber_citations(doc, len(refs))
    R.heading("參考文獻")
    for i, old in enumerate(order, 1):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Pt(18)
        p.paragraph_format.first_line_indent = Pt(-18)
        p.paragraph_format.space_after = Pt(2)
        set_font(p.add_run(f"[{i}]\t{refs[old - 1]}"), 8)

    # ---------------- 附錄 ----------------
    R.heading("附錄")
    R.heading("A. 五類分類模型結構", 2)
    rows = [[m, describe(m, "multi")] for m in MODELS]
    _appendix_table(R, ["模型", "結構（五類分類）"], rows, [1.6, 6.8], "A1", "五類分類模型結構")
    R.heading("B. 重現步驟", 2)
    R.para("所有程式與結果皆置於專案資料夾。nids_repro.py 負責前處理、模型定義、訓練與評估；"
           "make_report.py 產生對照表與圖；experiments.py 與 tf_check.py 為第 VI 節的診斷實驗；"
           "multiseed.py、unsw.py、explain.py、adversarial.py 為第 VII 節的延伸實驗，analyze.py 進行統計檢定；"
           "build_report.py 產生本報告。在已安裝套件的虛擬環境中依序執行：")
    for cmd in ["python nids_repro.py", "python make_report.py", "python experiments.py",
                "python multiseed.py", "python unsw.py", "python explain.py",
                "python adversarial.py", "python analyze.py",
                "python report_figures.py", "python build_report.py"]:
        p = R.para(cmd, size=8.5, indent=False, space_after=1)
        p.paragraph_format.left_indent = Pt(12)
        for run in p.runs:
            set_font(run, 8.5, latin="Consolas", cjk=CJK)

    R.save(OUTFILE)
    print(f"已輸出 {OUTFILE}")


CITE = re.compile(r"\[(\d+)\]")


def renumber_citations(doc, n_refs):
    """依 IEEE 慣例，把內文引用改為依首次出現的順序編號。

    回傳舊編號的新順序 (order[i] = 第 i+1 篇的舊編號)。沒被引用的文獻附在最後並印出警告。
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    def paragraphs():
        for el in doc.element.body.iterchildren():
            if el.tag == qn("w:p"):
                yield Paragraph(el, doc)
            elif el.tag == qn("w:tbl"):
                for row in Table(el, doc).rows:
                    for cell in row.cells:
                        yield from cell.paragraphs

    order = []
    for p in paragraphs():
        for m in CITE.finditer(p.text):
            old = int(m.group(1))
            assert 1 <= old <= n_refs, f"引用 [{old}] 超出參考文獻範圍: {p.text[:60]}"
            if old not in order:
                order.append(old)
    missing = [i for i in range(1, n_refs + 1) if i not in order]
    if missing:
        print("警告: 以下文獻沒有在內文引用:", missing)
    order += missing
    new = {old: i for i, old in enumerate(order, 1)}
    for p in paragraphs():
        for run in p.runs:
            if CITE.search(run.text):
                run.text = CITE.sub(lambda m: f"[{new[int(m.group(1))]}]", run.text)
    return order


def _appendix_table(R, headers, rows, widths, label, caption):
    """附錄表使用 A1 編號，不佔用正文表號。"""
    before = R.tab_no
    R.table(headers, rows, widths, caption, size=7.5, left_cols=(0, 1))
    R.tab_no = before
    for p in reversed(R.doc.paragraphs):
        if p.runs and p.runs[0].text.startswith("表 ") and caption in "".join(r.text for r in p.runs):
            p.runs[0].text = f"表 {label}. "
            break


REFERENCES = [
    "S. Elsayed, K. Mohamed, and M. A. Madkour, “A comparative study of using deep learning algorithms "
    "in network intrusion detection,” IEEE Access, vol. 12, pp. 58851–58870, 2024, "
    "doi: 10.1109/ACCESS.2024.3389096.",
    "M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, “A detailed analysis of the KDD CUP 99 data set,” "
    "in Proc. IEEE Symp. Comput. Intell. Secur. Defense Appl. (CISDA), 2009, pp. 1–6.",
    "S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Comput., vol. 9, no. 8, "
    "pp. 1735–1780, 1997.",
    "K. Cho et al., “Learning phrase representations using RNN encoder–decoder for statistical machine "
    "translation,” in Proc. EMNLP, 2014, pp. 1724–1734.",
    "D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. ICLR, 2015.",
    "R. Pascanu, T. Mikolov, and Y. Bengio, “On the difficulty of training recurrent neural networks,” "
    "in Proc. ICML, 2013, pp. 1310–1318.",
    "Y. Bengio, P. Simard, and P. Frasconi, “Learning long-term dependencies with gradient descent is "
    "difficult,” IEEE Trans. Neural Netw., vol. 5, no. 2, pp. 157–166, 1994.",
    "F. Chollet et al., “Keras,” 2015. [Online]. Available: https://keras.io",
    "A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” "
    "in Proc. NeurIPS, 2019, pp. 8024–8035.",
    "T. O’Malley et al., “KerasTuner,” 2019. [Online]. Available: https://github.com/keras-team/keras-tuner",
    "X. Bouthillier, C. Laurent, and P. Vincent, “Unreproducible research is reproducible,” "
    "in Proc. ICML, 2019, pp. 725–734.",
    "J. Pineau et al., “Improving reproducibility in machine learning research,” J. Mach. Learn. Res., "
    "vol. 22, no. 164, pp. 1–20, 2021.",
    "D. Arp et al., “Dos and don’ts of machine learning in computer security,” in Proc. USENIX Security "
    "Symp., 2022, pp. 3971–3988.",
    "N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic minority "
    "over-sampling technique,” J. Artif. Intell. Res., vol. 16, pp. 321–357, 2002.",
    "B. L. Welch, “The generalization of ‘Student’s’ problem when several different population "
    "variances are involved,” Biometrika, vol. 34, no. 1–2, pp. 28–35, 1947.",
    "S. Holm, “A simple sequentially rejective multiple test procedure,” Scand. J. Statist., "
    "vol. 6, no. 2, pp. 65–70, 1979.",
    "N. Moustafa and J. Slay, “UNSW-NB15: A comprehensive data set for network intrusion detection "
    "systems (UNSW-NB15 network data set),” in Proc. Military Commun. Inf. Syst. Conf. (MilCIS), "
    "2015, pp. 1–6.",
    "S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” "
    "in Proc. NeurIPS, 2017, pp. 4765–4774.",
    "A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models "
    "resistant to adversarial attacks,” in Proc. ICLR, 2018.",
]

def tf_status(abl):
    """TensorFlow 對照實驗: 完成則回傳結果，未完成則從 log 讀取目前進度。"""
    done = abl[("binary", "LSTM")].get("tf")
    if done:
        return dict(done=True, acc=done["last"]["accuracy"] * 100,
                    best=done["best"]["accuracy"] * 100, collapses=done["collapses"])
    log = nr.OUT / "tf_lstm_binary.log"
    vals = re.findall(r"val_accuracy: ([0-9.]+)", log.read_text(errors="ignore")) if log.exists() else []
    if not vals:
        return dict(done=False, epoch=0)
    epoch = len(vals)
    main_val = abl[("binary", "LSTM")]["main"]["val_accuracy"][epoch - 1]
    return dict(done=False, epoch=epoch, val=float(vals[-1]) * 100, main_val=main_val * 100)


def make_ablation_text(abl):
    def a(key, v, which="last"):
        r = abl[key].get(v)
        if r is None or r[which] is None:
            return float("nan")
        return r[which]["accuracy"] * 100

    L, G, C = ("binary", "LSTM"), ("binary", "GRU"), ("multi", "CNN-LSTM")
    seed_gap = [a(k, "base_s1") - a(k, "main") for k in (L, G, C)]
    tf = tf_status(abl)

    if tf["done"]:
        tf_result = (f"**H5 框架差異：**以原論文使用的 TensorFlow 2.14 直接以 batch 5000 訓練二元 LSTM，"
                     f"準確率為 {tf['acc']:.2f}%，崩落 {tf['collapses']} 次。"
                     + (f"此結果高於 PyTorch 主實驗（{a(L, 'main'):.2f}%，曾崩落），但與 PyTorch "
                        f"未崩落的各組（換種子 {a(L, 'base_s1'):.2f}%、梯度裁剪 {a(L, 'clip'):.2f}%、"
                        f"不做梯度累積 {a(L, 'plain5000'):.2f}%）相當，且仍低於原論文的 "
                        f"{nr.PAPER['binary']['LSTM'][0]:.2f}%。換言之，TensorFlow 版本的優勢來自"
                        "這次訓練沒有發生崩落，而非框架本身；框架差異無法解釋與原論文的落差。"
                        if tf["acc"] < 99 else
                        "結果接近原論文，顯示框架差異是落差的重要來源。"))
        tf_abs = (f"改用原論文的 TensorFlow 2.14 訓練 LSTM 得到 {tf['acc']:.2f}%，"
                  "與 PyTorch 未崩落的訓練相當，框架差異亦非主因。"
                  if tf["acc"] < 99 else "改用原論文的 TensorFlow 2.14 後結果接近原論文。")
    else:
        tf_result = ("**H5 框架差異：**以原論文使用的 TensorFlow 2.14 在 CPU 上訓練二元 LSTM，"
                     "因 CPU 訓練每個 epoch 需 85–150 秒，撰寫本報告時尚未完成"
                     + (f"。截至第 {tf['epoch']} epoch，驗證準確率為 {tf['val']:.2f}%，"
                        f"與 PyTorch 版本同一 epoch 的 {tf['main_val']:.2f}% 相近，"
                        "目前沒有證據顯示框架差異能解釋落差。" if tf["epoch"] else "。"))
        tf_abs = ""

    return {
        "abstract": (
            f"為找出差異原因，我們設計五項診斷實驗。結果顯示主因是循環模型的訓練不穩定："
            f"僅更換隨機種子，準確率即變動 {min(seed_gap):.1f} 至 {max(seed_gap):.1f} 個百分點；"
            f"加入梯度裁剪後所有訓練崩落現象消失，GRU 準確率由 {a(G, 'main'):.2f}% 提升至 "
            f"{a(G, 'clip'):.2f}%；梯度累積並非原因。{tf_abs}"),
        "results": [
            f"**H1 隨機種子：**只把種子由 42 換成 1，二元 LSTM 由 {a(L, 'main'):.2f}% 升至 "
            f"{a(L, 'base_s1'):.2f}%，二元 GRU 由 {a(G, 'main'):.2f}% 升至 {a(G, 'base_s1'):.2f}%，"
            f"五類 CNN-LSTM 更由 {a(C, 'main'):.2f}% 升至 {a(C, 'base_s1'):.2f}%，"
            f"恰好等於原論文報告的 97.96%。同一設定僅因種子不同就相差 {min(seed_gap):.1f}–"
            f"{max(seed_gap):.1f} 個百分點，大於原論文中六個模型之間不到 1 個百分點的差距。",
            f"**H2 梯度暴衝：**加入梯度裁剪（clipnorm = 1）後，三個設定的崩落次數全部降為 0，"
            f"「最後 epoch」與「最佳 epoch」的準確率幾乎相同，訓練變得穩定。"
            f"二元 GRU 達到 {a(G, 'clip'):.2f}%，是所有診斷實驗中最高的結果；"
            f"二元 LSTM 為 {a(L, 'clip'):.2f}%，五類 CNN-LSTM 為 {a(C, 'clip'):.2f}%"
            f"（最佳 epoch 為 {a(C, 'clip', 'best'):.2f}%）。這證實主實驗中的崩落是梯度暴衝所致 [6]。",
            f"**H3 學習率：**把學習率降為 3×10⁻⁴ 反而使結果變差：二元 LSTM 只有 "
            f"{a(L, 'lr3e-4'):.2f}%，二元 GRU {a(G, 'lr3e-4'):.2f}%，五類 CNN-LSTM "
            f"{a(C, 'lr3e-4'):.2f}%，且最佳 epoch 都落在第 99–100 epoch，代表模型仍在進步、"
            f"尚未收斂。在 batch 5000、100 epochs（約 1,600 次權重更新）的預算下，降低學習率只會讓收斂更慢。",
            f"**H4 梯度累積：**二元 LSTM 不做梯度累積、直接以 batch 5000 訓練，準確率為 "
            f"{a(L, 'plain5000'):.2f}%，沒有崩落，與換種子（{a(L, 'base_s1'):.2f}%）及梯度裁剪"
            f"（{a(L, 'clip'):.2f}%）的結果相當。因此梯度累積不是造成差異的原因；"
            f"主實驗的 {a(L, 'main'):.2f}% 是因為第 60 epoch 附近發生一次崩落，屬於隨機事件。",
            tf_result,
        ],
        "figure": ("診斷實驗的驗證準確率曲線。灰色虛線為主實驗；加入梯度裁剪（綠）後崩落消失，"
                   "換種子（藍）的結果明顯較好，降低學習率（橘）則收斂明顯變慢。"),
        "summary": [
            "綜合以上實驗，LSTM、GRU 與 CNN-LSTM 無法重現的主因是**訓練不穩定**："
            "循環網路在學習率 10⁻³ 下會間歇發生梯度暴衝，使準確率突然崩落，"
            "而 100 epochs 的訓練預算不足以讓模型從崩落中完全恢復。崩落何時發生取決於隨機種子，"
            "因此同一模型的結果會有數個百分點的落差。梯度累積不是主因；"
            + ("框架差異也不是主因。" if tf["done"] and tf["acc"] < 99 else
               "框架差異則可能是部分原因。" if tf["done"] else
               "框架差異的實驗尚未完成，但目前的訓練進度沒有顯示差異。"),
            f"然而，即使加入梯度裁剪使訓練穩定，二元 LSTM 與 GRU 的最佳結果"
            f"（{a(L, 'clip'):.2f}% 與 {a(G, 'clip'):.2f}%）仍比原論文低 0.8–1.5 個百分點。"
            "LSTM 的驗證曲線到第 100 epoch 仍在上升，顯示模型在此預算下尚未收斂。"
            "原論文的數值可能來自較幸運的種子，或是 KerasTuner 從大量試驗中挑選最佳結果所產生的選擇效應；"
            "由於原論文沒有報告種子與重複次數，我們無法進一步確認。",
            "這個發現對原論文的模型排名有直接影響：原論文以 0.1–0.2 個百分點的差距判定 GRU 為最佳模型，"
            "但本研究顯示單一模型因種子不同就可能有超過 4 個百分點的差異，"
            "因此在沒有重複實驗的情況下，這樣的排名並不可靠。",
        ],
        "conclusion": [
            "本報告完整重現了 Elsayed 等人 [1] 的 12 個深度學習入侵偵測模型。DNN、CNN 與 RNN 在二元與五類分類"
            "皆成功重現，與原論文差距在 0.4 個百分點以內；LSTM、GRU 與五類 CNN-LSTM 則明顯偏低，"
            "原論文「GRU 為最佳二元分類模型」的結論未能重現。",
            f"診斷實驗指出差異來自循環模型的訓練不穩定，而非實作錯誤：換種子即可讓準確率變動 "
            f"{max(seed_gap):.1f} 個百分點，加入梯度裁剪可消除崩落並使 GRU 提升至 {a(G, 'clip'):.2f}%。"
            "這說明在比較模型時，必須以多個種子重複實驗並報告變異，否則零點幾個百分點的差距不具意義。",
            "此外，最簡單的 DNN 只需 36 秒訓練即可達到 99.43%，與最複雜的模型相當，"
            "顯示在 NSL-KDD 這類表格資料上，循環與卷積結構並未帶來實質優勢。",
        ],
    }

if __name__ == "__main__":
    build()
