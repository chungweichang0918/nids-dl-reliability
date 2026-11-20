"""英文論文的 Word 排版工具 (IEEE 雙欄格式)。

- 引用用 [@key] 或 [@key1, @key2]，依首次出現的順序自動編號 (IEEE 慣例)
- 尚未完成的內容用 todo()，以黃色標示
- 表格為三線表，標題在上方 (TABLE I)；圖的標題在下方 (Fig. 1.)
"""
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"
BODY = 10
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"]
CITE = re.compile(r"\[@([\w\-]+(?:\s*,\s*@[\w\-]+)*)\]")
MARK = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|\{\{[^}]+\}\})")


def set_font(run, size=BODY, bold=None, italic=None, color=None, highlight=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(attr), FONT)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    if highlight:
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def set_columns(section, num, space_cm=0.5):
    cols = section._sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        section._sectPr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(int(space_cm * 567)))


class Paper:
    def __init__(self, refs):
        self.refs, self.order = refs, []
        self.doc = Document()
        self.fig_no = self.tab_no = self.sec_no = self.sub_no = 0
        normal = self.doc.styles["Normal"]
        normal.font.name, normal.font.size = FONT, Pt(BODY)
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        pf = normal.paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(2), 1.0
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(21.59), Cm(27.94)          # US Letter (IEEE)
        sec.top_margin = sec.bottom_margin = Cm(1.9)
        sec.left_margin = sec.right_margin = Cm(1.6)
        self._page_numbers(sec)

    def _page_numbers(self, sec):
        p = sec.footer.paragraphs[0]
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
            run._r.append(el)
        set_font(run, 8)

    # --- 引用 ---
    def cite(self, keys):
        nums = []
        for k in keys:
            if k not in self.refs:
                raise KeyError(f"未知的參考文獻 {k}")
            if k not in self.order:
                self.order.append(k)
            nums.append(self.order.index(k) + 1)
        return ", ".join(f"[{n}]" for n in sorted(nums))

    def resolve(self, text):
        return CITE.sub(lambda m: self.cite([k.strip().lstrip("@") for k in m.group(1).split(",")]), text)

    # --- 文字 ---
    def _runs(self, p, text, size=BODY, italic=False):
        """**粗體**、*斜體*、{{待補內容}} (黃色標示)。"""
        for part in MARK.split(self.resolve(text)):
            if not part:
                continue
            if part.startswith("**"):
                set_font(p.add_run(part[2:-2]), size, bold=True, italic=italic)
            elif part.startswith("{{"):
                set_font(p.add_run(f"[{part[2:-2]}]"), size, italic=italic, highlight=True)
            elif part.startswith("*"):
                set_font(p.add_run(part[1:-1]), size, italic=True)
            else:
                set_font(p.add_run(part), size, italic=italic)

    def para(self, text, indent=True, size=BODY, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=2,
             keep_next=False):
        p = self.doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.keep_with_next = keep_next
        if indent:
            p.paragraph_format.first_line_indent = Pt(BODY * 1.2)
        self._runs(p, text, size)
        return p

    def todo(self, text):
        return self.para("{{TODO: " + text + "}}", indent=False)

    def bullets(self, items, numbered=False):
        for i, item in enumerate(items, 1):
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Pt(14)
            p.paragraph_format.first_line_indent = Pt(-10)
            p.paragraph_format.space_after = Pt(1)
            set_font(p.add_run(f"{i}) " if numbered else "• "), BODY)
            self._runs(p, item)

    def section(self, title):
        self.sec_no += 1
        self.sub_no = 0
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(8), Pt(3)
        p.paragraph_format.keep_with_next = True
        set_font(p.add_run(f"{ROMAN[self.sec_no - 1]}. {title.upper()}"), BODY, bold=True)

    def unnumbered_section(self, title):
        """IEEE 的 Acknowledgment 等不編號章節。"""
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(8), Pt(3)
        p.paragraph_format.keep_with_next = True
        set_font(p.add_run(title.upper()), BODY, bold=True)

    def subsection(self, title):
        self.sub_no += 1
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(5), Pt(2)
        p.paragraph_format.keep_with_next = True
        set_font(p.add_run(f"{chr(64 + self.sub_no)}. {title}"), BODY, italic=True)

    def equation(self, text, number):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(3)
        set_font(p.add_run(text), BODY, italic=True)
        set_font(p.add_run(f" ({number})"), BODY)

    def columns(self, num):
        sec = self.doc.add_section(WD_SECTION.CONTINUOUS)
        set_columns(sec, num)

    # --- 圖表 ---
    def figure_placeholder(self, caption, height_cm=4.0, note=""):
        """尚未有最終結果的圖: 以灰框佔位，標題照常編號。"""
        self.fig_no += 1
        t = self.doc.add_table(rows=1, cols=1)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = t.rows[0].cells[0]
        tc = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:fill"), "EDEDED")
        tc.append(shd)
        tr = t.rows[0]._tr.get_or_add_trPr()
        h = OxmlElement("w:trHeight")
        h.set(qn("w:val"), str(int(height_cm * 567)))
        tr.append(h)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(p.add_run(f"Figure placeholder — {note}"), 8, italic=True, color="666666")
        return self._caption(f"Fig. {self.fig_no}. ", caption, after=True)

    def figure(self, path, width_cm, caption):
        self.fig_no += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        return self._caption(f"Fig. {self.fig_no}. ", caption, after=True)

    def _caption(self, label, caption, after):
        c = self.doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if len(caption) > 60 else WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.space_after = Pt(6 if after else 2)
        c.paragraph_format.space_before = Pt(2 if after else 6)
        c.paragraph_format.keep_with_next = not after
        set_font(c.add_run(label), 8)
        self._runs(c, caption, 8)
        return self.fig_no if after else self.tab_no

    def next_table(self):
        """下一個表的羅馬數字編號 (表格在引用之後才建立時使用)。"""
        return ROMAN[self.tab_no]

    def table(self, headers, rows, widths_cm, caption, size=8, left_cols=(0,)):
        self.tab_no += 1
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.keep_with_next = True
        cap.paragraph_format.space_before = Pt(6)
        set_font(cap.add_run(f"TABLE {ROMAN[self.tab_no - 1]}"), 8)
        cap2 = self.doc.add_paragraph()
        cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap2.paragraph_format.keep_with_next = True
        cap2.paragraph_format.space_after = Pt(2)
        self._runs(cap2, caption.upper() if len(caption) < 70 else caption, 8)

        t = self.doc.add_table(rows=len(rows) + 1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        pr = t._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "bottom"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "8")
            borders.append(el)
        pr.append(borders)
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        pr.append(layout)
        for r_i, values in enumerate([headers] + rows):
            row = t.rows[r_i]
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for c_i, val in enumerate(values):
                cell = row.cells[c_i]
                cell.width = Cm(widths_cm[c_i])
                tcpr = cell._tc.get_or_add_tcPr()
                mar = OxmlElement("w:tcMar")
                for side in ("left", "right"):
                    m = OxmlElement(f"w:{side}")
                    m.set(qn("w:w"), "40")
                    m.set(qn("w:type"), "dxa")
                    mar.append(m)
                tcpr.append(mar)
                if r_i == 0:
                    b = OxmlElement("w:tcBorders")
                    el = OxmlElement("w:bottom")
                    el.set(qn("w:val"), "single")
                    el.set(qn("w:sz"), "4")
                    b.append(el)
                    tcpr.append(b)
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.keep_with_next = r_i < len(rows)
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_i in left_cols else WD_ALIGN_PARAGRAPH.CENTER
                text = str(val)
                if r_i == 0:
                    set_font(p.add_run(self.resolve(text)), size, bold=True)
                else:
                    self._runs(p, text, size)
        spacer = self.doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(4)
        return self.tab_no

    def references(self):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(8), Pt(3)
        set_font(p.add_run("REFERENCES"), BODY, bold=True)
        for i, key in enumerate(self.order, 1):
            q = self.doc.add_paragraph()
            q.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            q.paragraph_format.left_indent = Pt(20)
            q.paragraph_format.first_line_indent = Pt(-20)
            q.paragraph_format.space_after = Pt(1)
            self._runs(q, f"[{i}]\t{self.refs[key]}", 8)
        unused = [k for k in self.refs if k not in self.order]
        return unused

    def save(self, path):
        self.doc.save(path)
