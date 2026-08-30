import re
import json
import base64
import csv
from pathlib import Path
from io import BytesIO
from typing import List, Dict
import requests
from pdf2image import convert_from_path
import pdfplumber
import fitz  # PyMuPDF
from pypdf import PdfReader
from PIL import Image

# ==================== 配置 ====================
SOURCE_PAPERS_DIR = Path(__file__).resolve().parent.parent / "source_papers"  # ES98C/source_papers
TEXT_OUTPUT_CSV = "extracted_data_text.csv"
IMAGE_OUTPUT_CSV = "extracted_data_image.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-vl:8b-instruct-q4_K_M"

# 主库 schema：24 元素（按周期表原子序数升序）+ 环境/目标/元数据字段。
# 单位统一用我们的口径：E_corr mV、I_corr μA/cm²、CR μm/年、Cl_M mol/L。
ELEMENTS = ["B", "C", "N", "Mg", "Al", "Si", "P", "S", "Ti", "V", "Cr", "Mn",
            "Fe", "Co", "Ni", "Cu", "Y", "Nb", "Mo", "Ce", "Gd", "Ta", "W", "Re"]
FIELD_NAMES = (["Alloy_Name"] + ELEMENTS +
               ["Electrolyte", "Cl_M", "pH", "Temp_C", "Test_Method", "Ref_Electrode",
                "E_corr_mV", "I_corr_uA_per_cm2", "ba_mV_per_dec", "bc_mV_per_dec", "CR_um_per_y",
                "Paper_Name", "DOI", "Notes"])


# ==================== 预筛选器（精简版） ====================
# ==================== 表格 bbox 辅助 ====================
def _caption_body(text: str, end_pos: int) -> list:
    """抓取 caption 之后、到下一个 "Table"/空行 为止的表体行。"""
    lines = text[end_pos:].split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1  # 跳过 caption 行尾的空白
    body = []
    for ln in lines[i:]:
        ln = ln.strip()
        if not ln:
            break
        if re.match(r"Table\s*\d+", ln, re.IGNORECASE):
            break
        body.append(ln)
    return body


def _caption_above(page, bbox) -> tuple:
    """取表格 bbox 上方最近的一块文字作为 caption，返回 (text, fitz.Rect | None)。"""
    x0, y0, x1, y1 = bbox
    blocks = [b for b in page.get_text("blocks") if b[4].strip() and b[3] <= y0 + 2]
    if not blocks:
        return "", None
    blocks.sort(key=lambda b: y0 - b[3])
    b = blocks[0]
    return b[4].strip(), fitz.Rect(b[0], b[1], b[2], b[3])


def _near_above(cap_rect, table_bbox) -> bool:
    """caption 是否在表格 bbox 上方 80pt 内且水平有交集（即 caption 属于该表）。"""
    x0, y0, x1, y1 = table_bbox
    horizontal_overlap = max(0, min(cap_rect.x1, x1) - max(cap_rect.x0, x0)) > 0
    return horizontal_overlap and 0 <= y0 - cap_rect.y1 <= 80


def _infer_bbox(page, cap_rect, next_cap_y=None):
    """无框线表：caption rect 到下一个 caption（或下方固定高度）之间文字的包围盒。"""
    pad = 4
    if next_cap_y is None:
        next_cap_y = cap_rect.y1 + 120  # 兜底：caption 下方 120pt
    words = page.get_text("words")
    xs = [cap_rect.x1]
    ys = [cap_rect.y1]
    for w in words:
        wx0, wy0, wx1, wy1 = w[:4]
        if cap_rect.y1 - 2 < wy1 < next_cap_y:
            xs.append(wx1)
            ys.append(wy1)
    x1 = min(page.rect.x1, max(xs) + pad)
    y1 = min(page.rect.y1, max(ys) + pad)
    return (max(0, cap_rect.x0 - pad), max(0, cap_rect.y0 - pad), x1, y1)


def _render_bbox(page, content_bbox, caption_rect):
    """合并表格内容 bbox 与 caption rect，加 padding，得到渲染裁剪框。"""
    x0, y0, x1, y1 = content_bbox
    if caption_rect is not None:
        x0 = min(x0, caption_rect.x0)
        y0 = min(y0, caption_rect.y0)
        x1 = max(x1, caption_rect.x1)
        y1 = max(y1, caption_rect.y1)
    pad = 6
    return (max(0, x0 - pad), max(0, y0 - pad),
            min(page.rect.x1, x1 + pad), min(page.rect.y1, y1 + pad))


# ==================== 预筛选器（精简版） ====================
def _is_bad_caption(caption: str) -> bool:
    """判断表头是否明显不是数据表（图注 / 作者行 / 版权行 / 链接等）。"""
    c = (caption or "").strip().lower()
    if not c:
        return False
    if re.match(r"^(fig\.?|figure)\b", c):  # 图注
        return True
    if re.search(r"et\s+al\.?|university|department|institute|©|copyright|https?://|\bdoi\b", c):
        return True  # 作者/机构/版权/链接
    return False


def locate_target_tables(pdf_path: str) -> List[Dict]:
    """定位三类目标表格：合金成分表、极化曲线表、E_corr表。

    返回每个表格的：页码、类型、表头、原始数据、前后文、bbox（PDF pt 坐标，供单表裁剪渲染）。

    识别方式（两种来源，各自带 bbox）：
    1. 有框线表：PyMuPDF ``find_tables()`` 拿精确 bbox + 文本矩阵。
    2. 无框线表："Table N. caption" 锚点 + 表体文字 search_for 推断 bbox。
    两种来源去重：caption 锚点识别的表若其区域已被 find_tables 覆盖则跳过。
    """
    targets = []

    table_keywords = {
        "composition": [r"composition", r"化学成分", r"wt%", r"wt\.?\s*%", r"元素含量"],
        "polarization": [r"polarization", r"potentiodynamic", r"极化曲线", r"Tafel"],
        "ecorr": [r"E_corr", r"i_corr", r"j_corr", r"Ecorr", r"icorr", r"腐蚀电流", r"βa", r"βc"],
    }

    # 只认 "Table N." / "Table N:" 后跟同一行非空 caption；排除正文里 "in Table 3"。
    caption_re = re.compile(r"Table\s*\d+\s*[.:]\s*([^\n]+)", re.IGNORECASE)

    def classify(snippet: str):
        flat = re.sub(r"\s+", "", snippet)  # 匹配被断行的表头如 "E\ncorr"
        for t_type, keywords in table_keywords.items():
            for kw in keywords:
                if re.search(kw, snippet, re.IGNORECASE) or re.search(kw, flat, re.IGNORECASE):
                    return t_type
        return None

    def context_around(text: str, anchor: str) -> str:
        """取 anchor 所在位置前后各 3 句。"""
        pos = text.find(anchor)
        if pos == -1:
            return ""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        char_count = 0
        for i, sent in enumerate(sentences):
            char_count += len(sent) + 1
            if char_count > pos:
                start_idx = max(0, i - 3)
                end_idx = min(len(sentences), i + 4)
                return " ".join(sentences[start_idx:end_idx])
        return ""

    doc = fitz.open(pdf_path)
    try:
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            seen: set = set()

            def add(t_type, caption, data, bbox, context=""):
                if _is_bad_caption(caption):
                    return  # 明显不是数据表（图注/作者行等），跳过
                key = (page_num, t_type, tuple(round(v, 1) for v in bbox))
                if key in seen:
                    return
                seen.add(key)
                targets.append({
                    "page": page_num,
                    "type": t_type,
                    "caption": caption,
                    "data": data,
                    "context": context,
                    "bbox": list(bbox),
                })

            # 方式一：有框线表（find_tables，精确 bbox）
            framed_bboxes = []
            try:
                ftabs = page.find_tables().tables
            except Exception:
                ftabs = []
            for t in ftabs:
                bb = t.bbox
                try:
                    matrix = t.extract()
                except Exception:
                    matrix = []
                tbl_text = " ".join(
                    " ".join(str(c) for c in row if c) for row in matrix
                )
                t_type = classify(tbl_text)
                if not t_type:
                    continue
                caption, cap_rect = _caption_above(page, bb)
                framed_bboxes.append(tuple(bb))
                add(t_type, caption, matrix, _render_bbox(page, bb, cap_rect),
                    context=context_around(text, caption or tbl_text[:40]))

            # 方式二：无框线表（caption 锚点，且未被 find_tables 覆盖）
            # 先收集所有 caption 锚点及其位置，用相邻锚点确定每个表的下边界。
            anchors = []
            for m in caption_re.finditer(text):
                caption = m.group(1).strip()
                t_type = classify(caption) or classify(text[m.end():m.end() + 400])
                if not t_type:
                    continue
                cap_rects = page.search_for(caption[:60])
                if not cap_rects:
                    continue
                anchors.append((m, caption, t_type, cap_rects[0]))

            for i, (m, caption, t_type, cap_rect) in enumerate(anchors):
                if any(_near_above(cap_rect, fb) for fb in framed_bboxes):
                    continue  # 该 caption 属于某个有框线表，已由方式一覆盖
                next_cap_y = anchors[i + 1][3].y0 if i + 1 < len(anchors) else None
                bbox = _infer_bbox(page, cap_rect, next_cap_y)
                data_lines = _caption_body(text, m.end())
                add(t_type, caption, data_lines, bbox,
                    context=context_around(text, caption))
    finally:
        doc.close()

    return targets


def _extract_paper_meta(pdf_path: str) -> tuple:
    """从 PDF 元数据 + 首页文本兜底提取 (paper_name, doi)。"""
    paper_name = doi = None
    try:
        doc = fitz.open(pdf_path)
        try:
            meta = doc.metadata or {}
            title = meta.get("title") or None
            author = meta.get("author") or None
            if title:
                paper_name = f"{author}, {title}" if author else title
        finally:
            if doc.page_count > 0:
                text = doc[0].get_text()
                m = re.search(r"10\.\d{4,9}/[^\s\"']+", text)
                if m:
                    doi = m.group(0).rstrip(".,;")
            doc.close()
    except Exception:
        pass
    return paper_name, doi


def _render_table_image(doc, table: Dict, dpi: int = 200) -> str:
    """按表格 bbox 裁剪渲染单张 PNG，返回 base64 字符串。"""
    page = doc[table["page"] - 1]
    x0, y0, x1, y1 = table["bbox"]
    clip = fitz.Rect(x0, y0, x1, y1) & page.rect
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    return base64.b64encode(pix.tobytes("png")).decode()


def _extract_table_text(doc, table: Dict) -> str:
    """用 PyMuPDF 把表格 bbox 内的文本重建成「列对齐」的等宽表格，供 VLM 逐行核对。

    背景：get_text("text") 把表格拍成一维，列结构丢失；简单按 x 排序用 "|" 拼接又只是
    左对齐，多级表头（参数 × 分组）的列归属、以及 PDF 里折行的单元格（"316L/Ti64" 折成
    "316L" + "/Ti64" 两行）都无法表达，VLM 会把参数名当成行维度展开。

    做法（通用，不写死表结构/列数/参数名）：
    1. 按 y 聚类分行、负号合并；
    2. 识别数据行，用其 x 中心确定列基准；
    3. 表头区：最下面覆盖全部数据列的行是「逐列层」（分组名，折行合并到同一列），
       再上面是「spanning 层」（参数名/单位，按相邻标题中点重复填充到覆盖的列）；
    4. 等宽输出，一级标题重复到每个子列、二级标题逐列、数据行逐列，上下垂直对齐。
    """
    page = doc[table["page"] - 1]
    bx0, by0, bx1, by1 = table["bbox"]
    rect = fitz.Rect(bx0, by0, bx1, by1) & page.rect
    raw = page.get_text("words", clip=rect)
    if not raw:
        return ""

    ws = [[w[0], w[1], w[2], w[4]] for w in raw]  # [x0, y0, x1, text]
    ws.sort(key=lambda a: (a[1], a[0]))

    # 合并被 PDF 拆开的孤立负号（"− 0.293" -> "-0.293"）
    merged = []
    i = 0
    while i < len(ws):
        x0, y0, x1, t = ws[i]
        if (t in ("-", "−", "–") and i + 1 < len(ws)
                and ws[i + 1][3] and ws[i + 1][3][0].isdigit()
                and ws[i + 1][0] - x1 <= 6.0 and abs(ws[i + 1][1] - y0) <= 1.0):
            merged.append([x0, y0, ws[i + 1][2], t + ws[i + 1][3]])
            i += 2
        else:
            merged.append([x0, y0, x1, t])
            i += 1
    ws = merged

    # 按 y 聚类成行（容差 6pt，小于数据行距）
    rows = [[ws[0]]]
    for w in ws[1:]:
        if w[1] - rows[-1][0][1] <= 6.0:
            rows[-1].append(w)
        else:
            rows.append([w])
    for r in rows:
        r.sort(key=lambda a: a[0])

    def _is_data(r):
        nums = sum(1 for w in r if re.match(r"^-?\d+([.,]\d+)?$", w[3]))
        return nums >= max(2, len(r) * 0.6)

    # caption = 含 "Table" 的行 + 紧随其后紧邻的一行（若非数据行）
    data_rows, header_rows, caption_rows = [], [], []
    t_idx = next((k for k, r in enumerate(rows)
                  if any(w[3].strip().lower() == "table" for w in r)), None)
    if t_idx is not None:
        caption_rows.append(rows[t_idx])
        start = t_idx + 1
        if start < len(rows) and not _is_data(rows[start]):
            caption_rows.append(rows[start])
            start += 1
    else:
        start = 0
    for r in rows[start:]:
        (data_rows if _is_data(r) else header_rows).append(r)

    if not data_rows:
        return "\n".join(" | ".join(w[3] for w in r) for r in rows)

    # 列基准：数据行列数（众数）+ 每列 x 中心中位数
    cnt = {}
    for r in data_rows:
        cnt[len(r)] = cnt.get(len(r), 0) + 1
    n_cols = max(cnt, key=cnt.get)
    aligned = [r for r in data_rows if len(r) == n_cols] or data_rows
    col_centers = []
    for k in range(n_cols):
        xs = [(a[k][0] + a[k][2]) / 2 for a in aligned if k < len(a)]
        col_centers.append(sum(xs) / len(xs) if xs else 0.0)
    if n_cols < 2:
        return "\n".join(" | ".join(w[3] for w in r) for r in rows)

    data_cs = col_centers[1:]

    def _assign(xc):
        return min(range(n_cols), key=lambda k: abs(xc - col_centers[k]))

    # 表头：从下往上累加，覆盖全部数据列的行 = 逐列层
    hs = sorted(header_rows, key=lambda r: -r[0][1])
    covered = [False] * n_cols
    covered[0] = True
    layer_ids, layer_rows = set(), []
    for r in hs:
        layer_ids.add(id(r))
        layer_rows.append(r)
        for w in r:
            covered[_assign((w[0] + w[2]) / 2)] = True
        if all(covered):
            break
    spanning_rows = sorted((r for r in header_rows if id(r) not in layer_ids),
                           key=lambda r: r[0][1])

    # 逐列层：折行合并（同列多词按 y 从上往下空格拼接）
    detail_row = [""] * n_cols
    cells = {}
    for r in layer_rows:
        for w in r:
            k = _assign((w[0] + w[2]) / 2)
            cells.setdefault(k, []).append((w[1], w[3]))
    for k, items in cells.items():
        items.sort(key=lambda tup: tup[0])
        detail_row[k] = " ".join(t for _, t in items)

    grid = []
    # spanning 层：重复填充到覆盖的数据列
    for r in spanning_rows:
        order = sorted(r, key=lambda w: (w[0] + w[2]) / 2)
        row_out = [""] * n_cols
        if len(data_cs) >= 2 and len(order) < len(data_cs):
            centers = [(w[0] + w[2]) / 2 for w in order]
            mids = [(centers[j] + centers[j + 1]) / 2 for j in range(len(centers) - 1)]
            lo_edge = data_cs[0] - (data_cs[1] - data_cs[0]) / 2
            hi_edge = data_cs[-1] + (data_cs[-1] - data_cs[-2]) / 2
            for j, (w, c) in enumerate(zip(order, centers)):
                lo = mids[j - 1] if j > 0 else lo_edge
                hi = mids[j] if j < len(mids) else hi_edge
                for k, dc in enumerate(data_cs):
                    if lo <= dc < hi:
                        row_out[1 + k] = w[3]
        else:
            for w in order:
                row_out[_assign((w[0] + w[2]) / 2)] = w[3]
        grid.append(row_out)

    grid.append(detail_row)
    for r in sorted(data_rows, key=lambda rr: rr[0][1]):
        row_out = [""] * n_cols
        for w in r:
            row_out[_assign((w[0] + w[2]) / 2)] = w[3]
        grid.append(row_out)

    widths = [max(len(row[k]) for row in grid) for k in range(n_cols)]
    lines = ["  ".join(row[k].ljust(widths[k]) for k in range(n_cols)).rstrip()
             for row in grid]
    caption = "\n".join(" ".join(w[3] for w in r) for r in caption_rows)
    return (caption + "\n" + "\n".join(lines)) if caption else "\n".join(lines)


_REF_ELECTRODE_KEYWORDS = [
    "reference electrode", "saturated calomel", "calomel", "SCE", "Ag/AgCl",
    "AgCl", "mercurous", "Hg/Hg", "Ag/Ag",
]


def _extract_ref_electrode_context(doc) -> str:
    """从全文捞出与参比电极相关的句子片段，供 VLM 读 Ref_Electrode（读不到返回空）。"""
    full = "\n".join(page.get_text() for page in doc)
    hits = set()
    for kw in _REF_ELECTRODE_KEYWORDS:
        for m in re.finditer(re.escape(kw), full, re.IGNORECASE):
            s = max(0, m.start() - 150)
            e = min(len(full), m.end() + 150)
            hits.add(full[s:e].replace("\n", " ").strip())
    return "\n".join(sorted(hits)[:8]) if hits else ""


# 单表提取 prompt：不写死任何表格结构/维度。单位换算交给代码，模型只抄数值+判单位。
_TABLE_EXTRACT_PROMPT = """你是一个专门从论文中提取电化学腐蚀数据的AI。请严格按照以下要求处理。

## 任务
下图是学术论文中的一张表格。请先识别这张表格的结构（分组表头、行/列维度、合并单元格），再从中提取电化学腐蚀实验数据，以 JSON 格式输出。

## 展开规则
这是一张完整表格，你必须输出表格中的**每一行**：一条 JSON 对应一行（或一个「维度组合」，如某合金×某时间）。逐行逐列全部遍历，不要只输出第一行、不要只输出第一个合金、不要省略后面任何行。
- 表格是多行×多列（例如多个合金 × 多个时间/温度/条件）时，为每个组合各输出一条，总条数 = 各维度取值数量之积。
- 多级表头（如 icorr/Ecorr/CR 各自对应多个合金的情况）可能需要你做一些组合，来将属于同一实验条件的数据统一到一条记录当中。
- 如果你能从数据本身以及横纵表头当中提取到看起来不属于下面提到的24项输出内容的参数（比如浸没时间），请不要丢弃，而是放入note当中。

## 提取规则

1. 合金成分：如提供wt%，提取以下 24 个元素：B,C,N,Mg,Al,Si,P,S,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Y,Nb,Mo,Ce,Gd,Ta,W,Re；未提供填null。如只有牌号则填写Alloy_Name，元素填null。
2. 电化学参数：E_corr_value + E_corr_unit、I_corr_value + I_corr_unit、ba_mV_per_dec、bc_mV_per_dec、CR_value + CR_unit（单位判断见下，只抄数值+判单位，**不要做乘法/小数点/10^x 换算**）。
3. 环境条件：Electrolyte, pH, Temp_C, Test_Method, Ref_Electrode。
4. 论文信息：Paper_Name, DOI（若表格图中读不到则填 null）
5. 严禁臆造：ba_mV_per_dec、bc_mV_per_dec 等字段，只有在该表里明确出现时才填写，否则一律 null。合金成分中若出现不在上方 24 元素列中的元素（如 Zr、Sn、Hf、O 等），必须把数值写进 Notes（如 "Zr:0.5"），不要丢弃。

## balance 的含义
表格里写 "balance"（或 "bal."、"余量"）表示：该元素是「余量」，等于 100 减去表内其它已列出元素的 wt% 之和。例如 Ni 13.6、Mo 2.9、Mn 2.0、Si 1.0、C 0.03、Fe balance，则 Fe = 100 − (13.6+2.9+2.0+1.0+0.03) = 80.47。此时该元素列**填字符串 "balance"**（不要填 null、不要填 100、不要填 0）。相关解释无需填入notes。

## 单位判断
E_corr_value 照抄表里的腐蚀电位原始数值；E_corr_unit 判断更像 "mV" 还是 "V"（绝对值 >10 通常 mV，<10 通常 V）。CR_value 照抄腐蚀速率原始数值；CR_unit 填 "mm/y" 或 "um/y"。**不要乘 1000、不要移动小数点、不要补 0。**

输出格式示例（每行一个JSON）：
{"Alloy_Name":"304L","B":null,"C":0.03,"N":null,"Mg":null,"Al":null,"Si":1.00,"P":null,"S":null,"Ti":null,"V":null,"Cr":18.00,"Mn":1.00,"Fe":69.71,"Co":null,"Ni":10.00,"Cu":null,"Y":null,"Nb":null,"Mo":null,"Ce":null,"Gd":null,"Ta":null,"W":null,"Re":null,"Electrolyte":"300 ppm NaCl","pH":7.8,"Temp_C":50,"Test_Method":"Potentiodynamic","Ref_Electrode":"SCE","E_corr_value":-0.288,"E_corr_unit":"V","I_corr_value":45.91,"I_corr_unit":"uA/cm2","ba_mV_per_dec":80.7,"bc_mV_per_dec":-186.2,"CR_value":531.7,"CR_unit":"um/y","Paper_Name":"Gaber et al. 2020","DOI":"10.4152/pea.202003127"}

解释：
Alloy_Name: 论文中给出的合金牌号或成分标识。直接填写。
B/C/N/Mg/Al/Si/P/S/Ti/V/Cr/Mn/Fe/Co/Ni/Cu/Y/Nb/Mo/Ce/Gd/Ta/W/Re: 各元素含量(wt%)。直接从论文提取wt%，若给出at%则换算为wt%，若同时给出则只取wt%；写 balance 则参见balance的含义。未提供则null。如果给出的成分表对同一种合金当中的同一种元素给出了不同的各元素含量，请将元素在各点处的含量取平均作为该合金当中该元素的含量输出。也就是说，对于一种合金，只能有一种确定的组分。
如果遇到合金当中包括不包含在以上元素当中的成分，请将这些无法写入元素列的内容写入notes当中。
Electrolyte: 电解质描述。直接填写原文描述。
pH: pH值。直接填写。未提供则null。
Temp_C: 测试温度(℃)。论文明确给出则直接填写；未明确指定则填25。
Test_Method: 测试方法。填真正采用的实验测试方法（如 Potentiodynamic polarization、EIS、线性极化等）。注意：「Tafel」只是对极化曲线做外推/拟合的数据处理方式，不是测试方法本身——若表格写 "obtained by Tafel"（或 "Tafel extrapolation"），测试方法通常是 Potentiodynamic，请结合论文正文/图注判断。
Ref_Electrode: 参比电极类型。优先从下方「论文正文片段」里读出（如 SCE、Ag/AgCl、饱和甘汞电极）；读不到就填 null。
E_corr_value / E_corr_unit: 见「单位判断」。原始数值照抄 + 判断 mV/V，不要换算。
I_corr_value / I_corr_unit: 腐蚀电流密度。I_corr_value 照抄表里的原始数值；I_corr_unit 判断该列单位——若 icorr 是 log 值（负的、或表头带 log/ln），填 "log10"；否则按表头单位填 "A/cm2"、"mA/cm2" 或 "uA/cm2"。不要换算，只抄数值+判单位。未提供则null。
ba_mV_per_dec: 阳极Tafel斜率(mV/dec)。未提供则null。
bc_mV_per_dec: 阴极Tafel斜率(mV/dec)。保持负值。未提供则null。
CR_value / CR_unit: 见「单位判断」。原始数值照抄 + 判断 mm/y 或 um/y，不要换算。
Paper_Name: 论文标题。
DOI: DOI号。填写完整DOI。未提供则null。
Notes: 备注。记录论文中未纳入其他列的补充信息，简明扼要，不要长篇大论。
重要：只输出JSON格式数据，字段必须严格限制在：Alloy_Name, B, C, N, Mg, Al, Si, P, S, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Y, Nb, Mo, Ce, Gd, Ta, W, Re, Electrolyte, pH, Temp_C, Test_Method, Ref_Electrode, E_corr_value, E_corr_unit, I_corr_value, I_corr_unit, ba_mV_per_dec, bc_mV_per_dec, CR_value, CR_unit, Paper_Name, DOI, Notes。不要输出其他字段。"""


def _parse_vlm_json(text: str) -> List[Dict]:
    """鲁棒解析 VLM 返回里的 JSON 对象。

    兼容：单个对象、JSON 数组、多个对象拼接、markdown 代码围栏、以及
    pretty-print 的多行对象（旧实现按行切、只认单行 `{...}`，会把多行对象整段丢弃）。
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    objs: List[Dict] = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\r\n,;":
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1  # 跳过一段非 JSON 文本后继续
            continue
        if isinstance(obj, dict):
            objs.append(obj)
        elif isinstance(obj, list):
            objs.extend(item for item in obj if isinstance(item, dict))
        idx = end
    return objs


_KNOWN_FIELDS = set(FIELD_NAMES) | {"E_corr_value", "E_corr_unit", "I_corr_value", "I_corr_unit", "CR_value", "CR_unit"}


def _fold_extra_fields_to_notes(rows: List[Dict]) -> List[Dict]:
    """把 schema 外的字段（如 V/N/P/S）折叠进 Notes，避免被 CSV 落盘时静默丢弃。"""
    for row in rows:
        extras = [f"{k}: {row.pop(k)}" for k in list(row.keys()) if k not in _KNOWN_FIELDS]
        if extras:
            existing = row.get("Notes")
            row["Notes"] = (f"{existing}; " if existing else "") + "; ".join(extras)
    return rows


def _as_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _unit_is_volts(u) -> bool:
    """True=伏特(需×1000), False=毫伏/未知(不乘)。"""
    s = str(u or "").lower().replace(" ", "")
    if "mv" in s:
        return False
    return "v" in s


def _unit_is_mm_per_y(u) -> bool:
    """True=mm/y(需×1000→μm/y), False=μm/y 或未知(不乘)。"""
    s = str(u or "").lower().replace(" ", "")
    return "mm" in s


def _compute_balance(rows: List[Dict]) -> List[Dict]:
    """把标记为 balance 的元素回填为 100 − 其余已列出元素之和。"""
    for row in rows:
        bal = [el for el in ELEMENTS
               if str(row.get(el)).strip().lower() in ("balance", "bal", "bal.")]
        if not bal:
            continue
        s = sum(_as_float(row.get(el)) or 0.0 for el in ELEMENTS if el not in bal)
        for el in bal:
            row[el] = round(100 - s, 2)
    return rows


def _i_corr_to_ua_per_cm2(value: float, unit) -> float:
    """把 VLM 的「原始数值 + 单位」换算成 μA/cm²。

    - "log10"/"log"：值是 log10(icorr)，10^x 即得 μA/cm²；
    - "A/cm2"：×1e6；"mA/cm2"：×1e3；"uA/cm2"/"μA/cm2"：×1。
    """
    u = str(unit or "").lower().replace(" ", "").replace("μ", "u").replace("µ", "u").replace("²", "2")
    if "log" in u:
        return 10.0 ** value
    if "ma/cm2" in u:
        return value * 1e3
    if "ua/cm2" in u:
        return value  # 微安，已是 μA/cm²
    if "a/cm2" in u:
        return value * 1e6
    return value  # 默认 μA/cm²


def _normalize_units(rows: List[Dict]) -> List[Dict]:
    """把 VLM 输出的「原始数值 + 单位判断」换算成规范字段（mV / μA/cm² / μm/年）。

    VLM 只负责抄原始数值、判单位（mV/V、log10/A/mA/uA、mm/y/um/y），乘法/10^x
    换算在这里由代码完成，避免模型口算时漏 0 / 点错小数点。
    """
    for row in rows:
        val = row.pop("E_corr_value", None)
        unit = row.pop("E_corr_unit", None)
        v = _as_float(val)
        if v is not None:
            row["E_corr_mV"] = v * 1000 if _unit_is_volts(unit) else v

        ival = row.pop("I_corr_value", None)
        iunit = row.pop("I_corr_unit", None)
        iv = _as_float(ival)
        if iv is not None:
            row["I_corr_uA_per_cm2"] = _i_corr_to_ua_per_cm2(iv, iunit)

        # 兜底：负的电流密度物理上不可能，必是 log10 值被误判（VLM 判错单位、
        # 或直接给了负的 I_corr_uA_per_cm2）→ 10^x 转成 μA/cm²。
        cur = row.get("I_corr_uA_per_cm2")
        cv = _as_float(cur)
        if cv is not None and cv < 0:
            row["I_corr_uA_per_cm2"] = round(10.0 ** cv, 6)

        cval = row.pop("CR_value", None)
        cunit = row.pop("CR_unit", None)
        c = _as_float(cval)
        if c is not None:
            row["CR_um_per_y"] = c * 1000 if _unit_is_mm_per_y(cunit) else c
    return rows


_SALT_CL = {
    "NaCl": (58.44, 1), "KCl": (74.55, 1), "HCl": (36.46, 1),
    "FeCl3": (162.20, 3), "FeCl2": (126.75, 2), "CaCl2": (110.98, 2),
    "MgCl2": (95.21, 2), "AlCl3": (133.34, 3), "CuCl2": (134.45, 2),
    "ZnCl2": (136.29, 2), "NH4Cl": (53.49, 1), "LiCl": (42.39, 1),
}


def _salt_to_cl_molar(conc: float, unit, mm: float, cl: int) -> float:
    """单一氯盐的浓度 → 贡献的 [Cl-]（mol/L）。算术在代码里做，避免模型口算出错。"""
    u = str(unit or "").lower().replace(" ", "")
    if "mm" in u:
        return conc * 1e-3 * cl
    if "wt" in u or "%" in u:
        return conc * 10.0 / mm * cl      # 假设密度≈水 1 g/mL
    if "ppm" in u:
        return conc * 1e-3 / mm * cl      # mg/L → g/L
    if "g/l" in u:
        return conc / mm * cl
    return conc * cl                       # 默认 M


def compute_cl_molarity(electrolyte: str) -> float | None:
    """用本地 VLM 解析电解质的氯盐成分，代码换算成 [Cl-]（mol/L）。

    VLM 只做「识别氯盐 + 抄浓度/单位」（解析难、正则难覆盖），乘法换算交给代码，
    避免模型口算漏 0 / 点错小数点（与 _normalize_units 同一原则）。
    """
    text = str(electrolyte or "").strip()
    if not text:
        return None
    prompt = (
        "你是电化学实验数据整理助手。下面是某腐蚀实验的「电解质」描述，格式任意。\n\n"
        f"电解质：{text}\n\n"
        "请识别其中是否含氯离子 Cl⁻：若含，列出其中的氯盐成分（盐名、浓度数值、浓度单位），"
        "只输出 JSON：{\"salts\": [{\"salt\": \"NaCl\", \"conc\": 3.5, \"unit\": \"wt%\"}]}\n"
        "若不含氯（如 H2SO4、NaOH、Na2SO4、KSCN、H2O2、纯水等），输出 {\"salts\": null}。\n\n"
        "规则：\n"
        "- 只列含氯的盐（NaCl/KCl/HCl/FeCl3/CaCl2/MgCl2/AlCl3 等）；不含氯的盐（Na2SO4、KSCN、H2O2 等）不要列。\n"
        "- conc 直接抄原文字面数值，unit 从原文判断（wt%、%、ppm、M、mM、g/L），不要做任何换算。\n"
        "- 一个电解质里多个氯盐都列出来（如 \"3.5 wt% NaCl + 0.2 M HCl\" → 两条）。\n"
        "- 若电解质是俗称、列不出盐名（如 \"seawater\"），直接输出 {\"cl_molar\": 0.55}（给估计值）；判断不出就 {\"cl_molar\": null}。\n"
        "只输出 JSON，不要多余文字。"
    )
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME, "prompt": prompt, "stream": False, "temperature": 0.05,
        }, timeout=120)
        raw = resp.json().get("response", "").strip()
    except Exception as exc:
        print(f"    Cl⁻ 解析失败({text[:30]!r}): {exc}")
        return None
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    # 俗称兜底：模型直接给 cl_molar（seawater 等）
    if "cl_molar" in obj and obj.get("cl_molar") is not None:
        v = _as_float(obj.get("cl_molar"))
        if v is not None and 0.0 <= v <= 6.0:
            return round(v, 4)
    # 正常路径：代码按盐表换算
    salts = obj.get("salts")
    if not isinstance(salts, list) or not salts:
        return None
    total = 0.0
    for s in salts:
        if not isinstance(s, dict):
            return None
        name = str(s.get("salt", "")).strip()
        if name not in _SALT_CL:
            return None
        conc = _as_float(s.get("conc"))
        if conc is None or conc < 0:
            return None
        mm, cl = _SALT_CL[name]
        total += _salt_to_cl_molar(conc, s.get("unit"), mm, cl)
    if total <= 0 or total > 6.0:
        return None
    return round(total, 4)


_cl_cache: Dict[str, float] = {}


def _fill_cl_molarity(rows: List[Dict]) -> List[Dict]:
    """为每条记录补 Cl_M（氯离子摩尔浓度），按电解质字符串缓存、幂等（已有则跳过）。"""
    for row in rows:
        if row.get("Cl_M") is not None:
            continue
        el = str(row.get("Electrolyte") or "").strip()
        if not el:
            continue
        if el not in _cl_cache:
            _cl_cache[el] = compute_cl_molarity(el)
        row["Cl_M"] = _cl_cache[el]
    return rows


def _call_vlm_for_table(img_b64: str, table_text: str, ref_electrode_ctx: str, paper_name, doi) -> List[Dict]:
    """对单张表格图调用 VLM，返回解析出的 JSON 行列表。"""
    prompt = _TABLE_EXTRACT_PROMPT
    if table_text and table_text.strip():
        prompt += (
            "\n\n## 表格原文（已从 PDF 抽取，与图片同源）\n"
            + table_text.strip()
            + "\n\n请依据上面的原文逐行核对、逐行展开，不要漏掉任何一行。"
        )
    if ref_electrode_ctx and ref_electrode_ctx.strip():
        prompt += (
            "\n\n## 论文正文片段（用于读 Ref_Electrode）\n"
            + ref_electrode_ctx.strip()
            + "\n\n若片段中没写参比电极，Ref_Electrode 填 null。"
        )
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "images": [img_b64],
            "temperature": 0.05,
            "stream": False,
        }, timeout=1200)
        full_response = response.json().get("response", "").strip()
    except Exception as exc:
        print(f"    VLM 调用失败: {exc}")
        return []

    data_list = _parse_vlm_json(full_response)
    for row in data_list:
        # 兜底回填 Paper_Name / DOI（单表图里 VLM 通常读不到）
        if paper_name and not row.get("Paper_Name"):
            row["Paper_Name"] = paper_name
        if doi and not row.get("DOI"):
            row["DOI"] = doi

    return data_list


def extract_from_pdf(pdf_path: str) -> List[Dict]:
    """主流程：定位目标表格 -> 单表裁剪渲染 -> 单表单次 VLM -> 汇总 JSON 行。"""
    print("Step 1: 定位目标表格...")
    tables = locate_target_tables(pdf_path)

    if not tables:
        print("未找到目标表格，跳过")
        return []

    print(f"找到 {len(tables)} 个目标表格")
    for t in tables:
        print(f"  - 第{t['page']}页, 类型: {t['type']}, bbox: {t['bbox']}, 表头: {t['caption'][:40]}...")

    paper_name, doi = _extract_paper_meta(pdf_path)

    doc = fitz.open(pdf_path)
    ref_ctx = _extract_ref_electrode_context(doc)
    data_list = []
    try:
        for idx, table in enumerate(tables, 1):
            img_b64 = _render_table_image(doc, table, dpi=200)
            table_text = _extract_table_text(doc, table)
            print(f"Step 3[{idx}/{len(tables)}]: 调用模型识别第{table['page']}页 {table['type']} 表...")
            rows = _call_vlm_for_table(img_b64, table_text, ref_ctx, paper_name, doi)
            rows = _fold_extra_fields_to_notes(rows)
            rows = _compute_balance(rows)
            rows = _normalize_units(rows)
            print(f"    -> {len(rows)} 条")
            data_list.extend(rows)
    finally:
        doc.close()

    # 给「有牌号、无成分」的 ecorr 记录补成分（候选来自同批成分表记录）
    data_list = enrich_table_records(data_list)
    # 从 Electrolyte 文本解析氯离子摩尔浓度 Cl_M
    data_list = _fill_cl_molarity(data_list)

    return data_list


def write_to_csv(data_list, output_path, append=True):
    if not data_list:
        print("没有提取到数据")
        return

    fieldnames = list(FIELD_NAMES)

    file_exists = Path(output_path).exists()
    mode = 'a' if append else 'w'

    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or not append:
            writer.writeheader()

        # 过滤多余字段
        cleaned_data = []
        for row in data_list:
            cleaned_row = {key: row.get(key) for key in fieldnames}
            cleaned_data.append(cleaned_row)
        writer.writerows(cleaned_data)

    print(f"已写入 {len(cleaned_data)} 行数据到 {output_path}")


# ==================== 图像通路补全 ====================


def _match_alloy_names_vlm(image_names: List[str], candidates: Dict) -> Dict[str, str]:
    """让本地模型把图像通路的合金名模糊匹配到文本通路的候选合金。

    ``image_names`` 是图像通路每条记录里的 Alloy_Name；``candidates`` 是
    文本通路抽到的 {合金名: 成分dict}。返回 {图像名: 候选名} 映射（匹配不上的
    value 为 null）。
    """
    prompt = f"""你是电化学论文数据整理助手。下面是同一篇论文里出现的合金名。

图像通路（来自图注）给出的合金名：{json.dumps(image_names, ensure_ascii=False)}

文本通路（来自论文成分表）给出的候选合金及成分：{json.dumps(candidates, ensure_ascii=False)}

请判断每个「图像通路合金名」对应哪个「候选合金」（允许模糊匹配，例如
"316L" 对应 "316L stainless steel" 或 "AISI 316L"；"Ti6Al4V" 对应 "Ti-6Al-4V"）。
只输出 JSON，格式：{{"图像通路合金名": "候选合金名", ...}}，匹配不上的 value 用 null。"""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.05,
        }, timeout=120)
        raw = resp.json().get("response", "").strip()
    except Exception:
        return {}

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


def enrich_image_records(image_records: List[Dict], table_records: List[Dict]) -> List[Dict]:
    """给图像通路记录补成分 + Paper_Name/DOI。

    - 成分：由本地模型把图像通路 Alloy_Name 模糊匹配到文本通路候选，再填元素列。
    - Paper_Name/DOI：直接取文本通路任意一条（同一篇论文）。
    匹配不上的成分留 null，Alloy_Name 已保留在 Notes 里备查。
    """
    if not image_records:
        return image_records

    # 候选合金（成分至少有一个非 null 元素）。
    candidates: Dict = {}
    for r in table_records:
        name = r.get("Alloy_Name")
        if not name:
            continue
        comp = {e: r.get(e) for e in ELEMENTS if r.get(e) is not None}
        if comp and name not in candidates:
            candidates[name] = comp

    paper_name = next((r.get("Paper_Name") for r in table_records if r.get("Paper_Name")), None)
    doi = next((r.get("DOI") for r in table_records if r.get("DOI")), None)

    image_names = sorted({r.get("Alloy_Name") for r in image_records if r.get("Alloy_Name")})
    name_map = _match_alloy_names_vlm(image_names, candidates) if image_names and candidates else {}

    for r in image_records:
        r["Paper_Name"] = paper_name
        r["DOI"] = doi
        target = name_map.get(r.get("Alloy_Name"))
        if target and target in candidates:
            for e in ELEMENTS:
                r[e] = candidates[target].get(e)
            if r.get("Notes"):
                r["Notes"] += f"; 成分来自文本通路({target})"
    return image_records


def enrich_table_records(records: List[Dict]) -> List[Dict]:
    """给文本通路里「有牌号、无成分」的记录（如 ecorr 表的 316L/Ti64）补成分。

    候选来自同一批记录里「有成分」的成分表记录，用 VLM 模糊匹配牌号（同图像通路）。
    """
    if not records:
        return records

    # 候选合金（成分至少有一个非 null 元素）
    candidates: Dict = {}
    for r in records:
        name = r.get("Alloy_Name")
        if not name:
            continue
        comp = {e: r.get(e) for e in ELEMENTS if r.get(e) is not None}
        if comp and name not in candidates:
            candidates[name] = comp

    if not candidates:
        return records

    # 目标：有牌号但无任何元素成分
    targets = [r for r in records
               if r.get("Alloy_Name") and not any(r.get(e) is not None for e in ELEMENTS)]
    if not targets:
        return records

    target_names = sorted({r.get("Alloy_Name") for r in targets})
    name_map = _match_alloy_names_vlm(target_names, candidates)

    for r in targets:
        target = name_map.get(r.get("Alloy_Name"))
        if target and target in candidates:
            for e in ELEMENTS:
                r[e] = candidates[target].get(e)
            if r.get("Notes"):
                r["Notes"] += f"; 成分来自成分表({target})"
    return records


def process_one_paper(pdf_path, text_csv=TEXT_OUTPUT_CSV, image_csv=IMAGE_OUTPUT_CSV) -> tuple:
    """处理一篇论文：文本通路 → 存 text CSV；图像通路 → 补全 → 存 image CSV。

    流式落盘、不累积：文本一做完就 append 进 text CSV，图像一做完就 append 进 image CSV，
    谁也不等谁。图像通路每篇论文用独立 output_root，避免 page_XXX 互相覆盖。
    """
    from curve_pipeline import run_image_leg

    pdf_path = Path(pdf_path)
    print(f"\n[paper] 处理: {pdf_path.name}")

    # 文本通路（extract_from_pdf 内部已补 Cl_M）
    table_data = extract_from_pdf(str(pdf_path))
    write_to_csv(table_data, text_csv, append=True)
    print(f"[paper]   文本 {len(table_data)} 条 -> {text_csv}")

    # 图像通路：半成品 → 补成分/Paper/DOI → 补 Cl_M
    image_root = Path("image_leg_output") / pdf_path.stem
    image_records = run_image_leg(str(pdf_path), output_root=image_root)
    image_records = enrich_image_records(image_records, table_data)
    image_records = _fill_cl_molarity(image_records)
    write_to_csv(image_records, image_csv, append=True)
    print(f"[paper]   图像 {len(image_records)} 条 -> {image_csv}")
    return table_data, image_records


if __name__ == "__main__":
    pdfs = sorted(SOURCE_PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"[main] {SOURCE_PAPERS_DIR} 下没有 PDF，跳过")
    else:
        # 清空旧输出，避免上次运行残留
        for p in (TEXT_OUTPUT_CSV, IMAGE_OUTPUT_CSV):
            Path(p).unlink(missing_ok=True)
        print(f"[main] 共 {len(pdfs)} 篇论文，串行处理")
        for pdf in pdfs:
            process_one_paper(pdf)
        print(f"\n[main] 完成 -> {TEXT_OUTPUT_CSV} / {IMAGE_OUTPUT_CSV}")