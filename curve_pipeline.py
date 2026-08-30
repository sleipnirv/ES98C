"""
Image leg of the ES98C pipeline: PDF page -> Tafel panels -> extracted curves -> records.

This module turns the *chart* extraction path (previously exercised only in
``test_pipeline.ipynb``) into a callable pipeline that ``project_v2.py`` can
invoke alongside the existing table/text leg.

Flow per paper:
1. Pre-filter figure pages (regex on "Fig. N" caption + polarization keywords).
2. Render each candidate page to PNG.
3. ``analyze_page`` (VLM) -> title / per-panel condition / Tafel classification.
4. Full-page ChartDete -> axis boxes -> crop individual Tafel panels.
5. ``run_pipeline`` (LineFormer + ChartDete + VLM axis OCR + coordinate convert).
6. ``match_legends_for_directory`` -> series -> legend-label binding.
7. ``data_cleaner`` -> fit Butler-Volmer Tafel asymptotes -> E_corr / log_i_corr / b_a / b_c.
8. Binding VLM (ONE call per panel) -> alloy / electrolyte / environment fields
   + the panel's axis units.  The unit interpretation is a per-panel constant,
   so the numeric conversion is applied to *all* series of that panel at once
   (never per point, never per series).
9. Emit one record per labelled series, tagged "来自图像" in Notes.

The output records are *half-finished*: alloy composition (Fe/Cr/Ni/...),
Paper_Name and DOI are deliberately left empty here — ``project_v2.py`` fills
them from its own text/table extraction (composition lookup + paper metadata).
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Optional

import requests

# --- Local imports (require this repo's Codes/ on sys.path) ----------------
# Light imports only at module load; the heavy chart_extractor bits
# (torch / mmdet / transformers) are imported lazily inside functions so that
# cheap helpers like ``select_figure_pages`` stay importable without them.
from page_analyzer.analyzer import analyze_page, crop_tafel_panels
from data_cleaner.bv_cleaner import clean_labeled_series


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
VLM_MODEL = "qwen3-vl:8b-instruct-q4_K_M"

# 39-column schema, shared with project_v2.py (24 元素周期表序 + Cl_M)。
FIELD_NAMES = [
    "Alloy_Name",
    "B", "C", "N", "Mg", "Al", "Si", "P", "S", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Y", "Nb", "Mo", "Ce", "Gd", "Ta", "W", "Re",
    "Electrolyte", "Cl_M", "pH", "Temp_C", "Test_Method", "Ref_Electrode",
    "E_corr_mV", "I_corr_uA_per_cm2", "ba_mV_per_dec", "bc_mV_per_dec", "CR_um_per_y",
    "Paper_Name", "DOI", "Notes",
]

# Figure-page pre-filter: a "Fig. N" caption plus at least one corrosion keyword.
FIG_CAPTION_RE = re.compile(r"\bfig(?:ure)?s?\.?\s*\d+", re.IGNORECASE)
POLAR_KEYWORD_RE = re.compile(
    r"polarization|potentiodynamic|tafel|corrosion\s*potential|current\s*density"
    r"|极化曲线|动电位|塔菲尔|腐蚀电位|电流密度",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Step 1: figure-page selection
# ---------------------------------------------------------------------------
def select_figure_pages(pdf_path: str | Path) -> list[int]:
    """Return 1-indexed page numbers that look like polarization-figure pages.

    Cheap and deterministic: no rendering, no VLM.  ``analyze_page`` re-checks
    each candidate later, so false positives (a page that only *mentions*
    "Fig. 3" in prose) are dropped there.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        import pdfplumber

        pages: list[int] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for idx, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if FIG_CAPTION_RE.search(text) and POLAR_KEYWORD_RE.search(text):
                    pages.append(idx)
        return pages

    reader = PdfReader(str(pdf_path))
    pages: list[int] = []
    for idx, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if FIG_CAPTION_RE.search(text) and POLAR_KEYWORD_RE.search(text):
            pages.append(idx)
    return pages


# ---------------------------------------------------------------------------
# Step 2: render a page to PNG
# ---------------------------------------------------------------------------
def render_page(pdf_path: str | Path, page_num: int, out_dir: str | Path, dpi: int = 200) -> Path:
    import fitz

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    try:
        if page_num < 1 or page_num > len(doc):
            raise ValueError(f"Page {page_num} out of range for {pdf_path}")
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi)
    finally:
        doc.close()
    out_path = out_dir / f"page_{page_num:03d}.png"
    pix.save(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Step 4: full-page ChartDete (only to locate axis boxes for panel cropping)
# ---------------------------------------------------------------------------
def run_fullpage_chartdete(page_png: str | Path, out_dir: str | Path) -> Path:
    """Run ChartDete on the *whole* page and return its bounding_boxes.json.

    ``crop_tafel_panels`` needs ``x_axis_area`` / ``y_axis_area`` at page scale
    to spatially pair the axes of each panel.  This is a separate, heavier step
    from the per-panel ChartDete run inside ``run_pipeline``.
    """
    page_png = Path(page_png)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ChartDete.inference() processes a whole directory, so give it one image
    # per run.  A per-page staging dir keeps successive pages from piling up.
    staging = out_dir / f"_staging_{page_png.stem}"
    staging.mkdir(parents=True, exist_ok=True)
    staged = staging / page_png.name
    if staged.resolve() != page_png.resolve():
        import shutil

        shutil.copy2(page_png, staged)

    from chart_extractor.src.plextract.local.chartdete import ChartDete

    det = ChartDete(device="cpu")
    det.inference(str(staging), str(out_dir))

    bb_path = out_dir / page_png.name / "chartdete" / "bounding_boxes.json"
    if not bb_path.exists():
        raise FileNotFoundError(f"Full-page ChartDete produced no output at {bb_path}")
    return bb_path


# ---------------------------------------------------------------------------
# Step 8: binding VLM (one call per panel)
# ---------------------------------------------------------------------------
_BIND_PROMPT = """You are extracting structured corrosion data from one panel of a scientific figure.

Given:
- Panel condition (free text): {condition}
- Page context (extracted from surrounding text): {context}
- Figure title/caption: {title}
- X-axis title: {x_title}
- Y-axis title: {y_title}

Determine these fields and the axis units, then answer in this exact JSON shape:
{{
  "Alloy_Name": "e.g. 316L",
  "Electrolyte": "e.g. 0.9% NaCl",
  "Ref_Electrode": "SCE / Ag/AgCl / ... or null",
  "Test_Method": "e.g. Potentiodynamic",
  "pH": null,
  "Temp_C": 25,
  "x_unit": "V",
  "y_is_log10": true,
  "y_unit": "A/cm2"
}}

Rules:
- Alloy_Name: the alloy grade mentioned in the condition/caption (e.g. "316L", "Ti6Al4V"). Use null only if truly absent.
- Electrolyte: the medium and concentration (e.g. "3.5% NaCl").
- Ref_Electrode / Test_Method: take from the context/title when present, else null.
- pH: number if stated, else null. Temp_C: number if stated, else 25.
- x_unit: the potential unit on the x-axis, exactly one of "V" or "mV".
- y_is_log10: true if the y-axis is log10 scale, false if natural log or linear.
- y_unit: the current-density unit (before any log), exactly one of "A/cm2", "mA/cm2", "uA/cm2".

Do not invent data. Use null for genuinely missing values."""


def bind_panel_vlm(condition: str, context: dict, title: Optional[str],
                   x_title: str, y_title: str) -> dict:
    """Ask the VLM to parse one panel's condition + axis units (single call)."""
    prompt = _BIND_PROMPT.format(
        condition=condition or "",
        context=json.dumps(context or {}, ensure_ascii=False),
        title=title or "",
        x_title=x_title or "",
        y_title=y_title or "",
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": VLM_MODEL, "prompt": prompt, "stream": False, "temperature": 0.05},
            timeout=120,
        )
        raw = resp.json().get("response", "").strip()
    except Exception as exc:
        raise RuntimeError(f"Binding VLM failed: {exc}") from exc

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            result = json.loads(raw[start:end + 1])
        else:
            raise RuntimeError(f"Binding VLM did not return valid JSON:\n{raw[:500]}")

    # Defaults so a partially-filled response still yields usable records.
    result.setdefault("Alloy_Name", None)
    result.setdefault("Electrolyte", None)
    result.setdefault("Ref_Electrode", None)
    result.setdefault("Test_Method", None)
    result.setdefault("pH", None)
    result.setdefault("Temp_C", 25)
    result.setdefault("x_unit", "V")
    result.setdefault("y_is_log10", True)
    result.setdefault("y_unit", "A/cm2")
    return result


# ---------------------------------------------------------------------------
# 轴朝向检测：极化曲线通常 x=电位、y=log 电流，但有的论文旋转 90°（x=电流、y=电位）。
# 检测电位在哪个轴，旋转的就交换数据点 x/y + 交换标题，统一成标准朝向，下游不用改。
# ---------------------------------------------------------------------------
_POTENTIAL_KW = ("potential", "voltage", "电位", "电压")
_CURRENT_KW = ("current", "density", "电流", "密度", "log")


def _axis_looks_potential(title: str):
    """规则判断轴是否为电位轴。True/False/None（None=判不出）。"""
    t = (title or "").lower()
    pot = any(k in t for k in _POTENTIAL_KW)
    cur = any(k in t for k in _CURRENT_KW)
    if pot and not cur:
        return True
    if cur and not pot:
        return False
    return None


def _vlm_detect_potential_axis(x_title: str, y_title: str) -> str:
    """规则判不出时，让 VLM 判断电位在哪个轴。失败默认 'x'（标准朝向）。"""
    prompt = (
        "判断一个电化学极化曲线图中，横轴和纵轴哪个是「电位(电压)」、哪个是「电流密度」。\n"
        f"横轴标题：{x_title}\n纵轴标题：{y_title}\n"
        "只输出 JSON：{\"potential_axis\": \"x\"} 或 {\"potential_axis\": \"y\"}。"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": VLM_MODEL, "prompt": prompt, "stream": False, "temperature": 0.05},
            timeout=60,
        )
        raw = resp.json().get("response", "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
        obj = json.loads(raw)
        return "y" if obj.get("potential_axis") == "y" else "x"
    except Exception:
        return "x"


def detect_potential_axis(x_title: str, y_title: str) -> str:
    """返回 'x' 或 'y'：电位在哪个轴。规则优先（含排除法），判不出才 VLM。"""
    xp = _axis_looks_potential(x_title)
    yp = _axis_looks_potential(y_title)
    if xp is True and yp is not True:
        return "x"
    if yp is True and xp is not True:
        return "y"
    if xp is False and yp is not False:  # x 明确是电流 → y 是电位
        return "y"
    if yp is False and xp is not False:  # y 明确是电流 → x 是电位
        return "x"
    return _vlm_detect_potential_axis(x_title, y_title)  # 双 True/False/None 才求助模型


# ---------------------------------------------------------------------------
# Step 8b: apply the panel's unit interpretation to every series at once
# ---------------------------------------------------------------------------
_Y_UNIT_TO_UA_PER_CM2 = {
    "a/cm2": 1e6,
    "a/cm²": 1e6,
    "ma/cm2": 1e3,
    "ma/cm²": 1e3,
    "ua/cm2": 1.0,
    "ua/cm²": 1.0,
    "μa/cm2": 1.0,
    "μa/cm²": 1.0,
}


def convert_params(params: dict, unit_info: dict) -> dict:
    """Convert one series' fitted BV params to the CSV's absolute units.

    ``params`` comes straight from ``data_cleaner.fit_tafel_bv`` (E_corr in the
    x-axis unit, log_i_corr in the y-axis log, b_a/b_c as positive V/decade).
    The unit interpretation is a *per-panel* constant (``unit_info``), so this
    is pure arithmetic applied uniformly — the VLM already did the judging.
    """
    x_to_mv = 1000.0 if str(unit_info.get("x_unit", "V")).strip().lower() == "v" else 1.0

    log_val = float(params["log_i_corr"])
    i_corr = (10.0 ** log_val) if unit_info.get("y_is_log10", True) else math.exp(log_val)
    y_scale = _Y_UNIT_TO_UA_PER_CM2.get(
        str(unit_info.get("y_unit", "A/cm2")).strip().lower(), 1e6
    )

    # b_c is stored as a positive magnitude; the CSV schema wants it negative.
    return {
        "E_corr_mV": round(float(params["E_corr"]) * x_to_mv, 4),
        "I_corr_uA_per_cm2": round(i_corr * y_scale, 4),
        "ba_mV_per_dec": round(float(params["b_a"]) * x_to_mv, 4),
        "bc_mV_per_dec": round(-float(params["b_c"]) * x_to_mv, 4),
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run_image_leg(
    pdf_path: str | Path,
    *,
    output_root: Optional[str | Path] = None,
    dpi: int = 200,
) -> list[dict]:
    """Run the whole image leg and return the per-series records.

    Returns a list of dicts in the 24-field schema (composition / Paper_Name /
    DOI left empty — filled later by ``project_v2.py``).
    """
    pdf_path = Path(pdf_path)
    if output_root is None:
        output_root = Path.cwd() / "image_leg_output"
    output_root = Path(output_root)
    pages_dir = output_root / "pages"
    fullpage_dir = output_root / "fullpage"
    crops_dir = output_root / "crops"
    extract_dir = output_root / "extract"
    for d in (pages_dir, fullpage_dir, crops_dir, extract_dir):
        d.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    # 1. Which pages have polarization figures?
    figure_pages = select_figure_pages(pdf_path)
    print(f"[image] figure candidate pages: {figure_pages}")
    if not figure_pages:
        print("[image] no figure pages found; image leg produces nothing.")
        return records

    for page_num in figure_pages:
        try:
            records.extend(_process_page(page_num, pdf_path, dpi,
                                         pages_dir, fullpage_dir, crops_dir, extract_dir))
        except Exception as exc:
            print(f"[image] page {page_num} FAILED: {exc}")

    _write_csv(records, output_root / "extracted_data_image.csv")
    print(f"[image] wrote {len(records)} records -> {output_root / 'extracted_data_image.csv'}")
    return records


def _process_page(page_num, pdf_path, dpi, pages_dir, fullpage_dir, crops_dir, extract_dir) -> list[dict]:
    records: list[dict] = []

    # 2. Render the page.
    page_png = render_page(pdf_path, page_num, pages_dir, dpi=dpi)
    print(f"[image] page {page_num}: rendered -> {page_png.name}")

    # 3. VLM page analysis (also confirms which panels are Tafel).
    findings = analyze_page(str(page_png))

    # 保存 page analysis，供测试脚本/下游重绘复用（避免重跑 VLM）。
    analysis_path = fullpage_dir / f"page_{page_num:03d}_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, ensure_ascii=False, indent=2)

    tafel = [p for p in findings.get("panels", []) if p.get("is_tafel")]
    if not tafel:
        print(f"[image] page {page_num}: no Tafel panels (VLM), skipped.")
        return records
    print(f"[image] page {page_num}: {len(tafel)} Tafel panels, title={findings.get('title')!r}")

    # 4. Full-page ChartDete -> axis boxes -> crop panels.
    bb_path = run_fullpage_chartdete(page_png, fullpage_dir)
    page_crops_dir = crops_dir / f"page_{page_num:03d}"
    cropped = crop_tafel_panels(
        page_png, findings, page_crops_dir, chartdete_bboxes=bb_path
    )
    if not cropped:
        print(f"[image] page {page_num}: cropping produced no panels.")
        return records

    # 5-6. Chart extraction + legend binding, batched over all panels of the page.
    from chart_extractor.src.plextract.local.app import run_pipeline
    from chart_extractor.src.plextract.local.legend_matcher import match_legends_for_directory

    page_extract_dir = extract_dir / f"page_{page_num:03d}"
    run_pipeline(str(page_crops_dir), str(page_extract_dir))
    match_legends_for_directory(str(page_crops_dir), str(page_extract_dir))

    # 7-9. Clean, bind, convert, emit — panel by panel.
    title = findings.get("title")
    for panel in sorted(cropped, key=lambda c: c["panel_id"]):
        records.extend(_process_panel(panel, title, page_extract_dir))
    return records


def _process_panel(panel: dict, title: Optional[str], page_extract_dir: Path) -> list[dict]:
    pid = panel["panel_id"]
    img_name = Path(panel["image_path"]).name
    sub = page_extract_dir / img_name

    data_path = sub / "converted_datapoints" / "data.json"
    mapping_path = sub / "legend_mapping.json"
    titles_path = sub / "axis_titles.json"

    if not data_path.exists():
        print(f"[image] {pid}: missing data.json (extraction failed), skipped.")
        return []
    if not mapping_path.exists():
        print(f"[image] {pid}: missing legend_mapping.json, skipped.")
        return []

    with open(data_path) as f:
        data = json.load(f)
    with open(mapping_path) as f:
        sm = json.load(f).get("series_mapping", {})

    x_title = y_title = ""
    if titles_path.exists():
        with open(titles_path, encoding="utf-8") as f:
            t = json.load(f)
        x_title = t.get("x_title", "")
        y_title = t.get("y_title", "")

    # 若纵轴是电位（Tafel 图旋转 90°），把数据点 x/y 对调、标题也对调，统一成标准朝向。
    if detect_potential_axis(x_title, y_title) == "y":
        for pts in data.values():
            for pt in pts:
                pt["x"], pt["y"] = pt["y"], pt["x"]
        x_title, y_title = y_title, x_title

    # Only series that got a matched (non-empty) label are kept.
    cleaned = clean_labeled_series(data, sm)

    condition = panel.get("condition") or ""
    context = panel.get("context") or {}

    # ONE binding VLM call for the whole panel (units + environment fields).
    try:
        unit_info = bind_panel_vlm(condition, context, title, x_title, y_title)
    except Exception as exc:
        print(f"[image] {pid}: binding VLM failed ({exc}); using defaults.")
        unit_info = {
            "Alloy_Name": None, "Electrolyte": None, "Ref_Electrode": None,
            "Test_Method": None, "pH": None, "Temp_C": 25,
            "x_unit": "V", "y_is_log10": True, "y_unit": "A/cm2",
        }

    records: list[dict] = []
    for series_key, res in cleaned.items():
        if res.skipped or not res.label:
            continue
        conv = convert_params(res.params, unit_info)

        note_parts = [
            "来自图像",
            f"浸没时间={res.label}",
            f"condition={condition}" if condition else None,
            f"原始log_i_corr={res.params['log_i_corr']:.4f}",
            f"y轴={y_title}" if y_title else None,
        ]
        notes = "; ".join(p for p in note_parts if p)

        record = {k: None for k in FIELD_NAMES}
        record.update({
            "Alloy_Name": unit_info.get("Alloy_Name"),
            "Electrolyte": unit_info.get("Electrolyte"),
            "pH": unit_info.get("pH"),
            "Temp_C": unit_info.get("Temp_C"),
            "Test_Method": unit_info.get("Test_Method"),
            "Ref_Electrode": unit_info.get("Ref_Electrode"),
            "E_corr_mV": conv["E_corr_mV"],
            "I_corr_uA_per_cm2": conv["I_corr_uA_per_cm2"],
            "ba_mV_per_dec": conv["ba_mV_per_dec"],
            "bc_mV_per_dec": conv["bc_mV_per_dec"],
            "CR_um_per_y": None,          # filled by main pipeline (author value) or left null
            "Notes": notes,
        })
        records.append(record)

    print(f"[image] {pid}: {len(records)} records from {len(cleaned)} labelled series.")
    return records


def _write_csv(records: list[dict], path: str | Path):
    import csv

    if not records:
        print("[image] no records to write.")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in FIELD_NAMES})


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "validation papers/Comparative_study_of_electrochemical_Corrosion_of_.pdf"
    run_image_leg(pdf)
