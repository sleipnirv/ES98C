"""
Page-level VLM analysis: figure detection, panel cropping, and Tafel classification.

This module is a *utility* called by ``project_v2.py``.  It does NOT make
decisions — it faithfully reports everything it finds on a single page so
that the main pipeline can aggregate, cross-validate and filter with
global (full-paper) context.

Responsibilities
----------------
1. Detect chart / figure regions on a scanned PDF page image.
2. Count sub-panels per figure, describe their grid layout.
3. Classify each panel as Tafel (polarization curve) or not.
4. Extract experimental context from surrounding text (alloy, electrolyte, …).
5. Crop individual Tafel panels into standalone images ready for
   ``chart_extractor``.

All VLM calls go through Ollama (qwen3-vl), the same model used by
``legend_matcher`` and ``project_v2``.
"""

import base64
import itertools
import json
import io
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

# ── Configuration (shared with the rest of the project) ──────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
VLM_MODEL = "qwen3-vl:8b-instruct-q4_K_M"

_PAGE_ANALYSIS_PROMPT = """Analyze this scientific paper page image carefully.

IMPORTANT: A single page often contains MULTIPLE separate figures (e.g. a mechanical scratch-test figure, an SEM image, AND a Tafel / potentiodynamic-polarization figure). The Tafel/polarization figure is the PRIMARY target: identify it by its caption, which contains keywords such as "Tafel", "polarization", "potentiodynamic" or "corrosion". You MUST detect EVERY figure and every panel on the whole page — do NOT stop after finding one figure (e.g. do not stop at a scratch test just because it is near the top). The Tafel panels have log current on one axis and potential on the other, with V-shaped curves; they may sit below or beside other figures.

1. How many separate chart/graph panels (sub-figures) are visible across the whole page (including all distinct figures)? Ignore text paragraphs, only count actual data plots with axes.
2. Describe the grid layout (e.g., "2 columns x 3 rows", or "1 column x 4 rows").
3. For each panel, assign a panel_id (top-to-bottom, left-to-right: panel_0, panel_1, ...) and answer:
   - Is it a Tafel plot (potentiodynamic polarization curve)?
     Tafel plots have: log(current) on y-axis, potential (V) on x-axis, and a characteristic V-shaped curve.
   - What legend labels are visible on this panel? Read them exactly.
   - What experimental condition does this panel represent (alloy, electrolyte, concentration, treatment)? Infer it from ANYWHERE on the page — the figure title/caption, the panel's own sub-title, or surrounding text. Report as free text (e.g. "316L in 0.9 % NaCl"), or null if not discernible.
   - Estimate the bounding box of this panel in pixels: [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right. Include the plot area, axis tick labels, axis titles, and the legend. Exclude neighbouring panels.
4. What experimental conditions are mentioned in the page text surrounding the figures?
   Look for: alloy name / composition, electrolyte, concentration, temperature, pH, immersion time, reference electrode.
   Only report what is EXPLICITLY stated.
5. What is the figure's title or caption? Report the caption of the TAFEL / polarization figure specifically — the one whose caption contains "Tafel", "polarization", "potentiodynamic" or "corrosion". Do NOT report the caption of other figures (scratch test, SEM, etc.). The caption is usually BELOW the figure. Read it exactly as written (e.g. "Fig. 17. ..."). Put it in the top-level "title" field, or null if absent.

Respond in this JSON format only:
{
  "title": "Fig.3. Potentiodynamic polarization curves in NaCl solutions, a) 316L in 0.9 %, ...",
  "panels": [
    {
      "panel_id": "panel_0",
      "position": "top-left",
      "is_tafel": true,
      "bbox": [120, 140, 400, 320],
      "legend_labels": ["0 min", "30 min", "1h", "3h", "5h", "24 h"],
      "condition": "316L in 0.9 % NaCl"
    }
  ],
  "grid": {"cols": 2, "rows": 3},
  "context": {
    "alloy": "316L",
    "electrolyte": "3.5 wt.% NaCl",
    "temperature": null,
    "pH": null,
    "immersion_time": "0 min - 24 h",
    "reference_electrode": "SCE",
    "notes": "aerated"
  }
}

Be precise. Use null for missing values. Do not invent data."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_page(image_path: str | Path) -> dict:
    """Analyse a single paper-page image with VLM.

    Args:
        image_path: Path to a PNG / JPEG image of one paper page.

    Returns:
        A dict with keys ``panels``, ``grid`` and ``context`` (see the
        prompt above for the schema).  ``panels`` is the primary output —
        each entry records position, Tafel classification and legend labels.
    """
    image_path = Path(image_path)

    img_b64 = _encode_image(image_path)

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": VLM_MODEL,
                "prompt": _PAGE_ANALYSIS_PROMPT,
                "images": [img_b64],
                "stream": False,
                "temperature": 0.05,
            },
            timeout=120,
        )
        raw = resp.json().get("response", "").strip()
    except Exception as exc:
        raise RuntimeError(f"VLM page analysis failed for {image_path}: {exc}") from exc

    # The VLM may wrap the JSON in markdown fences.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Best-effort: try to extract the first JSON object.
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(raw[start:end])
        else:
            raise RuntimeError(f"VLM did not return valid JSON:\n{raw[:500]}")

    _validate_findings(result, image_path.name)
    return result


def crop_tafel_panels(
    image_path: str | Path,
    findings: dict,
    output_dir: str | Path,
    *,
    chartdete_bboxes: Optional[str | Path] = None,
    padding_ratio: float = 0.03,
) -> list[dict]:
    """Crop Tafel panels from a page image.

    Strategy (tried in order):
    1. If ``chartdete_bboxes`` is provided, use ChartDete's axis detections
       (``y_axis_area`` + ``x_axis_area``) to locate individual panels via
       spatial pairing.
    2. Otherwise, fall back to the VLM per-panel ``bbox`` estimates.

    Only panels where ``is_tafel`` is ``True`` are cropped.

    Args:
        image_path: Path to the full page image.
        findings: Dict returned by :func:`analyze_page`.
        output_dir: Directory to write cropped panel images.
        chartdete_bboxes: Optional path to a ChartDete ``bounding_boxes.json``
            for precise axis-based panel detection.
        padding_ratio: Fraction of panel dimension added as margin
            (default 3 %).

    Returns:
        List of dicts per cropped Tafel panel.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    page_h, page_w = image.shape[:2]
    grid = findings.get("grid", {})
    cols = max(1, grid.get("cols", 1))
    rows = max(1, grid.get("rows", 1))
    tafel_panels = [p for p in findings.get("panels", []) if p.get("is_tafel")]
    panel_count = cols * rows

    # ── Determine per-panel bounding boxes ──────────────────────────
    if chartdete_bboxes is not None:
        panel_bboxes = _panel_bboxes_from_axes(
            Path(chartdete_bboxes), panel_count, page_w, page_h, padding_ratio,
        )
    else:
        panel_bboxes = None  # signal: use per-panel fallback

    cropped = []
    for panel in tafel_panels:
        pid = panel["panel_id"]
        idx = int(pid.split("_")[-1])

        if panel_bboxes is not None and idx < len(panel_bboxes):
            x1, y1, x2, y2 = panel_bboxes[idx]
        else:
            # Fallback: VLM per-panel bbox → uniform grid.
            bbox = panel.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                # VLM bbox 常只框 plot 区、漏掉轴刻度/轴标题；向外扩 15% 把它们框进来
                pad_x = (x2 - x1) * 0.15
                pad_y = (y2 - y1) * 0.15
                x1, y1 = x1 - pad_x, y1 - pad_y
                x2, y2 = x2 + pad_x, y2 + pad_y
            else:
                row, col = idx // cols, idx % cols
                cell_w = page_w // cols
                cell_h = page_h // rows
                x1 = col * cell_w
                y1 = row * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h

        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(page_w, int(x2)), min(page_h, int(y2))

        crop = image[y1:y2, x1:x2]
        out_name = f"{image_path.stem}_{pid}.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), crop)

        cropped.append({
            "panel_id": pid,
            "image_path": str(out_path),
            "legend_labels": panel.get("legend_labels", []),
            "condition": panel.get("condition"),
            "context": findings.get("context", {}),
            "title": findings.get("title"),
        })

    return cropped


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_image(image_path: Path) -> str:
    """Read an image and return a base64 data-URI string."""
    if image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        with open(image_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")

    # Try OpenCV for other formats.
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _validate_findings(findings: dict, img_name: str) -> None:
    """Lightweight sanity checks on the VLM response."""
    panels = findings.get("panels")
    if not isinstance(panels, list) or len(panels) == 0:
        raise RuntimeError(f"No panels found by VLM in {img_name}")

    findings.setdefault("title", None)

    for p in panels:
        if "panel_id" not in p:
            raise RuntimeError(f"Panel missing panel_id in {img_name}: {p}")
        if "is_tafel" not in p:
            p["is_tafel"] = False  # default
        if "condition" not in p:
            p["condition"] = None  # default


def _panel_bboxes_from_axes(
    chartdete_path: Path,
    panel_count: int,
    page_w: int,
    page_h: int,
    padding_ratio: float = 0.03,
) -> list[tuple[int, int, int, int]]:
    """Compute panel bounding boxes from ChartDete axis detections.

    1. Hard-filter ``y_axis_area`` (H > W) and ``x_axis_area`` (W > H) by
       maximum size.
    2. Pick the *panel_count* most similar detections of each type by
       intra-group dimension variance.
    3. Sort by centre-y and group into rows; within each row pair the
       leftmost y-axis with the leftmost x-axis.
    4. Each (y, x) pair produces a bbox ``[y.x1, y.y1, x.x2, x.y2]``.
    5. Apply a small padding for tolerance.

    Falls back to a uniform grid (``page_w//cols × page_h//rows``) when
    ChartDete does not provide enough axis detections.
    """
    if not chartdete_path.exists():
        return []

    with open(chartdete_path) as f:
        bb = json.load(f)

    max_area = (page_w * page_h) / panel_count / 2

    # ── Stage 1: hard-filter candidates ─────────────────────────────
    def _filter(detections, aspect_ok):
        kept = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2, c = det
            dw, dh = x2 - x1, y2 - y1
            if dw * dh > max_area:
                continue
            if not aspect_ok(dw, dh):
                continue
            kept.append({
                "i": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": c, "w": dw, "h": dh,
                "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            })
        return kept

    y_cands = _filter(bb.get("y_axis_area", []), lambda w, h: h > w)
    x_cands = _filter(bb.get("x_axis_area", []), lambda w, h: w > h)

    if len(y_cands) < panel_count or len(x_cands) < panel_count:
        return []  # signal fallback

    # ── Stage 2: pick most-similar group of k ───────────────────────
    def _pick_best(candidates, k):
        best_score = float("inf")
        best_group = None
        for combo in itertools.combinations(candidates, k):
            ws = [c["w"] for c in combo]
            hs = [c["h"] for c in combo]
            score = np.std(ws) / np.mean(ws) + np.std(hs) / np.mean(hs)
            if score < best_score:
                best_score = score
                best_group = combo
        return list(best_group) if best_group else []

    y_sel = _pick_best(y_cands, panel_count)
    x_sel = _pick_best(x_cands, panel_count)

    if len(y_sel) < panel_count or len(x_sel) < panel_count:
        return []

    # ── Stage 3: sort & group into rows ─────────────────────────────
    y_sel.sort(key=lambda c: (c["cy"], c["cx"]))
    x_sel.sort(key=lambda c: (c["cy"], c["cx"]))

    def _group_rows(items):
        rows_list = []
        current = [items[0]]
        for item in items[1:]:
            if abs(item["cy"] - current[-1]["cy"]) < 40:
                current.append(item)
            else:
                rows_list.append(sorted(current, key=lambda c: c["cx"]))
                current = [item]
        rows_list.append(sorted(current, key=lambda c: c["cx"]))
        return rows_list

    y_rows = _group_rows(y_sel)
    x_rows = _group_rows(x_sel)

    # ── Stage 4: pair & compute bboxes ──────────────────────────────
    bboxes = []
    for row_idx in range(len(y_rows)):
        for col_idx in range(len(y_rows[row_idx])):
            yc = y_rows[row_idx][col_idx]
            xc = x_rows[row_idx][col_idx]

            px1 = yc["x1"]
            py1 = yc["y1"]
            px2 = xc["x2"]
            py2 = xc["y2"]

            # Small padding for tolerance (proportional to panel size).
            pad_w = (px2 - px1) * padding_ratio
            pad_h = (py2 - py1) * padding_ratio
            pad = max(pad_w, pad_h, 5)

            bboxes.append((
                max(0, int(px1 - pad)),
                max(0, int(py1 - pad)),
                min(page_w, int(px2 + pad)),
                min(page_h, int(py2 + pad)),
            ))

    return bboxes
