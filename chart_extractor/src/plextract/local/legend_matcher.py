"""
Match LineFormer series to ChartDete legend entries.

This module:
1. Pairs legend_patch boxes with nearby legend_label boxes.
2. OCRs the legend_label text.
3. Samples colors from each legend_patch and from each LineFormer series.
4. Matches series to legend entries by color similarity in LAB space.
5. Writes a legend_mapping.json file.
"""

import json
import os
import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image
from scipy.spatial.distance import cdist

from ..utils import logger

try:
    from .trocr import OCRModel  # noqa: F401 — optional, kept for fallback
except ImportError:
    OCRModel = None  # type: ignore[assignment]


# Minimum confidence for a detection to be considered.
CONFIDENCE_THRESHOLD = 0.5

# Colour-distance thresholds for confidence grading (weighted LAB distance).
CONFIDENCE_HIGH = 10.0    # near-certain match
CONFIDENCE_MEDIUM = 25.0  # likely correct
CONFIDENCE_LOW = 40.0     # plausible but risky — may still be a wrong label
# Distances above CONFIDENCE_LOW are rejected outright.


def _box_center(box: list[float]) -> tuple[float, float]:
    """Return the center of a bounding box [x1, y1, x2, y2, conf]."""
    x1, y1, x2, y2, _ = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_iou_horz(box_a: list[float], box_b: list[float]) -> float:
    """
    Compute a horizontal IoU-like overlap score.

    Useful when checking whether a label sits to the right of a patch.
    Returns overlap fraction of the vertical range.
    """
    _, y1_a, _, y2_a, _ = box_a
    _, y1_b, _, y2_b, _ = box_b

    y_top = max(y1_a, y1_b)
    y_bottom = min(y2_a, y2_b)
    if y_bottom <= y_top:
        return 0.0

    overlap = y_bottom - y_top
    height_a = y2_a - y1_a
    height_b = y2_b - y1_b
    denom = min(height_a, height_b)
    if denom <= 0:
        return 0.0
    return overlap / denom


def _pair_patches_with_labels(
    patches: list[list[float]],
    labels: list[list[float]],
    min_horz_overlap: float = 0.3,
) -> list[tuple[list[float], Optional[list[float]]]]:
    """
    Pair each legend_patch with the most likely legend_label.

    Pairing rules (in order of priority):
    1. The label is to the right of the patch and shares vertical overlap.
    2. Fall back to nearest center distance.

    Returns a list of (patch_box, label_box_or_None) tuples.
    """
    if not patches:
        return []

    used_labels = set()
    pairs: list[tuple[list[float], Optional[list[float]]]] = []

    # Sort patches top-to-bottom, then left-to-right for stable ordering.
    sorted_patches = sorted(patches, key=lambda b: (b[1], b[0]))

    for patch in sorted_patches:
        patch_x2 = patch[2]
        patch_center = _box_center(patch)

        # Candidates: labels to the right of the patch with vertical overlap.
        right_candidates = []
        for idx, label in enumerate(labels):
            if idx in used_labels:
                continue
            label_x1 = label[0]
            if label_x1 < patch_x2:
                continue
            overlap = _box_iou_horz(patch, label)
            if overlap >= min_horz_overlap:
                label_center = _box_center(label)
                distance = np.linalg.norm(
                    np.array(patch_center) - np.array(label_center)
                )
                right_candidates.append((idx, label, distance, overlap))

        if right_candidates:
            # Prefer candidates with good vertical overlap, then distance.
            right_candidates.sort(key=lambda x: (-x[3], x[2]))
            chosen_idx, chosen_label, _, _ = right_candidates[0]
            used_labels.add(chosen_idx)
            pairs.append((patch, chosen_label))
            continue

        # Fallback: nearest unused label by center distance.
        best_idx: Optional[int] = None
        best_distance: float = float("inf")
        for idx, label in enumerate(labels):
            if idx in used_labels:
                continue
            label_center = _box_center(label)
            distance = np.linalg.norm(
                np.array(patch_center) - np.array(label_center)
            )
            if distance < best_distance:
                best_distance = distance
                best_idx = idx

        if best_idx is not None:
            used_labels.add(best_idx)
            pairs.append((patch, labels[best_idx]))
        else:
            pairs.append((patch, None))

    return pairs


OLLAMA_URL = "http://localhost:11434/api/generate"
VLM_MODEL = "qwen3-vl:8b-instruct-q4_K_M"


VLM_OCR_PROMPT = (
    "This is the legend area of a scientific chart. "
    "Read ALL legend labels visible in this image, from top to bottom. "
    "Output one label per line, exactly as written. "
    "Only output the labels, nothing else."
)


def _ocr_labels_vlm_full_area(
    image: np.ndarray,
    legend_area_box: list[float],
    n_expected: int,
) -> list[str]:
    """OCR all legend labels at once by sending the full legend area to VLM.

    Sending the whole legend region (typically ~80×130 px) gives the VLM enough
    context to avoid hallucination, unlike tiny per-label crops (~16×13 px).
    Labels are read top-to-bottom, matching the vertical sort order of patches.
    """
    x1, y1, x2, y2 = [max(0, int(v)) for v in legend_area_box[:4]]
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        logger.warning("Legend area crop is empty.")
        return [""] * n_expected

    _, encoded = cv2.imencode(".png", crop)
    img_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": VLM_MODEL,
                "prompt": VLM_OCR_PROMPT,
                "images": [img_b64],
                "stream": False,
            },
            timeout=60,
        )
        text = resp.json().get("response", "").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        logger.info(f"VLM OCR full-area result ({len(lines)} lines): {lines}")
        if len(lines) < n_expected:
            logger.warning(
                f"VLM returned {len(lines)} labels but {n_expected} were expected. "
                "Padding with empty strings."
            )
            lines += [""] * (n_expected - len(lines))
        return lines[:n_expected]
    except Exception as e:
        logger.warning(f"VLM OCR (full-area) failed: {e}")
        return [""] * n_expected


def _extract_label_texts(
    image: np.ndarray,
    pairs: list[tuple[list[float], Optional[list[float]]]],
    ocr_model: Optional[OCRModel] = None,
    legend_area_box: Optional[list[float]] = None,
) -> list[tuple[list[float], Optional[str]]]:
    """
    OCR legend labels and return (patch_box, text) pairs.

    Uses VLM (qwen3-vl) on the full legend area when ``legend_area_box`` is
    provided — the VLM reads all labels top-to-bottom in one call, and results
    are mapped to patches by vertical position order.

    Falls back to per-label TrOCR when the legend area is unavailable.
    """
    n_labels = sum(1 for _, lb in pairs if lb is not None)

    # ── Preferred path: VLM on full legend area ────────────────────
    if legend_area_box is not None and n_labels > 0:
        vlm_texts = _ocr_labels_vlm_full_area(image, legend_area_box, n_labels)
        result: list[tuple[list[float], Optional[str]]] = []
        text_idx = 0
        for patch_box, label_box in pairs:
            if label_box is None:
                result.append((patch_box, None))
            else:
                text = vlm_texts[text_idx] if text_idx < len(vlm_texts) else None
                result.append((patch_box, text or None))
                text_idx += 1
        return result

    # ── Fallback: TrOCR on individual label crops ──────────────────
    if ocr_model is None and OCRModel is not None:
        ocr_model = OCRModel()
    if ocr_model is None:
        # No OCR available at all — return patches with no text.
        return [(patch_box, None) for patch_box, _ in pairs]

    temp_dir = Path("_legend_ocr_temp")
    temp_dir.mkdir(exist_ok=True)

    crop_paths: list[tuple[int, Path, list[float]]] = []
    for idx, (_, label_box) in enumerate(pairs):
        if label_box is None:
            continue
        x1, y1, x2, y2, _ = label_box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        crop_path = temp_dir / f"legend_label_{idx}.png"
        cv2.imwrite(str(crop_path), cropped)
        crop_paths.append((idx, crop_path, label_box))

    ocr_results = {}
    if crop_paths:
        paths = [str(p) for _, p, _ in crop_paths]
        for path, text in ocr_model.inference_batch(paths):
            ocr_results[path] = text.strip()

    result = []
    for idx, (patch_box, label_box) in enumerate(pairs):
        if label_box is None:
            result.append((patch_box, None))
            continue
        matching = [p for i, p, _ in crop_paths if i == idx]
        if not matching:
            result.append((patch_box, None))
            continue
        crop_path = matching[0]
        text = ocr_results.get(str(crop_path), "").strip() or None
        result.append((patch_box, text))

    # Clean up temp files.
    for _, crop_path, _ in crop_paths:
        try:
            crop_path.unlink()
        except OSError:
            pass
    try:
        temp_dir.rmdir()
    except OSError:
        pass

    return result


def _sample_patch_color(image: np.ndarray, box: list[float]) -> np.ndarray:
    """Sample the dominant colour inside a legend patch in LAB space.

    Picks the pixel furthest from white in the left half of the patch, which
    mirrors the *furthest-from-white* strategy used for series line sampling
    and avoids systematic brightness offsets between the two.
    """
    x1, y1, x2, y2, _ = box
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)

    h, w = roi.shape[:2]
    # Restrict to the left half where the coloured mark lives.
    left_region = roi[:, : max(1, w // 2)]
    pixels = left_region.reshape(-1, 3).astype(np.float32)

    # Pick the top 20% furthest-from-white pixels and take their median
    # (same heuristic as series sampling — avoids systematic bias between
    # legend colour and series colour while being robust to single outliers).
    white_ref = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    dists = np.linalg.norm(pixels - white_ref, axis=1)
    top_k = max(1, len(dists) // 5)
    top_idx = np.argpartition(dists, -top_k)[-top_k:]
    median_bgr = np.median(pixels[top_idx], axis=0)

    median_bgr_uint8 = np.uint8([[median_bgr]])
    lab = cv2.cvtColor(median_bgr_uint8, cv2.COLOR_BGR2LAB)[0, 0]
    return lab.astype(np.float32)


_POTENTIAL_KW = ("potential", "voltage", "电位", "电压")
_CURRENT_KW = ("current", "density", "电流", "密度", "log")


def _potential_axis(x_title: str, y_title: str) -> str:
    """返回电位在哪个轴（'x' 或 'y'）。规则判断，判不出默认 'x'（标准朝向）。

    与 curve_pipeline.detect_potential_axis 逻辑一致，但只做规则、不调 VLM，避免颜色
    采样阶段额外多一次模型调用，也避免 legend_matcher 反向依赖 curve_pipeline。
    """
    def _looks_potential(t):
        t = (t or "").lower()
        pot = any(k in t for k in _POTENTIAL_KW)
        cur = any(k in t for k in _CURRENT_KW)
        if pot and not cur:
            return True
        if cur and not pot:
            return False
        return None

    xp, yp = _looks_potential(x_title), _looks_potential(y_title)
    if xp is True:
        return "x"
    if yp is True:
        return "y"
    if xp is False:
        return "y"  # x 明确是电流 → y 是电位
    if yp is False:
        return "x"  # y 明确是电流 → x 是电位
    return "x"  # 判不出，默认标准朝向


def _sample_series_color(
    image: np.ndarray,
    points: list[dict[str, int]],
    sample_stride: int = 3,
    exclude_middle_frac: float = 1.0 / 3.0,
    exclude_middle_axis: str = "x",
) -> Optional[np.ndarray]:
    """
    Sample the colour of a LineFormer series in LAB space.

    At each sampled coordinate a 7x7 window is taken and the single pixel
    furthest from white is kept.  This tolerates LineFormer coordinate
    offsets of up to ~3 px while avoiding the dilution that a mean over the
    window causes for thin (1-2 px) lines.  Pixels that are still too close
    to white (> 230 per channel) are skipped.  The result is the median LAB
    over all sampled points.

    The middle ``exclude_middle_frac`` of the *potential* axis range is
    skipped: in a Tafel plot all curves converge near the corrosion
    potential, so pixels there are contaminated by overlapping lines.  Only
    the two outer thirds (where curves are well separated) are used.  The
    potential axis is usually ``x`` (standard orientation), but rotated
    90-degree plots have it on ``y``.
    """
    if not points:
        return None

    # Middle-band exclusion on the potential axis (x or y).
    coords = [p[exclude_middle_axis] for p in points]
    c_min, c_max = min(coords), max(coords)
    span = c_max - c_min
    if span > 0:
        lo_cut = c_min + span * exclude_middle_frac
        hi_cut = c_max - span * exclude_middle_frac
    else:
        lo_cut = hi_cut = None

    lab_values = []
    h, w = image.shape[:2]
    white_ref = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    for i, point in enumerate(points):
        if i % sample_stride != 0:
            continue
        px = int(round(point["x"]))
        py = int(round(point["y"]))

        coord = px if exclude_middle_axis == "x" else py
        if lo_cut is not None and lo_cut <= coord <= hi_cut:
            continue  # crowded middle band — skip

        x1 = max(0, px - 3)
        y1 = max(0, py - 3)
        x2 = min(w, px + 4)
        y2 = min(h, py + 4)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Pick the pixel furthest from white in this window.
        dists = np.linalg.norm(roi.astype(np.float32) - white_ref, axis=2)
        best_idx = np.argmax(dists.reshape(-1))
        best_pixel = roi.reshape(-1, 3)[best_idx]

        if best_pixel[0] > 230 and best_pixel[1] > 230 and best_pixel[2] > 230:
            continue

        best_uint8 = np.uint8([[best_pixel]])
        lab = cv2.cvtColor(best_uint8, cv2.COLOR_BGR2LAB)[0, 0]
        lab_values.append(lab.astype(np.float32))

    if not lab_values:
        return None

    return np.median(lab_values, axis=0)


def _series_shape_features(series: list[dict[str, int]]) -> np.ndarray:
    """
    Extract shape features for clustering LineFormer series.

    Features are chosen to be robust to pixel coordinates and to capture the
    characteristic Tafel / Butler-Volmer shape of polarization curves.
    """
    if not series:
        return np.zeros(6, dtype=np.float32)

    xs = np.array([p["x"] for p in series], dtype=np.float32)
    ys = np.array([p["y"] for p in series], dtype=np.float32)

    # y is in image coordinates (downwards positive). Invert so curves rise.
    ys_inv = -ys

    # Avoid log of non-positive values.
    ys_log = np.log(np.clip(ys_inv - ys_inv.min() + 1.0, 1e-6, None))

    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()

    # Linear fit on inverted y.
    if len(xs) >= 2:
        slope, _ = np.polyfit(xs, ys_inv, 1)
        slope_log, _ = np.polyfit(xs, ys_log, 1)
    else:
        slope = 0.0
        slope_log = 0.0

    # Capture the "V" shape: minimum y position (inverted = peak current).
    y_min_inv = ys_inv.min()
    y_max_inv = ys_inv.max()

    # x position of the minimum (the corrosion potential region).
    x_at_min = xs[ys_inv.argmin()]

    return np.array(
        [x_range, y_range, slope, slope_log, y_min_inv, x_at_min],
        dtype=np.float32,
    )


def _kmeans(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 42) -> np.ndarray:
    """Minimal KMeans implementation using only numpy."""
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    if k >= n_samples:
        return np.arange(n_samples)

    indices = rng.choice(n_samples, k, replace=False)
    centroids = X[indices].copy()

    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            points = X[labels == j]
            if len(points) > 0:
                centroids[j] = points.mean(axis=0)
            else:
                centroids[j] = X[rng.choice(n_samples)]
    return labels


def _merge_series_by_color(
    series_list: list[list[dict[str, int]]],
    series_colors: list[Optional[np.ndarray]],
    n_target: int,
) -> tuple[list[list[dict[str, int]]], list[int]]:
    """
    Merge LineFormer series by colour similarity using hierarchical clustering.

    Each series starts as its own cluster.  The two clusters with the smallest
    colour distance are repeatedly merged until exactly *n_target* clusters
    remain (or no more merges are possible).  Series whose colour could not be
    sampled (``None``) are kept as singletons.

    Returns:
        (merged_series, label_map) — *label_map[idx]* gives the final group id
        for the original series at index idx.
    """
    n = len(series_list)
    if n <= n_target:
        return series_list, list(range(n))

    # Initial clusters: each series is its own cluster (index → set of series).
    clusters: list[set[int]] = [{i} for i in range(n)]
    # Representative colour for each cluster.
    rep_color: list[Optional[np.ndarray]] = list(series_colors)

    # Merge until we reach the target.
    while len(clusters) > n_target:
        # Find the closest pair of clusters that both have valid colours.
        best_i, best_j, best_dist = -1, -1, float("inf")
        for ii in range(len(clusters)):
            ci = rep_color[ii]
            if ci is None:
                continue
            for jj in range(ii + 1, len(clusters)):
                cj = rep_color[jj]
                if cj is None:
                    continue
                # Same L-downweighted distance used for matching (0.3× L, 1.0× a,b).
                diff = (ci - cj) * np.array([0.3, 1.0, 1.0], dtype=np.float32)
                d = float(np.linalg.norm(diff))
                if d < best_dist:
                    best_dist, best_i, best_j = d, ii, jj

        if best_i == -1:
            break  # cannot merge any more.

        # Merge cluster j into i.
        clusters[best_i] |= clusters[best_j]
        # Re-compute the representative colour as the mean of members' colours.
        member_colors = [series_colors[idx] for idx in clusters[best_i]
                         if series_colors[idx] is not None]
        if member_colors:
            rep_color[best_i] = np.mean(member_colors, axis=0)
        # Remove cluster j.
        clusters.pop(best_j)
        rep_color.pop(best_j)

    # Build label map from final clusters.
    labels = [-1] * n
    for gid, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = gid

    # Build merged series list.
    merged: list[list[dict[str, int]]] = []
    for cluster in clusters:
        group_pts: list[dict[str, int]] = []
        for idx in sorted(cluster):
            group_pts.extend(series_list[idx])
        group_pts.sort(key=lambda p: p["x"])
        merged.append(group_pts)

    logger.info(
        f"Merged {n} series into {len(merged)} groups by colour "
        f"(target={n_target})."
    )
    return merged, labels


def _match_series_to_legends(
    series_colors: list[Optional[np.ndarray]],
    legend_colors: list[np.ndarray],
) -> list[tuple[int, float]]:
    """
    Match each series to the closest legend patch by LAB color distance.

    Uses the Hungarian algorithm (linear_sum_assignment) to find a global
    one-to-one matching between series and legends, rather than greedily
    assigning each series to its nearest legend. This avoids cases where an
    early series "steals" a legend that is a better match for a later series.

    Returns a list of (legend_index, distance) for each series.
    """
    if not legend_colors:
        return [(-1, float("inf")) for _ in series_colors]

    valid_series: list[tuple[int, np.ndarray]] = []
    for idx, color in enumerate(series_colors):
        if color is not None:
            valid_series.append((idx, color))

    if not valid_series:
        return [(-1, float("inf")) for _ in series_colors]

    legend_matrix = np.vstack(legend_colors)
    valid_indices = [idx for idx, _ in valid_series]
    valid_colors = np.vstack([color for _, color in valid_series])

    # Build distance matrix: valid series x legends.
    # Weight L at 0.3× vs a,b at 1.0× so that chroma (a*b*) drives the
    # match.  Thin anti-aliased lines are systematically darker than the
    # solid legend squares, so full L contribution creates a bias.
    chroma_weights = np.array([0.3, 1.0, 1.0], dtype=np.float32)
    distances = cdist(
        valid_colors * chroma_weights,
        legend_matrix * chroma_weights,
        metric="euclidean",
    )

    # Global optimal assignment.
    from scipy.optimize import linear_sum_assignment

    series_idx, legend_idx = linear_sum_assignment(distances)

    # Build result for all original series.
    matches: list[tuple[int, float]] = [(-1, float("inf")) for _ in series_colors]
    for si, li in zip(series_idx, legend_idx):
        original_idx = valid_indices[si]
        matches[original_idx] = (int(li), float(distances[si, li]))

    return matches


def match_legends(
    image_path: str,
    output_dir: str,
    ocr_model: Optional[OCRModel] = None,
) -> dict:
    """
    Match LineFormer series to legend entries for a single chart image.

    Args:
        image_path: Path to the original chart image.
        output_dir: Directory where elcd has written outputs for this image.
        ocr_model: Optional pre-initialized OCRModel.

    Returns:
        Dictionary with legend mapping information.
    """
    image_name = Path(image_path).name
    image_output_dir = Path(output_dir) / image_name

    bounding_boxes_path = image_output_dir / "chartdete" / "bounding_boxes.json"
    lineformer_path = image_output_dir / "lineformer" / "coordinates.json"

    if not bounding_boxes_path.exists():
        raise FileNotFoundError(f"ChartDete output not found: {bounding_boxes_path}")
    if not lineformer_path.exists():
        raise FileNotFoundError(f"LineFormer output not found: {lineformer_path}")

    with open(bounding_boxes_path, "r") as f:
        bounding_boxes = json.load(f)

    with open(lineformer_path, "r") as f:
        all_lineseries = json.load(f)

    series_list = all_lineseries
    n_original_series = len(series_list)
    patches = [
        box for box in bounding_boxes.get("legend_patch", [])
        if box[4] >= CONFIDENCE_THRESHOLD
    ]
    labels = [
        box for box in bounding_boxes.get("legend_label", [])
        if box[4] >= CONFIDENCE_THRESHOLD
    ]

    logger.info(
        f"[{image_name}] Found {len(patches)} legend patches and {len(labels)} legend labels."
    )

    # Pair patches with labels and OCR the labels.
    pairs = _pair_patches_with_labels(patches, labels)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Get the highest-confidence legend_area for full-area VLM OCR.
    legend_areas = [
        box for box in bounding_boxes.get("legend_area", [])
        if box[4] >= CONFIDENCE_THRESHOLD
    ]
    legend_area_box = legend_areas[0] if legend_areas else None

    labeled_patches = _extract_label_texts(
        image, pairs, ocr_model, legend_area_box=legend_area_box,
    )

    # Build legend entries with colors.
    legend_entries: list[dict] = []
    for patch_box, text in labeled_patches:
        color = _sample_patch_color(image, patch_box)
        legend_entries.append(
            {
                "patch_box": patch_box,
                "label_text": text,
                "lab_color": color.tolist(),
            }
        )

    # Determine which axis is the potential axis, so the crowded middle band
    # is skipped on the correct axis (rotated 90° plots have potential on y).
    exclude_axis = "x"
    titles_path = image_output_dir / "axis_titles.json"
    if titles_path.exists():
        try:
            with open(titles_path, encoding="utf-8") as f:
                titles = json.load(f)
            if _potential_axis(
                titles.get("x_title", ""), titles.get("y_title", "")
            ) == "y":
                exclude_axis = "y"
        except Exception:
            pass

    # Sample colours for each original series FIRST, before any merging.
    original_colors: list[Optional[np.ndarray]] = []
    for series in series_list:
        original_colors.append(
            _sample_series_color(image, series, exclude_middle_axis=exclude_axis)
        )

    # If LineFormer produced more series than there are legend patches, merge
    # series whose sampled colours are nearly identical.  Shape-based KMeans
    # cannot tell curves apart in Tafel plots where all curves share the same
    # "V" shape, so colour is a much stronger grouping signal here.
    merge_info: Optional[list[int]] = None
    if len(patches) > 0 and len(series_list) > len(patches):
        series_list, merge_info = _merge_series_by_color(
            series_list, original_colors, n_target=len(patches)
        )

    # Re-sample colours after merging (merged series combine points from
    # multiple segments, giving richer colour statistics).
    series_colors: list[Optional[np.ndarray]] = []
    for series in series_list:
        series_colors.append(
            _sample_series_color(image, series, exclude_middle_axis=exclude_axis)
        )

    # Match series to legend entries.
    legend_colors = [
        np.array(entry["lab_color"], dtype=np.float32)
        for entry in legend_entries
    ]

    # Debug: log sampled colors.
    for i, (entry, lc) in enumerate(zip(legend_entries, legend_colors)):
        logger.info(
            f"[{image_name}] legend_{i} ('{entry['label_text']}'): "
            f"LAB=[{lc[0]:.0f}, {lc[1]:.0f}, {lc[2]:.0f}]"
        )
    for i, sc in enumerate(series_colors):
        if sc is not None:
            logger.info(
                f"[{image_name}] series_{i}: "
                f"LAB=[{sc[0]:.0f}, {sc[1]:.0f}, {sc[2]:.0f}]"
            )
        else:
            logger.info(f"[{image_name}] series_{i}: color=None")

    matches = _match_series_to_legends(series_colors, legend_colors)

    # Build mapping.
    mapping: dict = {
        "image": image_name,
        "legend_entries": legend_entries,
        "series_mapping": {},
        "clustering": {
            "n_original_series": n_original_series,
            "n_merged_curves": len(series_list),
            "n_legends": len(patches),
        },
    }

    def _confidence(d: float) -> str:
        if d <= CONFIDENCE_HIGH:
            return "high"
        elif d <= CONFIDENCE_MEDIUM:
            return "medium"
        elif d <= CONFIDENCE_LOW:
            return "low"
        return "rejected"

    used_legends: set[int] = set()
    for series_idx, (legend_idx, distance) in enumerate(matches):
        series_key = f"series_{series_idx}"
        conf = _confidence(distance)

        if legend_idx == -1 or conf == "rejected":
            mapping["series_mapping"][series_key] = {
                "legend_index": None,
                "label": None,
                "distance": round(distance, 2),
                "confidence": conf,
                "matched": False,
            }
            continue

        if legend_idx in used_legends:
            mapping["series_mapping"][series_key] = {
                "legend_index": None,
                "label": None,
                "distance": round(distance, 2),
                "confidence": "rejected",
                "matched": False,
                "reason": "legend_already_assigned",
            }
            continue

        used_legends.add(legend_idx)
        mapping["series_mapping"][series_key] = {
            "legend_index": legend_idx,
            "label": legend_entries[legend_idx]["label_text"],
            "distance": round(distance, 2),
            "confidence": conf,
            "matched": True,
        }

    # Write output.
    output_path = image_output_dir / "legend_mapping.json"
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    logger.info(f"[{image_name}] Legend mapping saved to {output_path}")
    return mapping


def match_legends_for_directory(
    input_dir: str,
    output_dir: str,
) -> dict[str, dict]:
    """
    Run legend matching for all images in an input directory.

    Args:
        input_dir: Directory containing input chart images.
        output_dir: Directory where elcd has written outputs.

    Returns:
        Dictionary mapping image name to legend mapping result.
    """
    ocr_model = OCRModel() if OCRModel is not None else None
    results: dict[str, dict] = {}

    input_path = Path(input_dir)
    for image_file in sorted(input_path.iterdir()):
        if image_file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue

        try:
            result = match_legends(str(image_file), output_dir, ocr_model)
            results[image_file.name] = result
        except Exception as e:
            logger.error(f"Legend matching failed for {image_file.name}: {e}")

    return results


VLM_AXIS_X_PROMPT = (
    "This is the x-axis of a scientific chart. "
    "Read ALL tick labels from left to right. "
    "Output one value per line, exactly as shown. "
    "Only output the values, nothing else."
)

VLM_AXIS_Y_PROMPT = (
    "This is the y-axis of a scientific chart. "
    "Read ALL tick labels from bottom to top. "
    "Output one value per line, exactly as shown. "
    "Only output the values, nothing else."
)


def ocr_axis_labels_vlm(
    image_path: str,
    bounding_boxes: dict,
    label_coordinates: dict,
) -> dict[str, str]:
    """OCR axis tick labels using VLM on full axis-area crops.

    Crops the x_axis_area / y_axis_area regions from the original image and
    sends each to qwen3-vl in a single call, then maps the returned label
    lines to individual crop filenames by positional order (left→right for
    x, bottom→top for y).

    Args:
        image_path: Path to the original chart image.
        bounding_boxes: Parsed ``bounding_boxes.json`` dict.
        label_coordinates: Parsed ``label_coordinates.json`` dict.

    Returns:
        Dict mapping crop-filename → OCR text, in the same format as
        the old per-crop TrOCR results.  Ready to be written to
        ``axis_label_texts.json``.
    """
    CONF = 0.5
    results: dict[str, str] = {}

    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"[ocr_axis] Cannot load image: {image_path}")
        return results

    def _vlm_crop(area_key: str, prompt: str, label_filter: str,
                   sort_index: int, reverse: bool = False):
        areas = [b for b in bounding_boxes.get(area_key, []) if b[4] >= CONF]
        if not areas:
            logger.info(f"[ocr_axis] No {area_key} detected, skipping.")
            return

        # Collect and sort individual label crop filenames by position.
        label_files = sorted(
            [k for k in label_coordinates if label_filter in k],
            key=lambda k: label_coordinates[k][sort_index],
            reverse=reverse,
        )
        if not label_files:
            logger.info(f"[ocr_axis] No individual {label_filter} crops found, skipping.")
            return

        x1, y1, x2, y2 = [max(0, int(v)) for v in areas[0][:4]]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return

        _, encoded = cv2.imencode(".png", crop)
        img_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": VLM_MODEL,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False,
                },
                timeout=60,
            )
            text = resp.json().get("response", "").strip()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            logger.info(
                f"[ocr_axis] VLM {area_key} → {len(lines)} labels, "
                f"expected {len(label_files)}: {lines}"
            )
            for i, fname in enumerate(label_files):
                results[fname] = lines[i] if i < len(lines) else ""
        except Exception as e:
            logger.warning(f"[ocr_axis] VLM failed for {area_key}: {e}")

    _vlm_crop("x_axis_area", VLM_AXIS_X_PROMPT, "xlabel", sort_index=0)
    _vlm_crop("y_axis_area", VLM_AXIS_Y_PROMPT, "ylabel", sort_index=1, reverse=True)

    return results


VLM_TITLE_PROMPT = (
    "This is an axis title of a scientific chart. "
    "Read the title text exactly as written. "
    "Output only the text, nothing else."
)


def ocr_axis_titles_vlm(
    image_path: str,
    bounding_boxes: dict,
) -> dict[str, str]:
    """OCR the x/y-axis titles and the chart title using VLM.

    ChartDete detects the title regions (``x_title`` / ``y_title`` /
    ``chart_title``); this sends each region to the VLM and returns the
    recognized text so callers can label axes with what was actually read
    instead of assuming units, and get the chart's own title.

    Args:
        image_path: Path to the original chart image.
        bounding_boxes: Parsed ``bounding_boxes.json`` dict.

    Returns:
        Dict ``{"x_title": str, "y_title": str, "chart_title": str}`` — empty
        string when no title box was detected or OCR failed.
    """
    CONF = 0.5
    titles = {"x_title": "", "y_title": "", "chart_title": ""}

    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"[ocr_title] Cannot load image: {image_path}")
        return titles

    for area_key in ("x_title", "y_title", "chart_title"):
        areas = [b for b in bounding_boxes.get(area_key, []) if b[4] >= CONF]
        if not areas:
            logger.info(f"[ocr_title] No {area_key} detected, skipping.")
            continue

        # Use the highest-confidence title box.
        box = max(areas, key=lambda b: b[4])
        x1, y1, x2, y2 = [max(0, int(v)) for v in box[:4]]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        _, encoded = cv2.imencode(".png", crop)
        img_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": VLM_MODEL,
                    "prompt": VLM_TITLE_PROMPT,
                    "images": [img_b64],
                    "stream": False,
                },
                timeout=60,
            )
            text = resp.json().get("response", "").strip()
            titles[area_key] = text
            logger.info(f"[ocr_title] {area_key} → {text!r}")
        except Exception as e:
            logger.warning(f"[ocr_title] VLM failed for {area_key}: {e}")

    return titles
