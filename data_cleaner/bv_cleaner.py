"""
Butler-Volmer based outlier cleaning for extracted polarization curves.

The extracted (E, log|i|) points of a polarization curve form a "V": two
near-linear Tafel branches meeting at the corrosion potential E_corr.
Extraction error is concentrated in the *middle* of the curve (around
E_corr), where several curves converge, colours bleed into one another and
LineFormer splits/merges segments.  The two *outer* branches are well
separated and therefore trustworthy.

This module fits the two Tafel asymptotes of the Butler-Volmer equation using
only the outer thirds of the x-range (the middle third is excluded from the
fit), then uses that fitted "V" as the reference curve.  Any point whose
residual from that reference exceeds a robust multiple (3x) of the side-point
noise (MAD) is dropped.

The fitted two lines ARE the Butler-Volmer equation in its Tafel limit::

    log10|i| = log10 i_corr + (E - E_corr) / b_a     (anodic,  E > E_corr)
    log10|i| = log10 i_corr + (E_corr - E) / b_c     (cathodic, E < E_corr)

so the intersection gives E_corr and i_corr, and the slopes give b_a / b_c.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Minimum number of points before a fit is even attempted.
MIN_POINTS = 8
# Minimum number of points required on each side for the Tafel fits.
MIN_SIDE_POINTS = 3
# Fraction of the x-range in the middle that is excluded from fitting.
EXCLUDE_MIDDLE_FRAC = 1.0 / 3.0
# Outlier threshold: this many robust sigmas from the fitted curve.
SIGMA_MULT = 3.0
# Robust sigma = MAD * this scale factor (converts MAD to std for gaussians).
MAD_TO_SIGMA = 1.4826


@dataclass
class CleanResult:
    """Outcome of cleaning one series."""

    series_key: str
    label: str
    skipped: bool
    points_before: list
    points_after: list
    points_removed: list
    segments: list
    params: dict
    n_before: int
    n_after: int
    n_removed: int
    sigma: float
    threshold: float


def fit_tafel_bv(
    points: list[dict],
    exclude_middle_frac: float = EXCLUDE_MIDDLE_FRAC,
) -> Optional[dict]:
    """Fit the two Tafel asymptotes of the Butler-Volmer equation.

    Only the outer ``1 - exclude_middle_frac`` fraction of the x-range is used
    (the middle fraction is treated as unreliable and ignored).

    Returns a dict of fitted parameters, or ``None`` if the curve cannot be
    fit reliably (too few points, missing a side, or no clean "V" shape).
    """
    if len(points) < MIN_POINTS:
        return None

    pts = sorted(points, key=lambda p: p["x"])
    xs = np.asarray([p["x"] for p in pts], dtype=float)
    ys = np.asarray([p["y"] for p in pts], dtype=float)

    x_min, x_max = float(xs.min()), float(xs.max())
    span = x_max - x_min
    if span <= 0:
        return None

    lo = x_min + span * exclude_middle_frac
    hi = x_max - span * exclude_middle_frac

    left = xs < lo
    right = xs > hi
    if left.sum() < MIN_SIDE_POINTS or right.sum() < MIN_SIDE_POINTS:
        return None

    # Cathodic branch (E < E_corr): y = m_c * x + c_c, slope m_c < 0.
    m_c, c_c = np.polyfit(xs[left], ys[left], 1)
    # Anodic branch (E > E_corr): y = m_a * x + c_a, slope m_a > 0.
    m_a, c_a = np.polyfit(xs[right], ys[right], 1)
    m_c, c_c, m_a, c_a = float(m_c), float(c_c), float(m_a), float(c_a)

    # A well-formed "V" needs a negative cathodic slope and positive anodic
    # slope.  Anything else means the two sides do not form the expected V.
    if not (m_c < 0 < m_a):
        return None

    # Tafel slopes (V/decade), positive by construction.
    b_c = -1.0 / m_c
    b_a = 1.0 / m_a

    # Corrosion potential = intersection of the two Tafel lines.
    e_corr = (c_c - c_a) / (m_a - m_c)
    log_i_corr = m_a * e_corr + c_a

    return {
        "E_corr": e_corr,
        "log_i_corr": log_i_corr,
        "b_a": b_a,
        "b_c": b_c,
        "m_a": m_a,
        "c_a": c_a,
        "m_c": m_c,
        "c_c": c_c,
    }


def _model_y(xs: np.ndarray, params: dict) -> np.ndarray:
    """Reference "correct" curve: the fitted two-line Tafel V."""
    e_corr = params["E_corr"]
    m_c, c_c = params["m_c"], params["c_c"]
    m_a, c_a = params["m_a"], params["c_a"]
    return np.where(xs <= e_corr, m_c * xs + c_c, m_a * xs + c_a)


def _split_segments(points: list, keep: np.ndarray) -> list:
    """Split kept points into contiguous runs, breaking at removed points.

    ``points`` and ``keep`` must be aligned (same length, both sorted by x).
    This lets the caller draw each surviving run as its own line instead of
    having matplotlib bridge across the deleted middle.
    """
    segments: list = []
    current: list = []
    for p, k in zip(points, keep):
        if k:
            current.append(p)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def clean_series(
    points: list[dict],
    series_key: str = "",
    label: str = "",
    exclude_middle_frac: float = EXCLUDE_MIDDLE_FRAC,
    sigma_mult: float = SIGMA_MULT,
) -> CleanResult:
    """Clean one extracted polarization curve.

    Returns a :class:`CleanResult`.  If the curve cannot be fit (``skipped``),
    ``points_after`` equals ``points_before`` and nothing is removed.
    """
    n_before = len(points)

    def _empty(skipped: bool) -> CleanResult:
        pts_sorted = sorted(points, key=lambda p: p["x"])
        return CleanResult(
            series_key=series_key,
            label=label,
            skipped=skipped,
            points_before=pts_sorted,
            points_after=pts_sorted,
            points_removed=[],
            segments=[pts_sorted],
            params={},
            n_before=n_before,
            n_after=n_before,
            n_removed=0,
            sigma=0.0,
            threshold=0.0,
        )

    params = fit_tafel_bv(points, exclude_middle_frac)
    if params is None:
        return _empty(skipped=True)

    pts = sorted(points, key=lambda p: p["x"])
    xs = np.asarray([p["x"] for p in pts], dtype=float)
    ys = np.asarray([p["y"] for p in pts], dtype=float)

    resid = ys - _model_y(xs, params)

    # Estimate the trusted noise level from the SIDE points only: the middle
    # is unreliable, so its (large) residuals must not inflate sigma.
    x_min, x_max = float(xs.min()), float(xs.max())
    span = x_max - x_min
    lo = x_min + span * exclude_middle_frac
    hi = x_max - span * exclude_middle_frac
    side = (xs < lo) | (xs > hi)
    side_resid = resid[side]

    med = float(np.median(side_resid))
    mad = float(np.median(np.abs(side_resid - med)))
    sigma = mad * MAD_TO_SIGMA

    # Degenerate fallback: if the side points are noise-free (sigma ~ 0),
    # fall back to the side std, then to a tiny floor, so a zero estimate
    # does not delete the whole curve.
    if sigma < 1e-12:
        sigma = float(np.std(side_resid))
    if sigma < 1e-12:
        sigma = 1e-9

    threshold = sigma_mult * sigma
    keep = np.abs(resid) <= threshold

    kept = [pts[i] for i in range(len(pts)) if keep[i]]
    removed = [pts[i] for i in range(len(pts)) if not keep[i]]
    segments = _split_segments(pts, keep)

    return CleanResult(
        series_key=series_key,
        label=label,
        skipped=False,
        points_before=pts,
        points_after=kept,
        points_removed=removed,
        segments=segments,
        params=params,
        n_before=n_before,
        n_after=len(kept),
        n_removed=len(removed),
        sigma=sigma,
        threshold=threshold,
    )


def clean_labeled_series(
    data: dict,
    series_mapping: dict,
    **kwargs,
) -> dict:
    """Clean every series that has a matched (non-empty) label.

    ``data`` is the dict read from ``converted_datapoints/data.json``;
    ``series_mapping`` is the ``series_mapping`` field of
    ``legend_mapping.json``.  Only series with a non-empty label (the ones the
    notebook keeps) are cleaned; rejected / unmatched series are ignored.
    """
    results: dict = {}
    for series_key, points in data.items():
        info = series_mapping.get(series_key, {})
        label = info.get("label")
        if not label:
            continue
        results[series_key] = clean_series(
            points, series_key=series_key, label=label, **kwargs
        )
    return results
