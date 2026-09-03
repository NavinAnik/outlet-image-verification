"""Scoring: how well each image fits its own outlet folder.

Each image is scored by the *leave-one-out median cosine similarity* to the
other images in its folder:

  - Cosine, because DINOv2 CLS vectors compare by direction (identity), and the
    vectors are unit-norm so cosine is just a dot product.
  - Leave-one-out, so an image is never compared to itself.
  - Median, not mean, so a folder with several fakes doesn't drag every image's
    reference down — the honest majority still defines "normal". This is what
    makes the method robust to multiple planted images in one folder.

Higher score = fits the outlet; lower = suspicious. Flagging lives in the
dual-threshold logic (next milestone); this module only produces the score.
"""

from __future__ import annotations

import numpy as np

# Defaults ship here and are calibrated on injected outliers (eval milestone).
DEFAULT_K = 3.0      # relative test: flag if score < folder_median - k * MAD
DEFAULT_TAU = 0.45   # absolute floor: an image must also be below this to flag
_MAD_SCALE = 1.4826  # scales MAD to be std-consistent for normal data
_MAD_EPS = 1e-6      # below this the folder is too tight for a relative test


def _l2_normalize(emb: np.ndarray) -> np.ndarray:
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)


def loo_median_similarity(emb: np.ndarray) -> np.ndarray:
    """Leave-one-out median cosine similarity per row.

    Input:  (N, D) embeddings (need not be pre-normalized).
    Output: (N,) scores; row i is the median cosine similarity of image i to the
            other N-1 images. A folder with N < 2 yields nan (can't judge fit).
    """
    emb = np.asarray(emb, dtype=np.float64)
    n = emb.shape[0]
    if n == 0:
        return np.empty(0)
    if n == 1:
        return np.array([np.nan])
    unit = _l2_normalize(emb)
    sim = unit @ unit.T
    np.fill_diagonal(sim, np.nan)  # exclude self from each row
    return np.nanmedian(sim, axis=1)


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def flag_outliers(
    scores: np.ndarray, k: float = DEFAULT_K, tau: float = DEFAULT_TAU
) -> np.ndarray:
    """Boolean flag per image within one folder.

    An image is flagged only when it is BOTH:
      - a within-folder low outlier: score < median - k * (scaled MAD), and
      - below the absolute floor: score < tau.

    Requiring both keeps clean folders empty: the relative test alone would flag
    the lowest image in *every* folder, and the floor alone can't tell a merely
    varied outlet from a fake. Small (n < 3) or too-tight (MAD ~ 0) folders can't
    support the relative test, so they fall back to the absolute floor alone.
    NaN scores (single-image folders) are never flagged.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    flags = np.zeros(n, dtype=bool)
    if n == 0:
        return flags

    valid = ~np.isnan(scores)
    below_floor = valid & (scores < tau)
    if n < 3:
        return below_floor  # too few images to judge relative fit

    med = np.median(scores[valid])
    spread = _mad(scores[valid]) * _MAD_SCALE
    if spread < _MAD_EPS:
        return below_floor  # folder too tight for a meaningful relative test

    relative = valid & (scores < med - k * spread)
    return relative & below_floor


def percentile_normalize(raw_suspicion: np.ndarray) -> np.ndarray:
    """Map raw suspicion (higher = more suspicious) to a dataset-relative [0, 1].

    Percentile rank across the whole dataset, so suspicion_score answers "how
    anomalous is this image compared to every other image we've seen", the
    min -> 0, the max -> 1. NaN (unjudgeable) -> 0. Feed this the global
    concatenation of (1 - similarity) over every image.
    """
    raw_suspicion = np.asarray(raw_suspicion, dtype=np.float64)
    out = np.zeros(raw_suspicion.shape[0], dtype=np.float64)
    valid = ~np.isnan(raw_suspicion)
    v = raw_suspicion[valid]
    if v.size == 0:
        return out
    if v.size == 1:
        out[valid] = 1.0
        return out
    ranks = np.empty(v.size)
    ranks[v.argsort()] = np.arange(v.size)
    out[valid] = ranks / (v.size - 1)
    return out


if __name__ == "__main__":  # self-check: python -m outlet_verify.scoring
    rng = np.random.default_rng(0)
    base = rng.normal(size=384)
    cluster = base + 0.01 * rng.normal(size=(6, 384))  # 6 near-identical images
    outlier = rng.normal(size=384)                      # 1 unrelated image
    emb = np.vstack([cluster, outlier])

    s = loo_median_similarity(emb)
    assert np.argmin(s) == 6, s                         # outlier scores lowest

    # Median robustness: with TWO fakes, both still score lowest (a mean would
    # let each fake inflate the other's reference).
    emb2 = np.vstack([cluster, outlier, rng.normal(size=384)])
    s2 = loo_median_similarity(emb2)
    assert set(np.argsort(s2)[:2].tolist()) == {6, 7}, s2

    # Edge cases don't crash.
    assert np.isnan(loo_median_similarity(np.ones((1, 4)))[0])
    assert loo_median_similarity(np.empty((0, 4))).shape == (0,)

    # --- flagging ---
    clean = np.array([0.60, 0.62, 0.59, 0.61, 0.60])
    assert not flag_outliers(clean).any()                       # clean -> empty
    one_bad = np.array([0.60, 0.62, 0.59, 0.61, 0.20])
    assert flag_outliers(one_bad).tolist() == [0, 0, 0, 0, 1]   # exactly the fake
    assert not flag_outliers(np.full(4, 0.9)).any()             # MAD~0, above floor
    assert flag_outliers(np.full(3, 0.30)).all()                # MAD~0, below floor
    assert flag_outliers(np.array([0.60, 0.30])).tolist() == [0, 1]  # n<3 -> floor
    assert not flag_outliers(np.array([np.nan])).any()          # single image

    # --- percentile normalization ---
    norm = percentile_normalize(np.array([0.1, 0.5, 0.9]))
    assert norm.tolist() == [0.0, 0.5, 1.0]
    assert percentile_normalize(np.array([0.1, np.nan, 0.9])).tolist() == [0.0, 0.0, 1.0]
    assert norm.min() >= 0.0 and norm.max() <= 1.0

    print(f"OK  cluster median ~{np.median(s[:6]):.3f}  outlier ~{s[6]:.3f}  "
          f"flagging+normalization verified")
