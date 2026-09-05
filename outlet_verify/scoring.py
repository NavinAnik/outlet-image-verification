"""Scoring: how well each image fits its own outlet folder.

Each image is scored by the mean cosine similarity to its *k nearest neighbours*
within the folder:

  - Cosine on unit-norm DINOv2 CLS vectors (a dot product), comparing by
    direction (identity).
  - Nearest neighbours, not all images: an outlet legitimately contains several
    kinds of shot (exterior, counter, signage). Scoring against the whole folder
    penalises a genuine counter photo for looking unlike the exterior shots;
    scoring against its most-similar folder-mates asks the right question —
    "does this resemble *its own kind* of image in this outlet?".
  - k scales with folder size (`k = clip(round(0.25*N), 3, 12)`): a few colluding
    fakes can't vouch for each other, because in a larger folder their k
    neighbours include real images that drag the mean back down. Small folders
    keep k=3.

Higher score = fits the outlet; lower = suspicious. On synthetic injections
(see eval/) this beat both median-to-all and a fixed small k, on single fakes
and coherent fake clusters alike.
"""

from __future__ import annotations

import numpy as np

# Defaults ship here and are calibrated on injected outliers (see eval/).
# The neighbourhood scales with folder size: k = clip(round(K_FRAC * N), K_MIN, K_MAX).
# Small enough to tolerate legit visual diversity, but folder-relative so a small
# coherent cluster of fakes can't become its own k-neighbourhood and self-vouch.
DEFAULT_K_FRAC = 0.25
DEFAULT_K_MIN = 3         # unchanged for small/median folders (median is 12 -> k=3)
DEFAULT_K_MAX = 12
DEFAULT_K = 3.0           # relative test: gap must exceed k * MAD (unrelated to kNN k)
DEFAULT_MIN_GAP = 0.10    # ...and at least this absolute gap below the folder median
DEFAULT_CEILING = 0.80    # never flag an image still this similar to the outlet
DEFAULT_TAU = 0.55        # absolute floor for small/degenerate-folder fallback
_MAD_SCALE = 1.4826       # scales MAD to be std-consistent for normal data
_MAD_EPS = 1e-6           # below this the folder is too tight for a relative test


def _l2_normalize(emb: np.ndarray) -> np.ndarray:
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)


def _k_for(n: int, k_frac: float, k_min: int, k_max: int) -> int:
    """Folder-relative neighbourhood size, clamped and never exceeding N-1."""
    return int(min(n - 1, np.clip(round(k_frac * n), k_min, k_max)))


def fit_scores(
    emb: np.ndarray,
    n_neighbors: int | None = None,
    k_frac: float = DEFAULT_K_FRAC,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
) -> np.ndarray:
    """Mean cosine similarity of each image to its most-similar folder-mates.

    The neighbourhood k scales with folder size — `k = clip(round(k_frac*N),
    k_min, k_max)` — so small folders keep k=k_min (median folder is 12 -> k=3)
    while large folders average over more images. That matters because a small
    coherent cluster of fakes (say 4 photos of a different shop) would otherwise
    *be* each other's k nearest neighbours and vouch for one another; a
    folder-relative k dilutes them with real images and exposes them. Pass an
    explicit `n_neighbors` to override.

    Input:  (N, D) embeddings (need not be pre-normalized).
    Output: (N,) scores; higher = fits the outlet, lower = suspicious. N < 2
            yields nan (can't judge fit).
    """
    emb = np.asarray(emb, dtype=np.float64)
    n = emb.shape[0]
    if n == 0:
        return np.empty(0)
    if n == 1:
        return np.array([np.nan])
    kk = min(n_neighbors, n - 1) if n_neighbors is not None else _k_for(n, k_frac, k_min, k_max)
    unit = _l2_normalize(emb)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)  # exclude self from each row
    topk = -np.sort(-sim, axis=1)[:, :kk]  # kk highest similarities per row
    return topk.mean(axis=1)


def canonicalize(emb4: np.ndarray, iters: int = 2) -> np.ndarray:
    """Pick each image's orientation that best agrees with its folder.

    Input:  (N, 4, D) — each image embedded at 0/90/180/270 (L2-normalized).
    Output: (N, D) — one vector per image, the rotation most aligned with the
            folder's consensus. This makes scoring rotation-invariant: a rotated
            real photo snaps into alignment with its outlet, while a genuine fake
            aligns poorly at every rotation and still scores low. What matters is
            internal consistency within the folder, not absolute uprightness.
    """
    emb4 = np.asarray(emb4, dtype=np.float64)
    n = emb4.shape[0]
    if n == 0:
        return np.empty((0, 0))
    if n < 2:
        return emb4[:, 0, :]  # can't judge orientation from a single image
    chosen = emb4[:, 0, :]  # start from as-stored (rotation 0)
    for _ in range(max(1, iters)):
        anchor = chosen.mean(axis=0)
        anchor /= np.linalg.norm(anchor) + 1e-12
        best_rot = np.argmax(emb4 @ anchor, axis=1)  # (N,) per-image best rotation
        chosen = emb4[np.arange(n), best_rot]
    return chosen


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def flag_outliers(
    scores: np.ndarray,
    k: float = DEFAULT_K,
    tau: float = DEFAULT_TAU,
    min_gap: float = DEFAULT_MIN_GAP,
    ceiling: float = DEFAULT_CEILING,
) -> np.ndarray:
    """Boolean flag per image within one folder.

    An image is flagged when it deviates from the outlet's *own* cohesion:

        gap = folder_median - score
        flag = gap >= max(k * scaledMAD, min_gap)   AND   score < ceiling

    Measuring the gap relative to the folder's median (in the folder's own MAD
    units) is what catches high-similarity fakes: different storefronts share
    domain features, so a fake can score, say, 0.76 in absolute terms yet sit far
    below a tight outlet's typical 0.88 — a strong relative outlier a fixed floor
    would miss. ``min_gap`` stops ultra-tight folders from flagging on noise, and
    ``ceiling`` never flags an image that is still highly similar overall (which
    keeps clean folders empty).

    Small (n < 3) or degenerate (MAD ~ 0) folders can't support the relative
    test, so they fall back to the absolute floor ``score < tau``. NaN scores
    (single-image folders) are never flagged.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    flags = np.zeros(n, dtype=bool)
    if n == 0:
        return flags

    valid = ~np.isnan(scores)
    if n < 3:
        return valid & (scores < tau)  # too few images to judge relative fit

    med = np.median(scores[valid])
    spread = _mad(scores[valid]) * _MAD_SCALE
    if spread < _MAD_EPS:
        return valid & (scores < tau)  # folder too tight for a relative test

    gap = med - scores
    threshold = max(k * spread, min_gap)
    return valid & (gap >= threshold) & (scores < ceiling)


def percentile_normalize(raw_suspicion: np.ndarray) -> np.ndarray:
    """Map raw suspicion (higher = more suspicious) to a dataset-relative [0, 1].

    Percentile rank across the whole dataset, so suspicion_score answers "how
    anomalous is this image compared to every other image we've seen", the
    min -> 0, the max -> 1. Tied values get the same (average-rank) score so
    equally-suspicious images are reported equally. NaN (unjudgeable) -> 0. Feed
    this the global concatenation of the folder gap (median - similarity).
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
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    ranks_sorted = np.arange(v.size, dtype=np.float64)
    start = 0  # average the ranks within each tied group
    for i in range(1, v.size + 1):
        if i == v.size or sv[i] != sv[start]:
            if i - start > 1:
                ranks_sorted[start:i] = (start + i - 1) / 2.0
            start = i
    ranks = np.empty(v.size)
    ranks[order] = ranks_sorted
    out[valid] = ranks / (v.size - 1)
    return out


if __name__ == "__main__":  # self-check: python -m outlet_verify.scoring
    rng = np.random.default_rng(0)
    base = rng.normal(size=384)
    cluster = base + 0.01 * rng.normal(size=(6, 384))  # 6 near-identical images
    outlier = rng.normal(size=384)                      # 1 unrelated image
    emb = np.vstack([cluster, outlier])

    s = fit_scores(emb)
    assert np.argmin(s) == 6, s                         # outlier scores lowest

    # Multi-fake robustness: with TWO unrelated fakes, both still score lowest —
    # k=3 means each fake needs 3 similar supporters to blend in, and it can't
    # find them.
    emb2 = np.vstack([cluster, outlier, rng.normal(size=384)])
    s2 = fit_scores(emb2)
    assert set(np.argsort(s2)[:2].tolist()) == {6, 7}, s2

    # Edge cases don't crash.
    assert np.isnan(fit_scores(np.ones((1, 4)))[0])
    assert fit_scores(np.empty((0, 4))).shape == (0,)

    # --- flagging (folder-relative gap rule) ---
    clean = np.array([0.60, 0.62, 0.59, 0.61, 0.60])
    assert not flag_outliers(clean).any()                       # clean -> empty
    one_bad = np.array([0.60, 0.62, 0.59, 0.61, 0.20])
    assert flag_outliers(one_bad).tolist() == [0, 0, 0, 0, 1]   # exactly the fake
    # High-similarity fake in a tight folder: above the old floor (0.55) but a
    # strong relative outlier below the ceiling -> now flagged (the M13 fix).
    tight = np.array([0.90, 0.91, 0.89, 0.90, 0.76])
    assert flag_outliers(tight).tolist() == [0, 0, 0, 0, 1]
    # Ceiling spares an outlier that is still very similar overall.
    assert not flag_outliers(np.array([0.98, 0.99, 0.97, 0.98, 0.85])).any()
    # min_gap suppresses a tiny-gap mild outlier.
    assert not flag_outliers(np.array([0.60, 0.61, 0.59, 0.60, 0.52])).any()
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
