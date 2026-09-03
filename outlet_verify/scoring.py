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

    print(f"OK  cluster median ~{np.median(s[:6]):.3f}  outlier ~{s[6]:.3f}")
