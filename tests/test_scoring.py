"""Unit tests for the scoring and dual-threshold flagging logic.

Pure logic on synthetic vectors/scores — no model or images required.
"""

import numpy as np

from outlet_verify.scoring import (
    DEFAULT_TAU,
    flag_outliers,
    loo_median_similarity,
    percentile_normalize,
)


def _cluster_plus_outliers(n_clean=6, n_out=1, dim=32, seed=0):
    """A tight cluster of near-identical vectors followed by unrelated ones."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim)
    clean = base + 0.01 * rng.normal(size=(n_clean, dim))
    out = rng.normal(size=(n_out, dim))
    return np.vstack([clean, out])  # outliers are the last n_out rows


# --- loo_median_similarity ---

def test_outlier_scores_lowest():
    emb = _cluster_plus_outliers(n_clean=6, n_out=1)
    s = loo_median_similarity(emb)
    assert int(np.argmin(s)) == 6  # the single outlier


def test_median_robust_to_multiple_outliers():
    # Two fakes: median keeps the honest majority as the reference, so BOTH
    # fakes still score lowest. A mean would let each fake prop up the other.
    emb = _cluster_plus_outliers(n_clean=6, n_out=2)
    s = loo_median_similarity(emb)
    assert set(np.argsort(s)[:2].tolist()) == {6, 7}


def test_empty_and_single_and_pair():
    assert loo_median_similarity(np.empty((0, 4))).shape == (0,)
    assert np.isnan(loo_median_similarity(np.ones((1, 4)))[0])  # n=1 unjudgeable
    # n=2: each score is the single pairwise cosine similarity (median of one).
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    s = loo_median_similarity(a)
    assert np.allclose(s, [0.0, 0.0])  # orthogonal -> cosine 0


# --- flag_outliers ---

def test_clean_folder_returns_empty():
    clean = np.array([0.60, 0.62, 0.59, 0.61, 0.60])
    assert not flag_outliers(clean).any()


def test_single_outlier_flagged_exactly():
    scores = np.array([0.60, 0.62, 0.59, 0.61, 0.20])
    assert flag_outliers(scores).tolist() == [0, 0, 0, 0, 1]


def test_both_conditions_required():
    # Strong within-folder outlier, but still ABOVE the absolute floor -> not
    # flagged. Proves the AND semantics that keeps varied-but-real folders clean.
    scores = np.array([0.90, 0.92, 0.91, 0.89, 0.60])
    assert scores[4] > DEFAULT_TAU
    assert not flag_outliers(scores).any()


def test_mad_zero_falls_back_to_floor():
    assert not flag_outliers(np.full(4, 0.90)).any()  # tight, above floor
    assert flag_outliers(np.full(3, 0.30)).all()      # tight, below floor


def test_small_folder_uses_floor_only():
    assert flag_outliers(np.array([0.60, 0.30])).tolist() == [0, 1]  # n=2
    assert not flag_outliers(np.array([np.nan])).any()               # n=1


# --- percentile_normalize ---

def test_normalize_bounds_and_monotonic():
    norm = percentile_normalize(np.array([0.1, 0.5, 0.9]))
    assert norm.tolist() == [0.0, 0.5, 1.0]
    assert norm.min() >= 0.0 and norm.max() <= 1.0
    # order preserved
    scrambled = np.array([0.9, 0.1, 0.5])
    assert np.argsort(percentile_normalize(scrambled)).tolist() == [1, 2, 0]


def test_normalize_handles_nan_and_singleton():
    assert percentile_normalize(np.array([0.1, np.nan, 0.9])).tolist() == [0.0, 0.0, 1.0]
    assert percentile_normalize(np.array([0.7])).tolist() == [1.0]
    assert percentile_normalize(np.empty(0)).shape == (0,)
