"""Unit tests for the scoring and dual-threshold flagging logic.

Pure logic on synthetic vectors/scores — no model or images required.
"""

import numpy as np

from outlet_verify.scoring import (
    DEFAULT_TAU,
    canonicalize,
    fit_scores,
    flag_outliers,
    percentile_normalize,
)


def _cluster_plus_outliers(n_clean=6, n_out=1, dim=32, seed=0):
    """A tight cluster of near-identical vectors followed by unrelated ones."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim)
    clean = base + 0.01 * rng.normal(size=(n_clean, dim))
    out = rng.normal(size=(n_out, dim))
    return np.vstack([clean, out])  # outliers are the last n_out rows


# --- fit_scores (kNN-mean) ---

def test_outlier_scores_lowest():
    emb = _cluster_plus_outliers(n_clean=6, n_out=1)
    s = fit_scores(emb)
    assert int(np.argmin(s)) == 6  # the single outlier


def test_robust_to_multiple_outliers():
    # Two unrelated fakes: with k=3 neighbours neither can find 3 similar
    # supporters, so BOTH still score lowest (they can't vouch for each other).
    emb = _cluster_plus_outliers(n_clean=6, n_out=2)
    s = fit_scores(emb)
    assert set(np.argsort(s)[:2].tolist()) == {6, 7}


def test_folder_relative_k_dilutes_self_vouching_cluster():
    # 34 "real" images + 6 near-identical "fake" images (a batched different shop).
    # At k=3 each fake's 3 neighbours are other fakes, so they self-vouch (~1.0);
    # a folder-relative k must include real images and drag their scores down.
    rng = np.random.default_rng(3)
    d = 32
    real = np.stack([_unit(rng.normal(size=d)) for _ in range(34)])  # spread out
    fbase = _unit(rng.normal(size=d))
    fake = np.stack([_unit(fbase + 0.02 * rng.normal(size=d)) for _ in range(6)])  # tight
    emb = np.vstack([real, fake])
    fake_idx = np.arange(34, 40)

    s3 = fit_scores(emb, n_neighbors=3)   # fixed small k: cluster self-vouches
    s = fit_scores(emb)                   # folder-relative k (=10 for N=40)
    assert s3[fake_idx].mean() > 0.9                    # fakes vouch each other at k=3
    assert (s[fake_idx] < s3[fake_idx]).all()          # every fake's score drops
    assert s3[fake_idx].mean() - s[fake_idx].mean() > 0.2  # meaningful dilution


def test_empty_and_single_and_pair():
    assert fit_scores(np.empty((0, 4))).shape == (0,)
    assert np.isnan(fit_scores(np.ones((1, 4)))[0])  # n=1 unjudgeable
    # n=2: k collapses to 1 neighbour, so the score is that one pairwise cosine.
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    s = fit_scores(a)
    assert np.allclose(s, [0.0, 0.0])  # orthogonal -> cosine 0


# --- canonicalize (rotation invariance) ---

def _unit(x):
    return x / np.linalg.norm(x)


def test_canonicalize_snaps_rotated_image_to_folder():
    rng = np.random.default_rng(1)
    d = 32
    u = _unit(rng.normal(size=d))  # the folder's true orientation

    def rand_rots():
        return np.stack([_unit(rng.normal(size=d)) for _ in range(4)])

    embs = []
    for _ in range(4):  # upright images: rotation 0 aligned to u
        r = rand_rots(); r[0] = _unit(u + 0.02 * rng.normal(size=d)); embs.append(r)
    r = rand_rots(); r[2] = _unit(u + 0.02 * rng.normal(size=d)); embs.append(r)  # rotated
    emb4 = np.stack(embs)  # (5, 4, d)

    canon = canonicalize(emb4)
    assert (canon @ u > 0.9).all()               # every image aligns with folder
    assert canon[4] @ emb4[4, 2] > 0.9           # rotated image snapped to its aligned rot
    assert canon[4] @ emb4[4, 0] < 0.5           # not its misaligned as-stored rot


def test_canonicalize_singleton_returns_rot0():
    emb4 = np.random.default_rng(0).normal(size=(1, 4, 8))
    assert np.array_equal(canonicalize(emb4), emb4[:, 0, :])


# --- flag_outliers ---

def test_clean_folder_returns_empty():
    clean = np.array([0.60, 0.62, 0.59, 0.61, 0.60])
    assert not flag_outliers(clean).any()


def test_single_outlier_flagged_exactly():
    scores = np.array([0.60, 0.62, 0.59, 0.61, 0.20])
    assert flag_outliers(scores).tolist() == [0, 0, 0, 0, 1]


def test_high_similarity_outlier_in_tight_folder_flagged():
    # The M13 regression case: an image well ABOVE the absolute floor (0.55) but a
    # strong relative outlier in a tight folder, below the ceiling -> flagged.
    # (Mirrors the real "different shop scoring 0.76" false negative.)
    scores = np.array([0.90, 0.91, 0.89, 0.90, 0.76])
    assert scores[4] > DEFAULT_TAU
    assert flag_outliers(scores).tolist() == [0, 0, 0, 0, 1]


def test_ceiling_spares_still_similar_outlier():
    # A relative outlier that is still very similar overall (>= ceiling) is not
    # flagged — keeps clean, tightly-clustered folders empty.
    scores = np.array([0.98, 0.99, 0.97, 0.98, 0.85])
    assert not flag_outliers(scores).any()


def test_min_gap_suppresses_tiny_gap_outlier():
    # A mild low image whose gap below the median is under min_gap is not flagged.
    scores = np.array([0.60, 0.61, 0.59, 0.60, 0.52])
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
