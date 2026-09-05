"""Synthetic-injection evaluation to calibrate the thresholds and report metrics.

There are no ground-truth labels, so we make our own: a "fake" is exactly what
the problem describes — a photo from a *different* outlet dropped into this
folder. We inject real images (their cached embeddings) from other outlets,
label those as positives, run the detector, and measure precision/recall.

Caveat: the original images are treated as negatives, but some are genuinely
odd (the very thing we're trying to catch). Those inflate the false-positive
count, so the reported precision is a *lower bound* — real precision is higher.

Usage:
    python -m eval.evaluate --data-dir dataset [--n-inject 1] [--trials 3] [--seed 0]
"""

from __future__ import annotations

import argparse

import numpy as np

from outlet_verify.embeddings import DEFAULT_MODEL, embed_folder_rotations
from outlet_verify.pipeline import find_outlet_dirs
from outlet_verify.scoring import (
    DEFAULT_CEILING,
    DEFAULT_MIN_GAP,
    canonicalize,
    fit_scores,
    flag_outliers,
)

MIN_GAP_GRID = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]


def _load(data_dir: str, model: str) -> list[np.ndarray]:
    dirs = find_outlet_dirs(data_dir)
    embs = []
    try:
        from tqdm import tqdm
        dirs = tqdm(dirs, desc="loading embeddings", unit="outlet")
    except ImportError:
        pass
    for d in dirs:
        names, emb4 = embed_folder_rotations(d, model=model)
        emb = canonicalize(emb4) if names else emb4
        if emb.shape[0] > 0:
            embs.append(emb)
    return embs


def _trial(embs: list[np.ndarray], n_inject: int, rng) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inject n_inject cross-outlet fakes into each folder; return (scores, labels)."""
    all_emb = np.vstack(embs)
    owner = np.concatenate([np.full(e.shape[0], i) for i, e in enumerate(embs)])
    out = []
    for i, emb in enumerate(embs):
        others = all_emb[owner != i]
        pick = rng.choice(others.shape[0], size=n_inject, replace=False)
        aug = np.vstack([emb, others[pick]])
        labels = np.array([0] * emb.shape[0] + [1] * n_inject)
        out.append((fit_scores(aug), labels))
    return out


def _cluster_trial(embs: list[np.ndarray], size: int, rng) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inject a *batch* of `size` images from ONE other outlet into each folder —
    a coherent fake cluster that self-vouches under a small fixed k."""
    out = []
    for i, emb in enumerate(embs):
        j = rng.choice([x for x in range(len(embs)) if x != i])
        src = embs[j]
        if src.shape[0] < size:
            continue
        pick = rng.choice(src.shape[0], size=size, replace=False)
        aug = np.vstack([emb, src[pick]])
        labels = np.array([0] * emb.shape[0] + [1] * size)
        out.append((fit_scores(aug), labels))
    return out


def _cluster_recall(embs: list[np.ndarray], size: int, seed: int) -> float:
    """Flag-recall on injected coherent fake clusters (uses shipped thresholds)."""
    trials = _cluster_trial(embs, size, np.random.default_rng(seed))
    tp = fn = 0
    for scores, labels in trials:
        flags = flag_outliers(scores).astype(int)
        tp += int(((flags == 1) & (labels == 1)).sum())
        fn += int(((flags == 0) & (labels == 1)).sum())
    return tp / (tp + fn) if tp + fn else 0.0


def _ranking_metrics(trials: list) -> dict:
    """Threshold-independent detector quality: is the injected fake surfaced at
    the top of its folder's suspicion ranking? Unaffected by tau/k, and a fairer
    read than precision/recall when the negatives contain real (unlabeled)
    outliers competing for the top ranks.
    """
    at1 = at3 = 0
    n = 0
    pctiles = []
    for scores, labels in trials:
        for j in np.where(labels == 1)[0]:
            below = int((scores < scores[j]).sum())  # more-suspicious images
            at1 += below == 0
            at3 += below < 3
            pctiles.append(below / max(1, len(scores) - 1))
            n += 1
    return {
        "recall@top1": at1 / n,
        "recall@top3": at3 / n,
        "median_rank_pctile": float(np.median(pctiles)),
    }


def _metrics(trials: list, min_gap: float, ceiling: float = DEFAULT_CEILING) -> dict:
    tp = fp = fn = 0
    for scores, labels in trials:
        flags = flag_outliers(scores, min_gap=min_gap, ceiling=ceiling).astype(int)
        tp += int(((flags == 1) & (labels == 1)).sum())
        fp += int(((flags == 1) & (labels == 0)).sum())
        fn += int(((flags == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def run(data_dir: str, model: str, n_inject: int, trials: int, seed: int) -> None:
    embs = _load(data_dir, model)
    all_trials = []
    for t in range(trials):
        all_trials += _trial(embs, n_inject, np.random.default_rng(seed + t))
    positives = sum(int(lbl.sum()) for _, lbl in all_trials)
    negatives = sum(int((lbl == 0).sum()) for _, lbl in all_trials)
    print(f"\n{len(embs)} outlets | {trials} trial(s) | injected {positives} fakes "
          f"vs {negatives} real images (base rate {positives/(positives+negatives):.1%})")

    rank = _ranking_metrics(all_trials)
    print(f"\n  ranking quality (threshold-independent):")
    print(f"    fake is most-suspicious in folder (recall@top1):  {rank['recall@top1']:.1%}")
    print(f"    fake among 3 most-suspicious     (recall@top3):   {rank['recall@top3']:.1%}")
    print(f"    median suspicion percentile of injected fakes:     {rank['median_rank_pctile']:.2f}")

    cr = _cluster_recall(embs, size=5, seed=seed)
    print(f"\n  coherent-cluster flag-recall (5 fakes from one outlet): {cr:.1%}")

    # Flag metrics (folder-relative gap rule): shipped default + a min_gap sweep.
    dflt = _metrics(all_trials, DEFAULT_MIN_GAP)
    print("\n                        precision  recall     f1     (TP/FP/FN)")
    print(f"  shipped  min_gap={DEFAULT_MIN_GAP} ceil={DEFAULT_CEILING}   "
          f"{dflt['precision']:.3f}    {dflt['recall']:.3f}   {dflt['f1']:.3f}   "
          f"({dflt['tp']}/{dflt['fp']}/{dflt['fn']})")

    best = None
    print(f"\n  min_gap sweep (ceiling={DEFAULT_CEILING}):")
    print("    min_gap  precision  recall    f1")
    for mg in MIN_GAP_GRID:
        m = _metrics(all_trials, mg)
        print(f"    {mg:.2f}     {m['precision']:.3f}     {m['recall']:.3f}   {m['f1']:.3f}")
        if best is None or m["f1"] > best[1]["f1"]:
            best = (mg, m)

    print(f"\n  Best-F1 min_gap={best[0]}  (F1={best[1]['f1']:.3f}). "
          f"Precision is a lower bound; see module docstring.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-inject", type=int, default=1, help="fakes injected per folder")
    ap.add_argument("--trials", type=int, default=3, help="repeats with different fakes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.data_dir, args.model, args.n_inject, args.trials, args.seed)


if __name__ == "__main__":
    main()
