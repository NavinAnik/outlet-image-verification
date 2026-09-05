"""End-to-end pipeline: dataset of outlet folders -> per-outlet JSON results.

For each outlet we embed its images (DINOv2, cached), score each image by the
mean cosine similarity to its nearest folder-mates, flag folder-relative
outliers, clear same-shop false positives with OCR, and normalize suspicion
globally so the score is dataset-relative. Every outlet is emitted, clean ones
with an empty ``flagged_images`` list.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .embeddings import DEFAULT_MODEL, embed_folder, embed_folder_rotations
from .scoring import (
    DEFAULT_CEILING,
    DEFAULT_K,
    DEFAULT_MIN_GAP,
    DEFAULT_TAU,
    canonicalize,
    fit_scores,
    flag_outliers,
    percentile_normalize,
)


def find_outlet_dirs(data_dir: Path) -> list[Path]:
    return sorted(p for p in Path(data_dir).iterdir() if p.is_dir())


def _reason(score: float) -> str:
    if np.isnan(score):
        return "single image in folder; fit cannot be assessed"
    return f"low visual similarity to the outlet's other images (fit score {score:.2f})"


def analyze(
    data_dir: Path,
    model: str = DEFAULT_MODEL,
    k: float = DEFAULT_K,
    tau: float = DEFAULT_TAU,
    min_gap: float = DEFAULT_MIN_GAP,
    ceiling: float = DEFAULT_CEILING,
    ranking: bool = True,
    reason_fn=None,
    ocr_filter: bool = True,
    rotation_invariant: bool = True,
) -> list[dict]:
    """Return a per-outlet record list conforming to the assignment schema.

    ``reason_fn(folder, names, flagged_idx, scores) -> {idx: str}`` optionally
    supplies richer explanations (wired up in a later milestone); when omitted a
    similarity-based fallback reason is used.
    """
    dirs = find_outlet_dirs(data_dir)
    try:
        from tqdm import tqdm
        dirs_iter = tqdm(dirs, desc="embedding outlets", unit="outlet")
    except ImportError:
        dirs_iter = dirs

    # Pass 1: embed + score every folder; collect raw suspicion (1 - similarity).
    per: list[tuple[Path, list[str], np.ndarray]] = []
    raw_chunks: list[np.ndarray] = []
    for d in dirs_iter:
        if rotation_invariant:
            names, emb4 = embed_folder_rotations(d, model=model)
            emb = canonicalize(emb4) if len(names) else emb4
        else:
            names, emb = embed_folder(d, model=model)
        scores = fit_scores(emb)
        per.append((d, names, scores))
        # Suspicion is the folder-relative gap (median - score), matching the
        # flag rule's basis, so a flagged high-similarity fake still reads as
        # suspicious globally (a global 1-similarity would understate it).
        gap = np.nanmedian(scores) - scores if np.isfinite(scores).any() else scores
        raw_chunks.append(np.where(np.isnan(scores), np.nan, gap))

    raw = np.concatenate(raw_chunks) if raw_chunks else np.empty(0)
    suspicion = percentile_normalize(raw)  # dataset-relative [0, 1]

    # Pass 2: vision flags per folder.
    flags_per = [
        flag_outliers(scores, k=k, tau=tau, min_gap=min_gap, ceiling=ceiling)
        for _, _, scores in per
    ]

    # Pass 2b: OCR signage corroboration clears same-shop false positives.
    if ocr_filter:
        _ocr_clear_flags(per, flags_per)

    # Pass 3: build per-outlet records.
    records: list[dict] = []
    off = 0
    for (folder, names, scores), flags in zip(per, flags_per):
        m = len(names)
        susp = suspicion[off:off + m]
        off += m
        order = np.argsort(-susp, kind="stable")  # most -> least suspicious

        reasons = reason_fn(folder, names, np.where(flags)[0], scores) if reason_fn else {}
        flagged = [
            {
                "file_name": names[i],
                "suspicion_score": round(float(susp[i]), 4),
                "reason": reasons.get(i, _reason(scores[i])),
            }
            for i in order if flags[i]
        ]
        rec = {"outlet_id": folder.name, "total_images": m, "flagged_images": flagged}
        if ranking:
            rec["ranking"] = [names[i] for i in order]
        records.append(rec)

    return records


def _ocr_clear_flags(per, flags_per, threshold=None) -> None:
    """Un-flag vision-flagged images whose signage text matches their outlet's
    *distinctive* text (same shop, photographed differently). Only flagged
    folders are OCR-ed. Degrades to a no-op if OCR is unavailable.
    """
    from . import ocr

    if not ocr.available():
        return
    if threshold is None:
        threshold = ocr.DEFAULT_OCR_CLEAR

    flagged = [j for j, f in enumerate(flags_per) if f.any()]
    if not flagged:
        return
    try:
        from tqdm import tqdm
        flagged_iter = tqdm(flagged, desc="OCR corroboration", unit="outlet")
    except ImportError:
        flagged_iter = flagged

    tokens_per: dict[int, dict[str, set]] = {}
    outlet_tokens: dict[str, set] = {}
    for j in flagged_iter:
        folder, names, _ = per[j]
        toks = {name: ocr.read_tokens(folder / name) for name in names}
        tokens_per[j] = toks
        outlet_tokens[folder.name] = set().union(*toks.values()) if toks else set()

    idf = ocr.compute_idf(outlet_tokens)  # promo text (many outlets) -> ~0
    for j in flagged:
        _, names, _ = per[j]
        flags = flags_per[j]
        toks = tokens_per[j]
        reference = set().union(
            *[toks[names[i]] for i in range(len(names)) if not flags[i]]
        ) if (~flags).any() else set()
        for i in range(len(names)):
            if flags[i] and ocr.corroboration(toks[names[i]], reference, idf) >= threshold:
                flags[i] = False  # same shop, just a different-looking photo


def write_results(records: list[dict], out_path: Path) -> None:
    Path(out_path).write_text(json.dumps(records, indent=2))
