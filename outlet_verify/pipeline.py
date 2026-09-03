"""End-to-end pipeline: dataset of outlet folders -> per-outlet JSON results.

For each outlet we embed its images (DINOv2, cached), score each image by
leave-one-out median similarity, flag with the dual threshold, and normalize
suspicion globally so the score is dataset-relative. Every outlet is emitted,
clean ones with an empty ``flagged_images`` list.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .embeddings import DEFAULT_MODEL, embed_folder
from .scoring import (
    DEFAULT_K,
    DEFAULT_TAU,
    flag_outliers,
    loo_median_similarity,
    percentile_normalize,
)


def find_outlet_dirs(data_dir: Path) -> list[Path]:
    return sorted(p for p in Path(data_dir).iterdir() if p.is_dir())


def _reason(score: float) -> str:
    if np.isnan(score):
        return "single image in folder; fit cannot be assessed"
    return f"low visual similarity to the outlet's other images (median cosine {score:.2f})"


def analyze(
    data_dir: Path,
    model: str = DEFAULT_MODEL,
    k: float = DEFAULT_K,
    tau: float = DEFAULT_TAU,
    ranking: bool = True,
    reason_fn=None,
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
    per: list[tuple[str, list[str], np.ndarray]] = []
    raw_chunks: list[np.ndarray] = []
    for d in dirs_iter:
        names, emb = embed_folder(d, model=model)
        scores = loo_median_similarity(emb)
        per.append((d.name, names, scores))
        raw_chunks.append(np.where(np.isnan(scores), np.nan, 1.0 - scores))

    raw = np.concatenate(raw_chunks) if raw_chunks else np.empty(0)
    suspicion = percentile_normalize(raw)  # dataset-relative [0, 1]

    # Pass 2: flag per folder and build records.
    records: list[dict] = []
    off = 0
    for outlet_id, names, scores in per:
        m = len(names)
        susp = suspicion[off:off + m]
        off += m
        flags = flag_outliers(scores, k=k, tau=tau)
        order = np.argsort(-susp, kind="stable")  # most -> least suspicious

        reasons = reason_fn(outlet_id, names, np.where(flags)[0], scores) if reason_fn else {}
        flagged = [
            {
                "file_name": names[i],
                "suspicion_score": round(float(susp[i]), 4),
                "reason": reasons.get(i, _reason(scores[i])),
            }
            for i in order if flags[i]
        ]
        rec = {"outlet_id": outlet_id, "total_images": m, "flagged_images": flagged}
        if ranking:
            rec["ranking"] = [names[i] for i in order]
        records.append(rec)

    return records


def write_results(records: list[dict], out_path: Path) -> None:
    Path(out_path).write_text(json.dumps(records, indent=2))
