"""CLI: python -m outlet_verify --data-dir dataset --out results.json"""

from __future__ import annotations

import argparse

from .embeddings import DEFAULT_MODEL
from .pipeline import analyze, write_results
from .reasons import CLIP_MODEL, clip_reason_fn
from .scoring import DEFAULT_K, DEFAULT_TAU


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="dataset", help="root of outlet folders")
    ap.add_argument("--out", default="results.json", help="output JSON path")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="DINOv2 model id")
    ap.add_argument("--k", type=float, default=DEFAULT_K, help="MAD multiplier (relative test)")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU, help="absolute similarity floor")
    ap.add_argument("--no-ranking", action="store_true", help="omit the optional ranking field")
    ap.add_argument("--no-clip", action="store_true", help="skip CLIP reasons (similarity-only)")
    ap.add_argument("--clip-model", default=CLIP_MODEL, help="CLIP model id for reasons")
    args = ap.parse_args()

    reason_fn = None if args.no_clip else clip_reason_fn(args.clip_model)
    records = analyze(
        args.data_dir, model=args.model, k=args.k, tau=args.tau,
        ranking=not args.no_ranking, reason_fn=reason_fn,
    )
    write_results(records, args.out)

    flagged = sum(len(r["flagged_images"]) for r in records)
    dirty = sum(1 for r in records if r["flagged_images"])
    imgs = sum(r["total_images"] for r in records)
    print(
        f"{len(records)} outlets, {imgs} images -> {flagged} flagged "
        f"across {dirty} outlets. Wrote {args.out}"
    )


if __name__ == "__main__":
    main()
