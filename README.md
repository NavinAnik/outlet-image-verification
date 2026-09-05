# Outlet Image Verification

Flag images that don't belong in an outlet's photo history — unsupervised, per
folder, using each outlet's own images as the reference. See
[`WRITEUP.md`](WRITEUP.md) for method, rationale, trade-offs, and limitations.

**Pipeline:** DINOv2 CLS embeddings (cached, rotation-canonical) → score each image
by mean cosine similarity to its k nearest neighbours in the folder → flag when an
image is *both* a within-folder outlier (folder-gap MAD test) *and* below a
similarity ceiling → global percentile normalization of the suspicion score → OCR
signage corroboration clears same-shop false positives → CLIP zero-shot supplies
the reason.

## Setup

A virtualenv is used because the models need a specific torch/torchvision combo.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
# Data layout: dataset/outlet_<id>/image_XXXX.jpg
.venv/bin/python -m outlet_verify --data-dir dataset --out results.json
```

By default this also runs an **OCR pass** (easyocr, bn+en) that clears same-shop
false positives by matching signage text — slower on first run (downloads models,
OCRs flagged folders; cached after). Disable with `--no-ocr`.

Flags:
`--model` (DINOv2 id), `--k` / `--min-gap` / `--ceiling` / `--tau` (flag thresholds),
`--no-clip` (skip CLIP reasons), `--no-ocr` (skip OCR corroboration),
`--no-rotinv` (skip rotation-invariant embedding), `--no-ranking`,
`--annotate-dir DIR` (visual mirror with suspicious photos stamped — see below).

First run downloads the models and embeds all images (~1–2 min on Apple MPS);
embeddings/OCR are cached to `.cache_embeddings/` and `.cache_ocr/`, so re-runs are
near-instant.

### Annotated visual output

```bash
# during a run:
.venv/bin/python -m outlet_verify --data-dir dataset --out results.json --annotate-dir output
# or standalone from an existing results.json (no re-embedding):
.venv/bin/python -m outlet_verify.annotate results.json dataset output
```

Mirrors `dataset/` under the original file names. Outlets that contain a flagged
image go to `DIR/_flagged_<outlet_id>/` (prefixed so they stand out and sort to the
top); clean outlets keep `DIR/<outlet_id>/`. Flagged images are stamped in place
with a red `SUSPICIOUS <score>` banner and the reason; clean images are copied
verbatim. `output/` is gitignored.

## Output schema (`results.json`)

A JSON array with one record per outlet — every outlet, clean ones with an empty
`flagged_images`:

```json
{
  "outlet_id": "outlet_045f93b1",
  "total_images": 17,
  "flagged_images": [
    {"file_name": "image_0003.jpg", "suspicion_score": 0.9863,
     "reason": "a product close-up, not the outlet (median cosine 0.34 ...)"}
  ],
  "ranking": ["image_0003.jpg", "image_0002_1.jpg", "..."]
}
```

`suspicion_score` is a dataset-relative percentile in [0, 1]; `ranking` orders all
images most→least suspicious (use it to triage beyond the hard flags).

## Evaluate (threshold calibration)

No ground-truth labels exist, so metrics come from injecting known cross-outlet
fakes and measuring how well they're caught/ranked:

```bash
.venv/bin/python -m eval.evaluate --data-dir dataset
```

## Test

```bash
.venv/bin/python -m pytest -q
```

## Layout

```text
outlet_verify/   embeddings.py · scoring.py · reasons.py · ocr.py · pipeline.py · annotate.py · __main__.py
eval/            evaluate.py        synthetic-injection calibration
tests/           test_scoring.py · test_ocr.py
results.json     final output for the full dataset (deliverable)
```
