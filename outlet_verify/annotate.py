"""Annotated visual output: mirror the dataset and stamp the suspicious photos.

Copies every image into ``out_dir/<outlet_id>/`` so the tree mirrors the input;
flagged images get a red ``SUSPICIOUS <score>`` banner with the reason drawn on
top, clean images are copied verbatim. Driven from the per-outlet records, so it
works both inside a pipeline run and standalone from an existing results.json.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .embeddings import list_images

_FONTS: dict[int, ImageFont.ImageFont] = {}


def _font(size: int) -> ImageFont.ImageFont:
    if size not in _FONTS:
        _FONTS[size] = ImageFont.load_default(size=size)  # Pillow >=10, no font file
    return _FONTS[size]


def _stamp(img: Image.Image, score: float, reason: str) -> Image.Image:
    """Draw a red banner ('SUSPICIOUS <score>' + wrapped reason) across the top."""
    draw = ImageDraw.Draw(img)
    w = img.width
    fs = max(16, w // 28)
    font = _font(fs)
    pad = fs // 2
    max_w = w - 2 * pad

    # Word-wrap the reason to the image width, measuring with the real font.
    lines: list[str] = []
    cur = ""
    for word in reason.split():
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)

    all_lines = [f"SUSPICIOUS  {score:.2f}"] + lines
    line_h = fs + fs // 4
    banner_h = pad * 2 + line_h * len(all_lines)
    draw.rectangle([0, 0, w, banner_h], fill=(200, 20, 20))
    y = pad
    for ln in all_lines:
        draw.text((pad, y), ln, fill="white", font=font)
        y += line_h
    return img


def annotate_dataset(records: list[dict], data_dir: Path, out_dir: Path) -> dict:
    """Write an annotated mirror of the dataset. Returns counts.

    Every image is copied under its original name. An outlet that has any flagged
    image is written to ``out_dir/_flagged_<outlet_id>/`` (prefixed so flagged
    outlets stand out and sort to the top); clean outlets keep ``<outlet_id>``.
    Flagged images are stamped in place (red ``SUSPICIOUS <score>`` banner +
    reason); clean images are copied verbatim.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    folders = images = stamped = 0
    for rec in records:
        src = data_dir / rec["outlet_id"]
        if not src.is_dir():
            continue
        flagged = {f["file_name"]: (f["suspicion_score"], f["reason"])
                   for f in rec["flagged_images"]}
        folder_name = f"_flagged_{rec['outlet_id']}" if flagged else rec["outlet_id"]
        dst = out_dir / folder_name
        dst.mkdir(parents=True, exist_ok=True)
        folders += 1
        for path in list_images(src):
            images += 1
            if path.name in flagged:
                score, reason = flagged[path.name]
                im = _stamp(Image.open(path).convert("RGB"), score, reason)
                im.save(dst / path.name, quality=90)  # stamped, same name, same folder
                stamped += 1
            else:
                shutil.copy2(path, dst / path.name)  # clean image, byte-identical
    return {"folders": folders, "images": images, "stamped": stamped}


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:  # annotate from an existing results.json
        import json
        results, data_dir, out_dir = sys.argv[1:4]
        recs = json.loads(Path(results).read_text())
        print("annotating from", results, "->", out_dir, "...")
        print(annotate_dataset(recs, data_dir, out_dir))
    else:  # self-check on synthetic images (no dataset needed)
        import filecmp
        import tempfile

        tmp = Path(tempfile.mkdtemp())
        src = tmp / "data" / "outlet_x"
        src.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (240, 320), (80, 120, 180)).save(src / f"image_{i}.jpg")
        clean = tmp / "data" / "outlet_y"  # no flags
        clean.mkdir()
        Image.new("RGB", (240, 320), (80, 120, 180)).save(clean / "image_0.jpg")
        recs = [
            {"outlet_id": "outlet_x", "total_images": 3,
             "flagged_images": [{"file_name": "image_0.jpg", "suspicion_score": 0.97,
                                 "reason": "a different storefront or location"}]},
            {"outlet_id": "outlet_y", "total_images": 1, "flagged_images": []},
        ]
        out = tmp / "out"
        counts = annotate_dataset(recs, tmp / "data", out)
        assert (out / "_flagged_outlet_x" / "image_0.jpg").exists()   # flagged folder prefixed
        assert (out / "outlet_y" / "image_0.jpg").exists()            # clean folder un-prefixed
        assert not (out / "outlet_x").exists()                        # flagged outlet not un-prefixed
        assert not filecmp.cmp(src / "image_0.jpg", out / "_flagged_outlet_x" / "image_0.jpg")  # stamped
        assert filecmp.cmp(src / "image_1.jpg", out / "_flagged_outlet_x" / "image_1.jpg")      # verbatim
        print("annotate self-check OK", counts)
