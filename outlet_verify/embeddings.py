"""DINOv2 CLS-token embeddings for outlet images, cached to disk.

Why DINOv2: the task is instance/location identity ("is this the *same* outlet"),
not semantic category, so self-supervised patch features beat CLIP/pHash here.
We use the CLS token (a global image descriptor), L2-normalized so cosine
similarity is just a dot product downstream.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "facebook/dinov2-small"
CACHE_DIR = Path(".cache_embeddings")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# DINOv2 preprocessing (from facebook/dinov2-small preprocessor_config.json),
# hand-rolled with PIL+NumPy to avoid transformers' AutoImageProcessor, whose
# "fast" path hard-requires torchvision and is fragile to pin against torch.
# (torchvision is still present in this env via easyocr; we simply don't route
# image preprocessing through it.)
_RESIZE_SHORT = 256
_CROP = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Loaded models are reused across folder calls: (model, device).
_MODELS: dict[str, tuple] = {}


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_model(name: str):
    if name not in _MODELS:
        from transformers import AutoModel

        device = _pick_device()
        model = AutoModel.from_pretrained(name).to(device).eval()
        _MODELS[name] = (model, device)
    return _MODELS[name]


def _preprocess(img: Image.Image) -> np.ndarray:
    """Resize shortest edge to 256 (bicubic), center-crop 224, rescale, normalize."""
    w, h = img.size
    scale = _RESIZE_SHORT / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = img.size
    left, top = (w - _CROP) // 2, (h - _CROP) // 2
    img = img.crop((left, top, left + _CROP, top + _CROP))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)  # HWC -> CHW


def list_images(folder: Path) -> list[Path]:
    """Image files directly in `folder`, sorted by name. Skips .DS_Store etc."""
    folder = Path(folder)
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _cache_key(path: Path, model: str) -> str:
    st = path.stat()
    # Path + mtime + size invalidates on edit without reading the file on hits.
    raw = f"{model}\0{path.resolve()}\0{st.st_mtime_ns}\0{st.st_size}"
    return hashlib.sha1(raw.encode()).hexdigest()


_ROTATIONS = (0, 90, 180, 270)  # DINOv2 isn't rotation-invariant; we embed all 4


def _embed_batch(paths: list[Path], model_name: str, rot: int = 0) -> np.ndarray:
    model, device = _get_model(model_name)
    imgs = [Image.open(p).convert("RGB") for p in paths]
    if rot:
        imgs = [im.rotate(rot, expand=True) for im in imgs]
    batch = np.stack([_preprocess(im) for im in imgs])
    pixel_values = torch.from_numpy(batch).to(device)
    with torch.inference_mode():
        cls = model(pixel_values=pixel_values).last_hidden_state[:, 0]  # CLS, (B, D)
    vecs = cls.float().cpu().numpy()
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12  # L2-normalize
    return vecs


def _cache_tag(model: str, rot: int) -> str:
    return model if rot == 0 else f"{model}@r{rot}"  # rot-0 tag == plain model (compat)


def _embed_folder_rot(
    folder: Path, model: str, rot: int, cache_dir: Path, batch_size: int
) -> tuple[list[str], np.ndarray]:
    """Embed every image in `folder` at one rotation; (N, D), L2-normalized."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = list_images(folder)
    if not paths:
        return [], np.empty((0, 0), dtype=np.float32)

    names = [p.name for p in paths]
    tag = _cache_tag(model, rot)
    vecs: list[np.ndarray | None] = [None] * len(paths)
    misses: list[int] = []
    for i, p in enumerate(paths):
        cf = cache_dir / f"{_cache_key(p, tag)}.npy"
        if cf.exists():
            vecs[i] = np.load(cf)
        else:
            misses.append(i)

    for start in range(0, len(misses), batch_size):
        idx = misses[start:start + batch_size]
        batch = _embed_batch([paths[i] for i in idx], model, rot=rot)
        for j, i in enumerate(idx):
            vecs[i] = batch[j]
            np.save(cache_dir / f"{_cache_key(paths[i], tag)}.npy", batch[j])

    return names, np.stack(vecs)  # type: ignore[arg-type]


def embed_folder(
    folder: Path,
    model: str = DEFAULT_MODEL,
    cache_dir: Path = CACHE_DIR,
    batch_size: int = 32,
) -> tuple[list[str], np.ndarray]:
    """Embed every image in `folder` (upright). (N, D), L2-normalized."""
    return _embed_folder_rot(folder, model, 0, cache_dir, batch_size)


def embed_folder_rotations(
    folder: Path,
    model: str = DEFAULT_MODEL,
    cache_dir: Path = CACHE_DIR,
    batch_size: int = 32,
) -> tuple[list[str], np.ndarray]:
    """Embed every image at 0/90/180/270. Returns (file_names, emb4) with emb4
    shaped (N, 4, D), L2-normalized. Used for rotation-invariant scoring."""
    names: list[str] = []
    mats: list[np.ndarray] = []
    for rot in _ROTATIONS:
        names, e = _embed_folder_rot(folder, model, rot, cache_dir, batch_size)
        mats.append(e)
    if not names:
        return [], np.empty((0, 0, 0), dtype=np.float32)
    return names, np.stack(mats, axis=1)  # (N, 4, D)


if __name__ == "__main__":  # self-check: python -m outlet_verify.embeddings <folder>
    import sys
    import time

    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset")
    if folder.is_dir() and not list_images(folder):
        folder = next(p for p in sorted(folder.iterdir()) if p.is_dir())
    print(f"Embedding {folder} ...")

    t0 = time.time()
    names, emb = embed_folder(folder)
    t1 = time.time()
    names2, emb2 = embed_folder(folder)  # should be all cache hits
    t2 = time.time()

    norms = np.linalg.norm(emb, axis=1)
    assert emb.shape[0] == len(names), "row/name mismatch"
    assert np.allclose(norms, 1.0, atol=1e-4), f"not L2-normalized: {norms[:3]}"
    assert np.array_equal(emb, emb2), "cache returned different vectors"
    print(f"  {emb.shape[0]} images -> embeddings {emb.shape}")
    print(f"  norms in [{norms.min():.4f}, {norms.max():.4f}] (expect ~1.0)")
    print(f"  cold: {t1 - t0:.2f}s   cached: {t2 - t1:.3f}s   OK")
