"""Human-readable flag reasons via CLIP zero-shot.

Division of labour: DINOv2 *decides* which images are suspicious (an identity
mismatch with the outlet's other photos); CLIP only *explains* the flag in words
a reviewer can read, by matching the flagged image against a short list of scene
descriptions. CLIP is never in the decision path, so if it can't load the
pipeline just falls back to a similarity-based reason.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CLIP_MODEL = "openai/clip-vit-base-patch32"

# (prompt given to CLIP, short reason phrase reported for a flag).
_CANDIDATES = [
    ("a photo of a store front or shop exterior", "a different storefront or location"),
    ("a photo of an indoor room or interior", "an indoor scene, unlike the outlet"),
    ("a photo of a person or a selfie", "a person or selfie, not the outlet"),
    ("a screenshot, document, or piece of paper", "a document or screenshot, not a storefront"),
    ("a close-up photo of a product or object", "a product close-up, not the outlet"),
    ("a photo of a street, road, or vehicles", "a street or vehicle scene, not the outlet"),
    ("a photo of nature, landscape, or scenery", "outdoor scenery, not the outlet"),
]

# CLIP preprocessing (from openai/clip-vit-base-patch32 preprocessor_config.json),
# hand-rolled to avoid torchvision, matching the embeddings module's approach.
_SIZE = 224
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_CLIP: dict[str, tuple] = {}  # name -> (model, device, normalized text features)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _preprocess(img: Image.Image) -> np.ndarray:
    w, h = img.size
    scale = _SIZE / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = img.size
    left, top = (w - _SIZE) // 2, (h - _SIZE) // 2
    img = img.crop((left, top, left + _SIZE, top + _SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)


def _load(name: str):
    if name not in _CLIP:
        from transformers import AutoTokenizer, CLIPModel

        device = _pick_device()
        model = CLIPModel.from_pretrained(name).to(device).eval()
        tok = AutoTokenizer.from_pretrained(name)
        inputs = tok([c[0] for c in _CANDIDATES], padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            tf = model.get_text_features(**inputs).pooler_output  # (num_prompts, 512)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        _CLIP[name] = (model, device, tf)
    return _CLIP[name]


def _label_for(path: Path, name: str) -> str:
    model, device, tf = _load(name)
    arr = _preprocess(Image.open(path).convert("RGB"))
    pixel_values = torch.from_numpy(arr[None]).to(device)
    with torch.inference_mode():
        imf = model.get_image_features(pixel_values=pixel_values).pooler_output  # (1, 512)
    imf = imf / imf.norm(dim=-1, keepdim=True)
    best = int((imf @ tf.T).squeeze(0).argmax())
    return _CANDIDATES[best][1]


def clip_reason_fn(model: str = CLIP_MODEL):
    """Return a ``reason_fn`` for the pipeline, or None if CLIP can't load.

    The returned fn signature matches ``pipeline.analyze``'s hook:
    ``fn(folder, names, flagged_idx, scores) -> {idx: reason_str}``.
    """
    try:
        _load(model)
    except Exception as e:  # missing weights, offline, incompatible env, ...
        warnings.warn(f"CLIP unavailable ({e}); using similarity-only reasons")
        return None

    def fn(folder, names, flagged_idx, scores) -> dict[int, str]:
        out: dict[int, str] = {}
        for i in flagged_idx:
            i = int(i)
            try:
                label = _label_for(Path(folder) / names[i], model)
                out[i] = f"{label} (median cosine {scores[i]:.2f} to the outlet's other images)"
            except Exception:
                pass  # leave this one to the caller's fallback reason
        return out

    return fn


if __name__ == "__main__":  # self-check: python -m outlet_verify.reasons
    # Graceful degradation: a bad model id yields None, not a crash.
    assert clip_reason_fn("nonexistent/clip-model-xyz") is None
    print("graceful degradation OK (bad model -> None)")

    fn = clip_reason_fn()
    assert fn is not None
    folder = Path("dataset/outlet_003a29a9")
    names = sorted(p.name for p in folder.glob("*.jpg"))
    idx = names.index("image_0001.jpg")  # the known outlier from earlier milestones
    reasons = fn(folder, names, [idx], np.full(len(names), 0.17))
    print("reason:", reasons[idx])
