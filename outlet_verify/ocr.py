"""OCR signage corroboration — a precision filter over the vision flags.

Vision (the gap rule) proposes flags with high recall but also flags legitimate
same-shop photos that merely look different (odd angle, dusk light, a motorbike
in the foreground). This module adds a second, non-visual signal: it reads the
signage TEXT and clears a flag when the image shares the outlet's *distinctive*
text (shop name, phone number) with the rest of the folder. A genuine fake has
different signage, so it keeps its flag.

Common promo text (every bKash agent shows the same "সুবিধা ডাবল" banner) is
down-weighted by inverse document frequency across outlets, so it cannot vouch
for a different-shop fake that happens to carry the same banner.

Everything degrades gracefully: if easyocr is unavailable or reads nothing, the
vision flags pass through unchanged.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .embeddings import _cache_key

CACHE_DIR = Path(".cache_ocr")
# Clear a flag when the shared normalized-IDF weight >= this (a unique shared
# token is worth ~1.0, so ~2.5 means "a couple of distinctive tokens in common").
# Calibrated on cross-outlet pairs: ~3% wrong-shop false-clear, while legitimate
# same-shop re-shots with readable signage score well above it. Because IDF is
# now log(N)-normalized, this threshold is stable regardless of how many outlets
# are flagged/OCR-ed (the earlier absolute bar silently no-oped when few were).
DEFAULT_OCR_CLEAR = 2.5
_MODEL = "easyocr-bn-en"  # cache-key namespace / version tag
_TOKEN_RE = re.compile(r"[^0-9a-zঀ-৿]+")  # keep digits, ascii, Bengali

_READER = []  # lazy singleton box: [] unset, [None] unavailable, [reader] ready


def _reader():
    if not _READER:
        try:
            import easyocr
            _READER.append(easyocr.Reader(["bn", "en"], gpu=False, verbose=False))
        except Exception:
            _READER.append(None)  # unavailable -> graceful no-op
    return _READER[0]


def available() -> bool:
    return _reader() is not None


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.split(text.lower()) if len(t) >= 2}


def read_tokens(path: Path, cache_dir: Path = CACHE_DIR) -> set[str]:
    """Normalized signage tokens for one image, cached to disk. Empty set if OCR
    is unavailable or reads nothing."""
    path = Path(path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cf = cache_dir / f"{_cache_key(path, _MODEL)}.json"
    if cf.exists():
        return set(json.loads(cf.read_text()))

    reader = _reader()
    if reader is None:
        return set()
    tokens: set[str] = set()
    try:
        for line in reader.readtext(str(path), detail=0):
            tokens |= _tokenize(line)
    except Exception:
        tokens = set()
    cf.write_text(json.dumps(sorted(tokens)))
    return tokens


def compute_idf(outlet_tokens: dict[str, set[str]]) -> dict[str, float]:
    """Normalized IDF: ``log(N/df) / log(N)`` in [0, 1]. Text shared across many
    outlets (the promo banner) -> ~0; text unique to one outlet (shop name, phone)
    -> ~1. Dividing by log(N) makes the scale independent of how many outlets were
    OCR-ed, so the clear threshold means the same thing whether 2 or 200 folders
    are flagged (a unique shared token is worth ~1.0 regardless of N)."""
    n = len(outlet_tokens)
    if n <= 1:
        return {}  # can't judge distinctiveness from a single outlet
    df: dict[str, int] = {}
    for toks in outlet_tokens.values():
        for t in toks:
            df[t] = df.get(t, 0) + 1
    logn = math.log(n)
    return {t: math.log(n / c) / logn for t, c in df.items()}


def corroboration(
    image_tokens: set[str], reference_tokens: set[str], idf: dict[str, float]
) -> float:
    """Total IDF weight of tokens the image shares with its folder's reference —
    high only when they share distinctive (rare) text."""
    return sum(idf.get(t, 0.0) for t in image_tokens & reference_tokens)


if __name__ == "__main__":  # self-check: pure logic, no model needed
    outlet_tokens = {
        "a": {"bkash", "subidha", "rsdrugs", "01711"},   # promo + distinctive
        "b": {"bkash", "subidha", "fivestar"},
        "c": {"bkash", "subidha", "grocery"},
    }
    idf = compute_idf(outlet_tokens)
    assert idf["bkash"] == 0.0 and idf["subidha"] == 0.0    # in every outlet -> 0
    assert idf["rsdrugs"] > 0.0                             # unique -> positive

    ref = {"rsdrugs", "01711"}                              # folder's distinctive text
    c_same = corroboration({"bkash", "rsdrugs"}, ref, idf)  # shares shop name
    c_fake = corroboration({"bkash", "subidha"}, ref, idf)  # shares only promo
    assert c_same > c_fake and c_fake == 0.0                # distinctive vs promo
    print("ocr logic self-check OK; idf(rsdrugs)=%.2f" % idf["rsdrugs"])
