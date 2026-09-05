"""Unit tests for the OCR corroboration logic (pure functions, no model)."""

from outlet_verify.ocr import compute_idf, corroboration


def _idf():
    # 'bkash'/'subidha' = promo shared by every outlet; others are distinctive.
    return compute_idf({
        "a": {"bkash", "subidha", "rsdrugs", "01711"},
        "b": {"bkash", "subidha", "fivestar", "01443"},
        "c": {"bkash", "subidha", "grocery"},
        "d": {"bkash", "subidha", "pharmacy"},
    })


def test_idf_downweights_common_text():
    idf = _idf()
    assert idf["bkash"] == 0.0 and idf["subidha"] == 0.0   # in all outlets
    assert idf["rsdrugs"] > 0.0                              # unique to one


def test_same_shop_cleared_fake_kept():
    idf = _idf()
    reference = {"rsdrugs", "01711", "bkash", "subidha"}    # outlet a's text
    same_shop = corroboration({"bkash", "rsdrugs"}, reference, idf)   # shares name
    promo_only = corroboration({"bkash", "subidha"}, reference, idf)  # shares promo
    fake_shop = corroboration({"bkash", "fivestar"}, reference, idf)  # other shop
    assert same_shop > promo_only
    assert promo_only == 0.0        # promo alone never corroborates
    assert fake_shop == 0.0         # a different shop's distinctive text doesn't match


def test_empty_inputs():
    assert compute_idf({}) == {}
    assert corroboration(set(), {"x"}, {"x": 1.0}) == 0.0
