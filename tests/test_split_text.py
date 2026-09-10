import pytest

from services.text_splitter import split_text


def test_empty_text_returns_no_chunks():
    assert split_text("") == []
    assert split_text("   ") == []


def test_short_text_stays_in_one_chunk():
    assert split_text("这是第一句。这是第二句。") == ["这是第一句。这是第二句。"]


def test_overlap_requires_smaller_value():
    with pytest.raises(ValueError):
        split_text("文本", chunk_size=5, chunk_overlap=5)


def test_long_text_splits_with_overlap():
    assert split_text("ABCDEFGHIJ", chunk_size=6, chunk_overlap=2) == [
        "ABCDEF",
        "EFGHIJ",
    ]
