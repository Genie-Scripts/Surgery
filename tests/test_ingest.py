"""src/ingest.py の回帰テスト。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ingest import (
    EXPECTED_COLUMNS,
    IngestResult,
    load,
    normalize,
    parse_operators,
    read_csv_with_fallback,
    split_lines,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_surgery.csv"


# --- split_lines ---------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (None, []),
        ("", []),
        ("   ", []),
        ("a", ["a"]),
        ("a\nb", ["a", "b"]),
        ("a\r\nb", ["a", "b"]),
        ("a\r\nb\nc", ["a", "b", "c"]),
        ("a\n\nb", ["a", "b"]),     # 連続改行は空行として除去
        ("\na\n", ["a"]),           # 前後の空行も除去
    ],
)
def test_split_lines(value, expected):
    assert split_lines(value) == expected


def test_split_lines_handles_nan():
    assert split_lines(float("nan")) == []


# --- parse_operators -----------------------------------------------------

def test_parse_operators_single():
    assert parse_operators("医師_001") == ("医師_001", [])


def test_parse_operators_multi_with_crlf():
    lead, assistants = parse_operators("医師_001\r\n医師_002\r\n医師_003")
    assert lead == "医師_001"
    assert assistants == ["医師_002", "医師_003"]


def test_parse_operators_empty():
    assert parse_operators(None) == (None, [])
    assert parse_operators("") == (None, [])


# --- read_csv_with_fallback ---------------------------------------------

def test_read_csv_with_fallback_utf8_sig():
    result = read_csv_with_fallback(FIXTURE)
    assert isinstance(result, IngestResult)
    assert result.encoding == "utf-8-sig"
    assert result.df.shape == (8, 14)
    assert tuple(result.df.columns) == EXPECTED_COLUMNS


def test_read_csv_with_fallback_cp932(tmp_path):
    src = pd.read_csv(FIXTURE, encoding="utf-8-sig", dtype=str, keep_default_na=True)
    out = tmp_path / "cp932.csv"
    src.to_csv(out, index=False, encoding="cp932")

    result = read_csv_with_fallback(out)
    assert result.encoding in ("cp932", "shift_jis")  # cp932 と shift_jis は近縁
    assert result.df.shape == src.shape


# --- normalize -----------------------------------------------------------

def test_normalize_dtypes_and_extra_columns():
    raw = read_csv_with_fallback(FIXTURE).df
    df = normalize(raw)

    assert df["手術実施日"].dtype.kind == "M"  # datetime64
    assert df["予定手術時間"].dtype == "Int64"
    assert "執刀医" in df.columns
    assert "助手リスト" in df.columns


def test_normalize_strips_carriage_returns():
    raw = read_csv_with_fallback(FIXTURE).df
    df = normalize(raw)
    for col in ("実施術者", "麻酔種別", "確定術式", "術後病名"):
        assert not df[col].dropna().str.contains("\r", regex=False).any(), col


def test_normalize_operator_split_first_row():
    raw = read_csv_with_fallback(FIXTURE).df
    df = normalize(raw)

    # row 0: 医師_001 / _002 / _003 の 3 名
    assert df.loc[0, "執刀医"] == "医師_001"
    assert df.loc[0, "助手リスト"] == ["医師_002", "医師_003"]

    # row 2: 単一術者
    assert df.loc[2, "執刀医"] == "医師_020"
    assert df.loc[2, "助手リスト"] == []


def test_normalize_missing_column_raises():
    raw = read_csv_with_fallback(FIXTURE).df.drop(columns=["申込区分"])
    with pytest.raises(ValueError, match="想定列が不足"):
        normalize(raw)


# --- load (一発エントリ) -----------------------------------------------

def test_load_round_trip():
    df = load(FIXTURE)
    assert len(df) == 8
    assert df["手術実施日"].min() == pd.Timestamp("2026-04-01")
    assert df["手術実施日"].max() == pd.Timestamp("2026-04-08")
    assert df["申込区分"].value_counts().to_dict() == {"通常": 6, "緊急": 1, "臨時": 1}
