"""src/anonymize.py の回帰テスト。

実データを使わず、tmp_path に小さな CSV を生成して動作を検証する。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.anonymize import (
    DEFAULT_OUT_FILENAME,
    OPERATOR_COLUMN,
    _next_serial,
    anonymize_dataframe,
    anonymize_directory,
    anonymize_operator_cell,
    load_master_key,
    save_master_key,
)
from src.ingest import EXPECTED_COLUMNS

# --- master_key 操作 ---------------------------------------------------


def test_load_master_key_missing_returns_empty(tmp_path: Path):
    assert load_master_key(tmp_path / "nonexistent.csv") == {}


def test_save_and_load_round_trip(tmp_path: Path):
    path = tmp_path / "key.csv"
    mapping = {"山田太郎": "医師_001", "佐藤花子": "医師_002"}
    save_master_key(path, mapping)
    assert load_master_key(path) == mapping


def test_load_master_key_invalid_columns_raises(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["x"], "id": ["y"]}).to_csv(path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="本名,匿名 ID"):
        load_master_key(path)


def test_next_serial_empty():
    assert _next_serial({}) == 1


def test_next_serial_skips_max():
    assert _next_serial({"a": "医師_001", "b": "医師_005", "c": "医師_003"}) == 6


def test_next_serial_ignores_non_pattern():
    assert _next_serial({"a": "医師_002", "b": "MD-001"}) == 3


# --- anonymize_operator_cell ------------------------------------------


def test_anonymize_cell_single_new_name():
    mapping: dict[str, str] = {}
    new: list[str] = []
    out = anonymize_operator_cell("山田太郎", mapping, new)
    assert out == "医師_001"
    assert mapping == {"山田太郎": "医師_001"}
    assert new == ["山田太郎"]


def test_anonymize_cell_multi_with_crlf():
    mapping = {"佐藤花子": "医師_005"}
    new: list[str] = []
    out = anonymize_operator_cell("山田太郎\r\n佐藤花子\r\n田中一郎", mapping, new)
    assert out == "医師_006\n医師_005\n医師_007"  # 既存佐藤=005, 山田=006, 田中=007
    assert mapping["山田太郎"] == "医師_006"
    assert mapping["田中一郎"] == "医師_007"
    assert set(new) == {"山田太郎", "田中一郎"}  # 佐藤は既存なので新規ではない


def test_anonymize_cell_empty_returns_none():
    assert anonymize_operator_cell(None, {}, []) is None
    assert anonymize_operator_cell("", {}, []) is None
    assert anonymize_operator_cell("   ", {}, []) is None


def test_anonymize_cell_stable_across_calls():
    mapping: dict[str, str] = {}
    new: list[str] = []
    anonymize_operator_cell("山田太郎", mapping, new)
    anonymize_operator_cell("山田太郎", mapping, new)
    assert mapping == {"山田太郎": "医師_001"}
    assert new == ["山田太郎"]  # 2 回目は new に追加されない


# --- anonymize_dataframe ----------------------------------------------


def _make_df(operators: list[str | None]) -> pd.DataFrame:
    """テスト用: 想定列を満たす最小 DataFrame を作る。"""
    n = len(operators)
    base: dict[str, list] = {col: [""] * n for col in EXPECTED_COLUMNS}
    base[OPERATOR_COLUMN] = operators
    return pd.DataFrame(base)


def test_anonymize_dataframe_only_touches_operator_column():
    df = _make_df(["山田太郎", "佐藤花子\n山田太郎"])
    df.loc[0, "確定術式"] = "腹腔鏡下胆嚢摘出術"
    df.loc[1, "確定術式"] = "腹腔鏡下虫垂切除術"

    out, new = anonymize_dataframe(df, {})

    # 実施術者は匿名化
    assert out.loc[0, OPERATOR_COLUMN] == "医師_001"
    assert out.loc[1, OPERATOR_COLUMN] == "医師_002\n医師_001"

    # 他列は変更なし
    assert out.loc[0, "確定術式"] == "腹腔鏡下胆嚢摘出術"
    assert out.loc[1, "確定術式"] == "腹腔鏡下虫垂切除術"

    assert set(new) == {"山田太郎", "佐藤花子"}


def test_anonymize_dataframe_missing_operator_column_raises():
    df = pd.DataFrame({"foo": [1, 2]})
    with pytest.raises(ValueError, match=OPERATOR_COLUMN):
        anonymize_dataframe(df, {})


# --- anonymize_directory (CP932 入出力, 複数ファイル) -----------------


def _write_raw_csv(path: Path, operators: list[str | None]) -> None:
    """raw 形式の CP932 CSV を書く。"""
    df = _make_df(operators)
    # 適当に重複防止用のキーも仕込む
    df["手術実施日"] = ["2026/04/01", "2026/04/02"][: len(df)]
    df["実施手術室"] = [f"OP-{i:02d}" for i in range(len(df))]
    df["入室時刻"] = [f"{8+i:02d}:00" for i in range(len(df))]
    df["確定術式"] = [f"術式_{i}" for i in range(len(df))]
    df.to_csv(path, index=False, encoding="cp932")


def test_anonymize_directory_end_to_end(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    out_dir = raw_dir / "anonymized"
    raw_dir.mkdir()
    key_path = tmp_path / "key.csv"

    _write_raw_csv(raw_dir / "batch1.csv", ["山田太郎", "佐藤花子\n山田太郎"])

    result = anonymize_directory(
        raw_dir=raw_dir,
        out_dir=out_dir,
        master_key_path=key_path,
    )

    assert result.rows_in == 2
    assert result.rows_out == 2
    assert result.total_operators == 2
    assert set(result.new_operators) == {"山田太郎", "佐藤花子"}
    assert result.output_path == out_dir / DEFAULT_OUT_FILENAME

    # 出力ファイルが存在し、UTF-8 BOM で読める
    assert result.output_path.exists()
    out_df = pd.read_csv(result.output_path, encoding="utf-8-sig", dtype=str)
    assert out_df.loc[0, OPERATOR_COLUMN] == "医師_001"
    assert out_df.loc[1, OPERATOR_COLUMN] == "医師_002\n医師_001"

    # master_key も書き出されている
    assert key_path.exists()
    assert load_master_key(key_path) == {"山田太郎": "医師_001", "佐藤花子": "医師_002"}


def test_anonymize_directory_stable_across_runs(tmp_path: Path):
    """2 バッチに分けても同じ医師は同じ ID を維持する。"""
    raw_dir = tmp_path / "raw"
    out_dir = raw_dir / "anonymized"
    raw_dir.mkdir()
    key_path = tmp_path / "key.csv"

    # バッチ 1: 山田 → 001, 佐藤 → 002
    _write_raw_csv(raw_dir / "batch1.csv", ["山田太郎", "佐藤花子"])
    anonymize_directory(raw_dir=raw_dir, out_dir=out_dir, master_key_path=key_path)
    mapping_after_b1 = load_master_key(key_path)

    # バッチ 1 を退避してバッチ 2 だけ走らせる
    (raw_dir / "batch1.csv").unlink()
    _write_raw_csv(raw_dir / "batch2.csv", ["田中一郎", "山田太郎"])
    result = anonymize_directory(raw_dir=raw_dir, out_dir=out_dir, master_key_path=key_path)

    final_mapping = load_master_key(key_path)
    # 既存医師の ID は不変、新規だけ採番
    assert final_mapping["山田太郎"] == mapping_after_b1["山田太郎"]
    assert final_mapping["佐藤花子"] == mapping_after_b1["佐藤花子"]
    assert final_mapping["田中一郎"] == "医師_003"
    assert result.new_operators == ["田中一郎"]


def test_anonymize_directory_dry_run_no_writes(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    out_dir = raw_dir / "anonymized"
    raw_dir.mkdir()
    key_path = tmp_path / "key.csv"

    _write_raw_csv(raw_dir / "batch1.csv", ["山田太郎"])

    result = anonymize_directory(
        raw_dir=raw_dir,
        out_dir=out_dir,
        master_key_path=key_path,
        dry_run=True,
    )

    assert result.output_path is None
    assert not (out_dir / DEFAULT_OUT_FILENAME).exists()
    assert not key_path.exists()  # dry-run では master_key も保存されない
    # ただし result.new_operators は計算される
    assert result.new_operators == ["山田太郎"]


def test_anonymize_directory_skips_anonymized_subdir(tmp_path: Path):
    """data/raw/anonymized/ サブディレクトリは入力対象外。"""
    raw_dir = tmp_path / "raw"
    out_dir = raw_dir / "anonymized"
    raw_dir.mkdir()
    out_dir.mkdir()
    key_path = tmp_path / "key.csv"

    _write_raw_csv(raw_dir / "real.csv", ["山田太郎"])
    # anonymized/ 内に古い出力があっても入力扱いされてはいけない
    _write_raw_csv(out_dir / "stale.csv", ["佐藤花子"])

    result = anonymize_directory(
        raw_dir=raw_dir,
        out_dir=out_dir,
        master_key_path=key_path,
    )
    assert result.rows_in == 1  # raw/real.csv のみ、stale.csv は無視
    assert "山田太郎" in result.new_operators
    assert "佐藤花子" not in result.new_operators


def test_anonymize_directory_no_files_raises(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="CSV ファイルがありません"):
        anonymize_directory(
            raw_dir=raw_dir,
            out_dir=raw_dir / "anonymized",
            master_key_path=tmp_path / "key.csv",
        )
