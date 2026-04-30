"""生データ CSV の匿名化処理。

`data/raw/*.csv`（CP932 想定）を読み込み、`実施術者` セル内の医師本名を
`医師_NNN` に置換して `data/raw/anonymized/anonymized_data.csv`（UTF-8 BOM）に出力する。

匿名化対象:
  - **`実施術者` 列のみ** — セル内の改行区切り各行を医師本名 → 安定 ID に置換
  - 他列（`手術実施日` `確定術式` `麻酔種別` 等）は無加工

医師本名 → ID 対応:
  - `config/master_key.csv` に CSV で永続化（**Git コミット厳禁**、`.gitignore` 済）
  - 列: `本名`, `匿名 ID`
  - 既存医師は対応表通り、新規医師は最大番号 + 1 で自動採番
  - 同じ医師は何度ファイルが入れ替わっても常に同じ ID を持つ（バッチ間安定性）

複数ファイル処理:
  - `data/raw/*.csv`（subdir は除外）を全て読み込み concat
  - 重複行除去キー: `手術実施日 + 実施手術室 + 入室時刻 + 確定術式`（要確認）
  - 出力は単一 CSV（`anonymized_data.csv`）

実行:
    python -m src.anonymize             # デフォルトパスで一括実行
    python -m src.anonymize --dry-run   # 書き込みせずに統計のみ表示
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ingest import EXPECTED_COLUMNS, read_csv_with_fallback

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUT_DIR = ROOT / "data" / "raw" / "anonymized"
DEFAULT_MASTER_KEY = ROOT / "config" / "master_key.csv"
DEFAULT_OUT_FILENAME = "anonymized_data.csv"

OPERATOR_COLUMN = "実施術者"
ID_PREFIX = "医師_"
ID_PATTERN = re.compile(r"^医師_(\d+)$")
DUP_KEY_COLUMNS = ("手術実施日", "実施手術室", "入室時刻", "確定術式")
LINE_SPLIT = re.compile(r"\r?\n")


@dataclass
class AnonymizationResult:
    rows_in: int
    rows_out: int
    rows_dedup: int
    files: list[Path]
    new_operators: list[str]      # 新規採番した本名
    total_operators: int          # master_key 全体の医師数
    output_path: Path | None      # dry-run では None


# ---- master_key 操作 ---------------------------------------------------


def load_master_key(path: Path) -> dict[str, str]:
    """対応表 CSV を読み込み {本名: 匿名 ID} の dict を返す。存在しなければ空 dict。"""
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if not {"本名", "匿名 ID"}.issubset(df.columns):
        raise ValueError(
            f"{path} は `本名,匿名 ID` の 2 列形式である必要があります（columns={list(df.columns)}）"
        )
    return dict(zip(df["本名"], df["匿名 ID"], strict=True))


def save_master_key(path: Path, mapping: dict[str, str]) -> None:
    """対応表 dict を CSV に保存（UTF-8 BOM、Excel 互換）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        sorted(mapping.items(), key=lambda kv: _id_to_serial(kv[1])),
        columns=["本名", "匿名 ID"],
    )
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _id_to_serial(anonymized_id: str) -> int:
    m = ID_PATTERN.match(anonymized_id)
    return int(m.group(1)) if m else 10**9  # 形式外は末尾


def _next_serial(mapping: dict[str, str]) -> int:
    """対応表内の最大採番番号 + 1（形式外の ID は無視）。"""
    serials = [
        int(m.group(1))
        for v in mapping.values()
        if (m := ID_PATTERN.match(v))
    ]
    return max(serials) + 1 if serials else 1


# ---- 匿名化本体 --------------------------------------------------------


def _split_lines(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [line for line in LINE_SPLIT.split(text) if line.strip()]


def anonymize_operator_cell(
    cell: object,
    mapping: dict[str, str],
    new_names: list[str],
) -> str | None:
    """`実施術者` 1 セルを匿名化する。新規医師は mapping を更新し、`new_names` に追記。

    出力は `\\n` 区切り（CR は使わない）。空セルは None を返す。
    """
    lines = _split_lines(cell)
    if not lines:
        return None

    out_lines: list[str] = []
    for name in lines:
        if name not in mapping:
            serial = _next_serial(mapping)
            mapping[name] = f"{ID_PREFIX}{serial:03d}"
            new_names.append(name)
        out_lines.append(mapping[name])

    return "\n".join(out_lines)


def anonymize_dataframe(
    df: pd.DataFrame,
    mapping: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """DataFrame の `実施術者` 列のみ匿名化。他列は変更しない。

    Returns:
        (anonymized_df, new_names): 新規採番した本名のリスト
    """
    if OPERATOR_COLUMN not in df.columns:
        raise ValueError(f"列 {OPERATOR_COLUMN!r} が見当たりません")

    new_names: list[str] = []
    out = df.copy()
    out[OPERATOR_COLUMN] = out[OPERATOR_COLUMN].apply(
        lambda v: anonymize_operator_cell(v, mapping, new_names)
    )
    return out, new_names


# ---- ディレクトリ走査 -------------------------------------------------


def _list_input_files(raw_dir: Path) -> list[Path]:
    """raw_dir 直下の *.csv のみ（サブディレクトリは除外）。"""
    return sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")


def _drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """重複行除去（重複キーが揃っていない場合は行をそのまま返す）。"""
    available = [c for c in DUP_KEY_COLUMNS if c in df.columns]
    if len(available) < len(DUP_KEY_COLUMNS):
        logger.warning(
            "重複除去キーが不揃い（揃っているキー: %s）→ 重複除去をスキップ", available
        )
        return df, 0
    before = len(df)
    out = df.drop_duplicates(subset=list(available), keep="first").reset_index(drop=True)
    return out, before - len(out)


def anonymize_directory(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    master_key_path: Path = DEFAULT_MASTER_KEY,
    out_filename: str = DEFAULT_OUT_FILENAME,
    dry_run: bool = False,
) -> AnonymizationResult:
    """`raw_dir` 直下の CSV をすべて読み込み、`out_dir/{out_filename}` に匿名化済みを書き出す。

    `data/raw/anonymized/` などの subdir は走査対象外。
    `dry_run=True` のときはファイルを書き込まず統計だけ計算する。
    """
    files = _list_input_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"{raw_dir} に CSV ファイルがありません")

    mapping = load_master_key(master_key_path)

    frames: list[pd.DataFrame] = []
    rows_in_total = 0
    for f in files:
        result = read_csv_with_fallback(f)
        logger.info("読込: %s (encoding=%s, rows=%d)", f.name, result.encoding, len(result.df))
        rows_in_total += len(result.df)
        missing = [c for c in EXPECTED_COLUMNS if c not in result.df.columns]
        if missing:
            raise ValueError(f"{f}: 想定列が不足: {missing}")
        frames.append(result.df)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    deduped, n_dup = _drop_duplicates(merged)
    if n_dup:
        logger.info("重複除去: %d 行", n_dup)

    anonymized, new_names = anonymize_dataframe(deduped, mapping)

    output_path: Path | None = None
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / out_filename
        # 出力は UTF-8 BOM（既存の `anonymized_data.csv` サンプルと同じ）
        # セル内改行は \n に統一して書き込む（pandas の to_csv はそのまま保持）
        anonymized.to_csv(output_path, index=False, encoding="utf-8-sig")
        save_master_key(master_key_path, mapping)

    return AnonymizationResult(
        rows_in=rows_in_total,
        rows_out=len(anonymized),
        rows_dedup=n_dup,
        files=files,
        new_operators=new_names,
        total_operators=len(mapping),
        output_path=output_path,
    )


# ---- CLI ----------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="raw CSV を匿名化して anonymized/ に出力")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--master-key", type=Path, default=DEFAULT_MASTER_KEY)
    parser.add_argument("--out-filename", default=DEFAULT_OUT_FILENAME)
    parser.add_argument("--dry-run", action="store_true", help="書き込みせず統計だけ表示")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    try:
        result = anonymize_directory(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            master_key_path=args.master_key,
            out_filename=args.out_filename,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1

    print()
    print("=== 匿名化結果 ===")
    print(f"入力ファイル: {len(result.files)} 件")
    for f in result.files:
        print(f"  - {f.name}")
    print(f"入力行: {result.rows_in:,} 件")
    print(f"重複除去: {result.rows_dedup:,} 件")
    print(f"出力行: {result.rows_out:,} 件")
    print(f"医師総数（master_key）: {result.total_operators:,} 名")
    print(f"今回の新規採番: {len(result.new_operators):,} 名")
    if result.output_path:
        print(f"出力: {result.output_path}")
    else:
        print("（dry-run のため未出力）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
