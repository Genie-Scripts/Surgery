"""CSV 読込・正規化ユーティリティ。

入力: 手術データ CSV（単一ファイル or ディレクトリ）
出力: 正規化済み DataFrame

実スキーマ（spec.md §2.1）:
  手術実施日, 実施診療科, 実施手術室, 麻酔科関与, 入外区分, 申込区分,
  実施術者, 麻酔種別, 入室時刻, 退室時刻, 予定手術時間, 予定手術時間(OR),
  確定術式, 術後病名

正規化内容:
  - 文字コードフォールバック (utf-8-sig → cp932 → shift_jis → utf-8)
  - 手術実施日 → datetime
  - 実施術者 → 先頭行=執刀医、2 行目以降=助手 のリスト
  - 確定術式・術後病名・麻酔種別 → セル内 \\r\\n を \\n に正規化
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 試行順: BOM 付き UTF-8 → CP932 → Shift-JIS → 素の UTF-8
ENCODING_FALLBACKS = ("utf-8-sig", "cp932", "shift_jis", "utf-8")

EXPECTED_COLUMNS = (
    "手術実施日",
    "実施診療科",
    "実施手術室",
    "麻酔科関与",
    "入外区分",
    "申込区分",
    "実施術者",
    "麻酔種別",
    "入室時刻",
    "退室時刻",
    "予定手術時間",
    "予定手術時間(OR)",
    "確定術式",
    "術後病名",
)

MULTILINE_COLUMNS = ("実施術者", "麻酔種別", "確定術式", "術後病名")

# 複数ファイル concat 時の重複行除去キー。
# 同一症例が複数の供給バッチに重複混入することを想定し、再現性高いキーで吸収する。
DUP_KEY_COLUMNS = ("手術実施日", "実施手術室", "入室時刻", "確定術式")

_LINE_SPLIT = re.compile(r"\r?\n")


@dataclass
class IngestResult:
    df: pd.DataFrame
    encoding: str
    source: Path


def read_csv_with_fallback(path: Path | str) -> IngestResult:
    """エンコーディングを順に試して最初に成功したものを返す。

    全候補で失敗した場合は最後の例外をそのまま再送出。
    """
    path = Path(path)
    last_error: Exception | None = None
    for enc in ENCODING_FALLBACKS:
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=True)
            return IngestResult(df=df, encoding=enc, source=path)
        except UnicodeDecodeError as e:
            last_error = e
    assert last_error is not None
    raise last_error


def split_lines(value: object) -> list[str]:
    """セル内の `\\r\\n` / `\\n` 混在改行を分解。空文字・NaN は空リスト。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [line for line in _LINE_SPLIT.split(text) if line.strip()]


def parse_operators(value: object) -> tuple[str | None, list[str]]:
    """実施術者セルを (執刀医, 助手リスト) に分解。

    先頭行=執刀医、2 行目以降=助手。値が空なら (None, []) を返す。
    """
    lines = split_lines(value)
    if not lines:
        return None, []
    return lines[0], lines[1:]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """期待スキーマに対する基本正規化。

    - 想定列の存在チェック（不足列は警告レベルで例外）
    - 手術実施日 → datetime64[ns]
    - 予定手術時間 → Int64（分単位）
    - MULTILINE_COLUMNS の `\\r\\n` を `\\n` に統一
    - 実施術者を `執刀医` / `助手リスト` 列に展開
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"想定列が不足: {missing}")

    out = df.copy()

    out["手術実施日"] = pd.to_datetime(out["手術実施日"], format="%Y/%m/%d", errors="coerce")
    out["予定手術時間"] = pd.to_numeric(out["予定手術時間"], errors="coerce").astype("Int64")

    for col in MULTILINE_COLUMNS:
        out[col] = out[col].apply(
            lambda v: "\n".join(split_lines(v)) if pd.notna(v) else v
        )

    operators = out["実施術者"].apply(parse_operators)
    out["執刀医"] = operators.apply(lambda x: x[0])
    out["助手リスト"] = operators.apply(lambda x: x[1])

    return out


def list_csv_files(directory: Path | str) -> list[Path]:
    """`directory` 直下の `*.csv` を昇順で返す。サブディレクトリは除外。

    `data/raw/anonymized/` のような出力先サブディレクトリが混じっても拾わない設計。
    """
    directory = Path(directory)
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    )


def _drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """`DUP_KEY_COLUMNS` で重複行除去。キーが揃わなければスキップして元 df を返す。"""
    available = [c for c in DUP_KEY_COLUMNS if c in df.columns]
    if len(available) < len(DUP_KEY_COLUMNS):
        logger.warning(
            "重複除去キーが不揃い（揃っているキー: %s）→ 重複除去をスキップ", available
        )
        return df, 0
    before = len(df)
    out = df.drop_duplicates(subset=list(available), keep="first").reset_index(drop=True)
    return out, before - len(out)


def load(path: Path | str) -> pd.DataFrame:
    """単一 CSV から正規化済み DataFrame を取得する一発エントリ。"""
    result = read_csv_with_fallback(path)
    return normalize(result.df)


def load_directory(directory: Path | str) -> pd.DataFrame:
    """ディレクトリ直下の全 CSV を concat → 重複除去 → 正規化して返す。

    ローカル実名版で「ファイル名任意・複数ファイル投入」を実現するためのエントリ。
    匿名化は行わない。`load` と同じ正規化を適用するため UI 側はファイル/ディレクトリを意識せずに済む。
    """
    directory = Path(directory)
    files = list_csv_files(directory)
    if not files:
        raise FileNotFoundError(f"{directory} に CSV ファイルがありません")

    frames: list[pd.DataFrame] = []
    for f in files:
        result = read_csv_with_fallback(f)
        logger.info("読込: %s (encoding=%s, rows=%d)", f.name, result.encoding, len(result.df))
        missing = [c for c in EXPECTED_COLUMNS if c not in result.df.columns]
        if missing:
            raise ValueError(f"{f}: 想定列が不足: {missing}")
        frames.append(result.df)

    merged = pd.concat(frames, ignore_index=True)
    deduped, n_dup = _drop_duplicates(merged)
    if n_dup:
        logger.info("重複除去: %d 行", n_dup)
    return normalize(deduped)


def load_auto(path: Path | str) -> pd.DataFrame:
    """`path` がディレクトリなら `load_directory`、ファイルなら `load` にディスパッチ。"""
    p = Path(path)
    if p.is_dir():
        return load_directory(p)
    return load(p)
