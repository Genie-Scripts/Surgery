"""カテゴリ抽出: regex 第 1 段。

`config/categories.yaml` を読み込み、`確定術式` 列にカテゴリタグを付与する。

ヒット規則:
  - patterns のいずれかに部分一致（regex / 大文字小文字を区別しない）するとカテゴリ候補
  - exclude_patterns のいずれかに当たれば除外
  - 結果は各カテゴリごとの bool 列として展開し、加えて以下のメタ列を付与:
      - `カテゴリヒット数`: int  (0 件 / 2 件以上は LLM 第 2 段の対象)
      - `LLM判定要`:        bool

第 2 段（LLM）は別モジュール（未実装）で `LLM判定要 == True` の行に対して実行する。
詳細は spec.md §7 を参照。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

CONFIG_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "categories.yaml"
CLASSIFY_TARGET_COLUMN = "確定術式"


@dataclass(frozen=True)
class CategoryRule:
    category_id: str
    display_name: str
    patterns: tuple[re.Pattern[str], ...]
    exclude_patterns: tuple[re.Pattern[str], ...]

    def matches(self, text: str) -> bool:
        if not text:
            return False
        if any(p.search(text) for p in self.exclude_patterns):
            return False
        return any(p.search(text) for p in self.patterns)


def load_rules(path: Path | str = CONFIG_DEFAULT_PATH) -> list[CategoryRule]:
    """YAML を読み込み、regex コンパイル済みのカテゴリルールを返す。"""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    rules: list[CategoryRule] = []
    for category_id, body in raw.items():
        patterns = tuple(
            re.compile(p, re.IGNORECASE) for p in body.get("patterns", []) or []
        )
        excludes = tuple(
            re.compile(p, re.IGNORECASE) for p in body.get("exclude_patterns", []) or []
        )
        rules.append(
            CategoryRule(
                category_id=category_id,
                display_name=body.get("display_name", category_id),
                patterns=patterns,
                exclude_patterns=excludes,
            )
        )
    return rules


def classify(
    df: pd.DataFrame,
    rules: list[CategoryRule] | None = None,
    target_column: str = CLASSIFY_TARGET_COLUMN,
) -> pd.DataFrame:
    """DataFrame にカテゴリ列を追加して返す。

    入力の `df` は変更しない。各カテゴリ ID と同名の bool 列を追加し、
    末尾に `カテゴリヒット数` (int) と `LLM判定要` (bool) を付与する。
    """
    if rules is None:
        rules = load_rules()
    if target_column not in df.columns:
        raise ValueError(f"対象列 {target_column!r} が存在しません")

    out = df.copy()
    text = out[target_column].fillna("")

    hit_count = pd.Series(0, index=out.index, dtype="int64")
    for rule in rules:
        col = text.apply(rule.matches)
        out[rule.category_id] = col
        hit_count = hit_count + col.astype("int64")

    out["カテゴリヒット数"] = hit_count
    out["LLM判定要"] = (hit_count == 0) | (hit_count >= 2)
    return out


def summarize(df_classified: pd.DataFrame, rules: list[CategoryRule]) -> pd.DataFrame:
    """カテゴリ別ヒット件数のサマリを返す（regex 第 1 段の精度ざっくり確認用）。"""
    rows = []
    for rule in rules:
        rows.append(
            {
                "category_id": rule.category_id,
                "display_name": rule.display_name,
                "件数": int(df_classified[rule.category_id].sum()),
            }
        )
    rows.append(
        {
            "category_id": "_LLM判定要",
            "display_name": "LLM 第 2 段に回す症例",
            "件数": int(df_classified["LLM判定要"].sum()),
        }
    )
    return pd.DataFrame(rows)
