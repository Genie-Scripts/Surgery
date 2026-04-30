"""LLM 第 2 段カテゴリ判定（spec.md §7.2）。

regex 第 1 段で 0 件ヒット または 2 件以上ヒット（カテゴリ重複）の症例に対し、
Swallow-8B などのローカル LLM で再判定する。

入力: `src.classify.classify()` 出力（`LLM判定要 == True` の行が対象）
出力: 入力 DataFrame のカテゴリ列を上書き（regex 結果は破棄し LLM 結果に置換）。
       `分類元` 列を追加し "regex" / "llm" / "llm_fallback" の値を入れる。

キャッシュ: `data/llm_cache/categories.json`
  - キー: sha256({確定術式, ルールバージョン, モデル名})
  - 値: {"categories": [<id>...], "model": str, "rule_version": str}
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
from pathlib import Path

import pandas as pd

from src.classify import CategoryRule
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

# プロンプト or rules を変えたらここを上げる（既存キャッシュを無効化）
RULE_VERSION: str = "v1"

CACHE_PATH_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "llm_cache" / "categories.json"

# ハードガード: regex / LLM どちらの判定でも、確定術式の文字列要件を満たさなければ強制 False。
#   required: 1 つも含まれなければ強制 False（保険請求文字列で一意に判定できる症例向け）
#   forbidden: 1 つでも含まれれば強制 False（"良性" 等の明確な除外条件向け）
# 悪性腫瘍/人工関節 は LLM の医療知識による拡張に価値があるが、明白な誤判定（"良性"明記症例、
# 良性 K コードの皮下腫瘍摘出術 K005）だけは決定論的に切る。
HARD_GUARD: dict[str, dict[str, list[str]]] = {
    "robot_assisted_davinci": {"required": ["手術用支援"]},
    "robot_assisted_other": {"required": ["ロボット"]},
    "malignant_tumor": {
        "forbidden": [
            "良性",                        # 明示的に良性と書かれているもの
            "皮膚、皮下腫瘍摘出術",        # K005（良性）。悪性は K006「皮膚悪性腫瘍切除術」が別途存在
        ],
    },
}


def _apply_hard_guard(ids: list[str], text: str) -> list[str]:
    """LLM/regex 判定結果から、確定術式の要件を満たさないカテゴリを除去。

    required (any-of): 1 つも含まれなければ除去
    forbidden (any-of): 1 つでも含まれれば除去
    """
    result = []
    for cid in ids:
        rule = HARD_GUARD.get(cid)
        if rule is None:
            result.append(cid)
            continue
        required = rule.get("required", [])
        forbidden = rule.get("forbidden", [])
        if required and not any(kw in text for kw in required):
            continue
        if forbidden and any(kw in text for kw in forbidden):
            continue
        result.append(cid)
    return result


def _build_prompt(text: str, rules: list[CategoryRule]) -> str:
    """カテゴリ判定プロンプトを組み立てる（spec §7.2）。

    出力規約は冒頭と末尾の両方で繰り返す（Swallow 系の format-following 緩和への保険）。
    """
    rule_lines = []
    for r in rules:
        rule_lines.append(f"- {r.category_id}: {r.display_name}")
    rules_block = "\n".join(rule_lines)

    return (
        "以下の手術について、該当するカテゴリを多ラベル形式で判定してください。\n\n"
        f"術式（複数行は同一手術カードの内訳・加算）:\n{text}\n\n"
        f"カテゴリ候補（この ID 以外は絶対に書かない）:\n{rules_block}\n\n"
        "出力規約（厳守）:\n"
        "- 出力は `CATEGORIES: [<id1>, <id2>, ...]` の 1 行のみ\n"
        "- 該当なしは `CATEGORIES: []`\n"
        "- 候補リスト以外の ID を絶対に書かない\n"
        "- 改行・コメント・推測を一切付けない\n"
        "- 角括弧と ID 以外の文字（コロンや日本語など）を ID に混入させない\n"
        "\n"
        f"再掲: 出力は `CATEGORIES: [...]` の 1 行のみ。ID は次のいずれかのみ: "
        f"{', '.join(r.category_id for r in rules)}\n"
    )


def _parse_response(text: str, valid_ids: set[str]) -> list[str] | None:
    """`CATEGORIES: [id1, id2]` 形式から ID リストを抽出。

    複数出現する場合は最後を採用（思考型モデルが CoT 中で自言及するケース対策）。
    valid_ids にない ID は除去するが、空リストとの区別のため `None` は返さず `[]` を返す。
    パース不能なら `None`。
    """
    matches = list(re.finditer(r"CATEGORIES[:：]\s*\[([^\]]*)\]", text))
    if not matches:
        return None
    inner = matches[-1].group(1).strip()
    if not inner:
        return []
    ids = [token.strip().strip("'\"") for token in inner.split(",")]
    return [i for i in ids if i in valid_ids]


def _cache_load(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("LLM キャッシュ読込失敗（空で続行）: %s", e)
        return {}


def _cache_save(path: Path, cache: dict[str, list[str]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("LLM キャッシュ書込失敗: %s", e)


def classify_one(
    text: str,
    rules: list[CategoryRule],
    client: LLMClient,
    cache: dict[str, list[str]] | None = None,
) -> tuple[list[str], str]:
    """1 件の `確定術式` テキストをカテゴリ ID リストに分類する。

    Returns:
        (カテゴリ ID リスト, 判定元) 判定元は "cache" / "llm" / "llm_fallback"

    LLM 呼び出し失敗時は空リスト + "llm_fallback"（regex 結果を維持したい場合は呼び出し側で）。
    """
    if cache is None:
        cache = {}

    valid_ids = {r.category_id for r in rules}
    key = LLMClient.cache_key(
        {
            "rule_version": RULE_VERSION,
            "model": client.config.model,
            "text": text,
        }
    )
    if key in cache:
        cached = [c for c in cache[key] if c in valid_ids]
        return _apply_hard_guard(cached, text), "cache"

    prompt = _build_prompt(text, rules)
    try:
        response = client.chat(prompt)
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as e:
        logger.warning("LLM 呼び出し失敗 → 空リストでフォールバック: %s", e)
        return [], "llm_fallback"

    parsed = _parse_response(response, valid_ids)
    if parsed is None:
        logger.warning("LLM 応答のパース失敗 → 空リストでフォールバック: %r", response[:200])
        return [], "llm_fallback"

    # キャッシュには raw 判定を保存（ガードは取り出し時に毎回適用）。
    # こうしておけばガードの基準を変えてもキャッシュを再利用できる。
    cache[key] = parsed
    return _apply_hard_guard(parsed, text), "llm"


def classify_with_llm(
    df: pd.DataFrame,
    rules: list[CategoryRule],
    client: LLMClient,
    cache_path: Path = CACHE_PATH_DEFAULT,
    target_column: str = "確定術式",
    progress: bool = True,
) -> pd.DataFrame:
    """`LLM判定要 == True` の行だけ LLM で再判定し、カテゴリ列を上書きする。

    入力 `df` は `src.classify.classify()` の出力を想定（カテゴリ列・LLM判定要 列が付いている）。
    返り値は `df` のコピーに `分類元` 列を追加。値は "regex" / "cache" / "llm" / "llm_fallback"。
    LLM が利用不可（サーバ未起動・モデル未取得）の場合は regex 結果のまま `分類元 = "regex"` を付けて返す。
    """
    if "LLM判定要" not in df.columns:
        raise ValueError("classify() の出力 (LLM判定要 列付き) を渡してください")

    out = df.copy()
    out["分類元"] = "regex"

    target_idx = out.index[out["LLM判定要"]]
    n_target = len(target_idx)
    if n_target == 0:
        return out

    if not client.is_available():
        logger.warning("LLM 利用不可 → regex 第 1 段の結果のみで進行")
        return out

    cache = _cache_load(cache_path)
    cache_size_before = len(cache)
    valid_ids = [r.category_id for r in rules]

    for n_done, idx in enumerate(target_idx, start=1):
        text = out.at[idx, target_column]
        if not isinstance(text, str) or not text:
            out.at[idx, "分類元"] = "llm_fallback"
            for cid in valid_ids:
                out.at[idx, cid] = False
            continue

        ids, source = classify_one(text, rules, client, cache)
        # regex 結果（最低保証）と LLM 結果を OR 結合（ハードガードは最終一括適用で行う）。
        # こうすれば LLM が拾い損ねた regex 確実ケースを保持しつつ、
        # 文字列要件を満たさない過剰検出は最終パスで一括除去できる。
        for cid in valid_ids:
            regex_says_true = bool(out.at[idx, cid])
            llm_says_true = cid in ids
            out.at[idx, cid] = regex_says_true or llm_says_true
        out.at[idx, "分類元"] = source

        if progress and n_done % 25 == 0:
            logger.info("LLM 判定 進捗 %d / %d", n_done, n_target)

    if len(cache) > cache_size_before:
        _cache_save(cache_path, cache)
        logger.info("LLM キャッシュ %d → %d 件に更新", cache_size_before, len(cache))

    # ハードガードを全行に最終適用 (regex-only 行も含む)
    text_col = out[target_column].fillna("")
    for cid, rule in HARD_GUARD.items():
        if cid not in out.columns:
            continue
        required = rule.get("required", [])
        forbidden = rule.get("forbidden", [])
        if required:
            mask = ~text_col.apply(lambda t, kws=required: any(kw in t for kw in kws))
            out.loc[mask, cid] = False
        if forbidden:
            mask = text_col.apply(lambda t, kws=forbidden: any(kw in t for kw in kws))
            out.loc[mask, cid] = False

    # ヒット数の再計算
    out["カテゴリヒット数"] = out[valid_ids].sum(axis=1).astype("int64")
    return out
