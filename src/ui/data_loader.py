"""データロード共通コンポーネント。

`load_pipeline` を `@st.cache_data` で 1 度だけ実行し、複数ページ間で同一の DataFrame を共有する。
キャッシュは関数オブジェクト単位なので、各ページから同じシンボルで import すればキャッシュが効く。
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.aggregate import is_general_anesthesia
from src.classify import classify, load_rules
from src.classify_llm import classify_with_llm
from src.ingest import list_csv_files, load_auto
from src.llm_client import LLMClient

ROOT = Path(__file__).resolve().parent.parent.parent
LLM_CONFIG = ROOT / "config" / "llm_config.yaml"

# 既定パス: 環境変数 SURGERY_CSV が指定されていれば優先（ローカル実名版で使用）。
# 値はファイル/ディレクトリどちらでも可。ディレクトリなら直下の *.csv を全てマージする。
# 未指定時は 匿名化フロー出力 → ルート直下サンプル の順にフォールバック。
_PIPELINE_OUTPUT = ROOT / "data" / "raw" / "anonymized" / "anonymized_data.csv"
_ROOT_SAMPLE = ROOT / "anonymized_data.csv"
_ENV_CSV = os.getenv("SURGERY_CSV")
DEFAULT_CSV = (
    Path(_ENV_CSV).expanduser()
    if _ENV_CSV
    else (_PIPELINE_OUTPUT if _PIPELINE_OUTPUT.exists() else _ROOT_SAMPLE)
)

# カテゴリ ID → 表示用日本語ラベル（UI 全体で統一）
CATEGORY_LABELS: dict[str, str] = {
    "malignant_tumor": "悪性腫瘍",
    "artificial_joint": "人工関節",
    "robot_assisted_davinci": "ロボット支援(ダヴィンチ系)",
    "robot_assisted_other": "ロボット支援(非ダヴィンチ系)",
}


def _csv_signature(path: str) -> tuple[tuple[str, float, int], ...]:
    """CSV/ディレクトリ配下 CSV 群の (name, mtime, size) を cache key 用に返す。

    Streamlit の `@st.cache_data` は引数値で判定するため、ファイルを差し替えても
    `path` 文字列が同じだと古い結果を返してしまう。シグネチャを別引数として
    渡すことで、ファイル更新時にキャッシュを自動失効させる。
    ディレクトリの場合は直下 *.csv 全てのシグネチャを合算するので、
    1 ファイル増減・置換でも再読込される。
    """
    p = Path(path)
    if not p.exists():
        return ()
    if p.is_dir():
        files = list_csv_files(p)
        return tuple((f.name, f.stat().st_mtime, f.stat().st_size) for f in files)
    st_ = p.stat()
    return ((p.name, st_.st_mtime, st_.st_size),)


@st.cache_data(show_spinner="CSV 読込・カテゴリ分類中...")
def _load_pipeline_cached(
    path: str, run_llm: bool, _signature: tuple[tuple[str, float, int], ...]
) -> tuple[pd.DataFrame, str]:
    df = load_auto(path)
    df = classify(df)

    if run_llm:
        client = LLMClient(LLM_CONFIG)
        if client.is_available():
            df = classify_with_llm(df, load_rules(), client)
            status = "LLM 第 2 段適用済 (キャッシュ＋ハードガード経由)"
        else:
            df["分類元"] = "regex"
            status = "oMLX 未起動/利用不可 → regex 第 1 段のみ"
    else:
        df["分類元"] = "regex"
        status = "LLM スキップ → regex 第 1 段のみ"

    df["全身麻酔"] = df["麻酔種別"].apply(is_general_anesthesia)
    return df, status


def load_pipeline(path: str, run_llm: bool) -> tuple[pd.DataFrame, str]:
    """CSV を読み込み、regex 第 1 段 → (任意) LLM 第 2 段の順で分類して返す。

    返り値: (df, status_message)。status は UI 表示用の状態説明。
    oMLX 未起動/利用不可の場合は run_llm=True でも regex のみに自動降格する。
    入力 CSV の mtime/size をキャッシュキーに含むため、ファイルを差し替えると
    自動で再読込される。
    """
    return _load_pipeline_cached(path, run_llm, _csv_signature(path))
