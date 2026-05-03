"""データロード共通コンポーネント。

`load_pipeline` を `@st.cache_data` で 1 度だけ実行し、複数ページ間で同一の DataFrame を共有する。
キャッシュは関数オブジェクト単位なので、各ページから同じシンボルで import すればキャッシュが効く。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.aggregate import is_general_anesthesia
from src.classify import classify, load_rules
from src.classify_llm import classify_with_llm
from src.ingest import load
from src.llm_client import LLMClient

ROOT = Path(__file__).resolve().parent.parent.parent
LLM_CONFIG = ROOT / "config" / "llm_config.yaml"

# 既定 CSV: 匿名化フロー出力 (data/raw/anonymized/anonymized_data.csv) があれば優先、
# なければルート直下のサンプルにフォールバック。
_PIPELINE_OUTPUT = ROOT / "data" / "raw" / "anonymized" / "anonymized_data.csv"
_ROOT_SAMPLE = ROOT / "anonymized_data.csv"
DEFAULT_CSV = _PIPELINE_OUTPUT if _PIPELINE_OUTPUT.exists() else _ROOT_SAMPLE

# カテゴリ ID → 表示用日本語ラベル（UI 全体で統一）
CATEGORY_LABELS: dict[str, str] = {
    "malignant_tumor": "悪性腫瘍",
    "artificial_joint": "人工関節",
    "robot_assisted_davinci": "ロボット支援(ダヴィンチ系)",
    "robot_assisted_other": "ロボット支援(非ダヴィンチ系)",
}


@st.cache_data(show_spinner="CSV 読込・カテゴリ分類中...")
def load_pipeline(path: str, run_llm: bool) -> tuple[pd.DataFrame, str]:
    """CSV を読み込み、regex 第 1 段 → (任意) LLM 第 2 段の順で分類して返す。

    返り値: (df, status_message)。status は UI 表示用の状態説明。
    Ollama 未起動の場合は run_llm=True でも regex のみに自動降格する。
    """
    df = load(path)
    df = classify(df)

    if run_llm:
        client = LLMClient(LLM_CONFIG)
        if client.is_available():
            df = classify_with_llm(df, load_rules(), client)
            status = "LLM 第 2 段適用済 (キャッシュ＋ハードガード経由)"
        else:
            df["分類元"] = "regex"
            status = "Ollama 未起動 → regex 第 1 段のみ"
    else:
        df["分類元"] = "regex"
        status = "LLM スキップ → regex 第 1 段のみ"

    df["全身麻酔"] = df["麻酔種別"].apply(is_general_anesthesia)
    return df, status
