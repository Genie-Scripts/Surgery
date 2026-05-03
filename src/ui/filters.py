"""サイドバーフィルタ共通コンポーネント。

各ページから `render_sidebar_filters()` を呼ぶと、サイドバーに共通フィルタを描画し
`(df_全件, df_フィルタ後, state辞書)` を返す。

widget には明示的な `key=` を指定しているため、`st.session_state` 経由で値が
ページ切替を跨いで維持される。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.ui.data_loader import DEFAULT_CSV, load_pipeline


def render_sidebar_filters() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """共通サイドバーを描画して `(df_full, df_filtered, state)` を返す。

    state のキー:
      - csv_name: 読込ファイル名 (caption 用)
      - pipeline_status: LLM 状態の人間可読文字列
      - operator_mode_label: "執刀医のみ" or "執刀医＋助手を含む"
      - operator_mode: "lead_only" or "all"
      - apply_codes, departments: list[str]
      - ga_only: bool
      - enable_compare, period_a, period_b
    """
    st.sidebar.title("フィルタ")

    csv_path = st.sidebar.text_input("CSV パス", value=str(DEFAULT_CSV), key="csv_path")
    csv_p = Path(csv_path)
    if not csv_p.exists():
        st.error(f"ファイルが見つかりません: {csv_p}")
        st.stop()

    run_llm = st.sidebar.checkbox(
        "LLM 第 2 段を適用 (Swallow-8B + ハードガード)", value=True, key="run_llm"
    )
    df, pipeline_status = load_pipeline(str(csv_p), run_llm)

    operator_mode_label = st.sidebar.radio(
        "執刀医モード",
        options=["執刀医のみ", "執刀医＋助手を含む"],
        key="operator_mode_label",
    )
    operator_mode = "lead_only" if operator_mode_label == "執刀医のみ" else "all"

    apply_codes_options = sorted(df["申込区分"].dropna().unique().tolist())
    apply_codes = st.sidebar.multiselect(
        "申込区分",
        options=apply_codes_options,
        default=apply_codes_options,
        key="apply_codes",
    )

    departments_all = sorted(df["実施診療科"].dropna().unique().tolist())
    departments = st.sidebar.multiselect(
        "実施診療科",
        options=departments_all,
        default=departments_all,
        key="departments",
    )

    ga_only = st.sidebar.checkbox("全身麻酔のみ", value=False, key="ga_only")

    period_a = None
    period_b = None
    with st.sidebar.expander("期間比較", expanded=False):
        enable_compare = st.checkbox("有効化", value=False, key="enable_compare")
        if enable_compare and not df.empty:
            date_min = df["手術実施日"].min().date()
            date_max = df["手術実施日"].max().date()
            midpoint = date_min + (date_max - date_min) / 2
            period_a = st.date_input(
                "期間 A（前期間）",
                value=(date_min, midpoint),
                min_value=date_min,
                max_value=date_max,
                key="period_a",
            )
            period_b = st.date_input(
                "期間 B（後期間）",
                value=(midpoint, date_max),
                min_value=date_min,
                max_value=date_max,
                key="period_b",
            )

    mask = df["申込区分"].isin(apply_codes) & df["実施診療科"].isin(departments)
    if ga_only:
        mask = mask & df["全身麻酔"]
    df_f = df[mask].copy()

    state = {
        "csv_name": csv_p.name,
        "pipeline_status": pipeline_status,
        "operator_mode_label": operator_mode_label,
        "operator_mode": operator_mode,
        "apply_codes": apply_codes,
        "departments": departments,
        "ga_only": ga_only,
        "enable_compare": enable_compare,
        "period_a": period_a,
        "period_b": period_b,
    }
    return df, df_f, state


def render_caption(df: pd.DataFrame, df_f: pd.DataFrame, state: dict) -> None:
    """各ページのタイトル直下に表示する共通キャプション。"""
    st.caption(
        f"読込: `{state['csv_name']}` ／ "
        f"全 {len(df):,} 件中 {len(df_f):,} 件を表示中"
        f"（執刀医モード = {state['operator_mode_label']}"
        f"{' / 全身麻酔のみ' if state['ga_only'] else ''}）"
        f"｜ {state['pipeline_status']}"
    )
