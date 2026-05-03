"""Surgery Dashboard — ランディングページ。

実行:
    streamlit run app/main.py

Streamlit のマルチページ機能で `app/pages/` 配下の各ファイルが自動的にサイドバーの
ナビゲーションに登録される。本ファイルはトップページとして、データ読込状態と
ページ案内を表示する。
"""

from __future__ import annotations

import streamlit as st

from src.ui.data_loader import DEFAULT_CSV, load_pipeline

st.set_page_config(page_title="Surgery Dashboard", layout="wide")

st.title("Surgery Dashboard")

# 読み込み状況だけ取得 (フィルタは各ページで適用)
df, status = load_pipeline(str(DEFAULT_CSV), run_llm=True)
st.caption(f"読込: `{DEFAULT_CSV.name}` ／ {len(df):,} 件 ／ {status}")

# データの基本サマリ
period_min = df["手術実施日"].min().strftime("%Y-%m-%d") if not df.empty else "-"
period_max = df["手術実施日"].max().strftime("%Y-%m-%d") if not df.empty else "-"
n_doctors = (
    df["執刀医"].dropna().nunique()
    if "執刀医" in df.columns
    else 0
)
n_depts = df["実施診療科"].dropna().nunique() if "実施診療科" in df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("総件数", f"{len(df):,}")
c2.metric("期間", f"{period_min}\n〜 {period_max}")
c3.metric("執刀医数", f"{n_doctors:,}")
c4.metric("診療科数", f"{n_depts:,}")

st.divider()

st.markdown(
    """
### ページ案内

サイドバー上部のナビゲーションから各ページに移動できます:

- **全体 KPI** — 件数 / 総時間 / 平均時間 / 緊急比率の概観 + 月次推移
- **術者別** — 術者ランキング / 特定術者ドリルダウン / 期間比較
- **カテゴリ別** — 4 カテゴリ（悪性腫瘍 / 人工関節 / ダヴィンチ系ロボット / 非ダヴィンチ系ロボット）の件数と推移、分類元内訳
- **月次推移** — 月別の件数・時間・カテゴリ・緊急比率の時系列分析

各ページのサイドバーには共通フィルタ（CSV パス / 執刀医モード / 申込区分 /
実施診療科 / 全身麻酔のみ / 期間比較）が表示されます。フィルタ値は **ページ切替後も維持**されます。
"""
)
