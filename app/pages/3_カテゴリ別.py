"""Page 3: カテゴリ別。

4 カテゴリ（悪性腫瘍 / 人工関節 / ロボット支援ダヴィンチ系 / 非ダヴィンチ系）の
件数・構成比・分類元内訳・月次推移、診療科別件数を表示する。
"""

from __future__ import annotations

import streamlit as st

from src.aggregate import CATEGORY_COLUMNS, category_counts, category_monthly_trend
from src.ui.data_loader import CATEGORY_LABELS
from src.ui.filters import render_caption, render_sidebar_filters

st.set_page_config(page_title="カテゴリ別 - Surgery", layout="wide")

df, df_f, state = render_sidebar_filters()

st.title("カテゴリ別")
render_caption(df, df_f, state)

# ----- カテゴリ別件数（詳細） + 分類元 ---------------------------------

st.divider()
st.subheader("カテゴリ別件数")

cat = category_counts(df_f)
col_cat, col_src = st.columns([2, 1])

if cat.empty:
    col_cat.info("カテゴリ列がありません")
else:
    cat_disp = cat.assign(構成比=lambda d: (d["件数"] / max(len(df_f), 1) * 100).round(1))
    cat_disp.columns = ["カテゴリ", "件数", "構成比 (%)"]
    cat_disp["カテゴリ"] = cat_disp["カテゴリ"].map(CATEGORY_LABELS).fillna(
        cat_disp["カテゴリ"]
    )
    col_cat.dataframe(cat_disp, use_container_width=True, hide_index=True)
    n_target = int(df_f["LLM判定要"].sum()) if "LLM判定要" in df_f.columns else 0
    col_cat.caption(
        f"regex で 0 件ヒット or 2 件以上ヒット (LLM 第 2 段の対象): {n_target:,} 件"
    )

if "分類元" in df_f.columns:
    src_counts = (
        df_f["分類元"].value_counts().rename_axis("分類元").reset_index(name="件数")
    )
    col_src.markdown("**分類元の内訳**")
    col_src.dataframe(src_counts, use_container_width=True, hide_index=True)
    col_src.caption(
        "regex=第1段で1件ヒット確定 ／ cache=LLM 過去判定の流用 ／ "
        "llm=今回 LLM 呼出 ／ llm_fallback=LLM 失敗 or 確定術式欠損"
    )

# ----- カテゴリ別 月次推移 --------------------------------------------

st.divider()
st.subheader("カテゴリ別 月次推移")

cat_mt = category_monthly_trend(df_f)
if cat_mt.empty or len(cat_mt.columns) <= 1:
    st.info("該当データなし")
else:
    cat_mt_disp = cat_mt.set_index("手術実施月").rename(columns=CATEGORY_LABELS)
    st.line_chart(cat_mt_disp)
    st.caption("4 系列の折れ線で件数推移を表示。フィルタの影響を受ける。")

# ----- カテゴリ × 診療科 クロス --------------------------------------

st.divider()
st.subheader("カテゴリ × 診療科")

available = [c for c in CATEGORY_COLUMNS if c in df_f.columns]
if not available or df_f.empty:
    st.info("該当データなし")
else:
    cross = (
        df_f.groupby("実施診療科")[available]
        .sum()
        .astype("int64")
        .rename(columns=CATEGORY_LABELS)
    )
    cross["合計"] = cross.sum(axis=1)
    cross = cross.sort_values("合計", ascending=False)
    # 合計 0 の診療科は省略
    cross = cross[cross["合計"] > 0]
    cross_disp = cross.drop(columns="合計").reset_index().rename(columns={"実施診療科": "診療科"})
    if cross_disp.empty:
        st.info("カテゴリ該当の診療科なし")
    else:
        st.dataframe(cross_disp, use_container_width=True, hide_index=True)
        st.caption("各カテゴリの件数を診療科別に集計。合計件数の多い順に表示。")
