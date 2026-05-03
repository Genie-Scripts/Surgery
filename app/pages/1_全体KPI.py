"""Page 1: 全体 KPI。

件数・総時間・平均時間・緊急比率の 4 メトリクスと、件数・平均時間の月次推移、
カテゴリ別件数の簡易表（詳細はカテゴリ別ページへ）を表示する。
"""

from __future__ import annotations

import streamlit as st

from src.aggregate import category_counts, kpi_overall, monthly_trend
from src.ui.data_loader import CATEGORY_LABELS
from src.ui.filters import render_caption, render_sidebar_filters

st.set_page_config(page_title="全体 KPI - Surgery", layout="wide")

df, df_f, state = render_sidebar_filters()

st.title("全体 KPI")
render_caption(df, df_f, state)

# ----- KPI metrics --------------------------------------------------

st.divider()
k = kpi_overall(df_f)
c1, c2, c3, c4 = st.columns(4)
c1.metric("件数", f"{k['件数']:,}")
c2.metric("総手術時間 (時間)", f"{k['総手術時間_分'] / 60:,.1f}")
c3.metric("平均手術時間 (分)", f"{k['平均手術時間_分']:.1f}")
c4.metric("緊急比率", f"{k['緊急比率'] * 100:.1f}%")

# ----- 月次推移 ------------------------------------------------------

st.divider()
st.subheader("月次推移")

mt = monthly_trend(df_f)
if mt.empty:
    st.info("該当データなし")
else:
    mt_indexed = mt.set_index("手術実施月")
    col_a, col_b = st.columns(2)
    col_a.markdown("**件数**")
    col_a.line_chart(mt_indexed[["件数"]])
    col_b.markdown("**平均手術時間 (分)**")
    col_b.line_chart(mt_indexed[["平均手術時間_分"]])

# ----- カテゴリ別件数（簡易） + 分類元 -------------------------------

st.divider()
st.subheader("カテゴリ別件数（簡易）")

cat = category_counts(df_f)
col_left, col_right = st.columns([2, 1])

if cat.empty:
    col_left.info("カテゴリ列がありません")
else:
    cat_disp = cat.assign(構成比=lambda d: (d["件数"] / max(len(df_f), 1) * 100).round(1))
    cat_disp.columns = ["カテゴリ", "件数", "構成比 (%)"]
    cat_disp["カテゴリ"] = cat_disp["カテゴリ"].map(CATEGORY_LABELS).fillna(
        cat_disp["カテゴリ"]
    )
    col_left.dataframe(cat_disp, use_container_width=True, hide_index=True)
    col_left.caption("詳細は「カテゴリ別」ページを参照")

if "分類元" in df_f.columns:
    src_counts = (
        df_f["分類元"].value_counts().rename_axis("分類元").reset_index(name="件数")
    )
    col_right.markdown("**分類元の内訳**")
    col_right.dataframe(src_counts, use_container_width=True, hide_index=True)
