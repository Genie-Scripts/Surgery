"""Page 4: 月次推移。

時系列フォーカスで、件数 / 平均手術時間 / カテゴリ別件数 / 緊急比率の月次推移を
1 画面に並べて確認できる。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.aggregate import category_monthly_trend, monthly_trend
from src.ui.data_loader import CATEGORY_LABELS
from src.ui.filters import render_caption, render_sidebar_filters

st.set_page_config(page_title="月次推移 - Surgery", layout="wide")

df, df_f, state = render_sidebar_filters()

st.title("月次推移")
render_caption(df, df_f, state)

# ----- 件数 / 平均手術時間 -------------------------------------------

st.divider()
st.subheader("件数 / 平均手術時間")

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

# ----- カテゴリ別 月次推移 -------------------------------------------

st.divider()
st.subheader("カテゴリ別 月次推移")

cat_mt = category_monthly_trend(df_f)
if cat_mt.empty or len(cat_mt.columns) <= 1:
    st.info("該当データなし")
else:
    cat_mt_disp = cat_mt.set_index("手術実施月").rename(columns=CATEGORY_LABELS)
    st.line_chart(cat_mt_disp)

# ----- 緊急比率 月次推移 ---------------------------------------------

st.divider()
st.subheader("緊急比率 月次推移")

if df_f.empty or "申込区分" not in df_f.columns:
    st.info("該当データなし")
else:
    g = df_f.set_index("手術実施日").resample("MS")
    monthly_emergency = pd.DataFrame(
        {
            "総件数": g.size(),
            "緊急件数": g["申込区分"].apply(lambda s: int((s == "緊急").sum())),
        }
    )
    monthly_emergency["緊急比率(%)"] = (
        monthly_emergency["緊急件数"] / monthly_emergency["総件数"].replace(0, pd.NA) * 100
    ).astype("Float64").round(1)
    monthly_emergency = monthly_emergency.rename_axis("手術実施月")

    col_left, col_right = st.columns(2)
    col_left.markdown("**緊急比率 (%)**")
    col_left.line_chart(monthly_emergency[["緊急比率(%)"]])
    col_right.markdown("**緊急件数（実数）**")
    col_right.line_chart(monthly_emergency[["緊急件数"]])

# ----- 診療科別 件数 月次推移 ----------------------------------------

st.divider()
st.subheader("診療科別 件数 月次推移（上位 5 科）")

if df_f.empty:
    st.info("該当データなし")
else:
    top5 = df_f["実施診療科"].value_counts().head(5).index.tolist()
    if not top5:
        st.info("診療科の集計対象なし")
    else:
        df_top = df_f[df_f["実施診療科"].isin(top5)]
        pivot = (
            df_top.assign(月=df_top["手術実施日"].dt.to_period("M").dt.to_timestamp())
            .groupby(["月", "実施診療科"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=top5)
        )
        st.line_chart(pivot)
        st.caption("フィルタ後の件数で上位 5 診療科を抽出して表示。")
