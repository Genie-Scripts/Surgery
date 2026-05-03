"""Page 2: 術者別 KPI。

術者ランキング、特定術者ドリルダウン、期間比較（サイドバーで有効化時）を提供する。
"""

from __future__ import annotations

import streamlit as st

from src.aggregate import (
    category_counts,
    category_monthly_trend,
    expand_operators,
    kpi_overall,
    kpi_per_doctor,
    kpi_per_doctor_compare,
    monthly_trend,
)
from src.ui.data_loader import CATEGORY_LABELS
from src.ui.filters import render_caption, render_sidebar_filters

st.set_page_config(page_title="術者別 - Surgery", layout="wide")

df, df_f, state = render_sidebar_filters()

st.title(f"術者別 KPI ({state['operator_mode_label']})")
render_caption(df, df_f, state)

df_long = expand_operators(df_f, mode=state["operator_mode"])
per_doc = kpi_per_doctor(df_long)

# ----- ランキング -----------------------------------------------------

st.divider()
st.subheader("ランキング")
n_show = st.slider(
    "表示件数",
    min_value=10,
    max_value=max(10, len(per_doc)),
    value=min(20, len(per_doc)),
    key="page2_ranking_n",
)
st.dataframe(per_doc.head(n_show), use_container_width=True, hide_index=True)

# ----- 特定術者ドリルダウン -------------------------------------------

st.divider()
st.subheader("特定術者ドリルダウン")

if df_long.empty:
    st.info("対象術者なし（フィルタを緩めてください）")
else:
    doctor_counts = df_long.groupby("医師").size().sort_values(ascending=False)
    doctor_options = doctor_counts.index.tolist()
    doctor_labels = {d: f"{d} （{int(n):,} 件）" for d, n in doctor_counts.items()}

    selected_doctor = st.selectbox(
        "術者",
        options=doctor_options,
        format_func=lambda d: doctor_labels[d],
        help="件数降順。選択した術者に絞り込んで KPI と月次推移を表示します。",
        key="page2_drilldown_doctor",
    )

    df_d = df_long[df_long["医師"] == selected_doctor].copy()

    k_d = kpi_overall(df_d)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("件数", f"{k_d['件数']:,}")
    c2.metric("総手術時間 (時間)", f"{k_d['総手術時間_分'] / 60:,.1f}")
    c3.metric("平均手術時間 (分)", f"{k_d['平均手術時間_分']:.1f}")
    c4.metric("緊急比率", f"{k_d['緊急比率'] * 100:.1f}%")

    col_left, col_right = st.columns(2)

    mt_d = monthly_trend(df_d)
    if mt_d.empty:
        col_left.info("月次推移: データなし")
    else:
        col_left.markdown("**件数 月次推移**")
        col_left.line_chart(mt_d.set_index("手術実施月")[["件数"]])

    cat_mt_d = category_monthly_trend(df_d)
    if cat_mt_d.empty or len(cat_mt_d.columns) <= 1:
        col_right.info("カテゴリ別月次: データなし")
    else:
        col_right.markdown("**カテゴリ別 月次推移**")
        col_right.line_chart(
            cat_mt_d.set_index("手術実施月").rename(columns=CATEGORY_LABELS)
        )

    cat_d = category_counts(df_d)
    if not cat_d.empty:
        cat_d_disp = cat_d.assign(
            構成比=lambda d: (d["件数"] / max(len(df_d), 1) * 100).round(1)
        )
        cat_d_disp.columns = ["カテゴリ", "件数", "構成比 (%)"]
        cat_d_disp["カテゴリ"] = cat_d_disp["カテゴリ"].map(CATEGORY_LABELS).fillna(
            cat_d_disp["カテゴリ"]
        )
        st.markdown("**カテゴリ別件数（選択術者）**")
        st.dataframe(cat_d_disp, use_container_width=True, hide_index=True)

# ----- 期間比較 -------------------------------------------------------

if state["enable_compare"]:
    st.divider()
    st.subheader("期間比較")

    period_a = state["period_a"]
    period_b = state["period_b"]
    if (
        not isinstance(period_a, tuple)
        or len(period_a) != 2
        or not isinstance(period_b, tuple)
        or len(period_b) != 2
    ):
        st.info("サイドバーで期間 A・期間 B を範囲指定してください（開始〜終了の 2 点）")
    else:
        compare = kpi_per_doctor_compare(df_long, period_a, period_b)

        st.caption(
            f"期間 A: {period_a[0]} 〜 {period_a[1]} ／ "
            f"期間 B: {period_b[0]} 〜 {period_b[1]}（両端含む）。"
            f"件数_B 降順で表示。件数比率(%) は (B/A − 1)×100、A=0 のときは N/A。"
        )

        compare_disp = compare.copy()
        for col in (
            "件数比率(%)",
            "平均手術時間_分_A",
            "平均手術時間_分_B",
            "平均時間差_分",
        ):
            if col in compare_disp.columns:
                compare_disp[col] = compare_disp[col].astype("Float64").round(1)

        n_show_compare = st.slider(
            "表示件数",
            min_value=10,
            max_value=max(10, len(compare_disp)),
            value=min(20, len(compare_disp)),
            key="page2_compare_n",
        )
        st.dataframe(
            compare_disp.head(n_show_compare),
            use_container_width=True,
            hide_index=True,
        )
