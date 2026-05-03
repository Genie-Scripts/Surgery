"""Streamlit ダッシュボード エントリ。

実行:
    streamlit run app/main.py

サイドバーで CSV パスとフィルタを指定し、メインに全体 KPI / 月次推移 /
カテゴリ別件数 / 術者別 KPI を表示する。

このページは Phase 1 MVP の単一画面。spec §9 の app/pages/ 構造への分割は Phase 2 で。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.aggregate import (
    category_counts,
    category_monthly_trend,
    expand_operators,
    is_general_anesthesia,
    kpi_overall,
    kpi_per_doctor,
    monthly_trend,
)
from src.classify import classify, load_rules
from src.classify_llm import classify_with_llm
from src.ingest import load
from src.llm_client import LLMClient

ROOT = Path(__file__).resolve().parent.parent
LLM_CONFIG = ROOT / "config" / "llm_config.yaml"

# 既定 CSV: 匿名化フローの最終出力 (data/raw/anonymized/anonymized_data.csv) が存在すれば優先、
# なければルート直下のサンプル (anonymized_data.csv) にフォールバック。
_PIPELINE_OUTPUT = ROOT / "data" / "raw" / "anonymized" / "anonymized_data.csv"
_ROOT_SAMPLE = ROOT / "anonymized_data.csv"
DEFAULT_CSV = _PIPELINE_OUTPUT if _PIPELINE_OUTPUT.exists() else _ROOT_SAMPLE

CATEGORY_LABELS: dict[str, str] = {
    "malignant_tumor": "悪性腫瘍",
    "artificial_joint": "人工関節",
    "robot_assisted_davinci": "ロボット支援(ダヴィンチ系)",
    "robot_assisted_other": "ロボット支援(非ダヴィンチ系)",
}

st.set_page_config(page_title="Surgery Dashboard", layout="wide")


@st.cache_data(show_spinner="CSV 読込・カテゴリ分類中...")
def load_pipeline(path: str, run_llm: bool) -> tuple[pd.DataFrame, str]:
    """CSV を読み込み、regex 第 1 段 → (任意) LLM 第 2 段の順で分類して返す。

    返り値: (df, status_message)。status は UI 表示用の文字列。
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


# ----- Sidebar (filters) -------------------------------------------------

st.sidebar.title("フィルタ")

csv_path = st.sidebar.text_input("CSV パス", str(DEFAULT_CSV))
csv_p = Path(csv_path)
if not csv_p.exists():
    st.error(f"ファイルが見つかりません: {csv_p}")
    st.stop()

run_llm = st.sidebar.checkbox(
    "LLM 第 2 段を適用 (Swallow-8B + ハードガード)", value=True
)

df, pipeline_status = load_pipeline(str(csv_p), run_llm)

operator_mode_label = st.sidebar.radio(
    "執刀医モード",
    options=["執刀医のみ", "執刀医＋助手を含む"],
)
mode = "lead_only" if operator_mode_label == "執刀医のみ" else "all"

apply_codes = st.sidebar.multiselect(
    "申込区分",
    options=sorted(df["申込区分"].dropna().unique().tolist()),
    default=sorted(df["申込区分"].dropna().unique().tolist()),
)

departments_all = sorted(df["実施診療科"].dropna().unique().tolist())
departments = st.sidebar.multiselect(
    "実施診療科",
    options=departments_all,
    default=departments_all,
)

ga_only = st.sidebar.checkbox("全身麻酔のみ", value=False)

# ----- Apply filters -----------------------------------------------------

mask = df["申込区分"].isin(apply_codes) & df["実施診療科"].isin(departments)
if ga_only:
    mask = mask & df["全身麻酔"]

df_f = df[mask].copy()

# ----- Main --------------------------------------------------------------

st.title("Surgery Dashboard")
st.caption(
    f"読込: `{csv_p.name}` ／ 全 {len(df):,} 件中 {len(df_f):,} 件を表示中"
    f"（執刀医モード = {operator_mode_label}{' / 全身麻酔のみ' if ga_only else ''}）"
    f"｜ {pipeline_status}"
)

st.divider()
st.subheader("全体 KPI")

k = kpi_overall(df_f)
c1, c2, c3, c4 = st.columns(4)
c1.metric("件数", f"{k['件数']:,}")
c2.metric("総手術時間 (時間)", f"{k['総手術時間_分'] / 60:,.1f}")
c3.metric("平均手術時間 (分)", f"{k['平均手術時間_分']:.1f}")
c4.metric("緊急比率", f"{k['緊急比率'] * 100:.1f}%")

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

st.divider()
st.subheader("カテゴリ別 月次推移")

cat_mt = category_monthly_trend(df_f)
if cat_mt.empty or len(cat_mt.columns) <= 1:
    st.info("該当データなし")
else:
    cat_mt_disp = cat_mt.set_index("手術実施月").rename(columns=CATEGORY_LABELS)
    st.line_chart(cat_mt_disp)
    st.caption("線種ごとに件数推移を表示。フィルタ（執刀医モード／申込区分／診療科／全身麻酔）の影響を受ける。")

st.divider()
st.subheader("カテゴリ別件数")

cat = category_counts(df_f)
col_cat, col_src = st.columns([2, 1])

if cat.empty:
    col_cat.info("カテゴリ列がありません")
else:
    cat_disp = cat.assign(構成比=lambda d: (d["件数"] / max(len(df_f), 1) * 100).round(1))
    cat_disp.columns = ["カテゴリ", "件数", "構成比 (%)"]
    col_cat.dataframe(cat_disp, use_container_width=True, hide_index=True)
    n_target = int(df_f["LLM判定要"].sum()) if "LLM判定要" in df_f.columns else 0
    col_cat.caption(
        f"regex で 0 件ヒット or 2 件以上ヒット (LLM 第 2 段の対象): {n_target:,} 件"
    )

if "分類元" in df_f.columns:
    src_counts = (
        df_f["分類元"]
        .value_counts()
        .rename_axis("分類元")
        .reset_index(name="件数")
    )
    col_src.markdown("**分類元の内訳**")
    col_src.dataframe(src_counts, use_container_width=True, hide_index=True)
    col_src.caption(
        "regex=第1段で1件ヒット確定 ／ cache=LLM 過去判定の流用 ／ "
        "llm=今回 LLM 呼出 ／ llm_fallback=LLM 失敗 or 確定術式欠損"
    )

st.divider()
st.subheader(f"術者別 KPI ({operator_mode_label})")

df_long = expand_operators(df_f, mode=mode)
per_doc = kpi_per_doctor(df_long)

n_show = st.slider("表示件数", min_value=10, max_value=max(10, len(per_doc)), value=min(20, len(per_doc)))
st.dataframe(per_doc.head(n_show), use_container_width=True, hide_index=True)

# ----- 特定術者ドリルダウン --------------------------------------------

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
    )

    df_d = df_long[df_long["医師"] == selected_doctor].copy()

    # 個別 KPI
    k_d = kpi_overall(df_d)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("件数", f"{k_d['件数']:,}")
    c2.metric("総手術時間 (時間)", f"{k_d['総手術時間_分'] / 60:,.1f}")
    c3.metric("平均手術時間 (分)", f"{k_d['平均手術時間_分']:.1f}")
    c4.metric("緊急比率", f"{k_d['緊急比率'] * 100:.1f}%")

    # 月次推移 と カテゴリ別月次
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

    # カテゴリ別累計件数
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
