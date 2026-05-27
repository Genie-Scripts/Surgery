"""KPI 集計（spec.md §3.3）。

入力: `src.classify.classify()` の出力 DataFrame（カテゴリ列が付与済み）
出力: 各種 KPI の集計値（純関数）

関数構成:
  - is_general_anesthesia(s):           OQ-5 解決済みの全身麻酔判定
  - expand_operators(df, mode):         執刀医モード/全術者モードで long-form 化
  - kpi_overall(df):                    全体 KPI（件数/総時間/平均時間/緊急比率）
  - monthly_trend(df):                  月次推移（手術実施日基準）
  - kpi_per_doctor(df_long):            術者ごとの KPI
  - kpi_per_doctor_compare(...):        2 期間の術者別 KPI 比較
  - kpi_overall_period_compare(...):    2 期間の全体 KPI 比較
  - category_counts(df):                カテゴリ別件数
  - category_counts_period_compare(...):2 期間のカテゴリ別件数比較
  - category_monthly_trend(df):         カテゴリ × 月次の件数

PDF レポート向け（src/export_pdf.py で使用）:
  - month_end_cutoff(today):            月締めの集計終端日（前月末日）
  - fiscal_year_periods(today):         今年度 YTD と昨年同期の (start, end) ペア（表示用）
  - report_periods(today):              PDF レポートの全比較窓（3mo/12mo/12週）
  - kpi_overall_compare(df, recent, comparison):
                                         任意の 2 窓で 4 項目 KPI を比較
  - monthly_count_compare(...):         任意の 2 窓で月次件数を 12 行で並べる
  - monthly_avg_time_compare(...):      月次 平均手術時間
  - monthly_general_anesthesia_compare(...): 月次 全麻件数
  - monthly_category_compare(...):      カテゴリ別 月次件数
  - weekly_general_anesthesia(...):     週次 全麻件数（目標達成判定込み）
  - top_n_procedures(df, recent, comparison, n): 主要術式 top N
  - top_n_postop_diagnoses(df, recent, comparison, n): 主要術後病名 top N
  - kpi_per_doctor_compare_window(...): 執刀医ランキング（任意の 2 窓）
  - category_counts_compare_window(...):カテゴリ別件数（任意の 2 窓）
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import pandas as pd
from dateutil.relativedelta import relativedelta

OperatorMode = Literal["lead_only", "all"]

CATEGORY_COLUMNS = (
    "malignant_tumor",
    "artificial_joint",
    "robot_assisted_davinci",
    "robot_assisted_other",
)


def is_general_anesthesia(s: object) -> bool:
    """全身麻酔判定（OQ-5 解決済み式）。

    真となる条件（OR）:
      - 「全身麻酔(20分以上：吸入もしくは静脈麻酔薬)」を含む
      - 「全身麻酔」と「20分以上」の両方を含む

    `上下肢の伝達麻酔(全身麻酔時算定不可)` の誤検出を避けるため、
    単純な「全身麻酔」だけのマッチは使わない。
    """
    if not isinstance(s, str) or not s:
        return False
    if "全身麻酔(20分以上：吸入もしくは静脈麻酔薬)" in s:
        return True
    return "全身麻酔" in s and "20分以上" in s


def expand_operators(df: pd.DataFrame, mode: OperatorMode = "lead_only") -> pd.DataFrame:
    """術者軸で long-form 化する。

    - `lead_only`: 1 手術 = 1 行、医師 = 執刀医
    - `all`:       1 手術 × N 医師 = N 行、執刀医と助手を全展開

    追加列: `医師` (str), `役割` ("執刀医" / "助手")
    """
    df_lead = df.copy()
    df_lead["医師"] = df_lead["執刀医"]
    df_lead["役割"] = "執刀医"
    df_lead = df_lead[df_lead["医師"].notna()]

    if mode == "lead_only":
        return df_lead.reset_index(drop=True)

    df_assist = df.copy()
    df_assist["医師"] = df_assist["助手リスト"]
    df_assist = df_assist.explode("医師")
    df_assist = df_assist[df_assist["医師"].notna()]
    df_assist["役割"] = "助手"

    return pd.concat([df_lead, df_assist], ignore_index=True)


def kpi_overall(df: pd.DataFrame) -> dict[str, float | int]:
    """全体 KPI: 件数 / 総手術時間 / 平均手術時間 / 緊急比率。"""
    n = len(df)
    times = df["予定手術時間"].dropna()
    return {
        "件数": int(n),
        "総手術時間_分": int(times.sum()) if len(times) else 0,
        "平均手術時間_分": float(times.mean()) if len(times) else 0.0,
        "緊急比率": float((df["申込区分"] == "緊急").sum() / n) if n else 0.0,
    }


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """月次推移（手術実施日基準）。

    返却列: 手術実施月 (Period[M] -> Timestamp), 件数, 平均手術時間_分, 総手術時間_分

    注: spec §3.3 で「中途月は土日祝日除外の営業日換算」を予定しているが、
    本関数は素の月次集計のみ。営業日換算は呼び出し側で別途乗算する。
    """
    if df.empty:
        return pd.DataFrame(columns=["手術実施月", "件数", "平均手術時間_分", "総手術時間_分"])

    g = df.set_index("手術実施日").resample("MS")
    out = pd.DataFrame(
        {
            "件数": g.size(),
            "平均手術時間_分": g["予定手術時間"].mean(),
            "総手術時間_分": g["予定手術時間"].sum(),
        }
    )
    return out.rename_axis("手術実施月").reset_index()


def kpi_per_doctor(df_long: pd.DataFrame) -> pd.DataFrame:
    """術者ごとの KPI。`df_long` は `expand_operators` の出力を想定。

    同一手術に対し 1 医師につき 1 行存在する前提（`mode="all"` なら助手分も加算）。
    """
    if df_long.empty:
        return pd.DataFrame(
            columns=["医師", "件数", "総手術時間_分", "平均手術時間_分", "緊急件数"]
        )

    g = df_long.groupby("医師", dropna=False)
    return (
        pd.DataFrame(
            {
                "件数": g.size(),
                "総手術時間_分": g["予定手術時間"].sum(),
                "平均手術時間_分": g["予定手術時間"].mean(),
                "緊急件数": g["申込区分"].apply(lambda s: int((s == "緊急").sum())),
            }
        )
        .reset_index()
        .sort_values("件数", ascending=False, ignore_index=True)
    )


def kpi_per_doctor_compare(
    df_long: pd.DataFrame,
    period_a: tuple[date, date],
    period_b: tuple[date, date],
    date_column: str = "手術実施日",
) -> pd.DataFrame:
    """期間 A と期間 B の術者別 KPI を比較する。

    `df_long` は `expand_operators` の出力を想定。
    各期間は両端含む `(start, end)` のタプル。

    返却列:
      - 医師
      - 件数_A, 件数_B, 件数差 (B - A)
      - 件数比率(%) ((B/A - 1) * 100、A=0 は NaN)
      - 平均手術時間_分_A, 平均手術時間_分_B, 平均時間差_分 (B - A)
      - 緊急件数_A, 緊急件数_B
    """

    def _slice(start: date, end: date) -> pd.DataFrame:
        mask = (df_long[date_column] >= pd.Timestamp(start)) & (
            df_long[date_column] <= pd.Timestamp(end)
        )
        return df_long[mask]

    df_a = _slice(*period_a)
    df_b = _slice(*period_b)

    kpi_a = kpi_per_doctor(df_a).set_index("医師").add_suffix("_A")
    kpi_b = kpi_per_doctor(df_b).set_index("医師").add_suffix("_B")

    merged = kpi_a.join(kpi_b, how="outer")

    # int 列の NaN は 0 で埋める（その医師は当該期間に手術なし）
    for col in ("件数_A", "件数_B", "緊急件数_A", "緊急件数_B"):
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype("int64")

    merged["件数差"] = merged["件数_B"] - merged["件数_A"]
    # 件数比率: A=0 のときは NaN (新規参入の医師など)
    merged["件数比率(%)"] = (
        (merged["件数_B"] / merged["件数_A"].replace(0, pd.NA)) - 1
    ) * 100

    if "平均手術時間_分_A" in merged.columns and "平均手術時間_分_B" in merged.columns:
        merged["平均時間差_分"] = (
            merged["平均手術時間_分_B"] - merged["平均手術時間_分_A"]
        )

    # 列順を整える（存在する列のみ）
    column_order = [
        "件数_A",
        "件数_B",
        "件数差",
        "件数比率(%)",
        "平均手術時間_分_A",
        "平均手術時間_分_B",
        "平均時間差_分",
        "緊急件数_A",
        "緊急件数_B",
    ]
    ordered = [c for c in column_order if c in merged.columns]

    return (
        merged[ordered]
        .sort_values("件数_B", ascending=False, kind="stable")
        .reset_index()
    )


def kpi_overall_period_compare(
    df: pd.DataFrame,
    period_a: tuple[date, date],
    period_b: tuple[date, date],
    date_column: str = "手術実施日",
) -> dict[str, dict[str, float | int]]:
    """期間 A と期間 B の全体 KPI を比較する。

    各期間は両端含む `(start, end)` のタプル。
    返却辞書: `{"A": kpi_overall(...), "B": kpi_overall(...), "diff": {k: B-A for k}}`。
    """

    def _slice(start: date, end: date) -> pd.DataFrame:
        mask = (df[date_column] >= pd.Timestamp(start)) & (
            df[date_column] <= pd.Timestamp(end)
        )
        return df[mask]

    a = kpi_overall(_slice(*period_a))
    b = kpi_overall(_slice(*period_b))
    diff = {k: b[k] - a[k] for k in a}
    return {"A": a, "B": b, "diff": diff}


def category_counts(df: pd.DataFrame) -> pd.DataFrame:
    """カテゴリ列ごとの件数（True 件数）。"""
    rows = []
    for col in CATEGORY_COLUMNS:
        if col in df.columns:
            rows.append({"カテゴリ": col, "件数": int(df[col].sum())})
    return pd.DataFrame(rows)


def category_counts_period_compare(
    df: pd.DataFrame,
    period_a: tuple[date, date],
    period_b: tuple[date, date],
    date_column: str = "手術実施日",
) -> pd.DataFrame:
    """期間 A と期間 B のカテゴリ別件数を比較する。

    返却列: `カテゴリ`, `件数_A`, `件数_B`, `件数差` (B - A)。
    両期間でいずれのカテゴリも 0 件のときは行を含む（カテゴリ列が `df` に存在する限り）。
    """

    def _slice(start: date, end: date) -> pd.DataFrame:
        mask = (df[date_column] >= pd.Timestamp(start)) & (
            df[date_column] <= pd.Timestamp(end)
        )
        return df[mask]

    a = category_counts(_slice(*period_a)).rename(columns={"件数": "件数_A"})
    b = category_counts(_slice(*period_b)).rename(columns={"件数": "件数_B"})
    if a.empty and b.empty:
        return pd.DataFrame(columns=["カテゴリ", "件数_A", "件数_B", "件数差"])

    merged = a.merge(b, on="カテゴリ", how="outer").fillna(0)
    merged["件数_A"] = merged["件数_A"].astype("int64")
    merged["件数_B"] = merged["件数_B"].astype("int64")
    merged["件数差"] = merged["件数_B"] - merged["件数_A"]
    return merged


def category_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """カテゴリ別件数の月次集計（手術実施日基準）。

    返却列: `手術実施月` (Timestamp) と、`df` に存在する各カテゴリ ID 列（件数）。
    `df` が空、もしくはカテゴリ列が 1 つも無いときは `手術実施月` のみの空 DataFrame を返す。
    """
    available = [c for c in CATEGORY_COLUMNS if c in df.columns]
    if df.empty or not available:
        return pd.DataFrame(columns=["手術実施月", *available])

    g = df.set_index("手術実施日").resample("MS")
    out = g[available].sum().astype("int64")
    return out.rename_axis("手術実施月").reset_index()


# ---------------------------------------------------------------------------
# PDF レポート向け（月締め・任意の比較窓）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FiscalYearPeriods:
    """今年度 YTD と昨年同期間（表示メタデータ用、PDF 本体の比較窓は ReportPeriods）。"""

    fiscal_year: int  # 今年度（4 月開始の年）。例: 2026/4〜2027/3 なら 2026
    cutoff: date  # 集計終端（前月末日）
    ytd: tuple[date, date]  # (4/1/FY, cutoff)
    last_year: tuple[date, date]  # (4/1/(FY-1), 同日数前の終端)


@dataclass(frozen=True)
class ReportPeriods:
    """PDF レポートの各セクションが使う比較窓。

    cutoff = today の前月末日。各窓は両端含む (start, end) のタプル。

    用途別の対応:
      - Page 1 KPI カード:         recent_3mo  vs  yoy_3mo   (季節性キャンセル)
      - 月次折れ線 (件数/平均/全麻/カテゴリ): recent_12mo vs prior_12mo
      - 週次 全麻 vs 目標:         weekly_12w  (単一窓・直近 12 週)
      - 表/カテゴリバー (ランキング、術式・病名 Top10):
                                   recent_3mo  vs  prior_3mo (順次比較)
    """

    fiscal_year: int  # cutoff が属する会計年度（表示用）
    cutoff: date
    recent_3mo: tuple[date, date]
    prior_3mo: tuple[date, date]
    yoy_3mo: tuple[date, date]
    recent_12mo: tuple[date, date]
    prior_12mo: tuple[date, date]
    weekly_12w: tuple[date, date]


def month_end_cutoff(today: date) -> date:
    """月締め集計の終端 = 当日の前月末日。

    例: today = 2026-05-27 → 2026-04-30
        today = 2026-05-01 → 2026-04-30
        today = 2026-05-31 → 2026-04-30
    """
    first_of_month = today.replace(day=1)
    return first_of_month - timedelta(days=1)


def fiscal_year_periods(today: date) -> FiscalYearPeriods:
    """月締めで今年度 YTD と昨年同期間を導出する（表示メタ用）。

    - 年度 = 4 月開始
    - cutoff = 前月末日（`month_end_cutoff`）
    - cutoff が 1〜3 月の場合、年度はその前年（例: 2026-02-28 → FY 2025）
    - 昨年同期は (4/1/(FY-1), cutoff の 1 年前同日)。閏日 2/29 は 2/28 に丸める
    """
    cutoff = month_end_cutoff(today)
    fiscal_year = cutoff.year if cutoff.month >= 4 else cutoff.year - 1
    ytd_start = date(fiscal_year, 4, 1)
    ly_start = date(fiscal_year - 1, 4, 1)
    try:
        ly_end = cutoff.replace(year=cutoff.year - 1)
    except ValueError:
        ly_end = cutoff.replace(year=cutoff.year - 1, day=28)
    return FiscalYearPeriods(
        fiscal_year=fiscal_year,
        cutoff=cutoff,
        ytd=(ytd_start, cutoff),
        last_year=(ly_start, ly_end),
    )


def _start_of_month(d: date) -> date:
    return d.replace(day=1)


def _shift_back_months(d: date, months: int) -> date:
    """`d` の年月から `months` ヶ月前の同月 1 日を返す（日付は無視）。"""
    return _start_of_month(d) - relativedelta(months=months)


def _months_ago(d: date, months: int) -> date:
    """`d` から純粋に `months` ヶ月前。閏日は relativedelta が自動で 28/29 を丸める。"""
    return d - relativedelta(months=months)


def report_periods(today: date) -> ReportPeriods:
    """PDF レポートで使う比較窓を一括導出する。

    - cutoff = 前月末日
    - recent_3mo: cutoff を含む 3 ヶ月 (cutoff の月 1 日から 2 ヶ月遡って 1 日, cutoff)
    - prior_3mo:  recent_3mo の直前 3 ヶ月
    - yoy_3mo:    recent_3mo の 1 年前同 3 ヶ月
    - recent_12mo: cutoff を含む 12 ヶ月
    - prior_12mo:  recent_12mo の直前 12 ヶ月
    - weekly_12w:  cutoff の週月曜を含む過去 12 週
    """
    cutoff = month_end_cutoff(today)
    fiscal_year = cutoff.year if cutoff.month >= 4 else cutoff.year - 1

    recent_3mo_start = _shift_back_months(cutoff, 2)  # 当月含めて 3 ヶ月遡る
    recent_3mo = (recent_3mo_start, cutoff)
    prior_3mo_end = recent_3mo_start - timedelta(days=1)
    prior_3mo_start = _shift_back_months(prior_3mo_end, 2)
    prior_3mo = (prior_3mo_start, prior_3mo_end)
    yoy_3mo = (_months_ago(recent_3mo_start, 12), _months_ago(cutoff, 12))

    recent_12mo_start = _shift_back_months(cutoff, 11)
    recent_12mo = (recent_12mo_start, cutoff)
    prior_12mo_end = recent_12mo_start - timedelta(days=1)
    prior_12mo_start = _shift_back_months(prior_12mo_end, 11)
    prior_12mo = (prior_12mo_start, prior_12mo_end)

    cutoff_ts = pd.Timestamp(cutoff)
    last_monday = cutoff_ts - pd.Timedelta(days=cutoff_ts.weekday())
    first_monday = last_monday - pd.Timedelta(days=11 * 7)
    weekly_12w = (first_monday.date(), cutoff)

    return ReportPeriods(
        fiscal_year=fiscal_year,
        cutoff=cutoff,
        recent_3mo=recent_3mo,
        prior_3mo=prior_3mo,
        yoy_3mo=yoy_3mo,
        recent_12mo=recent_12mo,
        prior_12mo=prior_12mo,
        weekly_12w=weekly_12w,
    )


def _slice_period(df: pd.DataFrame, period: tuple[date, date]) -> pd.DataFrame:
    mask = (df["手術実施日"] >= pd.Timestamp(period[0])) & (
        df["手術実施日"] <= pd.Timestamp(period[1])
    )
    return df.loc[mask]


def general_anesthesia_count(df: pd.DataFrame) -> int:
    """全身麻酔手術の件数。`麻酔種別` 列で is_general_anesthesia を判定。"""
    if df.empty or "麻酔種別" not in df.columns:
        return 0
    return int(df["麻酔種別"].apply(is_general_anesthesia).sum())


def kpi_overall_compare(
    df: pd.DataFrame,
    recent: tuple[date, date],
    comparison: tuple[date, date],
) -> dict[str, dict[str, float | int]]:
    """任意の 2 窓で 4 項目 KPI を返す。

    KPI: 件数 / 平均手術時間_分 / 緊急比率 / 全麻手術件数
    返却: {"recent": {...}, "comparison": {...}, "diff": {...}}
    """

    def _kpi(slice_df: pd.DataFrame) -> dict[str, float | int]:
        n = len(slice_df)
        times = (
            slice_df["予定手術時間"].dropna()
            if "予定手術時間" in slice_df.columns
            else pd.Series(dtype=float)
        )
        return {
            "件数": int(n),
            "平均手術時間_分": float(times.mean()) if len(times) else 0.0,
            "緊急比率": float((slice_df["申込区分"] == "緊急").sum() / n) if n else 0.0,
            "全麻手術件数": general_anesthesia_count(slice_df),
        }

    r = _kpi(_slice_period(df, recent))
    c = _kpi(_slice_period(df, comparison))
    diff = {k: r[k] - c[k] for k in r}
    return {"recent": r, "comparison": c, "diff": diff}


def _month_iter(window: tuple[date, date]) -> list[pd.Timestamp]:
    """window を含む各月の 1 日 (Timestamp) を昇順で返す。"""
    start = pd.Timestamp(_start_of_month(window[0]))
    end = pd.Timestamp(_start_of_month(window[1]))
    return list(pd.date_range(start, end, freq="MS"))


def _monthly_size(slice_df: pd.DataFrame) -> pd.Series:
    if slice_df.empty:
        return pd.Series(dtype="int64")
    return slice_df.set_index("手術実施日").resample("MS").size()


def _monthly_mean_time(slice_df: pd.DataFrame) -> pd.Series:
    if slice_df.empty:
        return pd.Series(dtype="float64")
    return slice_df.set_index("手術実施日").resample("MS")["予定手術時間"].mean()


def _monthly_sum(slice_df: pd.DataFrame, col: str) -> pd.Series:
    if slice_df.empty or col not in slice_df.columns:
        return pd.Series(dtype="int64")
    return slice_df.set_index("手術実施日").resample("MS")[col].sum().astype("int64")


def _align_two_windows(
    recent_window: tuple[date, date],
    prior_window: tuple[date, date],
    recent_series: pd.Series,
    prior_series: pd.Series,
    fill: float | int = 0,
) -> pd.DataFrame:
    """recent / prior 窓の月系列を「窓内のオフセット」で並列に並べる。

    `recent_series` / `prior_series` の値が pd.NA / NaN の場合は `fill` で置換する。
    （pd.NA は plotly→kaleido 経由の orjson でシリアライズできないため）

    返却列: 月オフセット, 月ラベル (YYYY-MM, recent 側), 直近, 前期
    """
    r_months = _month_iter(recent_window)
    p_months = _month_iter(prior_window)
    n = min(len(r_months), len(p_months))
    rows = []
    for i in range(n):
        rm = r_months[i]
        pm = p_months[i]
        r_val = recent_series.get(rm, fill)
        p_val = prior_series.get(pm, fill)
        if pd.isna(r_val):
            r_val = fill
        if pd.isna(p_val):
            p_val = fill
        rows.append(
            {
                "月オフセット": i,
                "月ラベル": rm.strftime("%Y-%m"),
                "直近": r_val,
                "前期": p_val,
            }
        )
    return pd.DataFrame(rows)


def monthly_count_compare(
    df: pd.DataFrame,
    recent_window: tuple[date, date],
    prior_window: tuple[date, date],
) -> pd.DataFrame:
    """月次件数を「直近 vs 前期」で並列に返す。

    返却列: 月オフセット, 月ラベル (YYYY-MM), 直近, 前期 (どちらも件数 int)
    """
    r = _monthly_size(_slice_period(df, recent_window))
    p = _monthly_size(_slice_period(df, prior_window))
    out = _align_two_windows(recent_window, prior_window, r, p, fill=0)
    if not out.empty:
        out["直近"] = out["直近"].astype("int64")
        out["前期"] = out["前期"].astype("int64")
    return out


def monthly_avg_time_compare(
    df: pd.DataFrame,
    recent_window: tuple[date, date],
    prior_window: tuple[date, date],
) -> pd.DataFrame:
    """月次 平均手術時間 (分) を「直近 vs 前期」で並列に返す。欠損月は NaN。"""
    r = _monthly_mean_time(_slice_period(df, recent_window))
    p = _monthly_mean_time(_slice_period(df, prior_window))
    return _align_two_windows(recent_window, prior_window, r, p, fill=float("nan"))


def monthly_general_anesthesia_compare(
    df: pd.DataFrame,
    recent_window: tuple[date, date],
    prior_window: tuple[date, date],
) -> pd.DataFrame:
    """月次 全麻件数を「直近 vs 前期」で並列に返す。"""

    def _ga_monthly(slice_df: pd.DataFrame) -> pd.Series:
        if slice_df.empty:
            return pd.Series(dtype="int64")
        ga = slice_df[_general_anesthesia_mask(slice_df)]
        if ga.empty:
            return pd.Series(dtype="int64")
        return ga.set_index("手術実施日").resample("MS").size().astype("int64")

    r = _ga_monthly(_slice_period(df, recent_window))
    p = _ga_monthly(_slice_period(df, prior_window))
    out = _align_two_windows(recent_window, prior_window, r, p, fill=0)
    if not out.empty:
        out["直近"] = out["直近"].astype("int64")
        out["前期"] = out["前期"].astype("int64")
    return out


def monthly_category_compare(
    df: pd.DataFrame,
    recent_window: tuple[date, date],
    prior_window: tuple[date, date],
    category: str,
) -> pd.DataFrame:
    """指定カテゴリの月次件数を「直近 vs 前期」で並列に返す。"""
    r = _monthly_sum(_slice_period(df, recent_window), category)
    p = _monthly_sum(_slice_period(df, prior_window), category)
    out = _align_two_windows(recent_window, prior_window, r, p, fill=0)
    if not out.empty:
        out["直近"] = out["直近"].astype("int64")
        out["前期"] = out["前期"].astype("int64")
    return out


def _general_anesthesia_mask(df: pd.DataFrame) -> pd.Series:
    if "麻酔種別" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["麻酔種別"].apply(is_general_anesthesia)


def weekly_general_anesthesia(
    df: pd.DataFrame,
    period: tuple[date, date],
    target: int | None = None,
) -> pd.DataFrame:
    """ISO 週 (月曜始まり) 単位で全麻手術件数を集計する。

    返却列:
      - 週開始日 (Timestamp, 月曜),
      - 週ラベル (str, 'MM/DD 週'),
      - 全麻件数 (int),
      - 目標 (int | NA),
      - 達成 (bool | NA)  # target あり時のみ
    `period` の範囲を含むすべての週（その週の月曜が含まれる範囲）を欠損なく返す。
    """
    start, end = period
    sliced = _slice_period(df, period)
    ga = sliced[_general_anesthesia_mask(sliced)] if not sliced.empty else sliced

    if not ga.empty:
        wk = (
            ga.assign(週開始日=ga["手術実施日"].dt.to_period("W-SUN").dt.start_time)
            .groupby("週開始日")
            .size()
        )
    else:
        wk = pd.Series(dtype="int64")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    first_monday = start_ts - pd.Timedelta(days=start_ts.weekday())
    last_monday = end_ts - pd.Timedelta(days=end_ts.weekday())
    all_weeks = pd.date_range(first_monday, last_monday, freq="W-MON")

    out = pd.DataFrame({"週開始日": all_weeks})
    out["週ラベル"] = out["週開始日"].dt.strftime("%m/%d 週")
    out["全麻件数"] = out["週開始日"].map(wk).fillna(0).astype("int64")
    if target is not None:
        out["目標"] = int(target)
        out["達成"] = out["全麻件数"] >= target
    return out


def _top_n_with_compare(
    df: pd.DataFrame,
    column: str,
    recent: tuple[date, date],
    comparison: tuple[date, date],
    n: int,
) -> pd.DataFrame:
    """指定列の value_counts top N を recent 窓で取り、comparison 窓の件数を併記。

    返却列: <column>, 直近件数, 比較件数, 差分
    """
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "直近件数", "比較件数", "差分"])

    r = _slice_period(df, recent)
    c = _slice_period(df, comparison)
    if r.empty:
        return pd.DataFrame(columns=[column, "直近件数", "比較件数", "差分"])

    r_counts = r[column].dropna().value_counts()
    if r_counts.empty:
        return pd.DataFrame(columns=[column, "直近件数", "比較件数", "差分"])

    top = r_counts.head(n)
    c_counts = c[column].dropna().value_counts() if not c.empty else pd.Series(dtype="int64")
    out = pd.DataFrame(
        {
            column: top.index,
            "直近件数": top.values.astype("int64"),
            "比較件数": [int(c_counts.get(k, 0)) for k in top.index],
        }
    )
    out["差分"] = out["直近件数"] - out["比較件数"]
    return out.reset_index(drop=True)


def top_n_procedures(
    df: pd.DataFrame,
    recent: tuple[date, date],
    comparison: tuple[date, date],
    n: int = 10,
) -> pd.DataFrame:
    """主要術式 top N (recent 基準、comparison 件数を併記)。"""
    return _top_n_with_compare(df, "確定術式", recent, comparison, n)


def top_n_postop_diagnoses(
    df: pd.DataFrame,
    recent: tuple[date, date],
    comparison: tuple[date, date],
    n: int = 10,
) -> pd.DataFrame:
    """主要術後病名 top N (recent 基準、comparison 件数を併記)。"""
    return _top_n_with_compare(df, "術後病名", recent, comparison, n)


def kpi_per_doctor_compare_window(
    df: pd.DataFrame,
    recent: tuple[date, date],
    comparison: tuple[date, date],
    mode: OperatorMode = "lead_only",
    top_n: int = 20,
) -> pd.DataFrame:
    """執刀医ランキング (recent 基準) + comparison 件数を返す。

    `mode="lead_only"`: 1 手術 = 1 行（執刀医）
        返却列: 順位, 医師, 直近件数, 比較件数, 差分, 平均時間_分, 緊急件数
    `mode="all"`: 1 手術 × N 医師（執刀＋助手）
        返却列: 順位, 医師, 直近執刀, 直近助手, 直近合計, 比較合計, 差分
    """
    long_df = expand_operators(df, mode)
    r = _slice_period(long_df, recent)
    c = _slice_period(long_df, comparison)

    if r.empty:
        cols = (
            ["順位", "医師", "直近件数", "比較件数", "差分", "平均時間_分", "緊急件数"]
            if mode == "lead_only"
            else ["順位", "医師", "直近執刀", "直近助手", "直近合計", "比較合計", "差分"]
        )
        return pd.DataFrame(columns=cols)

    if mode == "lead_only":
        g = r.groupby("医師", dropna=False)
        out = pd.DataFrame(
            {
                "直近件数": g.size(),
                "平均時間_分": g["予定手術時間"].mean(),
                "緊急件数": g["申込区分"].apply(lambda s: int((s == "緊急").sum())),
            }
        ).reset_index()
        c_counts = (
            c.groupby("医師", dropna=False).size()
            if not c.empty
            else pd.Series(dtype="int64")
        )
        out["比較件数"] = out["医師"].map(c_counts).fillna(0).astype("int64")
        out["差分"] = out["直近件数"].astype("int64") - out["比較件数"]
        out = out.sort_values("直近件数", ascending=False).head(top_n).reset_index(drop=True)
        out.insert(0, "順位", range(1, len(out) + 1))
        return out[["順位", "医師", "直近件数", "比較件数", "差分", "平均時間_分", "緊急件数"]]

    r_lead = r[r["役割"] == "執刀医"].groupby("医師", dropna=False).size()
    r_assist = r[r["役割"] == "助手"].groupby("医師", dropna=False).size()
    r_total = r.groupby("医師", dropna=False).size()
    c_total = (
        c.groupby("医師", dropna=False).size() if not c.empty else pd.Series(dtype="int64")
    )

    all_doctors = r_total.index
    out = pd.DataFrame(
        {
            "医師": all_doctors,
            "直近執刀": [int(r_lead.get(d, 0)) for d in all_doctors],
            "直近助手": [int(r_assist.get(d, 0)) for d in all_doctors],
            "直近合計": r_total.values.astype("int64"),
            "比較合計": [int(c_total.get(d, 0)) for d in all_doctors],
        }
    )
    out["差分"] = out["直近合計"] - out["比較合計"]
    out = out.sort_values("直近合計", ascending=False).head(top_n).reset_index(drop=True)
    out.insert(0, "順位", range(1, len(out) + 1))
    return out


def category_counts_compare_window(
    df: pd.DataFrame,
    recent: tuple[date, date],
    comparison: tuple[date, date],
) -> pd.DataFrame:
    """4 カテゴリそれぞれの (直近件数, 比較件数, 差分) を返す。

    返却列: カテゴリ, 直近件数, 比較件数, 差分
    """
    r = _slice_period(df, recent)
    c = _slice_period(df, comparison)
    rows = []
    for col in CATEGORY_COLUMNS:
        if col not in df.columns:
            continue
        r_n = int(r[col].sum()) if not r.empty else 0
        c_n = int(c[col].sum()) if not c.empty else 0
        rows.append({"カテゴリ": col, "直近件数": r_n, "比較件数": c_n, "差分": r_n - c_n})
    return pd.DataFrame(rows)
