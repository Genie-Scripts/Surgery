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
  - fiscal_year_periods(today):         今年度 YTD と昨年同期の (start, end) ペア
  - kpi_overall_yoy(df, periods):       4 項目 KPI の対前年同期
  - monthly_count_by_fiscal_year(...):  月次推移 2 系列（今年度/昨年度）
  - weekly_general_anesthesia(...):     週次 全麻件数（目標達成判定込み）
  - top_n_procedures(...):              主要術式 top N（対前年同期付き）
  - top_n_postop_diagnoses(...):        主要術後病名 top N（同上）
  - kpi_per_doctor_yoy(...):            執刀医ランキング（対前年同期付き）
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import pandas as pd

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
# PDF レポート向け（月締め・対前年同期）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FiscalYearPeriods:
    """今年度 YTD と昨年同期間（同日数）のペア。"""

    fiscal_year: int  # 今年度（4 月開始の年）。例: 2026/4〜2027/3 なら 2026
    cutoff: date  # 集計終端（前月末日）
    ytd: tuple[date, date]  # (4/1/FY, cutoff)
    last_year: tuple[date, date]  # (4/1/(FY-1), 同日数前の終端)


def month_end_cutoff(today: date) -> date:
    """月締め集計の終端 = 当日の前月末日。

    例: today = 2026-05-27 → 2026-04-30
        today = 2026-05-01 → 2026-04-30
        today = 2026-05-31 → 2026-04-30
    """
    first_of_month = today.replace(day=1)
    return first_of_month - timedelta(days=1)


def fiscal_year_periods(today: date) -> FiscalYearPeriods:
    """月締めで今年度 YTD と昨年同期間を導出する。

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
        # 2/29 → 前年は 2/28
        ly_end = cutoff.replace(year=cutoff.year - 1, day=28)
    return FiscalYearPeriods(
        fiscal_year=fiscal_year,
        cutoff=cutoff,
        ytd=(ytd_start, cutoff),
        last_year=(ly_start, ly_end),
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


def kpi_overall_yoy(df: pd.DataFrame, periods: FiscalYearPeriods) -> dict[str, dict[str, float | int]]:
    """今年度 YTD と昨年同期の 4 項目 KPI を返す。

    KPI: 件数 / 平均手術時間_分 / 緊急比率 / 全麻手術件数
    返却: {"ytd": {...}, "last_year": {...}, "diff": {...}}
    """

    def _kpi(slice_df: pd.DataFrame) -> dict[str, float | int]:
        n = len(slice_df)
        times = slice_df["予定手術時間"].dropna() if "予定手術時間" in slice_df.columns else pd.Series(dtype=float)
        return {
            "件数": int(n),
            "平均手術時間_分": float(times.mean()) if len(times) else 0.0,
            "緊急比率": float((slice_df["申込区分"] == "緊急").sum() / n) if n else 0.0,
            "全麻手術件数": general_anesthesia_count(slice_df),
        }

    ytd = _kpi(_slice_period(df, periods.ytd))
    ly = _kpi(_slice_period(df, periods.last_year))
    diff = {k: ytd[k] - ly[k] for k in ytd}
    return {"ytd": ytd, "last_year": ly, "diff": diff}


def monthly_count_by_fiscal_year(
    df: pd.DataFrame, periods: FiscalYearPeriods
) -> pd.DataFrame:
    """月次件数を「今年度 vs 昨年度」の 2 系列で返す（同月並列）。

    返却列: 月オフセット (int, 0=4 月, 1=5 月, ..., 11=3 月),
            月ラベル (str, 'MM 月'), 今年度件数 (int), 昨年度件数 (int)
    """
    cy = _slice_period(df, periods.ytd).copy()
    ly = _slice_period(df, periods.last_year).copy()

    def _monthly(slice_df: pd.DataFrame, start_year: int) -> pd.Series:
        if slice_df.empty:
            return pd.Series(dtype="int64")
        s = slice_df.set_index("手術実施日").resample("MS").size()
        # 月オフセット (4 月=0, ..., 3 月=11) でインデックス
        return s.rename_axis("month").rename(
            index=lambda ts: (ts.month - 4) % 12
        )

    cy_m = _monthly(cy, periods.fiscal_year)
    ly_m = _monthly(ly, periods.fiscal_year - 1)

    months = list(range(12))
    labels = [f"{(i + 3) % 12 + 1} 月" for i in months]
    out = pd.DataFrame(
        {
            "月オフセット": months,
            "月ラベル": labels,
            "今年度件数": [int(cy_m.get(i, 0)) for i in months],
            "昨年度件数": [int(ly_m.get(i, 0)) for i in months],
        }
    )
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

    # ISO 週の月曜起算で集計
    if not ga.empty:
        wk = (
            ga.assign(週開始日=ga["手術実施日"].dt.to_period("W-SUN").dt.start_time)
            .groupby("週開始日")
            .size()
        )
    else:
        wk = pd.Series(dtype="int64")

    # period を含む全週を生成（start の週の月曜 〜 end の週の月曜）
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


def _top_n_with_yoy(
    df: pd.DataFrame,
    column: str,
    periods: FiscalYearPeriods,
    n: int,
) -> pd.DataFrame:
    """指定列の value_counts top N を今年度 YTD ベースで取り、昨年同期件数を併記する。"""
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "今年度件数", "昨年同期件数", "差分"])

    cy = _slice_period(df, periods.ytd)
    ly = _slice_period(df, periods.last_year)

    if cy.empty:
        return pd.DataFrame(columns=[column, "今年度件数", "昨年同期件数", "差分"])

    cy_counts = cy[column].dropna().value_counts()
    if cy_counts.empty:
        return pd.DataFrame(columns=[column, "今年度件数", "昨年同期件数", "差分"])

    top = cy_counts.head(n)
    ly_counts = ly[column].dropna().value_counts() if not ly.empty else pd.Series(dtype="int64")

    out = pd.DataFrame(
        {
            column: top.index,
            "今年度件数": top.values.astype("int64"),
            "昨年同期件数": [int(ly_counts.get(k, 0)) for k in top.index],
        }
    )
    out["差分"] = out["今年度件数"] - out["昨年同期件数"]
    return out.reset_index(drop=True)


def top_n_procedures(df: pd.DataFrame, periods: FiscalYearPeriods, n: int = 10) -> pd.DataFrame:
    """主要術式 top N（今年度 YTD ベース、昨年同期件数を併記）。"""
    return _top_n_with_yoy(df, "確定術式", periods, n)


def top_n_postop_diagnoses(df: pd.DataFrame, periods: FiscalYearPeriods, n: int = 10) -> pd.DataFrame:
    """主要術後病名 top N（今年度 YTD ベース、昨年同期件数を併記）。"""
    return _top_n_with_yoy(df, "術後病名", periods, n)


def kpi_per_doctor_yoy(
    df: pd.DataFrame,
    periods: FiscalYearPeriods,
    mode: OperatorMode = "lead_only",
    top_n: int = 20,
) -> pd.DataFrame:
    """執刀医ランキングを今年度 YTD ベースで、昨年同期件数を併記して返す。

    `mode="lead_only"`: 1 手術 = 1 行（執刀医）
    `mode="all"`: 1 手術 × N 医師（執刀＋助手）

    返却列 (lead_only): 順位, 医師, 今年度件数, 昨年同期件数, 差分, 平均時間_分, 緊急件数
    返却列 (all):       順位, 医師, 今年執刀, 今年助手, 今年合計, 昨年合計, 差分
    """
    long_df = expand_operators(df, mode)
    cy = _slice_period(long_df, periods.ytd)
    ly = _slice_period(long_df, periods.last_year)

    if cy.empty:
        cols = (
            ["順位", "医師", "今年度件数", "昨年同期件数", "差分", "平均時間_分", "緊急件数"]
            if mode == "lead_only"
            else ["順位", "医師", "今年執刀", "今年助手", "今年合計", "昨年合計", "差分"]
        )
        return pd.DataFrame(columns=cols)

    if mode == "lead_only":
        g = cy.groupby("医師", dropna=False)
        cy_kpi = pd.DataFrame(
            {
                "今年度件数": g.size(),
                "平均時間_分": g["予定手術時間"].mean(),
                "緊急件数": g["申込区分"].apply(lambda s: int((s == "緊急").sum())),
            }
        ).reset_index()
        ly_counts = ly.groupby("医師", dropna=False).size() if not ly.empty else pd.Series(dtype="int64")
        cy_kpi["昨年同期件数"] = cy_kpi["医師"].map(ly_counts).fillna(0).astype("int64")
        cy_kpi["差分"] = cy_kpi["今年度件数"].astype("int64") - cy_kpi["昨年同期件数"]
        cy_kpi = cy_kpi.sort_values("今年度件数", ascending=False).head(top_n).reset_index(drop=True)
        cy_kpi.insert(0, "順位", range(1, len(cy_kpi) + 1))
        return cy_kpi[["順位", "医師", "今年度件数", "昨年同期件数", "差分", "平均時間_分", "緊急件数"]]

    # mode == "all"
    cy_lead = (cy[cy["役割"] == "執刀医"].groupby("医師", dropna=False).size()
               if not cy.empty else pd.Series(dtype="int64"))
    cy_assist = (cy[cy["役割"] == "助手"].groupby("医師", dropna=False).size()
                 if not cy.empty else pd.Series(dtype="int64"))
    cy_total = cy.groupby("医師", dropna=False).size() if not cy.empty else pd.Series(dtype="int64")
    ly_total = ly.groupby("医師", dropna=False).size() if not ly.empty else pd.Series(dtype="int64")

    all_doctors = cy_total.index
    out = pd.DataFrame(
        {
            "医師": all_doctors,
            "今年執刀": [int(cy_lead.get(d, 0)) for d in all_doctors],
            "今年助手": [int(cy_assist.get(d, 0)) for d in all_doctors],
            "今年合計": cy_total.values.astype("int64"),
            "昨年合計": [int(ly_total.get(d, 0)) for d in all_doctors],
        }
    )
    out["差分"] = out["今年合計"] - out["昨年合計"]
    out = out.sort_values("今年合計", ascending=False).head(top_n).reset_index(drop=True)
    out.insert(0, "順位", range(1, len(out) + 1))
    return out
