"""KPI 集計（spec.md §3.3）。

入力: `src.classify.classify()` の出力 DataFrame（カテゴリ列が付与済み）
出力: 各種 KPI の集計値（純関数）

関数構成:
  - is_general_anesthesia(s):   OQ-5 解決済みの全身麻酔判定
  - expand_operators(df, mode): 執刀医モード/全術者モードで long-form 化
  - kpi_overall(df):            全体 KPI（件数/総時間/平均時間/緊急比率）
  - monthly_trend(df):          月次推移（手術実施日基準）
  - kpi_per_doctor(df_long):    術者ごとの KPI
  - category_counts(df):        カテゴリ別件数
"""

from __future__ import annotations

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


def category_counts(df: pd.DataFrame) -> pd.DataFrame:
    """カテゴリ列ごとの件数（True 件数）。"""
    rows = []
    for col in CATEGORY_COLUMNS:
        if col in df.columns:
            rows.append({"カテゴリ": col, "件数": int(df[col].sum())})
    return pd.DataFrame(rows)


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
