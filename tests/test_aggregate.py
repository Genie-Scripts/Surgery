"""src/aggregate.py の回帰テスト。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.aggregate import (
    CATEGORY_COLUMNS,
    category_counts,
    category_counts_period_compare,
    category_monthly_trend,
    expand_operators,
    fiscal_year_periods,
    is_general_anesthesia,
    kpi_overall,
    kpi_overall_period_compare,
    kpi_overall_yoy,
    kpi_per_doctor,
    kpi_per_doctor_compare,
    kpi_per_doctor_yoy,
    month_end_cutoff,
    monthly_count_by_fiscal_year,
    monthly_trend,
    top_n_postop_diagnoses,
    top_n_procedures,
    weekly_general_anesthesia,
)

# --- is_general_anesthesia ----------------------------------------------


def test_ga_canonical_form():
    assert is_general_anesthesia("全身麻酔(20分以上：吸入もしくは静脈麻酔薬)")


def test_ga_separated_keywords():
    assert is_general_anesthesia("閉鎖循環式全身麻酔\n全身麻酔 20分以上")


def test_ga_rejects_disclaimer_text():
    # 単純な「全身麻酔」マッチで誤検出されないこと
    assert not is_general_anesthesia("上下肢の伝達麻酔(全身麻酔時算定不可)")


def test_ga_handles_nan_and_empty():
    assert not is_general_anesthesia(None)
    assert not is_general_anesthesia("")
    assert not is_general_anesthesia(float("nan"))


# --- expand_operators ---------------------------------------------------


def _make_df():
    return pd.DataFrame(
        {
            "手術実施日": pd.to_datetime(["2025-04-01", "2025-04-02", "2025-04-03"]),
            "予定手術時間": [30, 60, 90],
            "申込区分": ["通常", "緊急", "通常"],
            "執刀医": ["医師_001", "医師_002", None],
            "助手リスト": [["医師_002"], ["医師_003", "医師_004"], []],
        }
    )


def test_expand_operators_lead_only():
    out = expand_operators(_make_df(), "lead_only")
    assert len(out) == 2  # NaN 執刀医は落ちる
    assert (out["役割"] == "執刀医").all()
    assert set(out["医師"]) == {"医師_001", "医師_002"}


def test_expand_operators_all():
    out = expand_operators(_make_df(), "all")
    # 1 行目: lead 1 + asst 1 = 2、2 行目: lead 1 + asst 2 = 3、3 行目: 空 = 0 → 計 5
    assert len(out) == 5
    leads = out[out["役割"] == "執刀医"]
    assistants = out[out["役割"] == "助手"]
    assert len(leads) == 2
    assert len(assistants) == 3
    assert set(assistants["医師"]) == {"医師_002", "医師_003", "医師_004"}


# --- kpi_overall --------------------------------------------------------


def test_kpi_overall_basic():
    df = _make_df()
    k = kpi_overall(df)
    assert k["件数"] == 3
    assert k["総手術時間_分"] == 180
    assert k["平均手術時間_分"] == 60.0
    assert k["緊急比率"] == 1 / 3


def test_kpi_overall_empty():
    df = pd.DataFrame({"予定手術時間": [], "申込区分": []})
    k = kpi_overall(df)
    assert k == {"件数": 0, "総手術時間_分": 0, "平均手術時間_分": 0.0, "緊急比率": 0.0}


# --- monthly_trend ------------------------------------------------------


def test_monthly_trend_three_months():
    df = pd.DataFrame(
        {
            "手術実施日": pd.to_datetime(
                ["2025-04-15", "2025-04-20", "2025-05-10", "2025-06-01"]
            ),
            "予定手術時間": [30, 60, 120, 90],
        }
    )
    out = monthly_trend(df)
    assert list(out["手術実施月"].dt.strftime("%Y-%m")) == ["2025-04", "2025-05", "2025-06"]
    assert list(out["件数"]) == [2, 1, 1]
    assert out.loc[0, "平均手術時間_分"] == 45.0
    assert out.loc[0, "総手術時間_分"] == 90


def test_monthly_trend_empty():
    df = pd.DataFrame(columns=["手術実施日", "予定手術時間"])
    out = monthly_trend(df)
    assert out.empty
    assert list(out.columns) == ["手術実施月", "件数", "平均手術時間_分", "総手術時間_分"]


# --- kpi_per_doctor -----------------------------------------------------


def test_kpi_per_doctor_aggregates_by_doctor():
    df_long = pd.DataFrame(
        {
            "医師": ["医師_001", "医師_001", "医師_002"],
            "予定手術時間": [30, 60, 120],
            "申込区分": ["通常", "緊急", "緊急"],
        }
    )
    out = kpi_per_doctor(df_long)
    assert len(out) == 2
    a = out.loc[out["医師"] == "医師_001"].iloc[0]
    assert a["件数"] == 2
    assert a["総手術時間_分"] == 90
    assert a["平均手術時間_分"] == 45.0
    assert a["緊急件数"] == 1


# --- category_counts ----------------------------------------------------


def test_category_counts_only_includes_present_columns():
    df = pd.DataFrame(
        {"malignant_tumor": [True, True, False], "robot_assisted_davinci": [False, True, True]}
    )
    out = category_counts(df)
    assert list(out["カテゴリ"]) == ["malignant_tumor", "robot_assisted_davinci"]
    assert list(out["件数"]) == [2, 2]


# --- category_monthly_trend --------------------------------------------


def _make_categorized_df():
    return pd.DataFrame(
        {
            "手術実施日": pd.to_datetime(
                [
                    "2025-04-01",
                    "2025-04-15",
                    "2025-05-01",
                    "2025-05-20",
                    "2025-06-10",
                ]
            ),
            "malignant_tumor": [True, False, True, True, False],
            "artificial_joint": [False, True, False, False, True],
            "robot_assisted_davinci": [False, False, False, True, False],
            "robot_assisted_other": [False, True, False, False, True],
        }
    )


def test_category_monthly_trend_pivots_by_month():
    out = category_monthly_trend(_make_categorized_df())
    assert list(out["手術実施月"].dt.strftime("%Y-%m")) == [
        "2025-04",
        "2025-05",
        "2025-06",
    ]
    assert list(out["malignant_tumor"]) == [1, 2, 0]
    assert list(out["artificial_joint"]) == [1, 0, 1]
    assert list(out["robot_assisted_davinci"]) == [0, 1, 0]
    assert list(out["robot_assisted_other"]) == [1, 0, 1]


def test_category_monthly_trend_empty_df():
    df = pd.DataFrame(columns=["手術実施日", *CATEGORY_COLUMNS])
    out = category_monthly_trend(df)
    assert out.empty
    assert "手術実施月" in out.columns


def test_category_monthly_trend_no_category_columns():
    df = pd.DataFrame({"手術実施日": pd.to_datetime(["2025-04-01"])})
    out = category_monthly_trend(df)
    assert out.empty
    assert list(out.columns) == ["手術実施月"]


# --- kpi_per_doctor_compare --------------------------------------------


def _make_long_for_compare():
    """期間 A (4-5 月)、期間 B (6-7 月) で 3 名の医師の動きを作る。

    医師_001: A=2件 (30,60分) → B=3件 (40,50,60分)。緊急 A=1, B=0。
    医師_002: A=1件 (90分)   → B=0件
    医師_003: A=0件          → B=2件 (45,50分)。緊急 B=2
    """
    return pd.DataFrame(
        {
            "手術実施日": pd.to_datetime(
                [
                    "2025-04-15", "2025-05-05",  # 医師_001 A 期 2件
                    "2025-06-10", "2025-06-25", "2025-07-01",  # 医師_001 B 期 3件
                    "2025-04-20",  # 医師_002 A 期 1件
                    "2025-07-15", "2025-07-20",  # 医師_003 B 期 2件
                ]
            ),
            "医師": [
                "医師_001", "医師_001",
                "医師_001", "医師_001", "医師_001",
                "医師_002",
                "医師_003", "医師_003",
            ],
            "予定手術時間": [30, 60, 40, 50, 60, 90, 45, 50],
            "申込区分": [
                "緊急", "通常",
                "通常", "通常", "通常",
                "通常",
                "緊急", "緊急",
            ],
        }
    )


def test_kpi_per_doctor_compare_basic():
    out = kpi_per_doctor_compare(
        _make_long_for_compare(),
        period_a=(date(2025, 4, 1), date(2025, 5, 31)),
        period_b=(date(2025, 6, 1), date(2025, 7, 31)),
    )
    out_idx = out.set_index("医師")

    # 医師_001
    a = out_idx.loc["医師_001"]
    assert a["件数_A"] == 2 and a["件数_B"] == 3
    assert a["件数差"] == 1
    assert a["件数比率(%)"] == pytest.approx(50.0)
    assert a["平均手術時間_分_A"] == pytest.approx(45.0)  # (30+60)/2
    assert a["平均手術時間_分_B"] == pytest.approx(50.0)  # (40+50+60)/3
    assert a["平均時間差_分"] == pytest.approx(5.0)
    assert a["緊急件数_A"] == 1 and a["緊急件数_B"] == 0

    # 医師_002: A 期のみ → B=0、件数比率 -100%、B 平均は NaN
    b = out_idx.loc["医師_002"]
    assert b["件数_A"] == 1 and b["件数_B"] == 0
    assert b["件数差"] == -1
    assert b["件数比率(%)"] == pytest.approx(-100.0)
    assert pd.isna(b["平均手術時間_分_B"])

    # 医師_003: B 期のみ → A=0、件数比率は NaN (A=0 ガード)
    c = out_idx.loc["医師_003"]
    assert c["件数_A"] == 0 and c["件数_B"] == 2
    assert c["件数差"] == 2
    assert pd.isna(c["件数比率(%)"])
    assert pd.isna(c["平均手術時間_分_A"])
    assert c["平均手術時間_分_B"] == pytest.approx(47.5)
    assert c["緊急件数_A"] == 0 and c["緊急件数_B"] == 2


def test_kpi_per_doctor_compare_sorted_by_b():
    out = kpi_per_doctor_compare(
        _make_long_for_compare(),
        period_a=(date(2025, 4, 1), date(2025, 5, 31)),
        period_b=(date(2025, 6, 1), date(2025, 7, 31)),
    )
    # B 期降順: 医師_001 (3) > 医師_003 (2) > 医師_002 (0)
    assert list(out["医師"]) == ["医師_001", "医師_003", "医師_002"]


def test_kpi_per_doctor_compare_inclusive_endpoints():
    """期間端の日付も含まれること。"""
    df_long = pd.DataFrame(
        {
            "手術実施日": pd.to_datetime(["2025-04-01", "2025-04-30"]),
            "医師": ["医師_001", "医師_001"],
            "予定手術時間": [30, 60],
            "申込区分": ["通常", "通常"],
        }
    )
    out = kpi_per_doctor_compare(
        df_long,
        period_a=(date(2025, 4, 1), date(2025, 4, 30)),
        period_b=(date(2025, 5, 1), date(2025, 5, 31)),
    )
    a = out.set_index("医師").loc["医師_001"]
    assert a["件数_A"] == 2 and a["件数_B"] == 0


# --- kpi_overall_period_compare / category_counts_period_compare ---------


def _make_period_compare_df():
    """期間 A (1-2 月) で 3 件、期間 B (3-4 月) で 4 件、緊急混在のサンプル。"""
    return pd.DataFrame(
        {
            "手術実施日": pd.to_datetime(
                [
                    "2025-01-15", "2025-02-10", "2025-02-25",  # 期間 A: 3 件
                    "2025-03-05", "2025-03-20", "2025-04-08", "2025-04-30",  # 期間 B: 4 件
                ]
            ),
            "予定手術時間": [60, 90, 120, 30, 60, 90, 120],
            "申込区分": ["通常", "緊急", "通常", "通常", "通常", "緊急", "緊急"],
            "malignant_tumor": [True, False, True, True, True, False, False],
            "artificial_joint": [False, True, False, False, False, True, True],
            "robot_assisted_davinci": [False, False, False, False, False, False, False],
            "robot_assisted_other": [False, False, False, False, False, False, False],
        }
    )


def test_kpi_overall_period_compare_diff_is_b_minus_a():
    df = _make_period_compare_df()
    out = kpi_overall_period_compare(
        df,
        period_a=(date(2025, 1, 1), date(2025, 2, 28)),
        period_b=(date(2025, 3, 1), date(2025, 4, 30)),
    )
    a, b, diff = out["A"], out["B"], out["diff"]

    assert a["件数"] == 3 and b["件数"] == 4
    assert a["総手術時間_分"] == 270 and b["総手術時間_分"] == 300
    # A 緊急 1/3, B 緊急 2/4
    assert a["緊急比率"] == pytest.approx(1 / 3)
    assert b["緊急比率"] == pytest.approx(0.5)
    # diff = B - A
    assert diff["件数"] == 1
    assert diff["総手術時間_分"] == 30


def test_kpi_overall_period_compare_empty_periods_handled():
    df = _make_period_compare_df()
    out = kpi_overall_period_compare(
        df,
        period_a=(date(2024, 1, 1), date(2024, 12, 31)),  # データなし
        period_b=(date(2025, 1, 1), date(2025, 4, 30)),
    )
    assert out["A"]["件数"] == 0
    assert out["A"]["緊急比率"] == 0.0  # n=0 ガード
    assert out["B"]["件数"] == 7
    assert out["diff"]["件数"] == 7


def test_category_counts_period_compare_per_category_diff():
    df = _make_period_compare_df()
    out = category_counts_period_compare(
        df,
        period_a=(date(2025, 1, 1), date(2025, 2, 28)),
        period_b=(date(2025, 3, 1), date(2025, 4, 30)),
    ).set_index("カテゴリ")

    # 悪性腫瘍: A=2 (1/15, 2/25), B=2 (3/5, 3/20) → diff 0
    assert out.loc["malignant_tumor", "件数_A"] == 2
    assert out.loc["malignant_tumor", "件数_B"] == 2
    assert out.loc["malignant_tumor", "件数差"] == 0
    # 人工関節: A=1 (2/10), B=2 (4/8, 4/30) → diff +1
    assert out.loc["artificial_joint", "件数_A"] == 1
    assert out.loc["artificial_joint", "件数_B"] == 2
    assert out.loc["artificial_joint", "件数差"] == 1
    # 全 0 のカテゴリも欠落せず行が含まれる
    assert out.loc["robot_assisted_davinci", "件数_A"] == 0
    assert out.loc["robot_assisted_davinci", "件数_B"] == 0


def test_category_counts_period_compare_inclusive_endpoints():
    """期間端の日付（1/1, 2/28）も含まれること。"""
    df = _make_period_compare_df()
    out = category_counts_period_compare(
        df,
        period_a=(date(2025, 1, 15), date(2025, 2, 25)),  # 端 = サンプル日付
        period_b=(date(2025, 4, 30), date(2025, 4, 30)),
    ).set_index("カテゴリ")
    assert out.loc["malignant_tumor", "件数_A"] == 2  # 1/15 と 2/25 の両端含む
    assert out.loc["artificial_joint", "件数_B"] == 1  # 4/30 ピンポイント


# --- PDF レポート向け（月締め・対前年同期） ------------------------------


def test_month_end_cutoff_mid_month():
    assert month_end_cutoff(date(2026, 5, 27)) == date(2026, 4, 30)


def test_month_end_cutoff_first_day():
    assert month_end_cutoff(date(2026, 5, 1)) == date(2026, 4, 30)


def test_month_end_cutoff_last_day():
    assert month_end_cutoff(date(2026, 5, 31)) == date(2026, 4, 30)


def test_month_end_cutoff_january_returns_prev_year():
    assert month_end_cutoff(date(2026, 1, 15)) == date(2025, 12, 31)


def test_fiscal_year_periods_mid_fy():
    """5/27 の場合: FY=2026, cutoff=4/30, YTD=4/1〜4/30, 昨年=2025/4/1〜2025/4/30。"""
    p = fiscal_year_periods(date(2026, 5, 27))
    assert p.fiscal_year == 2026
    assert p.cutoff == date(2026, 4, 30)
    assert p.ytd == (date(2026, 4, 1), date(2026, 4, 30))
    assert p.last_year == (date(2025, 4, 1), date(2025, 4, 30))


def test_fiscal_year_periods_january_belongs_to_prev_fy():
    """1〜3 月の cutoff は前年度に属する。"""
    p = fiscal_year_periods(date(2026, 2, 15))
    # cutoff = 1/31, FY = 2025
    assert p.cutoff == date(2026, 1, 31)
    assert p.fiscal_year == 2025
    assert p.ytd == (date(2025, 4, 1), date(2026, 1, 31))
    assert p.last_year == (date(2024, 4, 1), date(2025, 1, 31))


def test_fiscal_year_periods_leap_day_safe():
    """2/29 cutoff の前年は 2/28 に丸める。"""
    p = fiscal_year_periods(date(2024, 3, 15))
    # cutoff = 2024/2/29
    assert p.cutoff == date(2024, 2, 29)
    assert p.last_year[1] == date(2023, 2, 28)


def _make_yoy_df():
    """1 診療科想定で 2025-04〜2026-04 にデータを散らした DataFrame。"""
    rows = []
    # 今年度 (2026-04) に 3 件、うち 2 件は全麻
    rows += [
        {
            "手術実施日": pd.Timestamp("2026-04-10"),
            "予定手術時間": 120,
            "申込区分": "通常",
            "麻酔種別": "全身麻酔(20分以上：吸入もしくは静脈麻酔薬)",
            "実施術者": "医A",
            "執刀医": "医A",
            "助手リスト": [],
            "確定術式": "術式X",
            "術後病名": "病名X",
            "malignant_tumor": True,
            "artificial_joint": False,
            "robot_assisted_davinci": False,
            "robot_assisted_other": False,
        },
        {
            "手術実施日": pd.Timestamp("2026-04-15"),
            "予定手術時間": 180,
            "申込区分": "緊急",
            "麻酔種別": "全身麻酔 20分以上",
            "実施術者": "医B",
            "執刀医": "医B",
            "助手リスト": ["医A"],
            "確定術式": "術式Y",
            "術後病名": "病名Y",
            "malignant_tumor": False,
            "artificial_joint": True,
            "robot_assisted_davinci": False,
            "robot_assisted_other": False,
        },
        {
            "手術実施日": pd.Timestamp("2026-04-22"),
            "予定手術時間": 90,
            "申込区分": "通常",
            "麻酔種別": "局所麻酔",
            "実施術者": "医A",
            "執刀医": "医A",
            "助手リスト": [],
            "確定術式": "術式X",
            "術後病名": "病名X",
            "malignant_tumor": True,
            "artificial_joint": False,
            "robot_assisted_davinci": False,
            "robot_assisted_other": False,
        },
    ]
    # 昨年同期 (2025-04) に 2 件、うち 1 件全麻
    rows += [
        {
            "手術実施日": pd.Timestamp("2025-04-12"),
            "予定手術時間": 150,
            "申込区分": "通常",
            "麻酔種別": "全身麻酔(20分以上：吸入もしくは静脈麻酔薬)",
            "実施術者": "医A",
            "執刀医": "医A",
            "助手リスト": [],
            "確定術式": "術式X",
            "術後病名": "病名X",
            "malignant_tumor": False,
            "artificial_joint": True,
            "robot_assisted_davinci": False,
            "robot_assisted_other": False,
        },
        {
            "手術実施日": pd.Timestamp("2025-04-20"),
            "予定手術時間": 60,
            "申込区分": "通常",
            "麻酔種別": "局所麻酔",
            "実施術者": "医B",
            "執刀医": "医B",
            "助手リスト": [],
            "確定術式": "術式Z",
            "術後病名": "病名Z",
            "malignant_tumor": False,
            "artificial_joint": False,
            "robot_assisted_davinci": False,
            "robot_assisted_other": False,
        },
    ]
    return pd.DataFrame(rows)


def test_kpi_overall_yoy_basic():
    df = _make_yoy_df()
    p = fiscal_year_periods(date(2026, 5, 27))
    k = kpi_overall_yoy(df, p)
    assert k["ytd"]["件数"] == 3
    assert k["ytd"]["全麻手術件数"] == 2
    assert k["ytd"]["緊急比率"] == pytest.approx(1 / 3)
    assert k["last_year"]["件数"] == 2
    assert k["last_year"]["全麻手術件数"] == 1
    assert k["diff"]["件数"] == 1
    assert k["diff"]["全麻手術件数"] == 1


def test_monthly_count_by_fiscal_year_all_12_months_present():
    df = _make_yoy_df()
    p = fiscal_year_periods(date(2026, 5, 27))
    out = monthly_count_by_fiscal_year(df, p)
    assert len(out) == 12
    # 4 月 (オフセット 0) のみ件数あり
    apr = out[out["月オフセット"] == 0].iloc[0]
    assert apr["今年度件数"] == 3
    assert apr["昨年度件数"] == 2
    # 他の月は 0
    others = out[out["月オフセット"] != 0]
    assert (others["今年度件数"] == 0).all()
    assert (others["昨年度件数"] == 0).all()


def test_weekly_general_anesthesia_with_target():
    df = _make_yoy_df()
    p = fiscal_year_periods(date(2026, 5, 27))
    wk = weekly_general_anesthesia(df, p.ytd, target=1)
    # 4/1〜4/30 を含む全週が出る（4 月の月曜は 6/13/20/27 と 3/30）
    assert not wk.empty
    assert "達成" in wk.columns
    # 全麻 2 件 (4/10 木, 4/15 水) → 4/6 週と 4/13 週 がそれぞれ 1 件
    total_ga = wk["全麻件数"].sum()
    assert total_ga == 2
    # target=1 で達成週が 2 週ある
    assert wk["達成"].sum() == 2


def test_weekly_general_anesthesia_no_target():
    df = _make_yoy_df()
    p = fiscal_year_periods(date(2026, 5, 27))
    wk = weekly_general_anesthesia(df, p.ytd, target=None)
    assert "達成" not in wk.columns
    assert "目標" not in wk.columns


def test_top_n_procedures_with_yoy():
    df = _make_yoy_df()
    p = fiscal_year_periods(date(2026, 5, 27))
    out = top_n_procedures(df, p, n=5)
    # 今年度: 術式X=2, 術式Y=1 / 昨年同期: 術式X=1, 術式Z=1
    out_by = out.set_index("確定術式")
    assert out_by.loc["術式X", "今年度件数"] == 2
    assert out_by.loc["術式X", "昨年同期件数"] == 1
    assert out_by.loc["術式X", "差分"] == 1
    assert out_by.loc["術式Y", "今年度件数"] == 1
    assert out_by.loc["術式Y", "昨年同期件数"] == 0


def test_top_n_postop_diagnoses_with_yoy():
    df = _make_yoy_df()
    p = fiscal_year_periods(date(2026, 5, 27))
    out = top_n_postop_diagnoses(df, p, n=5)
    out_by = out.set_index("術後病名")
    assert out_by.loc["病名X", "今年度件数"] == 2


def test_kpi_per_doctor_yoy_lead_only():
    df = _make_yoy_df()
    p = fiscal_year_periods(date(2026, 5, 27))
    out = kpi_per_doctor_yoy(df, p, mode="lead_only", top_n=10)
    assert list(out.columns) == [
        "順位", "医師", "今年度件数", "昨年同期件数", "差分", "平均時間_分", "緊急件数",
    ]
    by = out.set_index("医師")
    # 今年度: 医A=2(術式X×2), 医B=1
    assert by.loc["医A", "今年度件数"] == 2
    assert by.loc["医A", "昨年同期件数"] == 1
    assert by.loc["医B", "今年度件数"] == 1
    assert by.loc["医B", "緊急件数"] == 1


def test_kpi_per_doctor_yoy_all_includes_assistants():
    df = _make_yoy_df()
    p = fiscal_year_periods(date(2026, 5, 27))
    out = kpi_per_doctor_yoy(df, p, mode="all", top_n=10)
    assert list(out.columns) == [
        "順位", "医師", "今年執刀", "今年助手", "今年合計", "昨年合計", "差分",
    ]
    by = out.set_index("医師")
    # 医A: 執刀 2 件 + 助手 1 件 = 合計 3
    assert by.loc["医A", "今年執刀"] == 2
    assert by.loc["医A", "今年助手"] == 1
    assert by.loc["医A", "今年合計"] == 3
