"""src/aggregate.py の回帰テスト。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.aggregate import (
    CATEGORY_COLUMNS,
    category_counts,
    category_monthly_trend,
    expand_operators,
    is_general_anesthesia,
    kpi_overall,
    kpi_per_doctor,
    kpi_per_doctor_compare,
    monthly_trend,
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
