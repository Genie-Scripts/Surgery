"""src/export_html.py の回帰テスト。

純関数（期間導出 / Plotly figure ビルダー）と HTML 構造の smoke test を兼ねる。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.export_html import (
    PeriodPair,
    derive_default_periods,
    fig_category_bar,
    fig_emergency_ratio_monthly,
    fig_monthly_count,
    render_html,
)


def _make_full_year_df() -> pd.DataFrame:
    """2025-01-01 〜 2025-12-31 の各月に 1 件ずつ存在する 12 行データ。"""
    dates = pd.to_datetime([f"2025-{m:02d}-15" for m in range(1, 13)])
    n = len(dates)
    return pd.DataFrame(
        {
            "手術実施日": dates,
            "実施診療科": ["整形外科"] * n,
            "予定手術時間": [60] * n,
            "申込区分": ["通常"] * n,
            "malignant_tumor": [False] * n,
            "artificial_joint": [True] * n,
            "robot_assisted_davinci": [False] * n,
            "robot_assisted_other": [False] * n,
        }
    )


# --- derive_default_periods --------------------------------------------


def test_derive_default_periods_keys():
    df = _make_full_year_df()
    periods = derive_default_periods(df)
    assert [p.key for p in periods] == ["A", "B", "C"]
    assert all(isinstance(p, PeriodPair) for p in periods)


def test_derive_default_periods_a_is_3months_vs_prior_3months():
    df = _make_full_year_df()  # latest = 2025-12-15 → latest month 2025-12
    a = next(p for p in derive_default_periods(df) if p.key == "A")
    # newer: 2025-10-01 〜 2025-12-31 (3 ヶ月)
    assert a.newer == (date(2025, 10, 1), date(2025, 12, 31))
    # older: 2025-07-01 〜 2025-09-30 (前 3 ヶ月)
    assert a.older == (date(2025, 7, 1), date(2025, 9, 30))


def test_derive_default_periods_b_is_latest_month_vs_year_ago():
    df = _make_full_year_df()
    b = next(p for p in derive_default_periods(df) if p.key == "B")
    # newer: 2025-12-01 〜 2025-12-31
    assert b.newer == (date(2025, 12, 1), date(2025, 12, 31))
    # older: 2024-12-01 〜 2024-12-31 (前年同月)
    assert b.older == (date(2024, 12, 1), date(2024, 12, 31))


def test_derive_default_periods_c_is_6months_vs_prior_6months():
    df = _make_full_year_df()
    c = next(p for p in derive_default_periods(df) if p.key == "C")
    assert c.newer == (date(2025, 7, 1), date(2025, 12, 31))
    assert c.older == (date(2025, 1, 1), date(2025, 6, 30))


def test_derive_default_periods_handles_short_month():
    """30 日月（2 月など）でも月末が正しく取れること。"""
    df = pd.DataFrame(
        {
            "手術実施日": pd.to_datetime(["2024-02-29", "2025-02-28"]),
            "実施診療科": ["整形外科", "整形外科"],
            "予定手術時間": [60, 60],
            "申込区分": ["通常", "通常"],
        }
    )
    b = next(p for p in derive_default_periods(df) if p.key == "B")
    # 最新月は 2025-02、その前年同月は 2024-02 (うるう年)
    assert b.newer == (date(2025, 2, 1), date(2025, 2, 28))
    assert b.older == (date(2024, 2, 1), date(2024, 2, 29))


# --- figure builders smoke -----------------------------------------------


def test_fig_monthly_count_returns_figure_with_one_trace():
    fig = fig_monthly_count(_make_full_year_df())
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [1] * 12  # 各月 1 件


def test_fig_emergency_ratio_handles_missing_column():
    df = _make_full_year_df().drop(columns="申込区分")
    fig = fig_emergency_ratio_monthly(df)
    # データなしフォールバック: trace 0、annotation 1
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1


def test_fig_category_bar_empty_df_falls_back():
    df = pd.DataFrame({"手術実施日": pd.to_datetime([])})
    fig = fig_category_bar(df)
    assert len(fig.data) == 0


# --- render_html smoke ----------------------------------------------------


def test_render_html_includes_all_period_sections():
    df = _make_full_year_df()
    periods = derive_default_periods(df)
    html = render_html(df, periods)

    assert "<title>手術ダッシュボード</title>" in html
    # 3 期間セクション全部 + A だけが active
    assert 'id="period-A" class="period-section active"' in html
    assert 'id="period-B" class="period-section"' in html
    assert 'id="period-C" class="period-section"' in html
    # Plotly CDN が読み込まれている（include_plotlyjs="cdn" ではなくテンプレ側で参照）
    assert "cdn.plot.ly" in html
    # 切替 JS が存在
    assert 'name="period"' in html
    assert "addEventListener" in html


def test_render_html_period_overrides_apply():
    df = _make_full_year_df()
    custom = [
        PeriodPair(
            key="A",
            label="カスタム比較",
            older=(date(2025, 1, 1), date(2025, 3, 31)),
            newer=(date(2025, 10, 1), date(2025, 12, 31)),
        ),
        PeriodPair(
            key="B",
            label="ダミー B",
            older=(date(2025, 1, 1), date(2025, 1, 31)),
            newer=(date(2025, 12, 1), date(2025, 12, 31)),
        ),
        PeriodPair(
            key="C",
            label="ダミー C",
            older=(date(2025, 1, 1), date(2025, 6, 30)),
            newer=(date(2025, 7, 1), date(2025, 12, 31)),
        ),
    ]
    html = render_html(df, custom)
    assert "カスタム比較" in html
    assert "2025-01-01 〜 2025-03-31" in html  # A の older 表示
