"""公開用静的 HTML 出力（spec.md §8.2 / 機能 4）。

`data/aggregated/classified.parquet` を入力に、Plotly チャート + KPI カード +
カテゴリ集計 + 期間比較（A/B/C 3 択トグル）をまとめた単一 HTML を書き出す。

依存追加なし: Plotly は既に依存に含まれており、`fig.to_html(full_html=False,
include_plotlyjs=False)` で個別チャートを <div> として埋め込み、本モジュールの
f-string テンプレートでまとめる。Plotly.js 本体は CDN から 1 度だけ読み込む。

公開時の匿名化レベル: 術者軸（執刀医・助手）は出力に一切含めない。診療科×カテゴリ
までの集計に留める。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.aggregate import (
    CATEGORY_COLUMNS,
    category_counts,
    category_counts_period_compare,
    category_monthly_trend,
    kpi_overall,
    kpi_overall_period_compare,
    monthly_trend,
)

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

CATEGORY_LABELS: dict[str, str] = {
    "malignant_tumor": "悪性腫瘍",
    "artificial_joint": "人工関節",
    "robot_assisted_davinci": "ロボット支援(ダヴィンチ系)",
    "robot_assisted_other": "ロボット支援(非ダヴィンチ系)",
}


# ---------------------------------------------------------------------------
# Period derivation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeriodPair:
    """期間比較の 1 ペア。`older` は前期間、`newer` は最新期間。"""

    key: str  # "A" | "B" | "C"
    label: str  # 表示名
    older: tuple[date, date]
    newer: tuple[date, date]

    @property
    def older_str(self) -> str:
        return f"{self.older[0].isoformat()} 〜 {self.older[1].isoformat()}"

    @property
    def newer_str(self) -> str:
        return f"{self.newer[0].isoformat()} 〜 {self.newer[1].isoformat()}"


def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.normalize().replace(day=1)


def _month_offset(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return (ts - pd.DateOffset(months=months)).normalize()


def _month_end_date(ts: pd.Timestamp) -> date:
    return (ts + pd.offsets.MonthEnd(0)).date()


def derive_default_periods(df: pd.DataFrame) -> list[PeriodPair]:
    """データレンジから 3 種類の期間ペア（A/B/C）を導出する。

    基準点は `df["手術実施日"].max()` を含む月（暦月）。
      A: 最新 3 ヶ月 vs その前 3 ヶ月
      B: 最新 1 ヶ月 vs 前年同月
      C: 直近 6 ヶ月 vs その前 6 ヶ月

    全ペアとも (older, newer) で月初〜月末の暦月境界に揃える。
    """
    latest_month = _month_start(df["手術実施日"].max())

    def pair(key: str, label: str, span: int, gap: int = 0) -> PeriodPair:
        # newer: 最新月から遡って span ヶ月
        newer_start = _month_offset(latest_month, span - 1)
        newer_end = _month_end_date(latest_month)
        # older: newer の直前 span ヶ月（gap で間を空けることもできるが既定 0）
        older_end_month = _month_offset(latest_month, span + gap)
        older_start = _month_offset(latest_month, span * 2 - 1 + gap)
        return PeriodPair(
            key=key,
            label=label,
            older=(older_start.date(), _month_end_date(older_end_month)),
            newer=(newer_start.date(), newer_end),
        )

    a = pair("A", "最新 3 ヶ月 vs 前 3 ヶ月", span=3)
    c = pair("C", "直近 6 ヶ月 vs 前 6 ヶ月", span=6)

    # B: 最新 1 ヶ月 vs 前年同月
    b_newer_start = latest_month
    b_newer_end = _month_end_date(latest_month)
    b_older_start = _month_offset(latest_month, 12)
    b_older_end = _month_end_date(b_older_start)
    b = PeriodPair(
        key="B",
        label="最新 1 ヶ月 vs 前年同月",
        older=(b_older_start.date(), b_older_end),
        newer=(b_newer_start.date(), b_newer_end),
    )

    return [a, b, c]


# ---------------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------------


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "#888"},
    )
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=300,
    )
    return fig


def _common_layout(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(
        title=title,
        margin={"l": 50, "r": 30, "t": 50 if title else 20, "b": 40},
        height=height,
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font={"family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(showgrid=False, linecolor="#dee2e6")
    fig.update_yaxes(gridcolor="#f1f3f5", linecolor="#dee2e6")
    return fig


def fig_monthly_count(df: pd.DataFrame) -> go.Figure:
    mt = monthly_trend(df)
    if mt.empty:
        return _empty_fig("該当データなし")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mt["手術実施月"],
            y=mt["件数"],
            mode="lines+markers",
            line={"color": "#0d6efd", "width": 2.5},
            marker={"size": 7},
            name="件数",
        )
    )
    return _common_layout(fig, title="月次件数")


def fig_monthly_avg_time(df: pd.DataFrame) -> go.Figure:
    mt = monthly_trend(df)
    if mt.empty:
        return _empty_fig("該当データなし")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mt["手術実施月"],
            y=mt["平均手術時間_分"],
            mode="lines+markers",
            line={"color": "#fd7e14", "width": 2.5},
            marker={"size": 7},
            name="平均手術時間 (分)",
        )
    )
    return _common_layout(fig, title="月次 平均手術時間 (分)")


def fig_category_monthly(df: pd.DataFrame) -> go.Figure:
    cmt = category_monthly_trend(df)
    if cmt.empty or len(cmt.columns) <= 1:
        return _empty_fig("該当データなし")
    fig = go.Figure()
    palette = ["#0d6efd", "#198754", "#dc3545", "#6f42c1"]
    for i, col in enumerate(c for c in CATEGORY_COLUMNS if c in cmt.columns):
        fig.add_trace(
            go.Scatter(
                x=cmt["手術実施月"],
                y=cmt[col],
                mode="lines+markers",
                name=CATEGORY_LABELS.get(col, col),
                line={"color": palette[i % len(palette)], "width": 2},
                marker={"size": 6},
            )
        )
    return _common_layout(fig, title="カテゴリ別 月次件数", height=400)


def fig_emergency_ratio_monthly(df: pd.DataFrame) -> go.Figure:
    if df.empty or "申込区分" not in df.columns:
        return _empty_fig("該当データなし")
    g = df.set_index("手術実施日").resample("MS")
    monthly = pd.DataFrame(
        {
            "総件数": g.size(),
            "緊急件数": g["申込区分"].apply(lambda s: int((s == "緊急").sum())),
        }
    )
    monthly["緊急比率"] = monthly["緊急件数"] / monthly["総件数"].replace(0, pd.NA) * 100
    monthly = monthly.reset_index().rename(columns={"手術実施日": "手術実施月"})

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["手術実施月"],
            y=monthly["緊急比率"],
            mode="lines+markers",
            line={"color": "#dc3545", "width": 2.5},
            marker={"size": 7},
            name="緊急比率 (%)",
        )
    )
    return _common_layout(fig, title="月次 緊急比率 (%)")


def fig_dept_monthly_top5(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("該当データなし")
    top5 = df["実施診療科"].value_counts().head(5).index.tolist()
    if not top5:
        return _empty_fig("診療科の集計対象なし")

    df_top = df[df["実施診療科"].isin(top5)]
    pivot = (
        df_top.assign(月=df_top["手術実施日"].dt.to_period("M").dt.to_timestamp())
        .groupby(["月", "実施診療科"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=top5)
    )

    fig = go.Figure()
    palette = ["#0d6efd", "#198754", "#dc3545", "#fd7e14", "#6f42c1"]
    for i, dept in enumerate(top5):
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[dept],
                mode="lines+markers",
                name=str(dept),
                line={"color": palette[i % len(palette)], "width": 2},
                marker={"size": 6},
            )
        )
    return _common_layout(fig, title="診療科別 月次件数（上位 5 科）", height=400)


def fig_category_bar(df: pd.DataFrame) -> go.Figure:
    cat = category_counts(df)
    if cat.empty:
        return _empty_fig("該当データなし")
    cat = cat.assign(label=cat["カテゴリ"].map(CATEGORY_LABELS).fillna(cat["カテゴリ"]))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cat["label"],
            y=cat["件数"],
            marker_color="#0d6efd",
            text=cat["件数"],
            textposition="outside",
        )
    )
    return _common_layout(fig, title="カテゴリ別 件数")


def fig_period_compare_categories(df: pd.DataFrame, period: PeriodPair) -> go.Figure:
    cmp_df = category_counts_period_compare(df, period.older, period.newer)
    if cmp_df.empty:
        return _empty_fig("該当データなし")
    cmp_df = cmp_df.assign(
        label=cmp_df["カテゴリ"].map(CATEGORY_LABELS).fillna(cmp_df["カテゴリ"])
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cmp_df["label"],
            y=cmp_df["件数_A"],
            name=f"前期間 ({period.older_str})",
            marker_color="#adb5bd",
        )
    )
    fig.add_trace(
        go.Bar(
            x=cmp_df["label"],
            y=cmp_df["件数_B"],
            name=f"最新期間 ({period.newer_str})",
            marker_color="#0d6efd",
        )
    )
    fig.update_layout(barmode="group")
    return _common_layout(fig, title="カテゴリ別 件数比較", height=380)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _fig_div(fig: go.Figure, div_id: str) -> str:
    """Plotly Figure を <div> として埋め込み可能な HTML 文字列に変換する。"""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"displaylogo": False, "responsive": True},
    )


def _kpi_card(label: str, value: str, sub: str | None = None) -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {sub_html}
    </div>
    """


def _format_diff_count(diff: int) -> str:
    sign = "+" if diff >= 0 else ""
    cls = "kpi-diff-pos" if diff >= 0 else "kpi-diff-neg"
    return f'<span class="{cls}">{sign}{diff:,}</span>'


def _format_diff_minutes(diff: float) -> str:
    sign = "+" if diff >= 0 else ""
    cls = "kpi-diff-pos" if diff >= 0 else "kpi-diff-neg"
    return f'<span class="{cls}">{sign}{diff:.1f}</span>'


def _format_diff_pct(diff: float) -> str:
    sign = "+" if diff >= 0 else ""
    # 緊急比率の差分: 上昇が必ずしも良いとは限らないが、UI では上下で色分けするだけにする
    cls = "kpi-diff-pos" if diff >= 0 else "kpi-diff-neg"
    return f'<span class="{cls}">{sign}{diff * 100:.2f} pt</span>'


def _build_period_section(df: pd.DataFrame, period: PeriodPair) -> str:
    """1 期間ペア分の比較 HTML（KPI カード × 4 + カテゴリ別バーチャート + テーブル）。"""
    cmp_kpi = kpi_overall_period_compare(df, period.older, period.newer)
    a, b, diff = cmp_kpi["A"], cmp_kpi["B"], cmp_kpi["diff"]

    def _card(label: str, a_val: str, b_val: str, diff_html: str) -> str:
        return f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <table class="kpi-compare">
            <tr><td>前期間</td><td class="num">{a_val}</td></tr>
            <tr><td>最新期間</td><td class="num">{b_val}</td></tr>
            <tr><td>差分</td><td class="num">{diff_html}</td></tr>
          </table>
        </div>
        """

    cards = "\n".join(
        [
            _card(
                "件数",
                f"{a['件数']:,}",
                f"{b['件数']:,}",
                _format_diff_count(int(diff["件数"])),
            ),
            _card(
                "総手術時間 (時間)",
                f"{a['総手術時間_分'] / 60:,.1f}",
                f"{b['総手術時間_分'] / 60:,.1f}",
                _format_diff_minutes(diff["総手術時間_分"] / 60),
            ),
            _card(
                "平均手術時間 (分)",
                f"{a['平均手術時間_分']:.1f}",
                f"{b['平均手術時間_分']:.1f}",
                _format_diff_minutes(diff["平均手術時間_分"]),
            ),
            _card(
                "緊急比率",
                f"{a['緊急比率'] * 100:.1f}%",
                f"{b['緊急比率'] * 100:.1f}%",
                _format_diff_pct(diff["緊急比率"]),
            ),
        ]
    )

    cat_cmp = category_counts_period_compare(df, period.older, period.newer)
    cat_cmp = cat_cmp.assign(
        label=cat_cmp["カテゴリ"].map(CATEGORY_LABELS).fillna(cat_cmp["カテゴリ"])
    )
    rows = "\n".join(
        f"<tr><td>{r['label']}</td><td class='num'>{int(r['件数_A']):,}</td>"
        f"<td class='num'>{int(r['件数_B']):,}</td>"
        f"<td class='num'>{_format_diff_count(int(r['件数差']))}</td></tr>"
        for _, r in cat_cmp.iterrows()
    )
    cat_table = f"""
    <table class="cmp-table">
      <thead>
        <tr><th>カテゴリ</th><th class="num">前期間</th><th class="num">最新期間</th><th class="num">差分</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    """

    cat_chart = _fig_div(fig_period_compare_categories(df, period), f"chart-cmp-{period.key}")

    return f"""
    <div id="period-{period.key}" class="period-section{' active' if period.key == 'A' else ''}">
      <p class="period-info">
        <span class="period-info-label">前期間</span> {period.older_str}
        <span class="period-info-label">最新期間</span> {period.newer_str}
      </p>
      <div class="kpi-grid">
        {cards}
      </div>
      <div class="period-cat-row">
        <div class="period-cat-chart">{cat_chart}</div>
        <div class="period-cat-table">{cat_table}</div>
      </div>
    </div>
    """


def _build_category_cross_table(df: pd.DataFrame) -> str:
    available = [c for c in CATEGORY_COLUMNS if c in df.columns]
    if not available or df.empty:
        return '<p class="muted">該当データなし</p>'
    cross = (
        df.groupby("実施診療科")[available]
        .sum()
        .astype("int64")
        .rename(columns=CATEGORY_LABELS)
    )
    cross["合計"] = cross.sum(axis=1)
    cross = cross.sort_values("合計", ascending=False)
    cross = cross[cross["合計"] > 0].drop(columns="合計")
    if cross.empty:
        return '<p class="muted">カテゴリ該当の診療科なし</p>'

    headers = "".join(f"<th class='num'>{c}</th>" for c in cross.columns)
    rows = "\n".join(
        "<tr><td>{dept}</td>{cells}</tr>".format(
            dept=dept,
            cells="".join(f"<td class='num'>{int(v):,}</td>" for v in row),
        )
        for dept, row in cross.iterrows()
    )
    return f"""
    <table class="cross-table">
      <thead><tr><th>診療科</th>{headers}</tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """


def _build_category_summary_table(df: pd.DataFrame) -> str:
    cat = category_counts(df)
    if cat.empty:
        return '<p class="muted">該当データなし</p>'
    n = max(len(df), 1)
    cat = cat.assign(
        label=cat["カテゴリ"].map(CATEGORY_LABELS).fillna(cat["カテゴリ"]),
        ratio=(cat["件数"] / n * 100).round(1),
    )
    rows = "\n".join(
        f"<tr><td>{r['label']}</td><td class='num'>{int(r['件数']):,}</td>"
        f"<td class='num'>{r['ratio']:.1f}%</td></tr>"
        for _, r in cat.iterrows()
    )
    return f"""
    <table class="cmp-table">
      <thead><tr><th>カテゴリ</th><th class="num">件数</th><th class="num">構成比</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """


def render_html(df: pd.DataFrame, periods: list[PeriodPair]) -> str:
    """ダッシュボード HTML を組み立てて返す。"""
    k = kpi_overall(df)
    date_min = df["手術実施日"].min().date().isoformat()
    date_max = df["手術実施日"].max().date().isoformat()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    kpi_cards = "\n".join(
        [
            _kpi_card("件数", f"{k['件数']:,}"),
            _kpi_card("総手術時間 (時間)", f"{k['総手術時間_分'] / 60:,.1f}"),
            _kpi_card("平均手術時間 (分)", f"{k['平均手術時間_分']:.1f}"),
            _kpi_card("緊急比率", f"{k['緊急比率'] * 100:.1f}%"),
        ]
    )

    chart_count = _fig_div(fig_monthly_count(df), "chart-monthly-count")
    chart_avg = _fig_div(fig_monthly_avg_time(df), "chart-monthly-avg")
    chart_cat_monthly = _fig_div(fig_category_monthly(df), "chart-cat-monthly")
    chart_emerg = _fig_div(fig_emergency_ratio_monthly(df), "chart-emerg-monthly")
    chart_dept = _fig_div(fig_dept_monthly_top5(df), "chart-dept-monthly")
    chart_cat_bar = _fig_div(fig_category_bar(df), "chart-cat-bar")

    cat_summary = _build_category_summary_table(df)
    cat_cross = _build_category_cross_table(df)

    period_radios = "\n".join(
        f'<label><input type="radio" name="period" value="{p.key}"'
        f'{" checked" if p.key == "A" else ""}> {p.label}</label>'
        for p in periods
    )
    period_sections = "\n".join(_build_period_section(df, p) for p in periods)

    return _HTML_TEMPLATE.format(
        plotly_cdn=PLOTLY_CDN,
        date_min=date_min,
        date_max=date_max,
        n_total=f"{k['件数']:,}",
        generated_at=generated_at,
        kpi_cards=kpi_cards,
        chart_count=chart_count,
        chart_avg=chart_avg,
        chart_cat_monthly=chart_cat_monthly,
        chart_emerg=chart_emerg,
        chart_dept=chart_dept,
        chart_cat_bar=chart_cat_bar,
        cat_summary=cat_summary,
        cat_cross=cat_cross,
        period_radios=period_radios,
        period_sections=period_sections,
    )


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>手術ダッシュボード</title>
<script src="{plotly_cdn}"></script>
<style>
  :root {{
    --primary: #0d6efd;
    --text: #1a1a1a;
    --muted: #6c757d;
    --border: #dee2e6;
    --bg-soft: #f8f9fa;
    --pos: #198754;
    --neg: #dc3545;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Sans", "Yu Gothic", sans-serif;
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px;
    color: var(--text);
    background: #fff;
    line-height: 1.5;
  }}
  h1 {{
    font-size: 28px;
    margin: 0 0 4px;
    border-bottom: 2px solid var(--primary);
    padding-bottom: 12px;
  }}
  h2 {{
    font-size: 20px;
    margin: 48px 0 16px;
    border-left: 4px solid var(--primary);
    padding-left: 12px;
  }}
  .meta {{ color: var(--muted); font-size: 13px; margin: 8px 0 0; }}
  .muted {{ color: var(--muted); }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 16px 0;
  }}
  .kpi-card {{
    background: var(--bg-soft);
    border-radius: 8px;
    padding: 16px;
    border-left: 4px solid var(--primary);
  }}
  .kpi-label {{ color: var(--muted); font-size: 13px; }}
  .kpi-value {{ font-size: 28px; font-weight: 600; margin-top: 4px; }}
  .kpi-sub {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  .kpi-compare {{ width: 100%; margin-top: 8px; font-size: 14px; }}
  .kpi-compare td {{ padding: 4px 0; }}
  .kpi-compare td:first-child {{ color: var(--muted); }}
  .kpi-compare td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .kpi-diff-pos {{ color: var(--pos); font-weight: 600; }}
  .kpi-diff-neg {{ color: var(--neg); font-weight: 600; }}
  .chart-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 16px 0;
  }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ background: var(--bg-soft); font-weight: 600; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .cmp-table, .cross-table {{ margin: 12px 0; }}
  .cross-table td:first-child {{ font-weight: 500; }}
  .period-toggle {{
    display: flex;
    gap: 20px;
    align-items: center;
    margin: 16px 0 24px;
    padding: 12px 16px;
    background: var(--bg-soft);
    border-radius: 8px;
    flex-wrap: wrap;
  }}
  .period-toggle strong {{ color: var(--muted); font-size: 13px; }}
  .period-toggle label {{ cursor: pointer; user-select: none; }}
  .period-toggle input[type="radio"] {{ margin-right: 4px; }}
  .period-section {{ display: none; }}
  .period-section.active {{ display: block; }}
  .period-info {{
    color: var(--muted);
    font-size: 13px;
    margin: 8px 0 16px;
    padding: 8px 12px;
    background: var(--bg-soft);
    border-radius: 6px;
  }}
  .period-info-label {{
    display: inline-block;
    margin-right: 6px;
    color: var(--text);
    font-weight: 600;
  }}
  .period-cat-row {{
    display: grid;
    grid-template-columns: 3fr 2fr;
    gap: 16px;
    margin-top: 16px;
  }}
  @media (max-width: 900px) {{
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .chart-row, .period-cat-row {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<h1>手術ダッシュボード</h1>
<p class="meta">
  対象期間: {date_min} 〜 {date_max}　／　総件数: {n_total}　／　生成日時: {generated_at}
</p>

<h2>全体 KPI</h2>
<div class="kpi-grid">
{kpi_cards}
</div>

<h2>月次推移</h2>
<div class="chart-row">
  {chart_count}
  {chart_avg}
</div>
{chart_cat_monthly}
<div class="chart-row">
  {chart_emerg}
  {chart_dept}
</div>

<h2>カテゴリ別</h2>
<div class="chart-row">
  {chart_cat_bar}
  <div>{cat_summary}</div>
</div>
<h3 style="font-size:16px;margin:24px 0 8px;color:var(--muted);">カテゴリ × 診療科</h3>
{cat_cross}

<h2>期間比較</h2>
<div class="period-toggle">
  <strong>比較期間</strong>
  {period_radios}
</div>
{period_sections}

<script>
  document.querySelectorAll('input[name="period"]').forEach(function(radio) {{
    radio.addEventListener('change', function(e) {{
      document.querySelectorAll('.period-section').forEach(function(s) {{
        s.classList.remove('active');
      }});
      document.getElementById('period-' + e.target.value).classList.add('active');
      // 切替時に Plotly に resize を要求（display:none → block で寸法 0 のままになる対策）
      if (window.Plotly) {{
        document.querySelectorAll('#period-' + e.target.value + ' .js-plotly-plot').forEach(function(d) {{
          window.Plotly.Plots.resize(d);
        }});
      }}
    }});
  }});
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------


def export(
    parquet_path: Path,
    output_path: Path,
    period_overrides: list[PeriodPair] | None = None,
) -> Path:
    """parquet を読み込み、HTML を出力する。

    `period_overrides` が None の場合は `derive_default_periods` で A/B/C を導出する。
    """
    df = pd.read_parquet(parquet_path)
    periods = period_overrides or derive_default_periods(df)
    html = render_html(df, periods)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
