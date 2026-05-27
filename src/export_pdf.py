"""診療科別 PDF レポート出力。

ローカル実名版で `data/aggregated/classified.parquet`（実名 parquet）を入力に、
診療科ごとに 4 ページ A4 縦の PDF を `local/reports/YYYYMMDD/` 配下へ書き出す。

設計（spec 詰めの結果）:
  - 集計は月締め: 集計終端 = 当日の前月末日
  - すべての KPI を「今年度 YTD vs 昨年同期」の対比で表示（数字単体では意味薄い）
  - Page 2 = 週次 全麻手術件数 vs 目標（診療科 KPI として正式設定）
  - 目標は `config/department_targets.yaml` に保持（未設定の科は実績のみ）
  - 件数 < 30 の診療科はスキップ

実装:
  Plotly → PNG (kaleido) → base64 埋め込み → WeasyPrint で HTML→PDF
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import yaml

from src.aggregate import (
    CATEGORY_COLUMNS,
    ReportPeriods,
    category_counts_compare_window,
    kpi_overall_compare,
    kpi_per_doctor_compare_window,
    monthly_avg_time_compare,
    monthly_category_compare,
    monthly_count_compare,
    monthly_general_anesthesia_compare,
    report_periods,
    top_n_postop_diagnoses,
    top_n_procedures,
    weekly_general_anesthesia,
)

logger = logging.getLogger(__name__)

# 件数閾値: これ未満の診療科は出力しない
MIN_CASE_COUNT = 30

CATEGORY_LABELS: dict[str, str] = {
    "malignant_tumor": "悪性腫瘍",
    "artificial_joint": "人工関節",
    "robot_assisted_davinci": "ロボット支援(ダヴィンチ系)",
    "robot_assisted_other": "ロボット支援(非ダヴィンチ系)",
}

# 配色: アクセント青 + グレースケール
COLOR_PRIMARY = "#0d6efd"  # 直近側
COLOR_PRIMARY_LIGHT = "#9ec5fe"
COLOR_COMPARE = "#adb5bd"  # 比較側 (前期 / 昨年同期)
COLOR_TEXT = "#212529"
COLOR_MUTED = "#6c757d"
COLOR_OK = "#198754"
COLOR_NG = "#dc3545"
COLOR_GRID = "#e9ecef"


# ---------------------------------------------------------------------------
# 目標値設定の読込
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeptTarget:
    """診療科の週あたり全身麻酔手術件数 目標。"""

    weekly_general_anesthesia: int
    effective_from: date | None


def load_dept_targets(path: Path) -> dict[str, DeptTarget]:
    """`config/department_targets.yaml` を読み込んで {診療科: DeptTarget} を返す。

    ファイルが存在しない場合は空辞書を返す（目標未設定で実績のみ描画）。
    """
    if not path.exists():
        logger.warning("目標値ファイルが見つかりません: %s（実績のみで描画）", path)
        return {}

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, DeptTarget] = {}
    for dept, cfg in raw.items():
        if not isinstance(cfg, dict) or "weekly_general_anesthesia" not in cfg:
            logger.warning("目標値が不完全な診療科をスキップ: %s", dept)
            continue
        eff = cfg.get("effective_from")
        if isinstance(eff, str):
            eff = date.fromisoformat(eff)
        out[dept] = DeptTarget(
            weekly_general_anesthesia=int(cfg["weekly_general_anesthesia"]),
            effective_from=eff if isinstance(eff, date) else None,
        )
    return out


# ---------------------------------------------------------------------------
# Plotly チャート（PNG 出力）
# ---------------------------------------------------------------------------


def _common_layout(fig: go.Figure, title: str | None, height: int = 280) -> go.Figure:
    fig.update_layout(
        title=({"text": title, "font": {"size": 13, "color": COLOR_TEXT}} if title else None),
        margin={"l": 50, "r": 20, "t": 36 if title else 12, "b": 36},
        height=height,
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font={"family": "Hiragino Sans, Yu Gothic, sans-serif", "size": 11, "color": COLOR_TEXT},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0,
                "font": {"size": 10}},
        showlegend=True,
    )
    fig.update_xaxes(showgrid=False, linecolor=COLOR_GRID, tickfont={"size": 10})
    fig.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_GRID, tickfont={"size": 10},
                     rangemode="tozero")
    return fig


def _fig_to_data_uri(fig: go.Figure, width: int = 900, height: int = 280) -> str:
    """Plotly Figure を PNG (base64 data URI) に変換する。"""
    png = fig.to_image(format="png", width=width, height=height, scale=2)
    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _fig_monthly_count_rolling(df_dept: pd.DataFrame, periods: ReportPeriods) -> go.Figure:
    mt = monthly_count_compare(df_dept, periods.recent_12mo, periods.prior_12mo)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mt["月ラベル"], y=mt["直近"], mode="lines+markers",
            name="直近12ヶ月",
            line={"color": COLOR_PRIMARY, "width": 2.5}, marker={"size": 7},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mt["月ラベル"], y=mt["前期"], mode="lines+markers",
            name="その前12ヶ月",
            line={"color": COLOR_COMPARE, "width": 2, "dash": "dot"}, marker={"size": 6},
        )
    )
    return _common_layout(fig, title="月次 件数（直近12ヶ月 vs その前12ヶ月）")


def _fig_monthly_avg_time_rolling(df_dept: pd.DataFrame, periods: ReportPeriods) -> go.Figure:
    mt = monthly_avg_time_compare(df_dept, periods.recent_12mo, periods.prior_12mo)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=mt["月ラベル"], y=mt["直近"], mode="lines+markers",
                   name="直近12ヶ月",
                   line={"color": COLOR_PRIMARY, "width": 2.5}, marker={"size": 7},
                   connectgaps=False)
    )
    fig.add_trace(
        go.Scatter(x=mt["月ラベル"], y=mt["前期"], mode="lines+markers",
                   name="その前12ヶ月",
                   line={"color": COLOR_COMPARE, "width": 2, "dash": "dot"}, marker={"size": 6},
                   connectgaps=False)
    )
    return _common_layout(fig, title="月次 平均手術時間 (分)")


def _fig_weekly_ga_vs_target(
    df_dept: pd.DataFrame, periods: ReportPeriods, target: int | None
) -> go.Figure:
    wk = weekly_general_anesthesia(df_dept, periods.weekly_12w, target=target)
    if wk.empty:
        fig = go.Figure()
        fig.add_annotation(text="直近12週データなし", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font={"size": 13, "color": COLOR_MUTED})
        fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, height=240)
        return fig

    if target is not None:
        bar_colors = [COLOR_PRIMARY if v >= target else COLOR_COMPARE for v in wk["全麻件数"]]
    else:
        bar_colors = [COLOR_PRIMARY] * len(wk)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=wk["週ラベル"], y=wk["全麻件数"], marker_color=bar_colors,
               text=wk["全麻件数"], textposition="outside",
               name="実績")
    )
    if target is not None:
        fig.add_hline(y=target, line={"color": COLOR_NG, "width": 2, "dash": "dash"},
                      annotation_text=f"目標 {target} 件/週",
                      annotation_position="top right",
                      annotation_font={"color": COLOR_NG, "size": 10})

    title = "週次 全麻手術件数 vs 目標（直近12週）" if target is not None \
        else "週次 全麻手術件数（直近12週・目標未設定）"
    fig = _common_layout(fig, title=title, height=300)
    fig.update_layout(showlegend=False)
    return fig


def _fig_monthly_ga_rolling(df_dept: pd.DataFrame, periods: ReportPeriods) -> go.Figure:
    mt = monthly_general_anesthesia_compare(df_dept, periods.recent_12mo, periods.prior_12mo)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=mt["月ラベル"], y=mt["直近"],
                   mode="lines+markers", name="直近12ヶ月",
                   line={"color": COLOR_PRIMARY, "width": 2.5}, marker={"size": 7})
    )
    fig.add_trace(
        go.Scatter(x=mt["月ラベル"], y=mt["前期"],
                   mode="lines+markers", name="その前12ヶ月",
                   line={"color": COLOR_COMPARE, "width": 2, "dash": "dot"}, marker={"size": 6})
    )
    return _common_layout(fig, title="月次 全麻手術件数（直近12ヶ月 vs その前12ヶ月）")


def _fig_category_bar_3mo(df_dept: pd.DataFrame, periods: ReportPeriods) -> go.Figure:
    cc = category_counts_compare_window(df_dept, periods.recent_3mo, periods.prior_3mo)
    cats = cc["カテゴリ"].tolist() if not cc.empty else []
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    recent_vals = cc["直近件数"].tolist() if not cc.empty else []
    prior_vals = cc["比較件数"].tolist() if not cc.empty else []

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=prior_vals, name="その前3ヶ月",
                         marker_color=COLOR_COMPARE, text=prior_vals, textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=recent_vals, name="直近3ヶ月",
                         marker_color=COLOR_PRIMARY, text=recent_vals, textposition="outside"))
    fig.update_layout(barmode="group")
    return _common_layout(fig, title="カテゴリ別 件数（直近3ヶ月 vs その前3ヶ月）")


def _fig_category_monthly_rolling(df_dept: pd.DataFrame, periods: ReportPeriods) -> go.Figure:
    """4 カテゴリの月次推移を 1 枚に subplot (2x2)。"""
    from plotly.subplots import make_subplots  # noqa: PLC0415

    cats = [c for c in CATEGORY_COLUMNS if c in df_dept.columns]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[CATEGORY_LABELS.get(c, c) for c in cats[:4]],
        vertical_spacing=0.18, horizontal_spacing=0.10,
    )

    for i, cat in enumerate(cats[:4]):
        row, col = i // 2 + 1, i % 2 + 1
        mt = monthly_category_compare(df_dept, periods.recent_12mo, periods.prior_12mo, cat)
        fig.add_trace(
            go.Scatter(x=mt["月ラベル"], y=mt["直近"],
                       mode="lines+markers", line={"color": COLOR_PRIMARY, "width": 2},
                       marker={"size": 5}, showlegend=(i == 0), name="直近12ヶ月"),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(x=mt["月ラベル"], y=mt["前期"],
                       mode="lines+markers",
                       line={"color": COLOR_COMPARE, "width": 1.5, "dash": "dot"},
                       marker={"size": 5}, showlegend=(i == 0), name="その前12ヶ月"),
            row=row, col=col,
        )

    fig.update_annotations(font_size=11)
    fig.update_layout(
        height=440,
        margin={"l": 40, "r": 20, "t": 40, "b": 30},
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        font={"family": "Hiragino Sans, Yu Gothic, sans-serif", "size": 10, "color": COLOR_TEXT},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "xanchor": "left", "x": 0,
                "font": {"size": 10}},
    )
    fig.update_xaxes(showgrid=False, linecolor=COLOR_GRID, tickfont={"size": 9})
    fig.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_GRID, tickfont={"size": 9},
                     rangemode="tozero")
    return fig


# ---------------------------------------------------------------------------
# HTML パーツ
# ---------------------------------------------------------------------------


def _fmt_int(v: float | int) -> str:
    return f"{int(v):,}"


def _fmt_float(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}"


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _diff_span(diff: float | int, fmt: str = "int", unit: str = "") -> str:
    if isinstance(diff, float) and pd.isna(diff):
        return '<span class="muted">-</span>'
    cls = "diff-pos" if diff >= 0 else "diff-neg"
    sign = "+" if diff > 0 else ("±" if diff == 0 else "")
    if fmt == "int":
        body = f"{sign}{int(diff):,}"
    elif fmt == "minute":
        body = f"{sign}{diff:.1f}"
    elif fmt == "pct_point":
        body = f"{sign}{diff * 100:.1f}pt"
    else:
        body = f"{sign}{diff}"
    return f'<span class="{cls}">{body}{unit}</span>'


def _kpi_card(label: str, comp_label: str, recent_label: str,
              comp: str, recent: str, diff_html: str) -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <table class="kpi-yoy">
        <tr><td>{comp_label}</td><td class="num">{comp}</td></tr>
        <tr><td>{recent_label}</td><td class="num"><strong>{recent}</strong></td></tr>
        <tr><td>差分</td><td class="num">{diff_html}</td></tr>
      </table>
    </div>
    """


def _build_kpi_cards(kpi: dict) -> str:
    """KPI 4 枚 (直近3ヶ月 vs 昨年同3ヶ月)。`kpi` は kpi_overall_compare の戻り値。"""
    comp = kpi["comparison"]
    recent = kpi["recent"]
    diff = kpi["diff"]
    comp_label = "昨年同3ヶ月"
    recent_label = "直近3ヶ月"
    return (
        _kpi_card("件数", comp_label, recent_label,
                  _fmt_int(comp["件数"]), _fmt_int(recent["件数"]),
                  _diff_span(diff["件数"], "int"))
        + _kpi_card("平均手術時間 (分)", comp_label, recent_label,
                    _fmt_float(comp["平均手術時間_分"]),
                    _fmt_float(recent["平均手術時間_分"]),
                    _diff_span(diff["平均手術時間_分"], "minute"))
        + _kpi_card("緊急比率", comp_label, recent_label,
                    _fmt_pct(comp["緊急比率"]), _fmt_pct(recent["緊急比率"]),
                    _diff_span(diff["緊急比率"], "pct_point"))
        + _kpi_card("全麻手術件数", comp_label, recent_label,
                    _fmt_int(comp["全麻手術件数"]),
                    _fmt_int(recent["全麻手術件数"]),
                    _diff_span(diff["全麻手術件数"], "int"))
    )


def _build_doctor_table_lead(kpi_df: pd.DataFrame) -> str:
    if kpi_df.empty:
        return '<p class="muted">該当データなし</p>'
    rows = "\n".join(
        f"<tr><td class='num'>{int(r['順位'])}</td><td>{r['医師']}</td>"
        f"<td class='num'>{int(r['直近件数']):,}</td>"
        f"<td class='num'>{int(r['比較件数']):,}</td>"
        f"<td class='num'>{_diff_span(int(r['差分']), 'int')}</td>"
        f"<td class='num'>{r['平均時間_分']:.1f}</td>"
        f"<td class='num'>{int(r['緊急件数']):,}</td></tr>"
        for _, r in kpi_df.iterrows()
    )
    return f"""
    <table class="rank-table">
      <thead>
        <tr><th class="num">順位</th><th>医師</th>
            <th class="num">直近3ヶ月</th><th class="num">前3ヶ月</th><th class="num">差分</th>
            <th class="num">平均(分)</th><th class="num">緊急</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """


def _build_doctor_table_all(kpi_df: pd.DataFrame) -> str:
    if kpi_df.empty:
        return '<p class="muted">該当データなし</p>'
    rows = "\n".join(
        f"<tr><td class='num'>{int(r['順位'])}</td><td>{r['医師']}</td>"
        f"<td class='num'>{int(r['直近執刀']):,}</td>"
        f"<td class='num'>{int(r['直近助手']):,}</td>"
        f"<td class='num'><strong>{int(r['直近合計']):,}</strong></td>"
        f"<td class='num'>{int(r['比較合計']):,}</td>"
        f"<td class='num'>{_diff_span(int(r['差分']), 'int')}</td></tr>"
        for _, r in kpi_df.iterrows()
    )
    return f"""
    <table class="rank-table">
      <thead>
        <tr><th class="num">順位</th><th>医師</th>
            <th class="num">直近執刀</th><th class="num">直近助手</th><th class="num">直近合計</th>
            <th class="num">前3ヶ月合計</th><th class="num">差分</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """


def _build_topn_table(df_topn: pd.DataFrame, name_column: str, title: str) -> str:
    if df_topn.empty:
        return f'<h3>{title}</h3><p class="muted">該当データなし</p>'
    rows = "\n".join(
        f"<tr><td class='num'>{i + 1}</td><td>{r[name_column]}</td>"
        f"<td class='num'>{int(r['直近件数']):,}</td>"
        f"<td class='num'>{int(r['比較件数']):,}</td>"
        f"<td class='num'>{_diff_span(int(r['差分']), 'int')}</td></tr>"
        for i, (_, r) in enumerate(df_topn.iterrows())
    )
    return f"""
    <h3>{title}</h3>
    <table class="topn-table">
      <thead>
        <tr><th class="num">#</th><th>{name_column}</th>
            <th class="num">直近3ヶ月</th><th class="num">前3ヶ月</th><th class="num">差分</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """


# ---------------------------------------------------------------------------
# HTML テンプレート（A4 縦、4 ページ）
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>{dept} 手術実績レポート</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 14mm 14mm 16mm 14mm;
    @top-left {{
      content: "{dept} · 手術実績レポート";
      font-family: "Hiragino Sans", "Yu Gothic", sans-serif;
      font-size: 9pt; color: #6c757d;
    }}
    @bottom-left {{
      content: "生成 {generated_at} · 実名版（院内限り取扱）";
      font-family: "Hiragino Sans", "Yu Gothic", sans-serif;
      font-size: 8pt; color: #6c757d;
    }}
    @bottom-right {{
      content: "p." counter(page) "/" counter(pages);
      font-family: "Hiragino Sans", "Yu Gothic", sans-serif;
      font-size: 8pt; color: #6c757d;
    }}
  }}
  body {{
    font-family: "Hiragino Sans", "Yu Gothic", "Noto Sans CJK JP", sans-serif;
    color: #212529;
    font-size: 10pt;
    line-height: 1.5;
    margin: 0;
  }}
  .page {{
    page-break-after: always;
  }}
  .page:last-child {{ page-break-after: auto; }}
  h1 {{
    font-size: 18pt; margin: 0 0 6pt;
    color: #0d6efd;
    border-bottom: 2pt solid #0d6efd;
    padding-bottom: 4pt;
  }}
  h2 {{
    font-size: 13pt; margin: 14pt 0 6pt;
    border-left: 3pt solid #0d6efd;
    padding-left: 6pt;
  }}
  h3 {{
    font-size: 11pt; margin: 10pt 0 4pt;
    color: #495057;
  }}
  .meta {{
    color: #6c757d; font-size: 9pt;
    margin: 4pt 0 8pt;
  }}
  .muted {{ color: #6c757d; }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8pt;
    margin: 6pt 0 12pt;
  }}
  .kpi-card {{
    background: #f8f9fa;
    border-left: 3pt solid #0d6efd;
    padding: 6pt 8pt;
    border-radius: 3pt;
  }}
  .kpi-label {{ color: #6c757d; font-size: 9pt; }}
  .kpi-yoy {{ width: 100%; border-collapse: collapse; font-size: 9pt; margin-top: 4pt; }}
  .kpi-yoy td {{ padding: 1.5pt 0; }}
  .kpi-yoy td:first-child {{ color: #6c757d; }}
  .kpi-yoy td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .diff-pos {{ color: #198754; font-weight: 600; }}
  .diff-neg {{ color: #dc3545; font-weight: 600; }}
  img.chart {{ display: block; width: 100%; margin: 4pt 0 8pt; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 8.5pt;
    line-height: 1.3;
  }}
  th, td {{
    padding: 2pt 5pt;
    border-bottom: 0.5pt solid #dee2e6;
    text-align: left;
  }}
  th {{ background: #f8f9fa; color: #495057; font-weight: 600; font-size: 8pt; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .rank-table, .topn-table {{ margin: 3pt 0 8pt; }}
  .rank-table {{ font-size: 8pt; }}
  .rank-table th, .rank-table td {{ padding: 1.5pt 4pt; }}
  .target-summary {{
    background: #f8f9fa;
    padding: 6pt 10pt;
    border-radius: 3pt;
    font-size: 10pt;
    margin: 6pt 0;
  }}
  .target-summary strong {{ color: #0d6efd; }}
  .target-pair {{ display: flex; gap: 24pt; }}
  .target-pair > div {{ flex: 1; }}
</style>
</head>
<body>

<!-- Page 1: サマリ -->
<section class="page">
  <h1>{dept} 手術実績レポート</h1>
  <p class="meta">
    データ範囲 {date_min} 〜 {date_max}（月締め cutoff {cutoff}）　／
    KPI 比較: 直近3ヶ月 <strong>{recent_3mo_str}</strong> vs 昨年同3ヶ月 <strong>{yoy_3mo_str}</strong>
  </p>

  <h2>全体 KPI（直近3ヶ月 vs 昨年同3ヶ月）</h2>
  <div class="kpi-grid">
    {kpi_cards}
  </div>

  <img class="chart" src="{img_monthly_count}" alt="月次件数">
  <img class="chart" src="{img_monthly_avg}" alt="月次平均手術時間">
</section>

<!-- Page 2: 全麻手術件数 vs 目標 -->
<section class="page">
  <h2>全身麻酔手術件数 vs 目標（週次・直近12週）</h2>
  <div class="target-summary">
    {target_summary}
  </div>
  <img class="chart" src="{img_weekly_ga}" alt="週次全麻手術件数">
  <img class="chart" src="{img_monthly_ga}" alt="月次全麻手術件数">
</section>

<!-- Page 3: 術者ランキング -->
<section class="page">
  <h2>執刀医ランキング（執刀医のみ・上位 20、直近3ヶ月 vs その前3ヶ月）</h2>
  {table_lead}

  <h2>執刀＋助手 参加件数（上位 20、直近3ヶ月 vs その前3ヶ月）</h2>
  {table_all}
</section>

<!-- Page 4: カテゴリ・術式 -->
<section class="page">
  <h2>カテゴリ別 件数（直近3ヶ月 vs その前3ヶ月）</h2>
  <img class="chart" src="{img_category_bar}" alt="カテゴリ別件数">
  <img class="chart" src="{img_category_monthly}" alt="カテゴリ別月次">

  <h2>主要術式・術後病名（直近3ヶ月 上位 10、その前3ヶ月件数を併記）</h2>
  <div class="target-pair">
    <div>{table_procedures}</div>
    <div>{table_diagnoses}</div>
  </div>
</section>

</body>
</html>
"""


def _format_period(p: tuple[date, date]) -> str:
    return f"{p[0].isoformat()} 〜 {p[1].isoformat()}"


def _build_target_summary(
    df_dept: pd.DataFrame,
    periods: ReportPeriods,
    target: int | None,
) -> str:
    if target is None:
        return '<span class="muted">目標未設定（実績のみ表示）</span>'

    wk = weekly_general_anesthesia(df_dept, periods.weekly_12w, target=target)
    if wk.empty:
        return f"目標 <strong>{target} 件/週</strong>　｜　対象週なし"
    achieved = int(wk["達成"].sum())
    total = len(wk)
    pct = (achieved / total * 100) if total else 0.0
    return (
        f"目標 <strong>{target} 件/週</strong>　"
        f"｜　直近12週: 達成 <strong>{achieved}</strong> 週 / {total} 週（{pct:.1f}%）"
    )


def render_dept_html(
    df_dept: pd.DataFrame,
    dept: str,
    periods: ReportPeriods,
    target: int | None,
    generated_at: datetime,
) -> str:
    kpi = kpi_overall_compare(df_dept, periods.recent_3mo, periods.yoy_3mo)

    img_monthly_count = _fig_to_data_uri(
        _fig_monthly_count_rolling(df_dept, periods), width=950, height=240)
    img_monthly_avg = _fig_to_data_uri(
        _fig_monthly_avg_time_rolling(df_dept, periods), width=950, height=240)
    img_weekly_ga = _fig_to_data_uri(
        _fig_weekly_ga_vs_target(df_dept, periods, target), width=1000, height=300)
    img_monthly_ga = _fig_to_data_uri(
        _fig_monthly_ga_rolling(df_dept, periods), width=950, height=260)
    img_category_bar = _fig_to_data_uri(
        _fig_category_bar_3mo(df_dept, periods), width=950, height=240)
    img_category_monthly = _fig_to_data_uri(
        _fig_category_monthly_rolling(df_dept, periods), width=950, height=320)

    kpi_cards = _build_kpi_cards(kpi)

    table_lead = _build_doctor_table_lead(
        kpi_per_doctor_compare_window(
            df_dept, periods.recent_3mo, periods.prior_3mo,
            mode="lead_only", top_n=20))
    table_all = _build_doctor_table_all(
        kpi_per_doctor_compare_window(
            df_dept, periods.recent_3mo, periods.prior_3mo,
            mode="all", top_n=20))

    table_procedures = _build_topn_table(
        top_n_procedures(df_dept, periods.recent_3mo, periods.prior_3mo, n=10),
        "確定術式", "主要術式 top 10")
    table_diagnoses = _build_topn_table(
        top_n_postop_diagnoses(df_dept, periods.recent_3mo, periods.prior_3mo, n=10),
        "術後病名", "主要術後病名 top 10")

    target_summary = _build_target_summary(df_dept, periods, target)

    date_min = df_dept["手術実施日"].min().date().isoformat()
    date_max = df_dept["手術実施日"].max().date().isoformat()

    return _HTML_TEMPLATE.format(
        dept=dept,
        date_min=date_min,
        date_max=date_max,
        cutoff=periods.cutoff.isoformat(),
        recent_3mo_str=_format_period(periods.recent_3mo),
        yoy_3mo_str=_format_period(periods.yoy_3mo),
        generated_at=generated_at.strftime("%Y-%m-%d"),
        kpi_cards=kpi_cards,
        img_monthly_count=img_monthly_count,
        img_monthly_avg=img_monthly_avg,
        img_weekly_ga=img_weekly_ga,
        img_monthly_ga=img_monthly_ga,
        img_category_bar=img_category_bar,
        img_category_monthly=img_category_monthly,
        table_lead=table_lead,
        table_all=table_all,
        table_procedures=table_procedures,
        table_diagnoses=table_diagnoses,
        target_summary=target_summary,
    )


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------


def render_dept_pdf(
    df_dept: pd.DataFrame,
    dept: str,
    output_path: Path,
    periods: ReportPeriods,
    target: int | None,
    generated_at: datetime | None = None,
) -> Path:
    """1 診療科分の PDF を生成して書き出す。"""
    from weasyprint import HTML  # noqa: PLC0415

    if generated_at is None:
        generated_at = datetime.now()

    html = render_dept_html(df_dept, dept, periods, target, generated_at)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html).write_pdf(target=str(output_path))
    return output_path


def _cases_in_report_window(df_dept: pd.DataFrame, periods: ReportPeriods) -> int:
    """レポート最大窓（直近24ヶ月 = recent_12mo + prior_12mo）に入る件数を返す。

    PDF の各セクションは 3ヶ月窓 / 12ヶ月窓 / 12週窓を使う。24ヶ月窓は最も広く、
    どこかにデータがあるかどうかの判定に適切。
    """
    d = df_dept["手術実施日"]
    r12 = (d >= pd.Timestamp(periods.recent_12mo[0])) & (d <= pd.Timestamp(periods.recent_12mo[1]))
    p12 = (d >= pd.Timestamp(periods.prior_12mo[0])) & (d <= pd.Timestamp(periods.prior_12mo[1]))
    return int((r12 | p12).sum())


def export_all(
    parquet_path: Path,
    output_dir: Path,
    targets_path: Path,
    today: date | None = None,
    only_dept: str | None = None,
    min_cases: int = MIN_CASE_COUNT,
) -> list[Path]:
    """全診療科の PDF を出力する。

    - `only_dept`: 指定診療科のみ出力
    - `today` None なら現在日時を使用
    - 対象期間（直近24ヶ月）の件数 < `min_cases` の診療科は skip
    """
    import kaleido  # noqa: PLC0415

    df = pd.read_parquet(parquet_path)
    if today is None:
        today = date.today()
    periods = report_periods(today)
    targets = load_dept_targets(targets_path)

    if only_dept is not None:
        depts = [only_dept]
    else:
        depts = sorted(df["実施診療科"].dropna().unique().tolist())

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now()
    written: list[Path] = []

    kaleido.start_sync_server(silence_warnings=True)
    try:
        for dept in depts:
            df_d = df[df["実施診療科"] == dept]
            n_window = _cases_in_report_window(df_d, periods)
            if n_window < min_cases:
                logger.info(
                    "skip: %s (対象期間件数 %d < %d / 全期間 %d)",
                    dept, n_window, min_cases, len(df_d),
                )
                continue
            target = targets.get(dept)
            out = output_dir / f"{dept}.pdf"
            try:
                render_dept_pdf(
                    df_d, dept, out, periods,
                    target=target.weekly_general_anesthesia if target else None,
                    generated_at=generated_at,
                )
                written.append(out)
                logger.info("出力: %s (対象期間件数 %d)", out, n_window)
            except Exception:
                logger.exception("PDF 生成失敗: %s", dept)
    finally:
        kaleido.stop_sync_server()

    return written
