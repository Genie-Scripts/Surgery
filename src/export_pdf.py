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
    FiscalYearPeriods,
    fiscal_year_periods,
    kpi_overall_yoy,
    kpi_per_doctor_yoy,
    monthly_count_by_fiscal_year,
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
COLOR_PRIMARY = "#0d6efd"  # 今年度
COLOR_PRIMARY_LIGHT = "#9ec5fe"
COLOR_LAST_YEAR = "#adb5bd"  # 昨年度
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


def _fig_monthly_count_yoy(df_dept: pd.DataFrame, periods: FiscalYearPeriods) -> go.Figure:
    mt = monthly_count_by_fiscal_year(df_dept, periods)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mt["月ラベル"], y=mt["今年度件数"], mode="lines+markers",
            name=f"今年度 (FY{periods.fiscal_year})",
            line={"color": COLOR_PRIMARY, "width": 2.5}, marker={"size": 7},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mt["月ラベル"], y=mt["昨年度件数"], mode="lines+markers",
            name=f"昨年度 (FY{periods.fiscal_year - 1})",
            line={"color": COLOR_LAST_YEAR, "width": 2, "dash": "dot"}, marker={"size": 6},
        )
    )
    return _common_layout(fig, title="月次 件数（今年度 vs 昨年度）")


def _fig_monthly_avg_time_yoy(df_dept: pd.DataFrame, periods: FiscalYearPeriods) -> go.Figure:
    def _monthly_avg(slice_df: pd.DataFrame) -> pd.Series:
        if slice_df.empty:
            return pd.Series(dtype="float64")
        s = slice_df.set_index("手術実施日").resample("MS")["予定手術時間"].mean()
        return s.rename_axis("month").rename(index=lambda ts: (ts.month - 4) % 12)

    cy = df_dept[(df_dept["手術実施日"] >= pd.Timestamp(periods.ytd[0]))
                 & (df_dept["手術実施日"] <= pd.Timestamp(periods.ytd[1]))]
    ly = df_dept[(df_dept["手術実施日"] >= pd.Timestamp(periods.last_year[0]))
                 & (df_dept["手術実施日"] <= pd.Timestamp(periods.last_year[1]))]
    cy_m = _monthly_avg(cy)
    ly_m = _monthly_avg(ly)

    months = list(range(12))
    labels = [f"{(i + 3) % 12 + 1} 月" for i in months]
    cy_vals = [float(cy_m.get(i, float("nan"))) for i in months]
    ly_vals = [float(ly_m.get(i, float("nan"))) for i in months]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=labels, y=cy_vals, mode="lines+markers",
                   name=f"今年度 (FY{periods.fiscal_year})",
                   line={"color": COLOR_PRIMARY, "width": 2.5}, marker={"size": 7},
                   connectgaps=False)
    )
    fig.add_trace(
        go.Scatter(x=labels, y=ly_vals, mode="lines+markers",
                   name=f"昨年度 (FY{periods.fiscal_year - 1})",
                   line={"color": COLOR_LAST_YEAR, "width": 2, "dash": "dot"}, marker={"size": 6},
                   connectgaps=False)
    )
    return _common_layout(fig, title="月次 平均手術時間 (分)")


def _fig_weekly_ga_vs_target(
    df_dept: pd.DataFrame, periods: FiscalYearPeriods, target: int | None
) -> go.Figure:
    wk = weekly_general_anesthesia(df_dept, periods.ytd, target=target)
    if wk.empty:
        fig = go.Figure()
        fig.add_annotation(text="今年度データなし", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font={"size": 13, "color": COLOR_MUTED})
        fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, height=240)
        return fig

    if target is not None:
        bar_colors = [COLOR_PRIMARY if v >= target else COLOR_LAST_YEAR for v in wk["全麻件数"]]
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

    title = "週次 全麻手術件数 vs 目標" if target is not None else "週次 全麻手術件数（目標未設定）"
    fig = _common_layout(fig, title=title, height=300)
    fig.update_layout(showlegend=False)
    return fig


def _fig_monthly_ga_yoy(df_dept: pd.DataFrame, periods: FiscalYearPeriods) -> go.Figure:
    from src.aggregate import _general_anesthesia_mask  # noqa: PLC0415

    def _monthly(slice_df: pd.DataFrame) -> pd.Series:
        if slice_df.empty:
            return pd.Series(dtype="int64")
        ga = slice_df[_general_anesthesia_mask(slice_df)]
        if ga.empty:
            return pd.Series(dtype="int64")
        s = ga.set_index("手術実施日").resample("MS").size()
        return s.rename_axis("month").rename(index=lambda ts: (ts.month - 4) % 12)

    cy = df_dept[(df_dept["手術実施日"] >= pd.Timestamp(periods.ytd[0]))
                 & (df_dept["手術実施日"] <= pd.Timestamp(periods.ytd[1]))]
    ly = df_dept[(df_dept["手術実施日"] >= pd.Timestamp(periods.last_year[0]))
                 & (df_dept["手術実施日"] <= pd.Timestamp(periods.last_year[1]))]
    cy_m = _monthly(cy)
    ly_m = _monthly(ly)

    months = list(range(12))
    labels = [f"{(i + 3) % 12 + 1} 月" for i in months]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=labels, y=[int(cy_m.get(i, 0)) for i in months],
                   mode="lines+markers", name=f"今年度 (FY{periods.fiscal_year})",
                   line={"color": COLOR_PRIMARY, "width": 2.5}, marker={"size": 7})
    )
    fig.add_trace(
        go.Scatter(x=labels, y=[int(ly_m.get(i, 0)) for i in months],
                   mode="lines+markers", name=f"昨年度 (FY{periods.fiscal_year - 1})",
                   line={"color": COLOR_LAST_YEAR, "width": 2, "dash": "dot"}, marker={"size": 6})
    )
    return _common_layout(fig, title="月次 全麻手術件数（今年度 vs 昨年度）")


def _fig_category_bar_yoy(df_dept: pd.DataFrame, periods: FiscalYearPeriods) -> go.Figure:
    cy = df_dept[(df_dept["手術実施日"] >= pd.Timestamp(periods.ytd[0]))
                 & (df_dept["手術実施日"] <= pd.Timestamp(periods.ytd[1]))]
    ly = df_dept[(df_dept["手術実施日"] >= pd.Timestamp(periods.last_year[0]))
                 & (df_dept["手術実施日"] <= pd.Timestamp(periods.last_year[1]))]
    cats = [c for c in CATEGORY_COLUMNS if c in df_dept.columns]
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    cy_vals = [int(cy[c].sum()) if not cy.empty else 0 for c in cats]
    ly_vals = [int(ly[c].sum()) if not ly.empty else 0 for c in cats]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=ly_vals, name=f"昨年同期 (FY{periods.fiscal_year - 1})",
                         marker_color=COLOR_LAST_YEAR, text=ly_vals, textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=cy_vals, name=f"今年度YTD (FY{periods.fiscal_year})",
                         marker_color=COLOR_PRIMARY, text=cy_vals, textposition="outside"))
    fig.update_layout(barmode="group")
    return _common_layout(fig, title="カテゴリ別 件数（今年度 vs 昨年同期）")


def _fig_category_monthly_yoy(df_dept: pd.DataFrame, periods: FiscalYearPeriods) -> go.Figure:
    """4 カテゴリの月次推移を 1 枚に subplot (2x2)。"""
    from plotly.subplots import make_subplots  # noqa: PLC0415

    cats = [c for c in CATEGORY_COLUMNS if c in df_dept.columns]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[CATEGORY_LABELS.get(c, c) for c in cats[:4]],
        vertical_spacing=0.18, horizontal_spacing=0.10,
    )

    cy = df_dept[(df_dept["手術実施日"] >= pd.Timestamp(periods.ytd[0]))
                 & (df_dept["手術実施日"] <= pd.Timestamp(periods.ytd[1]))]
    ly = df_dept[(df_dept["手術実施日"] >= pd.Timestamp(periods.last_year[0]))
                 & (df_dept["手術実施日"] <= pd.Timestamp(periods.last_year[1]))]

    def _monthly_sum(slice_df: pd.DataFrame, col: str) -> dict[int, int]:
        if slice_df.empty:
            return {}
        s = slice_df.set_index("手術実施日").resample("MS")[col].sum()
        return {int((ts.month - 4) % 12): int(v) for ts, v in s.items()}

    months = list(range(12))
    labels = [f"{(i + 3) % 12 + 1} 月" for i in months]

    for i, cat in enumerate(cats[:4]):
        row, col = i // 2 + 1, i % 2 + 1
        cy_m = _monthly_sum(cy, cat)
        ly_m = _monthly_sum(ly, cat)
        fig.add_trace(
            go.Scatter(x=labels, y=[cy_m.get(m, 0) for m in months],
                       mode="lines+markers", line={"color": COLOR_PRIMARY, "width": 2},
                       marker={"size": 5}, showlegend=(i == 0), name="今年度"),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(x=labels, y=[ly_m.get(m, 0) for m in months],
                       mode="lines+markers",
                       line={"color": COLOR_LAST_YEAR, "width": 1.5, "dash": "dot"},
                       marker={"size": 5}, showlegend=(i == 0), name="昨年度"),
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


def _kpi_card_yoy(label: str, ly: str, ytd: str, diff_html: str) -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <table class="kpi-yoy">
        <tr><td>昨年同期</td><td class="num">{ly}</td></tr>
        <tr><td>今年度YTD</td><td class="num"><strong>{ytd}</strong></td></tr>
        <tr><td>差分</td><td class="num">{diff_html}</td></tr>
      </table>
    </div>
    """


def _build_kpi_cards(kpi: dict, periods: FiscalYearPeriods) -> str:
    ly, ytd, diff = kpi["last_year"], kpi["ytd"], kpi["diff"]
    return (
        _kpi_card_yoy("件数", _fmt_int(ly["件数"]), _fmt_int(ytd["件数"]),
                      _diff_span(diff["件数"], "int"))
        + _kpi_card_yoy("平均手術時間 (分)", _fmt_float(ly["平均手術時間_分"]),
                        _fmt_float(ytd["平均手術時間_分"]),
                        _diff_span(diff["平均手術時間_分"], "minute"))
        + _kpi_card_yoy("緊急比率", _fmt_pct(ly["緊急比率"]), _fmt_pct(ytd["緊急比率"]),
                        _diff_span(diff["緊急比率"], "pct_point"))
        + _kpi_card_yoy("全麻手術件数", _fmt_int(ly["全麻手術件数"]),
                        _fmt_int(ytd["全麻手術件数"]),
                        _diff_span(diff["全麻手術件数"], "int"))
    )


def _build_doctor_table_lead(kpi_df: pd.DataFrame) -> str:
    if kpi_df.empty:
        return '<p class="muted">該当データなし</p>'
    rows = "\n".join(
        f"<tr><td class='num'>{int(r['順位'])}</td><td>{r['医師']}</td>"
        f"<td class='num'>{int(r['今年度件数']):,}</td>"
        f"<td class='num'>{int(r['昨年同期件数']):,}</td>"
        f"<td class='num'>{_diff_span(int(r['差分']), 'int')}</td>"
        f"<td class='num'>{r['平均時間_分']:.1f}</td>"
        f"<td class='num'>{int(r['緊急件数']):,}</td></tr>"
        for _, r in kpi_df.iterrows()
    )
    return f"""
    <table class="rank-table">
      <thead>
        <tr><th class="num">順位</th><th>医師</th>
            <th class="num">今年度</th><th class="num">昨年同期</th><th class="num">差分</th>
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
        f"<td class='num'>{int(r['今年執刀']):,}</td>"
        f"<td class='num'>{int(r['今年助手']):,}</td>"
        f"<td class='num'><strong>{int(r['今年合計']):,}</strong></td>"
        f"<td class='num'>{int(r['昨年合計']):,}</td>"
        f"<td class='num'>{_diff_span(int(r['差分']), 'int')}</td></tr>"
        for _, r in kpi_df.iterrows()
    )
    return f"""
    <table class="rank-table">
      <thead>
        <tr><th class="num">順位</th><th>医師</th>
            <th class="num">今年執刀</th><th class="num">今年助手</th><th class="num">今年合計</th>
            <th class="num">昨年合計</th><th class="num">差分</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """


def _build_topn_table(df_topn: pd.DataFrame, name_column: str, title: str) -> str:
    if df_topn.empty:
        return f'<h3>{title}</h3><p class="muted">該当データなし</p>'
    rows = "\n".join(
        f"<tr><td class='num'>{i + 1}</td><td>{r[name_column]}</td>"
        f"<td class='num'>{int(r['今年度件数']):,}</td>"
        f"<td class='num'>{int(r['昨年同期件数']):,}</td>"
        f"<td class='num'>{_diff_span(int(r['差分']), 'int')}</td></tr>"
        for i, (_, r) in enumerate(df_topn.iterrows())
    )
    return f"""
    <h3>{title}</h3>
    <table class="topn-table">
      <thead>
        <tr><th class="num">#</th><th>{name_column}</th>
            <th class="num">今年度</th><th class="num">昨年同期</th><th class="num">差分</th></tr>
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
    対象期間 {date_min} 〜 {date_max}（月締め）　／
    比較 今年度YTD <strong>{ytd_str}</strong> vs 昨年同期 <strong>{ly_str}</strong>
  </p>

  <h2>全体 KPI（今年度 YTD vs 昨年同期）</h2>
  <div class="kpi-grid">
    {kpi_cards}
  </div>

  <img class="chart" src="{img_monthly_count}" alt="月次件数">
  <img class="chart" src="{img_monthly_avg}" alt="月次平均手術時間">
</section>

<!-- Page 2: 全麻手術件数 vs 目標 -->
<section class="page">
  <h2>全身麻酔手術件数 vs 目標（週次）</h2>
  <div class="target-summary">
    {target_summary}
  </div>
  <img class="chart" src="{img_weekly_ga}" alt="週次全麻手術件数">
  <img class="chart" src="{img_monthly_ga}" alt="月次全麻手術件数">
</section>

<!-- Page 3: 術者ランキング -->
<section class="page">
  <h2>執刀医ランキング（執刀医のみ、上位 20）</h2>
  {table_lead}

  <h2>執刀＋助手 参加件数（上位 20）</h2>
  {table_all}
</section>

<!-- Page 4: カテゴリ・術式 -->
<section class="page">
  <h2>カテゴリ別 件数</h2>
  <img class="chart" src="{img_category_bar}" alt="カテゴリ別件数">
  <img class="chart" src="{img_category_monthly}" alt="カテゴリ別月次">

  <h2>主要術式・術後病名（今年度 YTD ベース 上位 10）</h2>
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
    periods: FiscalYearPeriods,
    target: int | None,
) -> str:
    if target is None:
        return '<span class="muted">目標未設定（実績のみ表示）</span>'

    ytd_wk = weekly_general_anesthesia(df_dept, periods.ytd, target=target)
    ly_wk = weekly_general_anesthesia(df_dept, periods.last_year, target=target)

    def _summary(wk: pd.DataFrame) -> str:
        if wk.empty:
            return "対象週なし"
        achieved = int(wk["達成"].sum())
        total = len(wk)
        pct = (achieved / total * 100) if total else 0.0
        return f"達成 <strong>{achieved}</strong> 週 / {total} 週（{pct:.1f}%）"

    return (
        f"目標 <strong>{target} 件/週</strong>　"
        f"｜　今年度YTD: {_summary(ytd_wk)}　"
        f"｜　昨年同期: {_summary(ly_wk)}"
    )


def render_dept_html(
    df_dept: pd.DataFrame,
    dept: str,
    periods: FiscalYearPeriods,
    target: int | None,
    generated_at: datetime,
) -> str:
    kpi = kpi_overall_yoy(df_dept, periods)

    img_monthly_count = _fig_to_data_uri(
        _fig_monthly_count_yoy(df_dept, periods), width=950, height=240)
    img_monthly_avg = _fig_to_data_uri(
        _fig_monthly_avg_time_yoy(df_dept, periods), width=950, height=240)
    img_weekly_ga = _fig_to_data_uri(
        _fig_weekly_ga_vs_target(df_dept, periods, target), width=1000, height=300)
    img_monthly_ga = _fig_to_data_uri(
        _fig_monthly_ga_yoy(df_dept, periods), width=950, height=260)
    img_category_bar = _fig_to_data_uri(
        _fig_category_bar_yoy(df_dept, periods), width=950, height=240)
    img_category_monthly = _fig_to_data_uri(
        _fig_category_monthly_yoy(df_dept, periods), width=950, height=320)

    kpi_cards = _build_kpi_cards(kpi, periods)

    table_lead = _build_doctor_table_lead(
        kpi_per_doctor_yoy(df_dept, periods, mode="lead_only", top_n=20))
    table_all = _build_doctor_table_all(
        kpi_per_doctor_yoy(df_dept, periods, mode="all", top_n=20))

    table_procedures = _build_topn_table(
        top_n_procedures(df_dept, periods, n=10), "確定術式", "主要術式 top 10")
    table_diagnoses = _build_topn_table(
        top_n_postop_diagnoses(df_dept, periods, n=10), "術後病名", "主要術後病名 top 10")

    target_summary = _build_target_summary(df_dept, periods, target)

    date_min = df_dept["手術実施日"].min().date().isoformat()
    date_max = df_dept["手術実施日"].max().date().isoformat()

    return _HTML_TEMPLATE.format(
        dept=dept,
        date_min=date_min,
        date_max=date_max,
        ytd_str=_format_period(periods.ytd),
        ly_str=_format_period(periods.last_year),
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
    periods: FiscalYearPeriods,
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
    - 件数 < `min_cases` の診療科は skip
    """
    df = pd.read_parquet(parquet_path)
    if today is None:
        today = date.today()
    periods = fiscal_year_periods(today)
    targets = load_dept_targets(targets_path)

    if only_dept is not None:
        depts = [only_dept]
    else:
        depts = sorted(df["実施診療科"].dropna().unique().tolist())

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now()
    written: list[Path] = []

    for dept in depts:
        df_d = df[df["実施診療科"] == dept]
        n = len(df_d)
        if n < min_cases:
            logger.info("skip: %s (件数 %d < %d)", dept, n, min_cases)
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
            logger.info("出力: %s (件数 %d)", out, n)
        except Exception:
            logger.exception("PDF 生成失敗: %s", dept)

    return written
