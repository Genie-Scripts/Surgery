"""src/export_pdf.py の smoke test。

WeasyPrint/kaleido は重いので、本格テストは小さな DataFrame で 1 ファイル生成して
バイト列が妥当 PDF 構造を持つことだけ確認する。
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("weasyprint")
pytest.importorskip("kaleido")

from src.aggregate import report_periods
from src.export_pdf import (
    HOSPITAL_LABEL,
    MIN_CASE_COUNT,
    _hospital_weekly_target,
    export_all,
    load_dept_targets,
    render_dept_html,
    render_dept_pdf,
)


def _make_df(dept: str = "整形外科", n: int = 50, ly_n: int | None = None) -> pd.DataFrame:
    """件数 n の DataFrame を生成（直近3ヶ月窓に均等分布）。

    `ly_n` 省略時は n の 60% を昨年同3ヶ月に配置。テストで「件数 < 30 で skip」を
    検証する場合は ly_n=0 などで明示的に指定する。
    """
    if ly_n is None:
        ly_n = max(0, int(n * 0.6))
    base = pd.Timestamp("2026-04-01")
    rows = []
    for i in range(n):
        rows.append(
            {
                "手術実施日": base + pd.Timedelta(days=i % 25),
                "実施診療科": dept,
                "予定手術時間": 120 + (i % 5) * 30,
                "申込区分": "緊急" if i % 10 == 0 else "通常",
                "麻酔種別": (
                    "全身麻酔(20分以上：吸入もしくは静脈麻酔薬)"
                    if i % 2 == 0
                    else "局所麻酔"
                ),
                "実施術者": f"医師_{i % 5:03d}",
                "執刀医": f"医師_{i % 5:03d}",
                "助手リスト": [f"医師_{(i + 1) % 5:03d}"] if i % 3 == 0 else [],
                "確定術式": f"術式_{i % 4}",
                "術後病名": f"病名_{i % 3}",
                "malignant_tumor": i % 7 == 0,
                "artificial_joint": i % 4 == 0,
                "robot_assisted_davinci": False,
                "robot_assisted_other": i % 11 == 0,
            }
        )
    # 昨年同期
    for i in range(ly_n):
        rows.append(
            {
                "手術実施日": pd.Timestamp("2025-04-01") + pd.Timedelta(days=i % 25),
                "実施診療科": dept,
                "予定手術時間": 100 + (i % 4) * 30,
                "申込区分": "通常",
                "麻酔種別": (
                    "全身麻酔(20分以上：吸入もしくは静脈麻酔薬)"
                    if i % 3 == 0
                    else "局所麻酔"
                ),
                "実施術者": f"医師_{i % 4:03d}",
                "執刀医": f"医師_{i % 4:03d}",
                "助手リスト": [],
                "確定術式": f"術式_{i % 3}",
                "術後病名": f"病名_{i % 2}",
                "malignant_tumor": False,
                "artificial_joint": True,
                "robot_assisted_davinci": False,
                "robot_assisted_other": False,
            }
        )
    return pd.DataFrame(rows)


def test_render_dept_html_contains_key_sections():
    df = _make_df()
    periods = report_periods(date(2026, 5, 27))
    html = render_dept_html(df, "整形外科", periods, target=10, generated_at=datetime(2026, 5, 27))
    assert "整形外科 手術実績レポート" in html
    assert "全身麻酔手術件数 vs 目標（週次・直近12週）" in html
    assert "執刀医ランキング" in html
    assert "主要術式 top 10" in html
    assert "直近3ヶ月" in html
    # base64 PNG が 6 枚埋め込まれているはず
    assert html.count("data:image/png;base64,") == 6


def test_render_dept_pdf_writes_valid_pdf(tmp_path: Path):
    df = _make_df()
    periods = report_periods(date(2026, 5, 27))
    out = tmp_path / "test.pdf"
    written = render_dept_pdf(df, "整形外科", out, periods, target=10)
    assert written.exists()
    assert written.stat().st_size > 10_000  # 1 ページ以上の妥当サイズ
    # PDF マジックヘッダ
    assert written.read_bytes()[:4] == b"%PDF"


def test_export_all_skips_low_volume(tmp_path: Path):
    """件数 < min_cases の診療科は出力しない。"""
    df = pd.concat(
        [
            _make_df(dept="整形外科", n=50),
            _make_df(dept="内科", n=2, ly_n=0),  # スパース
        ],
        ignore_index=True,
    )
    parquet = tmp_path / "data.parquet"
    df.to_parquet(parquet)
    targets = tmp_path / "targets.yaml"
    targets.write_text("整形外科:\n  weekly_general_anesthesia: 5\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    written = export_all(parquet, out_dir, targets, today=date(2026, 5, 27))
    names = [p.name for p in written]
    assert "整形外科.pdf" in names
    assert "内科.pdf" not in names  # 2 件 < 30 で skip


def test_export_all_includes_hospital_by_default(tmp_path: Path):
    """デフォルトで病院全体レポートを全科に加えて出力する。"""
    df = pd.concat(
        [
            _make_df(dept="整形外科", n=50),
            _make_df(dept="泌尿器科", n=40),
        ],
        ignore_index=True,
    )
    parquet = tmp_path / "data.parquet"
    df.to_parquet(parquet)
    targets = tmp_path / "targets.yaml"
    targets.write_text(
        "整形外科:\n  weekly_general_anesthesia: 5\n"
        "泌尿器科:\n  weekly_general_anesthesia: 3\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    written = export_all(parquet, out_dir, targets, today=date(2026, 5, 27))
    names = [p.name for p in written]
    assert f"{HOSPITAL_LABEL}.pdf" in names
    assert "整形外科.pdf" in names
    assert "泌尿器科.pdf" in names
    # 病院全体は先頭に生成・列挙される
    assert names[0] == f"{HOSPITAL_LABEL}.pdf"


def test_export_all_no_hospital_excludes_hospital(tmp_path: Path):
    """include_hospital=False で病院全体を出力しない。"""
    df = _make_df(dept="整形外科", n=50)
    parquet = tmp_path / "data.parquet"
    df.to_parquet(parquet)
    targets = tmp_path / "targets.yaml"
    targets.write_text("整形外科:\n  weekly_general_anesthesia: 5\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    written = export_all(
        parquet, out_dir, targets, today=date(2026, 5, 27), include_hospital=False
    )
    names = [p.name for p in written]
    assert f"{HOSPITAL_LABEL}.pdf" not in names
    assert "整形外科.pdf" in names


def test_export_all_only_hospital(tmp_path: Path):
    """--dept 病院全体 で病院全体レポートのみ出力する。"""
    df = pd.concat(
        [_make_df(dept="整形外科", n=50), _make_df(dept="泌尿器科", n=40)],
        ignore_index=True,
    )
    parquet = tmp_path / "data.parquet"
    df.to_parquet(parquet)
    targets = tmp_path / "targets.yaml"
    targets.write_text("整形外科:\n  weekly_general_anesthesia: 5\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    written = export_all(
        parquet, out_dir, targets, today=date(2026, 5, 27), only_dept=HOSPITAL_LABEL
    )
    names = [p.name for p in written]
    assert names == [f"{HOSPITAL_LABEL}.pdf"]


def test_hospital_weekly_target_rounds_float_sum(tmp_path: Path):
    """病院全体の週次目標は各診療科目標（小数）の合計を四捨五入した整数。"""
    targets_path = tmp_path / "targets.yaml"
    targets_path.write_text(
        "整形外科:\n  weekly_general_anesthesia: 28.8\n"
        "泌尿器科:\n  weekly_general_anesthesia: 11.1\n",
        encoding="utf-8",
    )
    targets = load_dept_targets(targets_path)
    # 28.8 + 11.1 = 39.9 → 四捨五入で 40（個別科の切り捨て 28+11=39 ではない）
    assert _hospital_weekly_target(targets) == 40
    # 目標が 1 件もなければ None
    assert _hospital_weekly_target({}) is None


def test_hospital_weekly_target_round_half_up(tmp_path: Path):
    """ちょうど .5 は切り上げる（ROUND_HALF_UP、銀行丸めではない）。"""
    targets_path = tmp_path / "targets.yaml"
    targets_path.write_text(
        "A:\n  weekly_general_anesthesia: 2.0\n"
        "B:\n  weekly_general_anesthesia: 2.5\n",
        encoding="utf-8",
    )
    targets = load_dept_targets(targets_path)
    # 2.0 + 2.5 = 4.5 → 5（銀行丸めなら 4 になるが ROUND_HALF_UP で 5）
    assert _hospital_weekly_target(targets) == 5


def test_load_dept_targets_missing_file_returns_empty(tmp_path: Path):
    """存在しない YAML を渡すと空辞書（実績のみ描画）。"""
    out = load_dept_targets(tmp_path / "nope.yaml")
    assert out == {}


def test_min_case_count_default():
    """既定値が 30 件であること（運用要件）。"""
    assert MIN_CASE_COUNT == 30
