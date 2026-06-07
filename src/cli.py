"""統合 CLI エントリポイント。

サブコマンド:
  anonymize    生 data/raw/*.csv を匿名化して data/raw/anonymized/ に出力
  classify     匿名化済み CSV を読み込み、regex + LLM 第 2 段で分類して parquet 保存
  summary      classify 結果のカテゴリ別件数を表示
  export-html  公開用静的 HTML ダッシュボードを書き出す（spec §8.2）
  export-pdf   診療科別 PDF レポートを local/reports/ に書き出す（実名版）

実行:
  python -m src.cli anonymize [--dry-run]
  python -m src.cli classify  [--csv PATH] [--no-llm]
  python -m src.cli summary   [--parquet PATH]
  python -m src.cli export-html [--parquet PATH] [--output PATH]
  python -m src.cli export-pdf  [--parquet PATH] [--output-dir DIR] [--dept NAME]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from src.aggregate import category_counts
from src.anonymize import main as anonymize_main
from src.classify import classify, load_rules
from src.classify_llm import classify_with_llm
from src.ingest import load, load_directory
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "anonymized_data.csv"
DEFAULT_LLM_CONFIG = ROOT / "config" / "llm_config.yaml"
DEFAULT_OUTPUT_PARQUET = ROOT / "data" / "aggregated" / "classified.parquet"
DEFAULT_HTML_OUTPUT = ROOT / "docs" / "index.html"
DEFAULT_DEPT_TARGETS = ROOT / "config" / "department_targets.yaml"
DEFAULT_PDF_OUTPUT_DIR = ROOT / "local" / "reports"


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # kaleido / choreographer は INFO が冗長すぎるので WARNING に下げる
    if not verbose:
        for name in ("kaleido", "choreographer", "fontTools", "weasyprint"):
            logging.getLogger(name).setLevel(logging.WARNING)


def _cmd_classify(args: argparse.Namespace) -> int:
    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
        if not csv_dir.is_dir():
            logger.error("CSV ディレクトリが見つかりません: %s", csv_dir)
            return 1
        logger.info("読込ディレクトリ: %s", csv_dir)
        df = load_directory(csv_dir)
    else:
        csv = Path(args.csv)
        if not csv.exists():
            logger.error("CSV が見つかりません: %s", csv)
            return 1
        logger.info("読込: %s", csv)
        df = load(csv)

    df = classify(df)
    logger.info("regex 第 1 段: 全 %d 件 / LLM 判定要 %d 件", len(df), df["LLM判定要"].sum())

    if not args.no_llm:
        client = LLMClient(DEFAULT_LLM_CONFIG)
        if client.is_available():
            df = classify_with_llm(df, load_rules(), client)
            logger.info("LLM 第 2 段適用済 (キャッシュ＋ハードガード経由)")
        else:
            logger.warning("oMLX 未起動/利用不可 → regex 第 1 段のみで進行")
            df["分類元"] = "regex"
    else:
        df["分類元"] = "regex"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    logger.info("出力: %s", out_path)

    print()
    print("=== カテゴリ別件数 ===")
    print(category_counts(df).to_string(index=False))
    if "分類元" in df.columns:
        print()
        print("=== 分類元別件数 ===")
        print(df["分類元"].value_counts().to_string())
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    pq = Path(args.parquet)
    if not pq.exists():
        logger.error("parquet が見つかりません: %s （先に classify を実行してください）", pq)
        return 1
    df = pd.read_parquet(pq)
    print(f"行数: {len(df):,}")
    print()
    print("=== カテゴリ別件数 ===")
    print(category_counts(df).to_string(index=False))
    if "分類元" in df.columns:
        print()
        print("=== 分類元別件数 ===")
        print(df["分類元"].value_counts().to_string())
    return 0


def _parse_period_arg(spec: str) -> tuple[date, date]:
    """`YYYY-MM-DD..YYYY-MM-DD` を `(date, date)` に変換する。"""
    try:
        start_str, end_str = spec.split("..", 1)
        start = date.fromisoformat(start_str.strip())
        end = date.fromisoformat(end_str.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"期間指定の形式が不正です（'YYYY-MM-DD..YYYY-MM-DD' 形式）: {spec!r}"
        ) from exc
    if start > end:
        raise argparse.ArgumentTypeError(f"start > end になっています: {spec!r}")
    return (start, end)


def _cmd_export_html(args: argparse.Namespace) -> int:
    # import は呼び出し時のみ（plotly のロードを後回しにする）
    from src.export_html import PeriodPair, derive_default_periods, export

    pq = Path(args.parquet)
    if not pq.exists():
        logger.error(
            "parquet が見つかりません: %s （先に classify を実行してください）", pq
        )
        return 1

    periods = None
    if args.period_a or args.period_b or args.period_c:
        df = pd.read_parquet(pq)
        defaults = {p.key: p for p in derive_default_periods(df)}

        def _override(key: str, label: str, override: str | None) -> PeriodPair:
            if not override:
                return defaults[key]
            try:
                older_str, newer_str = override.split("/", 1)
            except ValueError as exc:
                raise SystemExit(
                    f"--period-{key.lower()} は 'OLDER/NEWER' 形式 "
                    f"（例 2025-01-01..2025-03-31/2025-04-01..2025-06-30）"
                ) from exc
            older = _parse_period_arg(older_str)
            newer = _parse_period_arg(newer_str)
            return PeriodPair(key=key, label=label, older=older, newer=newer)

        periods = [
            _override("A", "最新 3 ヶ月 vs 前 3 ヶ月", args.period_a),
            _override("B", "最新 1 ヶ月 vs 前年同月", args.period_b),
            _override("C", "直近 6 ヶ月 vs 前 6 ヶ月", args.period_c),
        ]

    out = Path(args.output)
    written = export(pq, out, periods)
    logger.info("出力: %s (%s bytes)", written, written.stat().st_size)
    return 0


def _cmd_export_pdf(args: argparse.Namespace) -> int:
    # import は呼び出し時のみ（weasyprint / kaleido のロードを後回しにする）
    from src.export_pdf import export_all

    pq = Path(args.parquet)
    if not pq.exists():
        logger.error(
            "parquet が見つかりません: %s （先に classify を実行してください）", pq
        )
        return 1

    targets_path = Path(args.targets)
    if not targets_path.exists():
        logger.warning(
            "目標値ファイルが見つかりません: %s（全診療科で目標未設定として描画します）",
            targets_path,
        )

    # 出力先: --output-dir 未指定なら local/reports/YYYYMMDD/
    today = date.today()
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = DEFAULT_PDF_OUTPUT_DIR / today.strftime("%Y%m%d")

    written = export_all(
        parquet_path=pq,
        output_dir=out_dir,
        targets_path=targets_path,
        today=today,
        only_dept=args.dept,
        min_cases=args.min_cases,
    )

    print()
    print(f"=== 出力先: {out_dir} ===")
    print(f"生成 PDF: {len(written)} ファイル（件数 < {args.min_cases} の診療科は skip）")
    for p in written:
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0 if written else 1


def _cmd_anonymize(args: argparse.Namespace) -> int:
    # anonymize はそれ自身に CLI を持つので、引数を組み立てて委譲する
    forwarded: list[str] = []
    if args.dry_run:
        forwarded.append("--dry-run")
    if args.verbose:
        forwarded.append("--verbose")
    return anonymize_main(forwarded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="surgery", description="Surgery Dashboard 統合 CLI"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG ログを表示")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_anon = sub.add_parser("anonymize", help="raw → 匿名化済み CSV 出力")
    p_anon.add_argument("--dry-run", action="store_true")
    p_anon.set_defaults(func=_cmd_anonymize)

    p_cls = sub.add_parser("classify", help="ingest + regex + (任意) LLM で分類して保存")
    p_cls.add_argument("--csv", default=str(DEFAULT_CSV), help=f"入力 CSV (default: {DEFAULT_CSV})")
    p_cls.add_argument(
        "--csv-dir",
        default=None,
        help="入力 CSV ディレクトリ（指定時は --csv より優先。実名版で data/raw/ を渡す用途）",
    )
    p_cls.add_argument("--no-llm", action="store_true", help="LLM 第 2 段をスキップ")
    p_cls.add_argument("--output", default=str(DEFAULT_OUTPUT_PARQUET))
    p_cls.set_defaults(func=_cmd_classify)

    p_sum = sub.add_parser("summary", help="classify の出力 parquet からサマリ表示")
    p_sum.add_argument("--parquet", default=str(DEFAULT_OUTPUT_PARQUET))
    p_sum.set_defaults(func=_cmd_summary)

    p_exp = sub.add_parser("export-html", help="公開用静的 HTML を docs/ に書き出す")
    p_exp.add_argument(
        "--parquet", default=str(DEFAULT_OUTPUT_PARQUET), help="入力 parquet（classify の出力）"
    )
    p_exp.add_argument("--output", default=str(DEFAULT_HTML_OUTPUT), help="出力 HTML パス")
    p_exp.add_argument(
        "--period-a",
        default=None,
        help="期間 A を上書き: 'OLDER/NEWER'（例 2025-11-01..2026-01-31/2026-02-01..2026-04-30）",
    )
    p_exp.add_argument("--period-b", default=None, help="期間 B 上書き（書式は --period-a と同じ）")
    p_exp.add_argument("--period-c", default=None, help="期間 C 上書き（書式は --period-a と同じ）")
    p_exp.set_defaults(func=_cmd_export_html)

    p_pdf = sub.add_parser(
        "export-pdf",
        help="診療科別 PDF レポートを local/reports/YYYYMMDD/ に書き出す（実名版）",
    )
    p_pdf.add_argument(
        "--parquet", default=str(DEFAULT_OUTPUT_PARQUET), help="入力 parquet（classify の出力）"
    )
    p_pdf.add_argument(
        "--output-dir",
        default=None,
        help=f"出力ディレクトリ（default: {DEFAULT_PDF_OUTPUT_DIR}/YYYYMMDD）",
    )
    p_pdf.add_argument(
        "--targets",
        default=str(DEFAULT_DEPT_TARGETS),
        help=f"目標値 YAML (default: {DEFAULT_DEPT_TARGETS})",
    )
    p_pdf.add_argument(
        "--dept", default=None, help="特定診療科のみ出力（未指定で全科 loop）"
    )
    p_pdf.add_argument(
        "--min-cases", type=int, default=30, help="この件数未満の診療科はスキップ (default: 30)"
    )
    p_pdf.set_defaults(func=_cmd_export_pdf)

    args = parser.parse_args(argv)

    # anonymize は自前で logging を初期化するためここでは起動しない
    if args.cmd != "anonymize":
        _setup_logging(args.verbose)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
