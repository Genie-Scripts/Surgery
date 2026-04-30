"""統合 CLI エントリポイント。

サブコマンド:
  anonymize    生 data/raw/*.csv を匿名化して data/raw/anonymized/ に出力
  classify     匿名化済み CSV を読み込み、regex + LLM 第 2 段で分類して parquet 保存
  summary      classify 結果のカテゴリ別件数を表示

実行:
  python -m src.cli anonymize [--dry-run]
  python -m src.cli classify  [--csv PATH] [--no-llm]
  python -m src.cli summary   [--parquet PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.aggregate import category_counts
from src.anonymize import main as anonymize_main
from src.classify import classify, load_rules
from src.classify_llm import classify_with_llm
from src.ingest import load
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "anonymized_data.csv"
DEFAULT_LLM_CONFIG = ROOT / "config" / "llm_config.yaml"
DEFAULT_OUTPUT_PARQUET = ROOT / "data" / "aggregated" / "classified.parquet"


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _cmd_classify(args: argparse.Namespace) -> int:
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
            logger.warning("Ollama 未起動 → regex 第 1 段のみで進行")
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
    p_cls.add_argument("--no-llm", action="store_true", help="LLM 第 2 段をスキップ")
    p_cls.add_argument("--output", default=str(DEFAULT_OUTPUT_PARQUET))
    p_cls.set_defaults(func=_cmd_classify)

    p_sum = sub.add_parser("summary", help="classify の出力 parquet からサマリ表示")
    p_sum.add_argument("--parquet", default=str(DEFAULT_OUTPUT_PARQUET))
    p_sum.set_defaults(func=_cmd_summary)

    args = parser.parse_args(argv)

    # anonymize は自前で logging を初期化するためここでは起動しない
    if args.cmd != "anonymize":
        _setup_logging(args.verbose)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
