#!/usr/bin/env bash
# 実名版データから診療科別 PDF を全科一括で生成する。
#
# フロー:
#   data/raw/*.csv (実名) → classify → data/aggregated/classified_realname.parquet
#   → export-pdf → local/reports/YYYYMMDD/{診療科}.pdf
#
# 出力は local/ 配下（.gitignore 対象）。件数 < 30 の診療科は自動 skip。
#
# 注意:
#   - 実名 parquet (data/aggregated/classified_realname.parquet) は .gitignore
#     により data/aggregated/*.parquet パターンで除外されている
#   - 公開用 (export-html) とは parquet ファイルを分けているので、本スクリプトは
#     公開ダッシュボードに影響しない

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PARQUET="data/aggregated/classified_realname.parquet"
RAW_DIR="data/raw"
TODAY="$(date +%Y%m%d)"
OUT_DIR="local/reports/${TODAY}"

# venv の python を優先
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "[all.sh] ERROR: python が見つかりません" >&2
    exit 1
fi

SKIP_CLASSIFY=0
NO_LLM=0
ONLY_DEPT=""
MIN_CASES=30

usage() {
    cat <<'EOF'
使い方: scripts/all.sh [OPTIONS]

実名版データから診療科別 PDF を全科一括で生成する。

OPTIONS:
  --skip-classify       classify をスキップ（既存 classified_realname.parquet を流用）
  --no-llm              LLM 第 2 段をスキップ（regex 第 1 段のみで分類）
  --dept NAME           特定診療科のみ出力
  --min-cases N         この件数未満の診療科はスキップ (default: 30)
  -h, --help            このヘルプを表示

出力:
  local/reports/YYYYMMDD/{診療科}.pdf
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-classify)  SKIP_CLASSIFY=1; shift ;;
        --no-llm)         NO_LLM=1; shift ;;
        --dept)           ONLY_DEPT="$2"; shift 2 ;;
        --min-cases)      MIN_CASES="$2"; shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *)
            echo "[all.sh] 不明なオプション: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

step() {
    printf '\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n' "$*"
}

# --- STEP 1: classify (実名 parquet) ---------------------------------------
if [ "$SKIP_CLASSIFY" = "1" ]; then
    if [ ! -f "${PARQUET}" ]; then
        echo "[all.sh] ERROR: --skip-classify 指定だが ${PARQUET} が存在しません" >&2
        exit 1
    fi
    step "STEP 1/2: classify (skipped, 既存 parquet を流用)"
else
    if [ ! -d "${RAW_DIR}" ]; then
        echo "[all.sh] ERROR: ${RAW_DIR} が見つかりません" >&2
        exit 1
    fi
    shopt -s nullglob
    RAW_CSVS=("${RAW_DIR}"/*.csv)
    shopt -u nullglob
    if [ "${#RAW_CSVS[@]}" -eq 0 ]; then
        echo "[all.sh] ERROR: ${RAW_DIR} 直下に *.csv がありません" >&2
        exit 1
    fi
    step "STEP 1/2: classify (実名 ${#RAW_CSVS[@]} ファイル → ${PARQUET})"
    classify_args=(--csv-dir "${RAW_DIR}" --output "${PARQUET}")
    if [ "$NO_LLM" = "1" ]; then
        classify_args+=(--no-llm)
    fi
    "$PYTHON" -m src.cli classify "${classify_args[@]}"
fi

# --- STEP 2: export-pdf ----------------------------------------------------
step "STEP 2/2: export-pdf → ${OUT_DIR}"
pdf_args=(--parquet "${PARQUET}" --output-dir "${OUT_DIR}" --min-cases "${MIN_CASES}")
if [ -n "${ONLY_DEPT}" ]; then
    pdf_args+=(--dept "${ONLY_DEPT}")
fi
"$PYTHON" -m src.cli export-pdf "${pdf_args[@]}"

# --- 完了 -------------------------------------------------------------------
if [ ! -d "${OUT_DIR}" ] || [ -z "$(ls -A "${OUT_DIR}" 2>/dev/null)" ]; then
    echo "[all.sh] WARNING: ${OUT_DIR} が空です" >&2
    exit 1
fi

step "完了"
echo "  生成先: ${OUT_DIR}"
echo "  Finder で開く: open ${OUT_DIR}"

# macOS なら Finder で開く
if command -v open >/dev/null 2>&1; then
    open "${OUT_DIR}" 2>/dev/null || true
fi
