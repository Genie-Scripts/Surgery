#!/usr/bin/env bash
# データ更新フロー: anonymize → classify → export-html
#
# 前提: data/raw/ に最新の生 CSV を配置済みであること
# 出力: docs/index.html (GitHub Pages 配信用)
#
# `git add docs/ && commit && push` は意図的に手動。生成 HTML をブラウザで
# 確認してから公開する運用を推奨（spec §8.2 / 機能 4）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# venv の python を優先（worktree から実行する場合は親リポジトリの venv も探す）。
# .venv が存在するなら必ずそれを使う。dangling symlink 等で壊れている場合に
# system python3（pandas 未導入）へ無言フォールバックすると
# 「ModuleNotFoundError: No module named 'pandas'」という紛らわしいエラーに
# なるため、その場合は再作成を促してエラー終了する。
_py_works() { [ -x "$1" ] && "$1" -c '' >/dev/null 2>&1; }

PARENT_VENV="$(cd ../../.. 2>/dev/null && pwd)/.venv/bin/python"
if [ -d ".venv" ]; then
    if _py_works ".venv/bin/python"; then
        PYTHON=".venv/bin/python"
    else
        echo "[deploy.sh] ERROR: .venv が壊れています（.venv/bin/python を実行できません）。" >&2
        echo "[deploy.sh]   再作成: rm -rf .venv && uv venv --python 3.11 && uv pip install -e ." >&2
        exit 1
    fi
elif _py_works "${PARENT_VENV}"; then
    PYTHON="${PARENT_VENV}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "[deploy.sh] ERROR: python が見つかりません" >&2
    exit 1
fi

SKIP_ANONYMIZE=0
NO_LLM=0
DRY_RUN=0

usage() {
    cat <<'EOF'
使い方: scripts/deploy.sh [OPTIONS]

データ更新フロー (anonymize → classify → export-html)

OPTIONS:
  --skip-anonymize  匿名化をスキップ（既存 anonymized_data.csv を流用）
  --no-llm          LLM 第 2 段をスキップ（regex 第 1 段のみで分類）
  --dry-run         anonymize はドライラン、以降は通常実行
  -h, --help        このヘルプを表示
EOF
}

for arg in "$@"; do
    case "$arg" in
        --skip-anonymize) SKIP_ANONYMIZE=1 ;;
        --no-llm)         NO_LLM=1 ;;
        --dry-run)        DRY_RUN=1 ;;
        -h|--help)        usage; exit 0 ;;
        *)
            echo "[deploy.sh] 不明なオプション: $arg" >&2
            usage >&2
            exit 1
            ;;
    esac
done

step() {
    printf '\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n' "$*"
}

# --- STEP 1: anonymize ----------------------------------------------------
if [ "$SKIP_ANONYMIZE" = "1" ]; then
    step "STEP 1/3: anonymize (skipped)"
else
    step "STEP 1/3: anonymize"
    if [ "$DRY_RUN" = "1" ]; then
        "$PYTHON" -m src.cli anonymize --dry-run
    else
        "$PYTHON" -m src.cli anonymize
    fi
fi

# --- STEP 2: classify -----------------------------------------------------
step "STEP 2/3: classify"
classify_args=()
if [ "$NO_LLM" = "1" ]; then
    classify_args+=(--no-llm)
fi
"$PYTHON" -m src.cli classify "${classify_args[@]}"

# --- STEP 3: export-html --------------------------------------------------
step "STEP 3/3: export-html"
"$PYTHON" -m src.cli export-html

# --- 確認案内 -------------------------------------------------------------
if [ ! -f "docs/index.html" ]; then
    echo "[deploy.sh] ERROR: docs/index.html が生成されませんでした" >&2
    exit 1
fi

step "完了: docs/index.html"
cat <<'EOF'

次のステップ（手動）:
  1) ブラウザで内容確認
       open docs/index.html
       # ローカルサーバ経由で見たい場合:
       python3 -m http.server 8765 --directory docs

  2) 問題なければ docs/ をコミット & push
       git add docs/index.html
       git commit -m "deploy: docs を更新"
       git push
EOF
