#!/usr/bin/env bash
# ローカル実名版 Streamlit を起動して http://localhost:8501 を開く。
#
# - 入力は data/raw/ 直下の *.csv（実名入り、ファイル名任意・複数可）
#   → SURGERY_CSV にディレクトリパスを渡し、UI 側で concat → 重複除去 → 正規化
# - Streamlit が既に 8501 を掴んでいる場合は強制再起動（最新 CSV で読み直し）
# - ログは /tmp/surgery_streamlit.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PORT=8501
URL="http://localhost:${PORT}"
LOG="/tmp/surgery_streamlit.log"
RAW_DIR="${ROOT}/data/raw"

if [ ! -d "${RAW_DIR}" ]; then
    osascript -e "display alert \"Surgery 実名版\" message \"生データ ディレクトリが見つかりません:\n${RAW_DIR}\" as critical" >/dev/null 2>&1 || true
    echo "[local_run.sh] ERROR: ${RAW_DIR} が見つかりません" >&2
    exit 1
fi

# data/raw/ 直下に *.csv が 1 つでもあるか（サブディレクトリは無視）
shopt -s nullglob
RAW_CSVS=("${RAW_DIR}"/*.csv)
shopt -u nullglob
if [ "${#RAW_CSVS[@]}" -eq 0 ]; then
    osascript -e "display alert \"Surgery 実名版\" message \"生 CSV が見つかりません:\n${RAW_DIR}/*.csv\" as critical" >/dev/null 2>&1 || true
    echo "[local_run.sh] ERROR: ${RAW_DIR} 直下に *.csv がありません" >&2
    exit 1
fi
echo "[local_run.sh] 取り込み対象 CSV: ${#RAW_CSVS[@]} 件"
for f in "${RAW_CSVS[@]}"; do
    echo "  - $(basename "$f")"
done

if [ -x ".venv/bin/streamlit" ]; then
    STREAMLIT=".venv/bin/streamlit"
elif command -v streamlit >/dev/null 2>&1; then
    STREAMLIT="streamlit"
else
    osascript -e 'display alert "Surgery 実名版" message "streamlit が見つかりません。.venv をセットアップしてください。" as critical' >/dev/null 2>&1 || true
    echo "[local_run.sh] ERROR: streamlit が見つかりません" >&2
    exit 1
fi

export SURGERY_CSV="${RAW_DIR}"

# 既存プロセスがあれば必ず止めてから起動し直す。
# Streamlit の @st.cache_data はプロセス内メモリ。.app をダブルクリック
# したら最新の CSV で読み直す、というメンタルモデルにしたいので強制再起動。
EXISTING_PIDS="$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)"
if [ -n "${EXISTING_PIDS}" ]; then
    echo "[local_run.sh] ポート ${PORT} を占有中のプロセスを停止: ${EXISTING_PIDS}"
    kill ${EXISTING_PIDS} 2>/dev/null || true
    # graceful 停止を最大 5 秒待ち、ダメなら SIGKILL
    for _ in $(seq 1 10); do
        sleep 0.5
        lsof -ti tcp:"${PORT}" >/dev/null 2>&1 || break
    done
    if lsof -ti tcp:"${PORT}" >/dev/null 2>&1; then
        kill -9 ${EXISTING_PIDS} 2>/dev/null || true
        sleep 1
    fi
fi

echo "[local_run.sh] Streamlit を起動: ${URL} (log: ${LOG})"
nohup "${STREAMLIT}" run app/main.py \
    --server.port "${PORT}" \
    --server.headless true \
    --browser.gatherUsageStats false \
    >"${LOG}" 2>&1 &
disown || true

# 起動完了をポーリング（最大 30 秒）
for _ in $(seq 1 60); do
    if curl -sf "${URL}" >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

open "${URL}"
