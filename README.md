# Surgery

外科手術データから **術者評価** と **カテゴリ別分析** を可視化するダッシュボード。

- **ローカル実名版**: 自分の Mac で Streamlit として動かす
- **匿名化 web 版**: 静的 HTML を GitHub Pages 等で配信（Phase 2）

詳細な要件は [spec.md](./spec.md) を参照。

---

## 概要

| 項目 | 内容 |
|---|---|
| 入力 | 手術データ CSV（術者 ID 事前匿名化済み） |
| 抽出カテゴリ | 悪性腫瘍 / 人工関節 / ロボット支援（MVP） |
| KPI | 件数 / 総手術時間 / 平均手術時間 / 月次推移 / 同僚比較 / 難度補正 |
| LLM | Ollama + Llama-3.1-Swallow-8B-Instruct-v0.5 (Q6_K) |
| UI | Streamlit |

---

## 必要環境

- macOS（Apple Silicon 推奨）
- Python 3.11 以上
- [Ollama](https://ollama.com/)（既導入想定）
- メモリ 16GB 以上（Q6_K 推奨は 64GB）

---

## セットアップ

### 1. リポジトリをクローン

```bash
cd /Users/genie/dev/ai-apps
git clone <remote> Surgery   # もしくはローカル初期化
cd Surgery
```

新規ローカル初期化の場合:

```bash
cd /Users/genie/dev/ai-apps/Surgery
git init
git add spec.md README.md
git commit -m "chore: initial spec and readme"
```

### 2. Python 仮想環境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'   # pyproject.toml の依存を取り込む（実装着手後）
```

### 3. Ollama モデルを pull

Phase A（カテゴリ自動判定）で使用する LLM をローカルにダウンロード：

```bash
ollama pull hf.co/mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf:Q6_K
```

Ollama サーバが未起動なら自動で起動する仕組みを `src/llm_client.py` に内包する予定（Outpatient-Dashboard と同パターン）。手動起動する場合：

```bash
ollama serve
```

### 4. 設定ファイル（実装着手後に整備）

```
config/
├── categories.yaml        # カテゴリ抽出ルール（YAML）
├── llm_config.yaml        # Ollama 接続設定
└── peers.csv              # 同僚比較対象（必要に応じ）
```

それぞれ雛形は実装フェーズで生成する。

### 5. データ配置

```bash
mkdir -p data/raw
cp /path/to/anonymized_data.csv data/raw/
```

> ⚠️ **`data/raw/` は `.gitignore` 対象**（Git にコミットしない）。

---

## 使い方

### Streamlit ダッシュボード

```bash
streamlit run app/main.py
```

ブラウザで `http://localhost:8501` を開くと、**マルチページ構成**でナビゲーションがサイドバーに表示される:

| ページ | 内容 |
|---|---|
| **main** | ランディング: 総件数・期間・執刀医数・診療科数のサマリ + 各ページ案内 |
| **全体 KPI** | 件数 / 総時間 / 平均時間 / 緊急比率 + 月次推移（件数・平均時間） |
| **術者別** | 術者ランキング + 特定術者ドリルダウン + 期間比較（サイドバー expander で有効化） |
| **カテゴリ別** | 4 カテゴリ件数（悪性腫瘍 / 人工関節 / ダヴィンチ系・非ダヴィンチ系ロボット）+ 分類元内訳 + カテゴリ別月次 + カテゴリ × 診療科クロス |
| **月次推移** | 件数・平均時間・カテゴリ・緊急比率・診療科別 top5 の時系列分析 |

サイドバー共通フィルタ（**ページ切替後も維持**される）:
- **CSV パス**: デフォルトは `data/raw/anonymized/anonymized_data.csv`（無ければルート直下の `anonymized_data.csv`）
- **LLM 第 2 段を適用**（Swallow-8B + ハードガード。Ollama 未起動時は自動で regex のみに降格）
- **執刀医モード**（`執刀医のみ` / `執刀医＋助手を含む`）
- **申込区分** / **実施診療科** / **全身麻酔のみ** で絞り込み
- **期間比較**（expander で展開、術者別ページに比較表が表示）

### CLI

```bash
# 1. 生 CSV を匿名化 (data/raw/*.csv → data/raw/anonymized/anonymized_data.csv)
python -m src.cli anonymize [--dry-run]

# 2. 匿名化済み CSV を読み込み、regex + LLM 第 2 段で分類して parquet 保存
python -m src.cli classify [--csv PATH] [--no-llm]

# 3. 保存済み parquet からカテゴリ・分類元のサマリを表示
python -m src.cli summary [--parquet PATH]

# ヘルプ
python -m src.cli --help
python -m src.cli <command> --help
```

`anonymize` は `src/anonymize.py` 単独実行 (`python -m src.anonymize`) でも同等。

### LLM 第 2 段の挙動

`src/classify_llm.py` は `LLM判定要 == True`（regex 0 件ヒット or 2 件以上ヒット）の症例に対して Swallow-8B を呼び出す。`config/llm_config.yaml` で接続先・モデルを設定。

- **キャッシュ**: `data/llm_cache/categories.json` (sha256 ベース、ルールバージョン込み)
- **OR 結合**: regex 結果と LLM 結果を OR 結合し、LLM が拾い損ねた regex 確実ケースを保護
- **ハードガード**（spec §7.2.2）: 規則ベースで一意判定可能なロボット 2 区分と「良性」明記の悪性腫瘍は決定論的に補正

### ローカル実名版ビルド（Phase 2）

```bash
./scripts/local_build.sh
```

`local/` 配下に静的レポートが生成される（`.gitignore` 対象）。

---

## ディレクトリ構成

```
Surgery/
├── README.md           # 本ファイル
├── spec.md             # 要件定義
├── CLAUDE.md           # Claude Code 向けプロジェクト指示書
├── pyproject.toml      # 依存と lint 設定
├── .gitignore
│
├── src/                # ロジック本体（pip install -e . で `src.*` 名前空間に解決）
│   ├── ingest.py       # CSV 読込・文字コード正規化（CP932 / UTF-8 自動判別）
│   ├── classify.py     # カテゴリ抽出 第 1 段（regex + categories.yaml）
│   ├── classify_llm.py # カテゴリ抽出 第 2 段（LLM + ハードガード + 永続キャッシュ）
│   ├── aggregate.py    # KPI 集計純関数群
│   ├── llm_client.py   # Ollama / OpenAI 互換クライアント
│   ├── anonymize.py    # 生 CSV → 匿名化済み CSV
│   ├── cli.py          # 統合 CLI（anonymize / classify / summary）
│   ├── ui/             # Streamlit 共通 UI（data_loader, filters）
│   └── core/           # 同僚比較 / 難度補正（Phase 2）
│
├── app/                # Streamlit マルチページ エントリ
│   ├── main.py         # ランディング
│   └── pages/          # 1_全体KPI / 2_術者別 / 3_カテゴリ別 / 4_月次推移
│
├── config/             # ルール・接続設定
├── data/
│   ├── raw/            # ※Gitコミット禁止
│   ├── normalized/
│   ├── aggregated/
│   └── llm_cache/
│
├── docs/               # 公開用静的HTML（Phase 2）
├── local/              # ローカル実名版出力 ※Gitコミット禁止
├── scripts/            # pull.sh / deploy.sh / local_build.sh
├── tests/              # pytest
└── tools/              # ラベリングUI 等（Phase 2）
```

完全なディレクトリ構成と各ファイルの役割は [spec.md §9](./spec.md) を参照。

---

## データ・セキュリティ

| 種別 | 場所 | Git 管理 |
|---|---|---|
| 生データ CSV | `data/raw/` | **コミット禁止** |
| 補助対応表（実名↔匿名） | `config/master_key.csv` 等 | **コミット禁止** |
| ローカル実名版出力 | `local/` | **コミット禁止** |
| 集計済みデータ | `data/aggregated/` | コミット可 |
| LLM キャッシュ | `data/llm_cache/` | 不要（再生成可） |

`.gitignore` で上記を除外する。新規ファイル追加前に `git status` で意図しない含有がないか必ず確認。

---

## 文字コード問題への対処

入力 CSV は **Windows 出力（Shift-JIS / CP932）→ Mac 編集（UTF-8）** ワークフローのため文字化けが起きやすい。`src/ingest.py` で以下のフォールバック順に試行する：

1. `utf-8-sig`（BOM 付き）
2. `cp932`
3. `shift_jis`
4. `utf-8`

判別失敗時は `chardet` で検出するエラー処理を入れる。詳細は実装時に。

---

## トラブルシューティング

### Ollama に接続できない

```bash
curl http://localhost:11434/
```

200 が返らない場合は `ollama serve` を別ターミナルで起動。

### Streamlit のポートが占有されている

```bash
streamlit run app/main.py --server.port 8502
```

### LLM 出力が定型文（フォールバック）ばかりになる

`config/llm_config.yaml` のモデル名を確認し、`ollama list` で pull 済みかチェック。Outpatient-Dashboard と同様、思考型モデルではなく **instruct** 系（Swallow / Qwen-Instruct）を使うこと。

---

## ライセンス

未定（個人プロジェクト）。

---

## 関連プロジェクト

- [Outpatient-Dashboard](../Outpatient-Dashboard/) — 外来効率化ダッシュボード（経営企画室向け）
- [Outpatient-Restructuring](../Outpatient-Restructuring/) — 外来枠再編診断

3 プロジェクトとも同一マシンで運用するが、Surgery はリポジトリ・依存・運用すべて独立。LLM クライアントの設計だけ Outpatient-Dashboard から踏襲する。
