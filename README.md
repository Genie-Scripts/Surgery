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

# 4. 公開用静的 HTML を docs/ に書き出す（GitHub Pages 配信用）
python -m src.cli export-html [--parquet PATH] [--output PATH] \
                              [--period-{a,b,c} 'OLDER/NEWER']

# 5. 診療科別 PDF レポート（実名版、ローカル限り取扱）
python -m src.cli export-pdf [--parquet PATH] [--output-dir DIR] \
                             [--dept 整形外科] [--targets config/department_targets.yaml]

# ヘルプ
python -m src.cli --help
python -m src.cli <command> --help
```

### 診療科別 PDF レポート（実名版）

`export-pdf` は実名版 `classified.parquet` を入力に、診療科ごとに 4 ページ A4 縦の PDF を `local/reports/YYYYMMDD/` 配下に書き出す（`local/` は `.gitignore` 対象）。レイアウトは以下:

- **p1 サマリ**: 4 KPI（件数 / 平均手術時間 / 緊急比率 / 全麻手術件数）を「今年度 YTD vs 昨年同期」で対比 ＋ 月次推移 2 系列（件数・平均手術時間）
- **p2 全麻手術件数 vs 目標**: 週次バー（達成週=青／未達=グレー、目標横線）＋ 月次 2 系列
- **p3 術者ランキング**: 執刀医のみ top20 ／ 執刀＋助手 top20、いずれも昨年同期件数併記
- **p4 カテゴリ・術式**: 4 カテゴリ件数（今年度 vs 昨年同期）＋ カテゴリ別月次 2x2 ＋ 主要術式 top10 ＋ 主要術後病名 top10

集計は **月締め**（集計終端 = 当日の前月末日）、年度は 4 月開始。件数 < 30 の診療科は出力しない。

目標値（週あたり全麻手術件数）は `config/department_targets.yaml` に診療科ごとに設定する。未設定の科は週次グラフで目標横線・達成率を表示せず実績のみ描画する。

#### 全科一括スクリプト

```bash
./scripts/all.sh                        # 実名版 data/raw/ → classify → 全科 PDF
./scripts/all.sh --skip-classify        # 既存 classified_realname.parquet を流用
./scripts/all.sh --no-llm               # LLM 第 2 段をスキップ
./scripts/all.sh --dept 整形外科        # 特定科のみ
```

`scripts/all.sh` は `data/raw/*.csv` を `--csv-dir` で `classify` に渡し、実名 parquet (`data/aggregated/classified_realname.parquet`、`.gitignore` 対象) を経由して `local/reports/YYYYMMDD/` に PDF を書き出す。公開用の `classified.parquet` とはファイルを分けているので、`export-html` の出力には影響しない。

#### 前提 native 依存

PDF レンダリングは WeasyPrint を使用するため Mac では以下の Homebrew パッケージが必要:

```bash
brew install pango  # cairo / harfbuzz / glib なども一緒に入る
```

PNG 化には `kaleido` を使用（pip 経由でインストール、追加 native 依存なし）。

`anonymize` は `src/anonymize.py` 単独実行 (`python -m src.anonymize`) でも同等。

### 公開フロー（機能 4 / spec §8.2 C ハイブリッド）

`scripts/deploy.sh` がデータ更新フロー（anonymize → classify → export-html）を一括実行する。

```bash
./scripts/deploy.sh                  # 標準フロー
./scripts/deploy.sh --skip-anonymize # 既存 anonymized_data.csv を流用
./scripts/deploy.sh --no-llm         # LLM 第 2 段をスキップ（regex のみ）
./scripts/deploy.sh --dry-run        # anonymize はドライラン
```

出力 `docs/index.html` は単一ファイルの Plotly ダッシュボード（術者軸非表示、A/B/C 3 期間比較トグル付き）。`git add docs/ && commit && push` は **意図的に手動** にしている（生成内容をブラウザで確認してから公開する運用）。GitHub Pages は repo Settings → Pages で「Branch: main / Folder: `/docs`」を選択。

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
│   ├── cli.py          # 統合 CLI（anonymize / classify / summary / export-html）
│   ├── export_html.py  # 公開用静的 HTML 生成（Plotly + 3 期間比較トグル）
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
├── docs/               # 公開用静的HTML（GitHub Pages 配信先 / 機能 4）
├── local/              # ローカル実名版出力 ※Gitコミット禁止
├── scripts/            # deploy.sh（anonymize → classify → export-html）
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
