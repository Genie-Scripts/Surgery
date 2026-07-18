"""ローカル LLM 第 2 段（カテゴリ判定）のモデル横並びベンチマーク。

目的:
  oMLX の配信モデル更新後に、現行 Swallow-8B と他モデル（Qwen3.6-27B 等）を同一
  プロンプトで比較し、置き替え判断の材料を出す。本番ロジック（src/classify_llm.
  _build_prompt / _parse_response）をそのまま流用するので、prompt 改修にも追従する。

評価軸:
  1. 正答（gold）: 人手ラベル付きケースに対する exact-match 正答率（raw LLM 判定）
  2. 整形追従（format）: 実データ曖昧ケースで `CATEGORIES: [...]` をパースできた率
  3. 一致度（agree）  : 同ケースで baseline モデル（既定 = 現行 Swallow）との exact 一致率
  4. レイテンシ       : 1 件あたり中央値 / 平均（秒）

使い方:
  python -m scripts.bench_llm                       # 既定モデル群・gold + 実データ 30 件
  python -m scripts.bench_llm --models A B C        # 比較モデルを指定
  python -m scripts.bench_llm --n 60 --csv data/raw/op.csv
  python -m scripts.bench_llm --baseline <model>    # 一致度の基準モデル

注: 評価対象は raw LLM 出力（ハードガード/regex OR 前）。モデル素の判定力を見るため。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from src.classify import classify, load_rules
from src.classify_llm import _build_prompt, _parse_response
from src.ingest import load
from src.llm_client import LLMClient

ROOT = Path(__file__).resolve().parent.parent
LLM_CONFIG = ROOT / "config" / "llm_config.yaml"

# 現行本番モデル（= 既定 baseline）。oMLX 配信 id（/v1/models と完全一致）。
CURRENT_MODEL = "Llama-3.1-Swallow-8B-Instruct-v0.5"

# 既定の比較対象（oMLX 配信中の候補。--models で上書き可）
DEFAULT_MODELS = [
    CURRENT_MODEL,
    "Qwen3.6-27B-OptiQ-4bit",
]

# 人手ラベル付き gold ケース。値は「raw LLM がこう答えるべき」期待 ID 集合。
# tests/test_classify_llm.py の確定ケース + 高確信の追加分。
GOLD: list[tuple[str, set[str]]] = [
    ("水晶体再建術（眼内レンズを挿入する場合）（その他）", set()),          # 白内障。該当なし
    ("硝子体内注射", set()),                                                  # 該当なし
    ("骨折観血的手術（大腿）", set()),                                        # 骨折整復。人工関節ではない
    ("付属器摘出術（良性・腹腔鏡）", set()),                                  # 良性明記
    ("脳腫瘍摘出術", {"malignant_tumor"}),                                    # 悪性想定（医療知識拡張）
    ("胃悪性腫瘍手術（噴門側胃切除術）", {"malignant_tumor"}),               # 悪性明記
    (
        "腹腔鏡下前立腺悪性腫瘍手術（内視鏡下手術用支援機器使用）",
        {"malignant_tumor", "robot_assisted_davinci"},
    ),
    ("人工股関節置換術", {"artificial_joint"}),                               # 人工関節
    (
        "人工関節置換術（膝）（ロボット）",
        {"artificial_joint", "robot_assisted_other"},
    ),
]


def sample_real_cases(csv_path: Path, n: int, seed: int = 0) -> list[str]:
    """実データから LLM 判定要（0/2+ ヒット）のユニーク確定術式を n 件サンプリング。"""
    df = load(csv_path)
    c = classify(df)
    amb = (
        c.loc[c["LLM判定要"], "確定術式"]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    return amb.sample(n=min(n, len(amb)), random_state=seed).tolist()


def run_model(
    client: LLMClient,
    model: str,
    gold: list[tuple[str, set[str]]],
    real: list[str],
    rules,
    valid_ids: set[str],
    max_tokens: int | None = None,
) -> dict:
    """1 モデル分を実行し、指標 dict と raw 予測（gold/real）を返す。

    max_tokens を渡すと config 既定を上書き（thinking 系に思考枠を与える用）。
    """
    client.config.model = model

    def _predict(text: str) -> frozenset[str] | None:
        """1 件叩いてパース結果を返す（パース不能/例外は None）。"""
        resp = client.chat(_build_prompt(text, rules), max_tokens=max_tokens)
        parsed = _parse_response(resp, valid_ids)
        return frozenset(parsed) if parsed is not None else None

    # --- gold: exact-match 正答率（誤答内訳のため予測も保持）---
    gold_correct = 0
    gold_parse_ok = 0
    gold_preds: list[tuple[str, set[str], frozenset[str] | None]] = []
    for text, expected in gold:
        try:
            got = _predict(text)
        except Exception as e:  # noqa: BLE001 — ベンチなので握りつぶして続行
            print(f"    [gold] {model} 呼び出し失敗: {e}", file=sys.stderr)
            got = None
        if got is not None:
            gold_parse_ok += 1
            if set(got) == expected:
                gold_correct += 1
        gold_preds.append((text, expected, got))

    # --- real: 整形追従率・レイテンシ・予測収集 ---
    latencies: list[float] = []
    parse_ok = 0
    preds: dict[str, frozenset[str] | None] = {}
    for text in real:
        t0 = time.perf_counter()
        try:
            got = _predict(text)
        except Exception as e:  # noqa: BLE001
            print(f"    [real] {model} 呼び出し失敗: {e}", file=sys.stderr)
            preds[text] = None
            continue
        latencies.append(time.perf_counter() - t0)
        if got is not None:
            parse_ok += 1
        preds[text] = got

    return {
        "model": model,
        "gold_total": len(gold),
        "gold_correct": gold_correct,
        "gold_parse_ok": gold_parse_ok,
        "gold_preds": gold_preds,
        "real_total": len(real),
        "parse_ok": parse_ok,
        "lat_median": statistics.median(latencies) if latencies else float("nan"),
        "lat_mean": statistics.mean(latencies) if latencies else float("nan"),
        "preds": preds,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ローカル LLM カテゴリ判定ベンチ")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="比較モデル名")
    ap.add_argument("--baseline", default=CURRENT_MODEL, help="一致度の基準モデル")
    ap.add_argument("--csv", default=str(ROOT / "data" / "raw" / "op.csv"))
    ap.add_argument("--n", type=int, default=30, help="実データ曖昧ケースのサンプル数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-tokens", type=int, default=None,
        help="max_tokens を上書き（thinking 系に思考枠を与える。例 2048）",
    )
    ap.add_argument(
        "--timeout", type=int, default=None,
        help="1 呼び出しのタイムアウト秒を上書き（大型/thinking 系向け。例 300）",
    )
    args = ap.parse_args(argv)

    rules = load_rules()
    valid_ids = {r.category_id for r in rules}

    client = LLMClient(LLM_CONFIG)
    if not client.is_available():
        print("oMLX が利用できません（サーバ未起動 / モデル未配信）", file=sys.stderr)
        return 1
    if args.timeout is not None:
        client.config.timeout = args.timeout

    real = sample_real_cases(Path(args.csv), args.n, args.seed)
    extra = []
    if args.max_tokens is not None:
        extra.append(f"max_tokens={args.max_tokens}")
    if args.timeout is not None:
        extra.append(f"timeout={args.timeout}s")
    print(f"gold: {len(GOLD)} 件 / 実データ曖昧サンプル: {len(real)} 件 "
          f"(csv={args.csv}, seed={args.seed}{', ' + ', '.join(extra) if extra else ''})\n")

    results = []
    for model in args.models:
        print(f"▶ {model} を実行中 ...", file=sys.stderr)
        results.append(
            run_model(client, model, GOLD, real, rules, valid_ids, args.max_tokens)
        )

    # baseline 予測（一致度の基準）
    base = next((r for r in results if r["model"] == args.baseline), None)

    # --- 結果テーブル ---
    print("\n| モデル | gold正答 | gold整形 | 実整形追従 | baseline一致 | 中央(s) | 平均(s) |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        gold_acc = f"{r['gold_correct']}/{r['gold_total']}"
        gold_fmt = f"{r['gold_parse_ok']}/{r['gold_total']}"
        fmt = f"{r['parse_ok']}/{r['real_total']} ({100*r['parse_ok']/max(r['real_total'],1):.0f}%)"
        if base is not None:
            both = [
                t for t in r["preds"]
                if r["preds"][t] is not None and base["preds"].get(t) is not None
            ]
            agree = sum(1 for t in both if r["preds"][t] == base["preds"][t])
            agree_s = (
                "— (基準)" if r["model"] == args.baseline
                else f"{agree}/{len(both)} ({100*agree/max(len(both),1):.0f}%)"
            )
        else:
            agree_s = "n/a"
        print(f"| `{r['model'].split('/')[-1]}` | {gold_acc} | {gold_fmt} | {fmt} | "
              f"{agree_s} | {r['lat_median']:.2f} | {r['lat_mean']:.2f} |")

    # --- gold の誤答内訳（run_model で取得済みの予測を再利用）---
    print("\n### gold 誤答内訳")
    for r in results:
        misses = [
            (text, exp, got)
            for text, exp, got in r["gold_preds"]
            if (set(got) if got is not None else None) != exp
        ]
        label = r["model"].split("/")[-1]
        if not misses:
            print(f"- `{label}`: 全問正解")
        else:
            print(f"- `{label}`: {len(misses)} 件誤答")
            for text, exp, got in misses:
                got_s = sorted(got) if got is not None else "PARSE_FAIL"
                print(f"    - {text[:42]} | 期待={sorted(exp) or '[]'} 実={got_s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
