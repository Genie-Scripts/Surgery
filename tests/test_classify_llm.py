"""src/classify_llm.py の回帰テスト。

oMLX は呼ばず、`FakeClient` で LLM レスポンスを差し替える。
"""

from __future__ import annotations

import time
import urllib.error
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.classify import classify, load_rules
from src.classify_llm import (
    _apply_hard_guard,
    _build_prompt,
    _parse_response,
    classify_one,
    classify_with_llm,
)

# --- _apply_hard_guard ---------------------------------------------------


def test_apply_hard_guard_no_rule_passes_through():
    assert _apply_hard_guard(["artificial_joint"], "人工関節置換術") == ["artificial_joint"]


def test_apply_hard_guard_required_present():
    assert _apply_hard_guard(
        ["robot_assisted_davinci"],
        "腹腔鏡下前立腺手術（内視鏡下手術用支援機器使用）",
    ) == ["robot_assisted_davinci"]


def test_apply_hard_guard_required_absent():
    assert _apply_hard_guard(["robot_assisted_davinci"], "腹腔鏡下胆嚢摘出術") == []


def test_apply_hard_guard_forbidden_ryousei():
    assert _apply_hard_guard(["malignant_tumor"], "卵巣腫瘍摘出術（良性・腹腔鏡）") == []


def test_apply_hard_guard_forbidden_skin_k005():
    assert _apply_hard_guard(
        ["malignant_tumor"], "皮膚、皮下腫瘍摘出術（露出部）（長径２ｃｍ未満）"
    ) == []


def test_apply_hard_guard_keeps_kept_drops_others():
    # 同時複数 ID で個別に判断される
    result = _apply_hard_guard(
        ["robot_assisted_other", "robot_assisted_davinci"],
        "人工関節置換術（膝）（ロボット）",
    )
    assert result == ["robot_assisted_other"]


def test_apply_hard_guard_real_davinci_passes_with_other_dropped():
    result = _apply_hard_guard(
        ["malignant_tumor", "robot_assisted_davinci"],
        "腹腔鏡下前立腺悪性腫瘍手術（内視鏡下手術用支援機器使用）",
    )
    assert set(result) == {"malignant_tumor", "robot_assisted_davinci"}


# --- _parse_response -----------------------------------------------------


VALID_IDS = {
    "malignant_tumor",
    "artificial_joint",
    "robot_assisted_davinci",
    "robot_assisted_other",
}


def test_parse_response_two_ids():
    assert _parse_response(
        "CATEGORIES: [malignant_tumor, robot_assisted_davinci]", VALID_IDS
    ) == ["malignant_tumor", "robot_assisted_davinci"]


def test_parse_response_empty_list():
    assert _parse_response("CATEGORIES: []", VALID_IDS) == []


def test_parse_response_invalid_ids_filtered():
    assert _parse_response(
        "CATEGORIES: [malignant_tumor, unknown_category]", VALID_IDS
    ) == ["malignant_tumor"]


def test_parse_response_no_match_returns_none():
    assert _parse_response("分類できません。", VALID_IDS) is None


def test_parse_response_quoted_ids():
    assert _parse_response(
        "CATEGORIES: ['malignant_tumor', \"robot_assisted_davinci\"]", VALID_IDS
    ) == ["malignant_tumor", "robot_assisted_davinci"]


def test_parse_response_full_width_colon():
    assert _parse_response("CATEGORIES：[malignant_tumor]", VALID_IDS) == [
        "malignant_tumor"
    ]


def test_parse_response_takes_last_match():
    text = (
        "考えると CATEGORIES: [artificial_joint] のように見えるが、\n"
        "最終的には\nCATEGORIES: [malignant_tumor]"
    )
    assert _parse_response(text, VALID_IDS) == ["malignant_tumor"]


def test_parse_response_no_brackets_single():
    # oMLX 版 Swallow が角括弧を落として返すケース
    assert _parse_response("CATEGORIES: malignant_tumor", VALID_IDS) == [
        "malignant_tumor"
    ]


def test_parse_response_no_brackets_comma_list():
    assert _parse_response(
        "CATEGORIES: malignant_tumor, robot_assisted_davinci", VALID_IDS
    ) == ["malignant_tumor", "robot_assisted_davinci"]


def test_parse_response_no_brackets_none_text_is_empty_list():
    # 角括弧なしで該当 ID 無し → パース不能(None)ではなく「該当なし([])」扱い
    assert _parse_response("CATEGORIES: 該当なし", VALID_IDS) == []


# --- _build_prompt -------------------------------------------------------


def test_build_prompt_contains_text_and_ids():
    rules = load_rules()
    prompt = _build_prompt("水晶体再建術", rules)
    assert "水晶体再建術" in prompt
    for r in rules:
        assert r.category_id in prompt
        assert r.display_name in prompt
    assert "CATEGORIES:" in prompt


# --- FakeClient ----------------------------------------------------------


class _FakeConfig:
    model = "fake-model"


class FakeClient:
    """LLMClient の最小モック。chat() が予め指定したレスポンスを返す。

    `responses_by_substring`: 確定術式テキストの部分一致で応答を切替える dict。
    `default_response`: どれにも当たらないときのデフォルト応答。
    `raise_exc`: 設定されていれば chat() で例外を送出する。
    """

    def __init__(
        self,
        responses_by_substring: dict[str, str] | None = None,
        default_response: str = "CATEGORIES: []",
        raise_exc: Exception | None = None,
        available: bool = True,
    ) -> None:
        self.responses_by_substring = responses_by_substring or {}
        self.default_response = default_response
        self.raise_exc = raise_exc
        self.available = available
        self.config = _FakeConfig()
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return self.available

    def chat(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(prompt)
        if self.raise_exc:
            raise self.raise_exc
        for substr, resp in self.responses_by_substring.items():
            if substr in prompt:
                return resp
        return self.default_response


# --- classify_one --------------------------------------------------------


@pytest.fixture
def rules():
    return load_rules()


def test_classify_one_cache_miss_calls_llm(rules):
    cache: dict[str, list[str]] = {}
    fake = FakeClient(default_response="CATEGORIES: [malignant_tumor]")
    ids, src = classify_one("脳腫瘍摘出術", rules, fake, cache)
    assert ids == ["malignant_tumor"]
    assert src == "llm"
    assert len(fake.calls) == 1
    assert len(cache) == 1


def test_classify_one_cache_hit_skips_llm(rules):
    cache: dict[str, list[str]] = {}
    fake = FakeClient(default_response="CATEGORIES: [malignant_tumor]")
    classify_one("脳腫瘍摘出術", rules, fake, cache)
    fake.calls.clear()
    ids, src = classify_one("脳腫瘍摘出術", rules, fake, cache)
    assert ids == ["malignant_tumor"]
    assert src == "cache"
    assert len(fake.calls) == 0


def test_classify_one_url_error_fallback(rules):
    fake = FakeClient(raise_exc=urllib.error.URLError("refused"))
    ids, src = classify_one("脳腫瘍摘出術", rules, fake, {})
    assert ids == []
    assert src == "llm_fallback"


def test_classify_one_parse_failure_fallback(rules):
    fake = FakeClient(default_response="分類できません。")
    ids, src = classify_one("脳腫瘍摘出術", rules, fake, {})
    assert ids == []
    assert src == "llm_fallback"


def test_classify_one_cache_stores_raw_guard_applied_on_read(rules):
    """キャッシュには raw を保存し、取り出し時にガードを適用するため、
    ガード違反データがキャッシュにあっても結果には漏れ出さない。"""
    cache: dict[str, list[str]] = {}
    fake = FakeClient(default_response="CATEGORIES: [robot_assisted_davinci]")
    ids, _ = classify_one("腹腔鏡下胆嚢摘出術", rules, fake, cache)
    assert ids == []

    # キャッシュには raw 判定 (ガード前) が残る
    cached_values = list(cache.values())
    assert ["robot_assisted_davinci"] in cached_values

    # キャッシュ取り出しで再現してもガードで除去される
    ids2, src2 = classify_one("腹腔鏡下胆嚢摘出術", rules, fake, cache)
    assert ids2 == []
    assert src2 == "cache"


# --- classify_with_llm (end-to-end with mock) ----------------------------


def _make_classified_df() -> pd.DataFrame:
    """ingest を経由せず、6 ケースを手で構築 → classify() に通す。"""
    cases = [
        # idx, 確定術式, 期待: regex / LLM / 最終 (ハードガード後)
        "水晶体再建術",  # regex 0 ヒット → LLM 行き
        "腹腔鏡下前立腺悪性腫瘍手術（内視鏡下手術用支援機器使用）",  # regex 2 ヒット → LLM 行き
        "卵巣腫瘍摘出術（良性・腹腔鏡）",  # regex 1 ヒット (腫瘍.*切除でなく良性。実は 0 ヒット)
        "脳腫瘍摘出術",  # regex 0 ヒット → LLM 行き (LLM が悪性判定)
        "腹腔鏡下胆嚢摘出術",  # regex 0 ヒット → LLM 行き (LLM が誤って davinci → ガード)
        "皮膚、皮下腫瘍摘出術（露出部）（長径２ｃｍ未満）",  # 0 ヒット (D2 ハードガード対象)
    ]
    df = pd.DataFrame({"確定術式": cases})
    return classify(df)


def test_classify_with_llm_full_pipeline(rules, tmp_path: Path):
    df = _make_classified_df()
    cache_path = tmp_path / "cache.json"

    fake = FakeClient(
        responses_by_substring={
            "水晶体再建術": "CATEGORIES: []",
            "前立腺悪性腫瘍": "CATEGORIES: [malignant_tumor, robot_assisted_davinci]",
            "卵巣腫瘍": "CATEGORIES: [malignant_tumor]",  # 誤判定 → 良性ガード
            "脳腫瘍": "CATEGORIES: [malignant_tumor]",
            "胆嚢摘出術": "CATEGORIES: [robot_assisted_davinci]",  # 誤判定 → 必須ガード
            "皮膚、皮下腫瘍": "CATEGORIES: [malignant_tumor]",  # 誤判定 → K005 ガード
        },
    )

    out = classify_with_llm(df, rules, fake, cache_path=cache_path)

    # 各行の最終判定
    assert not out.loc[0, "malignant_tumor"]  # 水晶体再建術
    assert out.loc[1, "malignant_tumor"]
    assert out.loc[1, "robot_assisted_davinci"]  # 手術用支援を含むので維持
    assert not out.loc[2, "malignant_tumor"]  # 良性ガードで除去
    assert out.loc[3, "malignant_tumor"]  # 脳腫瘍は LLM が拡張、ガード対象外
    assert not out.loc[4, "robot_assisted_davinci"]  # 手術用支援なしで除去
    assert not out.loc[5, "malignant_tumor"]  # 皮膚、皮下腫瘍摘出術は K005 ガードで除去

    # 分類元
    assert (out["分類元"] == "llm").all()  # 全件 LLM 経由

    # キャッシュが永続化されている
    assert cache_path.exists()


def test_classify_with_llm_preserves_regex_when_llm_drops(rules, tmp_path: Path):
    """regex で True だった判定は LLM が False と言っても OR 結合で維持される (ロボット系除く)。"""
    df = pd.DataFrame({"確定術式": ["手根管開放術\n人工関節置換術"]})  # 2 件regex ヒット候補
    df = classify(df)
    # 人工関節 regex ヒット
    assert df.loc[0, "artificial_joint"]

    fake = FakeClient(default_response="CATEGORIES: []")
    out = classify_with_llm(df, rules, fake, cache_path=tmp_path / "c.json")

    # LLM が空でも regex の artificial_joint は残る (OR 結合)
    assert out.loc[0, "artificial_joint"]


def test_classify_with_llm_skips_when_unavailable(rules, tmp_path: Path):
    """is_available() False のときは regex のみで返す。"""
    df = _make_classified_df()
    fake = FakeClient(available=False)
    out = classify_with_llm(df, rules, fake, cache_path=tmp_path / "c.json")
    # 全行 regex 維持
    assert (out["分類元"] == "regex").all()
    # LLM は呼ばれていない
    assert fake.calls == []


class SleepyFakeClient:
    """並列実行の実時間検証用: chat() 呼び出しごとに `sleep_seconds` 秒スリープする。"""

    def __init__(self, sleep_seconds: float, available: bool = True) -> None:
        self.sleep_seconds = sleep_seconds
        self.available = available
        self.config = _FakeConfig()

    def is_available(self) -> bool:
        return self.available

    def chat(self, prompt: str, **kwargs: Any) -> str:
        time.sleep(self.sleep_seconds)
        return "CATEGORIES: []"


def test_classify_with_llm_parallel_output_matches_serial(rules, tmp_path: Path):
    """max_workers=1 (逐次相当) と並列実行 (max_workers=4) で出力が完全一致することを検証する。

    重複テキストを含め、`分類元` の "llm"/"cache" 振り分け（最初の行だけ "llm"、
    以降は "cache"）が並列化で崩れないことを確認する。
    """
    cases = [
        "水晶体再建術",  # regex 0 ヒット → LLM
        "水晶体再建術",  # 重複
        "脳腫瘍摘出術",  # regex 0 ヒット → LLM (悪性判定)
        "脳腫瘍摘出術",  # 重複
        "脳腫瘍摘出術",  # 重複
        "腹腔鏡下胆嚢摘出術",  # regex 0 ヒット → LLM (誤判定 → ハードガードで除去)
    ]
    df = classify(pd.DataFrame({"確定術式": cases}))

    responses = {
        "水晶体再建術": "CATEGORIES: []",
        "脳腫瘍摘出術": "CATEGORIES: [malignant_tumor]",
        "胆嚢摘出術": "CATEGORIES: [robot_assisted_davinci]",
    }

    fake_serial = FakeClient(responses_by_substring=responses)
    out_serial = classify_with_llm(
        df, rules, fake_serial, cache_path=tmp_path / "serial.json", max_workers=1
    )

    fake_parallel = FakeClient(responses_by_substring=responses)
    out_parallel = classify_with_llm(
        df, rules, fake_parallel, cache_path=tmp_path / "parallel.json", max_workers=4
    )

    pd.testing.assert_frame_equal(out_serial, out_parallel)

    # 重複テキストの分類元振り分け（最初=llm, 以降=cache）が逐次実行時の挙動と一致
    assert list(out_serial["分類元"]) == ["llm", "cache", "llm", "cache", "cache", "llm"]


def test_classify_with_llm_runs_in_parallel(rules, tmp_path: Path):
    """並列実行が直列合計より明確に短いことを検証する（sleep 入り FakeClient）。"""
    n_unique = 8
    sleep_seconds = 0.15
    cases = [f"ダミー未知術式サンプル{i}のみ" for i in range(n_unique)]
    df = classify(pd.DataFrame({"確定術式": cases}))
    assert df["LLM判定要"].sum() == n_unique  # 全件 regex 0 ヒットで LLM 対象であること

    fake = SleepyFakeClient(sleep_seconds=sleep_seconds)
    start = time.monotonic()
    classify_with_llm(df, rules, fake, cache_path=tmp_path / "c.json", max_workers=4)
    elapsed = time.monotonic() - start

    serial_estimate = n_unique * sleep_seconds  # 1.2s
    # 4 並列の理論値は serial_estimate / 4 (=0.3s) 程度。オーバーヘッドを見込んでも
    # 直列合計を大きく下回ることを確認する（flaky にならない緩い閾値）。
    assert elapsed < serial_estimate * 0.6


def test_classify_with_llm_final_guard_pass_catches_regex_only_rows(rules, tmp_path: Path):
    """LLM 判定要 でない (regex 1 ヒット) 行も最終一括ガードで補正される。

    例: 確定術式 = "悪性腫瘍切除術 (良性・誤記)" は regex で malignant_tumor=True に
    なるが、forbidden に "良性" を含むので最終ガードで False に矯正される。
    """
    df = pd.DataFrame({"確定術式": ["悪性腫瘍切除術（良性扁桃肥大）"]})
    df = classify(df)
    assert df.loc[0, "malignant_tumor"]  # regex で True
    assert not df.loc[0, "LLM判定要"]  # 1 件ヒット = LLM 対象外

    fake = FakeClient(available=True, default_response="CATEGORIES: []")
    out = classify_with_llm(df, rules, fake, cache_path=tmp_path / "c.json")
    # 最終ガードで False に補正
    assert not out.loc[0, "malignant_tumor"]
