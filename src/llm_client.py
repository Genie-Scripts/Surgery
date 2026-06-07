"""oMLX(OpenAI 互換 API) 汎用クライアント。

Outpatient-Dashboard の `src/llm_client.py` を踏襲。プロジェクト固有のハイライト
生成ロジックは別モジュール（Surgery では `src/classify_llm.py`）に分離し、本ファイルは
汎用インフラ（lifecycle 管理・呼び出し・キャッシュキー）のみを担う。

バックエンドは oMLX（ローカル http://localhost:8000/v1、OpenAI 互換）。認証必須なので
`OMLX_API_KEY` env or `~/.omlx/settings.json` の `auth.api_key` を Bearer で送る。

機能:
  - YAML から接続設定 (`LLMConfig`) を読込
  - oMLX 未起動時は `open -a oMLX` で起動を試み `_STARTUP_TIMEOUT` 秒待機
  - モデルが配信済みかを `/v1/models` で確認（id 完全一致）
  - OpenAI 互換 chat completions を叩く `chat()` 一発呼び出し
  - `cache_key()` でプロンプト内容の sha256 ハッシュを返す（呼び出し側でキャッシュに使う）
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT: int = 30   # 秒: oMLX 起動待ちの上限
_HEALTH_INTERVAL: int = 1    # 秒: ヘルスチェックのポーリング間隔


def _resolve_api_key() -> str | None:
    """oMLX の API キーを解決する。

    優先順: 環境変数 OMLX_API_KEY → ~/.omlx/settings.json の auth.api_key。
    どちらも無ければ None（認証無しで送る = oMLX が 401 を返し得る）。
    """
    env = os.environ.get("OMLX_API_KEY")
    if env:
        return env
    try:
        settings = Path.home() / ".omlx" / "settings.json"
        data = json.loads(settings.read_text(encoding="utf-8"))
        return data.get("auth", {}).get("api_key") or None
    except (OSError, json.JSONDecodeError):
        return None


@dataclass
class LLMConfig:
    endpoint: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    system_prompt: str


class LLMClient:
    """oMLX(OpenAI 互換 API) の汎用クライアント。

    プロンプト構築・レスポンスパースは呼び出し側（Surgery の場合は classify_llm）の責務。
    """

    def __init__(self, config_path: Path, enabled: bool = True) -> None:
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self.config = LLMConfig(
            endpoint=raw["endpoint"],
            model=raw.get("model", "local-model"),
            temperature=float(raw.get("temperature", 0.3)),
            max_tokens=int(raw.get("max_tokens", 500)),
            timeout=int(raw.get("timeout", 120)),
            system_prompt=raw.get("system_prompt", ""),
        )
        self.enabled = enabled
        parsed = urlparse(self.config.endpoint)
        self._base_url = f"{parsed.scheme}://{parsed.netloc}"
        self._models_url = f"{self._base_url}/v1/models"
        self._api_key = _resolve_api_key()

    def _headers(self, *, json_body: bool = False) -> dict[str, str]:
        """oMLX へ送る共通ヘッダ（必要なら Content-Type、あれば Bearer 認証）。"""
        h: dict[str, str] = {}
        if json_body:
            h["Content-Type"] = "application/json"
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    # ---- public --------------------------------------------------------

    def is_available(self) -> bool:
        """サーバ起動 & モデル取得済みなら True。enabled=False の場合は常に False。"""
        if not self.enabled:
            return False
        if not self._ensure_server():
            return False
        return self._is_model_available()

    def chat(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """OpenAI 互換 chat completions を 1 ターンで叩き、応答テキストを返す。

        パラメータ未指定時は config の値を使う。例外（URLError / KeyError /
        JSONDecodeError / TimeoutError）は呼び出し側で処理する想定。
        """
        body = json.dumps(
            {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": (
                    temperature if temperature is not None else self.config.temperature
                ),
                "max_tokens": (
                    max_tokens if max_tokens is not None else self.config.max_tokens
                ),
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self.config.endpoint,
            data=body,
            headers=self._headers(json_body=True),
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]

    @staticmethod
    def cache_key(payload: dict[str, Any]) -> str:
        """sha256 でキャッシュキーを生成。`prompt_version` などはキー対象に含める想定。

        辞書を JSON で安定シリアライズしハッシュ化。先頭 16 文字を返す。
        """
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    # ---- oMLX lifecycle -----------------------------------------------

    def _is_server_up(self) -> bool:
        """oMLX が `/v1/models` に 200 を返すか確認する（認証付き）。"""
        try:
            req = urllib.request.Request(self._models_url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _start_omlx(self) -> bool:
        """oMLX.app を best-effort 起動し、最大 `_STARTUP_TIMEOUT` 秒待つ。

        oMLX は GUI アプリ（settings の auto_start_on_launch=true）なので `open -a oMLX`
        で起動を試みる。見つからなければ警告のみで False（呼び出し側が定型文フォールバック）。
        """
        logger.info("oMLX 未起動を検知 → `open -a oMLX` で起動を試みます")
        try:
            subprocess.Popen(
                ["open", "-a", "oMLX"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.warning(
                "`open` コマンドが見つかりません（macOS 以外？）。oMLX を手動起動してください。"
            )
            return False

        for elapsed in range(_STARTUP_TIMEOUT):
            time.sleep(_HEALTH_INTERVAL)
            if self._is_server_up():
                logger.info("oMLX 起動完了（%d 秒）", elapsed + 1)
                return True

        logger.warning("oMLX の起動が %d 秒でタイムアウトしました", _STARTUP_TIMEOUT)
        return False

    def _ensure_server(self) -> bool:
        if self._is_server_up():
            return True
        return self._start_omlx()

    def _is_model_available(self) -> bool:
        """設定モデルが oMLX で配信中か `/v1/models` で確認。確認 API 失敗時は楽観的 True。

        oMLX は id 完全一致を要求する（旧 Ollama 風のタグ名は 404 になる）。
        """
        try:
            req = urllib.request.Request(self._models_url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning("モデル一覧の取得に失敗しました（楽観的続行）: %s", e)
            return True

        available = [m.get("id") for m in data.get("data", [])]
        matched = self.config.model in available
        if not matched:
            logger.warning(
                "モデル '%s' が oMLX に見つかりません。配信中: %s",
                self.config.model,
                available,
            )
        return matched
