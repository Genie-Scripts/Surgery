"""Ollama / OpenAI 互換 API 汎用クライアント。

Outpatient-Dashboard の `src/llm_client.py` を踏襲。プロジェクト固有のハイライト
生成ロジックは別モジュール（Surgery では `src/classify_llm.py`）に分離し、本ファイルは
汎用インフラ（lifecycle 管理・呼び出し・キャッシュキー）のみを担う。

機能:
  - YAML から接続設定 (`LLMConfig`) を読込
  - Ollama 未起動時は `ollama serve` を自動起動して `_STARTUP_TIMEOUT` 秒待機
  - モデルがプル済みかをタグ API で確認
  - OpenAI 互換 chat completions を叩く `chat()` 一発呼び出し
  - `cache_key()` でプロンプト内容の sha256 ハッシュを返す（呼び出し側でキャッシュに使う）
"""

from __future__ import annotations

import hashlib
import json
import logging
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

_STARTUP_TIMEOUT: int = 30   # 秒: ollama serve 起動待ちの上限
_HEALTH_INTERVAL: int = 1    # 秒: ヘルスチェックのポーリング間隔


@dataclass
class LLMConfig:
    endpoint: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    system_prompt: str


class LLMClient:
    """Ollama / OpenAI 互換 API の汎用クライアント。

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
            headers={"Content-Type": "application/json"},
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

    # ---- ollama lifecycle ---------------------------------------------

    def _is_ollama_up(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self._base_url}/", timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _start_ollama(self) -> bool:
        """`ollama serve` を background 起動し、最大 `_STARTUP_TIMEOUT` 秒待つ。"""
        logger.info("Ollama 未起動を検知 → `ollama serve` を自動起動します")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.warning(
                "ollama コマンドが見つかりません。Ollama をインストールし PATH を通してください。"
            )
            return False

        for elapsed in range(_STARTUP_TIMEOUT):
            time.sleep(_HEALTH_INTERVAL)
            if self._is_ollama_up():
                logger.info("Ollama 起動完了（%d 秒）", elapsed + 1)
                return True

        logger.warning("Ollama の起動が %d 秒でタイムアウトしました", _STARTUP_TIMEOUT)
        return False

    def _ensure_server(self) -> bool:
        if self._is_ollama_up():
            return True
        return self._start_ollama()

    def _is_model_available(self) -> bool:
        """設定モデルが Ollama にプル済みか確認。確認 API 失敗時は楽観的 True。"""
        try:
            with urllib.request.urlopen(
                f"{self._base_url}/api/tags", timeout=5
            ) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning("モデル一覧の取得に失敗しました（楽観的続行）: %s", e)
            return True

        available = [m["name"] for m in data.get("models", [])]
        config_model = self.config.model
        base_name = config_model.split(":")[0]
        matched = any(
            m in (config_model, base_name) or m.startswith(base_name + ":")
            for m in available
        )
        if not matched:
            logger.debug("取得済みモデル一覧: %s", available)
        return matched
