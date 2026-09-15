#!/usr/bin/env python3
"""LLM token usage monitor for DGXTOP

Reads the Prometheus /metrics endpoint exposed by vLLM (OpenAI-compatible
servers such as vLLM, or any endpoint exposing ``vllm:*_tokens_total``) and
aggregates cumulative prompt/generation token counters with per-second rates.
Works entirely over HTTP with stdlib only — no extra dependencies.
"""

import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Optional

# Counter samples look like:
#   vllm:prompt_tokens_total{engine="0",model_name="llama-3-8b"} 1.2205153e+07
_COUNTER_RE = re.compile(
    r'^vllm:(?P<kind>prompt|generation)_tokens_total\{(?P<labels>[^}]*)\}\s+(?P<value>[0-9eE.+-]+)',
    re.MULTILINE,
)
_MODEL_RE = re.compile(r'model_name="([^"]+)"')


@dataclass
class LLMStats:
    """Snapshot of one polling cycle against the metrics endpoint."""

    model: str = ""
    prompt_tokens: float = 0.0
    generation_tokens: float = 0.0
    total_tokens: float = 0.0
    prompt_tokens_per_sec: float = 0.0
    generation_tokens_per_sec: float = 0.0
    total_tokens_per_sec: float = 0.0
    ok: bool = True
    error: str = ""
    last_seen: float = field(default_factory=time.time)


class LLMMonitor:
    """Poll an LLM server's Prometheus metrics and expose token usage."""

    def __init__(self, url: str = "http://127.0.0.1:8000/metrics", history_length: int = 40):
        self.url = url
        self.timeout = 3
        self._last_counters: Optional[Dict[str, float]] = None
        self._last_time: Optional[float] = None
        self._model: str = ""
        self.gen_rate_history: deque = deque(maxlen=history_length)
        self.total_rate_history: deque = deque(maxlen=history_length)
        self._last_error: Optional[str] = None

    def _fetch_text(self) -> Optional[str]:
        """GET the metrics URL; None on failure."""
        try:
            req = urllib.request.Request(self.url, headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001 - surface any transport error
            self._last_error = str(e)
            return None

    def _parse_counters(self, text: str) -> Optional[Dict[str, float]]:
        """Sum token counters across engines. Returns None when absent."""
        prompt = generation = 0.0
        found = False
        model = self._model
        for m in _COUNTER_RE.finditer(text):
            found = True
            value = float(m.group("value"))
            if m.group("kind") == "prompt":
                prompt += value
            else:
                generation += value
            if not model:
                lm = _MODEL_RE.search(m.group("labels"))
                if lm:
                    model = lm.group(1)
        if not found:
            self._last_error = "vllm token counters not found"
            return None
        self._model = model
        return {"prompt": prompt, "generation": generation}

    def get_stats(self) -> Optional[LLMStats]:
        """Poll metrics once and return current snapshot (None if never polled)."""
        now = time.time()
        text = self._fetch_text()
        if text is None:
            counters = self._last_counters
            if counters is None:
                return LLMStats(
                    ok=False,
                    error=self._last_error or "unreachable",
                    last_seen=now,
                )
            # Keep the last known totals, zero the rates.
        else:
            counters = self._parse_counters(text)
            if counters is None and self._last_counters is None:
                return LLMStats(
                    ok=False,
                    error=self._last_error or "parse error",
                    last_seen=now,
                )

        if counters is None:
            return None

        elapsed = None
        if self._last_time is not None:
            elapsed = now - self._last_time

        if (
            self._last_counters is not None
            and elapsed is not None
            and elapsed > 0
            and counters["prompt"] >= self._last_counters["prompt"]
            and counters["generation"] >= self._last_counters["generation"]
        ):
            p_rate = (counters["prompt"] - self._last_counters["prompt"]) / elapsed
            g_rate = (counters["generation"] - self._last_counters["generation"]) / elapsed
        else:
            # First sample after start-up, or counter reset (server restart).
            p_rate = g_rate = 0.0

        self._last_counters = counters
        self._last_time = now

        self.gen_rate_history.append(g_rate)
        total_rate = p_rate + g_rate
        self.total_rate_history.append(total_rate)

        return LLMStats(
            model=self._model,
            prompt_tokens=counters["prompt"],
            generation_tokens=counters["generation"],
            total_tokens=counters["prompt"] + counters["generation"],
            prompt_tokens_per_sec=p_rate,
            generation_tokens_per_sec=g_rate,
            total_tokens_per_sec=total_rate,
            ok=text is not None,
            error=self._last_error or "" if text is None else "",
            last_seen=now,
        )

    def get_history(self) -> Dict[str, deque]:
        return {
            "generation_rate": self.gen_rate_history,
            "total_rate": self.total_rate_history,
        }
