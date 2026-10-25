#!/usr/bin/env python3
"""LLM token usage monitor for DGXTOP

Reads the Prometheus /metrics endpoint exposed by vLLM (OpenAI-compatible
servers such as vLLM, or any endpoint exposing ``vllm:*_tokens_total``) and
aggregates cumulative prompt/generation token counters with per-second rates.
Works entirely over HTTP with stdlib only — no extra dependencies.
"""

import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Optional

# Counter samples look like:
#   vllm:prompt_tokens_total{engine="0",model_name="llama-3-8b"} 1.2205153e+07
#   velogb10_prompt_tokens_total 1.2205153e+07          (veloGB10, no labels)
#   tensorfold:prompt_tokens_total 1.2205153e+07        (TensorFold zig engine)
_COUNTER_RE = re.compile(
    r'^(?:vllm:|velogb10_|tensorfold:)(?P<kind>prompt|generation)_tokens_total'
    r'(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[0-9eE.+-]+)',
    re.MULTILINE,
)
_MODEL_RE = re.compile(r'model_name="([^"]+)"')
# True LIVE generation speed for veloGB10: its *_tokens_total counters and the
# decode histogram only commit when a request FINISHES, so nothing token-based
# moves mid-generation. The scheduler step gauge does: each decode step emits
# ~1+avg-accepted tokens (MTP speculative decoding). speed = d(steps)/dt *
# (spec_emitted_total / steps_total) tracks instant throughput. vLLM exposes
# neither, so token deltas remain the primary source everywhere.
_STEPS_RE = re.compile(r'^velogb10_scheduler_steps_total\s+([0-9eE.+-]+)', re.MULTILINE)
_EMIT_RE = re.compile(r'^velogb10_spec_emitted_tokens_total\s+([0-9eE.+-]+)', re.MULTILINE)
_GEN_RE = re.compile(r'^velogb10_generation_tokens_total\s+([0-9eE.+-]+)', re.MULTILINE)
_DECODE_SUM_RE = re.compile(r'^velogb10_decode_tokens_per_second_sum\s+([0-9eE.+-]+)', re.MULTILINE)
_DECODE_COUNT_RE = re.compile(r'^velogb10_decode_tokens_per_second_count\s+([0-9eE.+-]+)', re.MULTILINE)
_RUNNING_RE = re.compile(r'^(?:velogb10_requests_running|tensorfold:requests_running)'
                         r'(?:\{[^}]*\})?\s+([0-9eE.+-]+)', re.MULTILINE)
# TensorFold: totals commit at request finish too; tensorfold:decode_rounds_total
# advances per decode round, so d(rounds)/dt * (generation_total / rounds_total)
# is the live speed, same trick as veloGB10's step gauge.
_TF_ROUNDS_RE = re.compile(r'^tensorfold:decode_rounds_total\s+([0-9eE.+-]+)', re.MULTILINE)
_TF_GENTOK_RUN_RE = re.compile(
    r'^tensorfold:generation_tokens_running(?:\{[^}]*\})?\s+([0-9eE.+-]+)', re.MULTILINE)


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

    def __init__(self, url: str = "http://127.0.0.1:8000/metrics", history_length: int = 40,
                 api_key: str = ""):
        self.url = url
        # Bearer token for servers that gate /metrics behind an API key
        # (e.g. TensorFold with TENSORFOLD_API_KEY set). Env fallback lets
        # packaged binaries authenticate without config changes.
        self.api_key = api_key or os.environ.get("DGXTOP_LLM_API_KEY", "")
        self.timeout = 3
        self._last_counters: Optional[Dict[str, float]] = None
        self._last_time: Optional[float] = None
        self._model: str = ""
        self._model_tried: bool = False
        self._decode_hist = None
        self._decode_ema = None
        self._decode_time = None
        self.gen_rate_history: deque = deque(maxlen=history_length)
        self.total_rate_history: deque = deque(maxlen=history_length)
        self._last_error: Optional[str] = None

    def _auth_header(self, headers):
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _fetch_text(self) -> Optional[str]:
        """GET the metrics URL; None on failure."""
        try:
            req = urllib.request.Request(self.url,
                                         headers=self._auth_header({"Accept": "text/plain"}))
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001 - surface any transport error
            self._last_error = str(e)
            return None

    def _discover_model(self) -> str:
        """One-shot GET of the sibling /v1/models endpoint for the model name
        (servers like veloGB10 expose counters without a model_name label)."""
        base = self.url.rsplit("/metrics", 1)[0].rstrip("/")
        try:
            req = urllib.request.Request(base + "/v1/models",
                                         headers=self._auth_header({}))
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                import json

                data = json.load(resp).get("data") or []
                if data:
                    return data[0].get("id", "")
        except Exception:  # noqa: BLE001 - cosmetic field, never fatal
            pass
        return ""

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
                lm = _MODEL_RE.search(m.group("labels") or "")
                if lm:
                    model = lm.group(1)
        if not found:
            self._last_error = "token counters not found"
            return None
        if not model and not self._model_tried:
            self._model_tried = True
            model = self._discover_model()
        self._model = model
        return {"prompt": prompt, "generation": generation}

    def _decode_live(self, text: str, now: float):
        """True instant generation tokens/s for veloGB10, or None (vLLM etc.).

        All token-ish totals on veloGB10 commit only when a request finishes,
        so any token-delta rate is flat between requests. The scheduler step
        counter DOES advance continuously while decoding (~d steps/s), and each
        step emits on average emit_total/steps_total tokens under MTP
        speculative decoding. Instant speed therefore is
            d(steps)/dt * (emit_total / steps_total)
        smoothed with an EMA. Histogram deltas (fast requests committing
        between polls) refine the estimate when available."""
        st = _STEPS_RE.search(text)
        tf = None if st else _TF_ROUNDS_RE.search(text)
        if not st and not tf:
            return None
        if st:
            # veloGB10: step counter moves continuously during decode.
            qty = float(st.group(1))
            em = _EMIT_RE.search(text)
            emit = float(em.group(1)) if em else None
        else:
            # TensorFold commits every *_total at request finish (rounds too),
            # but tensorfold:generation_tokens_running — tokens currently held
            # by live streams, incl. in-flight drafts — moves every round.
            # d(running)/dt is the live token production rate directly.
            run_t = _TF_GENTOK_RUN_RE.search(text)
            qty = float(run_t.group(1)) if run_t else 0.0
            emit = None  # qty is already in tokens: multiplier 1
        r = _RUNNING_RE.search(text)
        running = float(r.group(1)) if r else 0.0

        inst = None
        elapsed = now - self._decode_time if self._decode_time is not None else 0.0
        if self._decode_hist is not None and elapsed > 0:
            prev = self._decode_hist[0]
            d = qty - prev
            if d >= 0:
                if emit is not None and qty > 0:
                    inst = (d / elapsed) * (emit / qty)
                else:
                    inst = d / elapsed
        self._decode_hist = (qty, emit)
        self._decode_time = now
        if inst is not None:
            self._decode_ema = (inst if self._decode_ema is None
                                else 0.3 * inst + 0.7 * self._decode_ema)
        if running <= 0:
            # Nothing generating: no live speed claim; keep EMA state so the
            # very first sample of the next request isn't blind.
            return None
        return self._decode_ema

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

        # veloGB10 commits generation/prompt *_tokens_total only when a request
        # FINISHES, so delta rates stay 0 for the whole decoding phase. While a
        # request is running, use the server's continuously-moving step gauge.
        if text is not None:
            live_mean = self._decode_live(text, now)
            if live_mean is not None:
                g_rate = max(g_rate, live_mean)

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
