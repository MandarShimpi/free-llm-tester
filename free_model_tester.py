#!/usr/bin/env python3
"""
Free AI Model Tester
Pings free provider models concurrently (optionally with pooled API keys) and
shows a live table of status, latency, context size, tier and quota.

A 401/403 still proves the endpoint is up and gives real latency.
A 200 means the model answered (with or without auth).

Usage:
  python free_model_tester.py                # live dashboard, refresh every 120s
  python free_model_tester.py --once         # single round, then exit
  python free_model_tester.py --interval 60 --timeout 10
  python free_model_tester.py --providers nvidia,groq --output runs/run1
"""

import argparse
import asyncio
import http.server
import json
import mimetypes
import os
import random
import re
import signal
import sys
import threading
import time
import urllib.parse
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

try:
    import aiohttp
except ImportError:
    print("Missing dependency: aiohttp. Install with:  pip install aiohttp rich")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Missing dependency: rich. Install with:  pip install aiohttp rich")
    sys.exit(1)

console = Console()

DEFAULT_INTERVAL = int(os.environ.get("FREE_MODEL_TESTER_INTERVAL", "120"))
DEFAULT_TIMEOUT = 15
DEFAULT_CONCURRENCY = 40
DEFAULT_PROVIDER_CONCURRENCY = 8  # cap when rpm is unknown
DEFAULT_PORT = int(os.environ.get("FREE_MODEL_TESTER_PORT", "8765"))
WEB_DIR = Path(__file__).resolve().parent / "web"
_RESULTS_PREFIX = os.environ.get("FREE_MODEL_TESTER_OUTPUT", "results")
_WEB_URL = None

MODELS_CACHE = "models_cache.json"
MODELS_CACHE_TTL = 3600  # seconds

DISCOVERY = {
    "nvidia":     {"url": "https://integrate.api.nvidia.com/v1/models"},
    "groq":       {"url": "https://api.groq.com/openai/v1/models"},
    "openrouter": {"url": "https://openrouter.ai/api/v1/models"},
    "mistral":    {"url": "https://api.mistral.ai/v1/models"},
    "googleai":   {"url": "https://generativelanguage.googleapis.com/v1beta/models"},
    "cohere":     {"url": "https://api.cohere.com/v1/models"},
}
# only these providers safely merge discovered ids into PROVIDERS: their /models
# list matches the chat endpoint and carries a reliable free signal. Others still
# refresh the cache (so we can see catalog size) but do not auto-merge, because
# their catalogs mix paid/non-chat models that 404 on /chat/completions.
DISCOVERY_MERGE_PROVIDERS = {
    p.strip() for p in os.environ.get(
        "FREE_MODEL_TESTER_DISCOVERY_MERGE", "openrouter").split(",") if p.strip()
}
DISCOVERY_BLOCKLIST = ("embed", "rerank", "reranking", "whisper", "tts", "audio",
                       "moderation", "guard", "safety", "ocr", "image", "vision-encoder")


# ---------------------------------------------------------------------------
# .env loader (optional, no hard dep on python-dotenv)
# ---------------------------------------------------------------------------

def _load_dotenv():
    for p in (os.path.join(Path(__file__).resolve().parent, ".env"),
              os.path.join(os.getcwd(), ".env"),
              os.path.join(os.path.expanduser("~"), ".env")):
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass


_load_dotenv()


def _get_keys(provider_key):
    """Return deduped list of keys for provider_key. Comma-separated env vars supported."""
    env_map = {
        "mistral": ["MISTRAL_API_KEYS", "MISTRAL_API_KEY", "mistral"],
        "nvidia": ["NVIDIA_API_KEY", "NVIDIA_API_KEYS"],
        "xkiro": ["XKIRO_API_KEY", "XKIRO_API_KEYS"],
        "groq": ["GROQ_API_KEY", "GROQ_API_KEYS"],
        "openrouter": ["OPENROUTER_API_KEY", "OPENROUTER_API_KEYS"],
        "deepinfra": ["DEEPINFRA_API_KEY", "DEEPINFRA_API_KEYS"],
        "plugsky": ["PLUGSKY_API_KEY", "PLUGSKY_API_KEYS"],
        "codestral": ["MISTRAL_API_KEYS", "MISTRAL_API_KEY"],
        "scaleway": ["SCALEWAY_API_KEY"],
        "googleai": ["GOOGLE_API_KEY"],
        "zai": ["ZAI_API_KEY"],
        "qwen": ["DASHSCOPE_API_KEY"],
        "cloudflare": ["CLOUDFLARE_API_TOKEN"],
        "ovhcloud": ["OVH_AI_ENDPOINTS_ACCESS_TOKEN"],
        "siliconflow": ["SILICONFLOW_API_KEY"],
        "kilo": ["KILO_API_KEY"],
        "requesty": ["REQUESTY_API_KEY"],
        "orcarouter": ["ORCAROUTER_API_KEY"],
        "fireworks": ["FIREWORKS_API_KEY"],
        "huggingface": ["HUGGINGFACE_API_KEY"],
        "together": ["TOGETHER_API_KEY"],
        "cohere": ["COHERE_API_KEY"],
        "sambanova": ["SAMBANOVA_API_KEY"],
        "minimax": ["MINIMAX_API_KEY"],
    }
    keys = []
    for env in env_map.get(provider_key, []):
        v = os.environ.get(env)
        if not v:
            continue
        for p in (s.strip() for s in v.split(",")):
            if p and p not in keys:
                keys.append(p)
    return keys


# Round-robin deques per provider
_key_deques = {}


def _next_key(provider_key):
    """Return the current key and advance the rotation."""
    keys = _get_keys(provider_key)
    if not keys:
        return None, 0
    dq = _key_deques.setdefault(provider_key, deque(keys))
    key = dq[0]
    dq.rotate(-1)
    return key, len(keys)


# Free-model filter: for these providers only test models known free
FREE_ONLY_PROVIDERS = {"xkiro", "openrouter"}
XKIRO_FREE_MODELS = {
    "openai/gpt-5.3-codex-spark", "mistralai/mistral-large-2512", "mistralai/mistral-medium-3.5",
    "mistralai/mistral-small-2603", "mistralai/codestral-2508", "mistralai/devstral-medium",
    "mistralai/ministral-14b", "mistralai/ministral-8b", "mistralai/ministral-3b",
    "minimax/minimax-m3:free", "minimax/minimax-m2.7:free", "minimax/minimax-m2.7-highspeed:free",
    "minimax/minimax-m2.5:free", "minimax/minimax-m2.5-highspeed:free", "minimax/minimax-m2.1:free",
    "minimax/minimax-m2.1-highspeed:free", "minimax/minimax-m2:free",
    "deepseek/deepseek-v4.1-flash:free", "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v3.2", "deepseek/deepseek-chat-v3.1",
    "sensenova/sensenova-6.8-flash-lite", "sensenova/sensenova-6.7-flash-lite",
}
DEEPINFRA_FREE_MODELS = {"meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen3-235B-A22B", "deepseek-ai/DeepSeek-V3.1"}


def _is_free_model(provider_key, model_id):
    if provider_key not in FREE_ONLY_PROVIDERS:
        return True
    if provider_key == "openrouter":
        return model_id.endswith(":free")
    if provider_key == "xkiro":
        return model_id in XKIRO_FREE_MODELS or ":free" in model_id or "stealth/" in model_id
    if provider_key == "deepinfra":
        return model_id in DEEPINFRA_FREE_MODELS
    return True


# ---------------------------------------------------------------------------
# MODEL AUTO-DISCOVERY  (/models endpoints)
# ---------------------------------------------------------------------------

def _normalize_model_ids(payload):
    ids = []
    if not isinstance(payload, dict):
        return ids
    raw = payload.get("data")
    if not isinstance(raw, list):
        raw = payload.get("models")
    if not isinstance(raw, list):
        return ids
    for item in raw:
        if isinstance(item, dict):
            mid = item.get("id") or item.get("name")
        else:
            mid = item
        if not isinstance(mid, str) or not mid:
            continue
        if mid.startswith("models/"):
            mid = mid[len("models/"):]
        ids.append(mid)
    return ids


def _merge_discovered(provider_key, ids):
    info = PROVIDERS.get(provider_key)
    if not info or provider_key not in DISCOVERY_MERGE_PROVIDERS:
        return 0
    known = {m[0] for m in info["models"]}
    added = 0
    for mid in ids:
        if mid in known:
            continue
        low = mid.lower()
        if any(bad in low for bad in DISCOVERY_BLOCKLIST):
            continue
        if not _is_free_model(provider_key, mid):
            continue
        short = mid.split("/")[-1].replace(":free", "")[:28]
        info["models"].append((mid, short, "?", "—"))
        known.add(mid)
        added += 1
    return added


async def discover_models(session, only_keys=None):
    cache_path = Path(__file__).resolve().parent / MODELS_CACHE
    now = time.time()
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
    total_added = 0
    cache_changed = False

    for pk, conf in DISCOVERY.items():
        if only_keys and pk not in only_keys:
            continue
        entry = cache.get(pk)
        if isinstance(entry, dict) and now - entry.get("ts", 0) < MODELS_CACHE_TTL:
            total_added += _merge_discovered(pk, entry.get("ids", []))
            continue
        headers = {"User-Agent": "free-model-tester/1.0"}
        keys = _get_keys(pk)
        if keys:
            headers["Authorization"] = f"Bearer {keys[0]}"
        if pk == "googleai":
            gkey = os.environ.get("GOOGLE_API_KEY")
            if not gkey:
                continue
            headers.pop("Authorization", None)
            headers["x-goog-api-key"] = gkey
        try:
            async with session.get(conf["url"], headers=headers,
                                   timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    continue
                payload = await resp.json(content_type=None)
        except Exception:
            continue
        ids = _normalize_model_ids(payload)
        if not ids:
            continue
        added = _merge_discovered(pk, ids)
        total_added += added
        cache[pk] = {"ts": now, "count": len(ids), "ids": ids,
                     "merge": pk in DISCOVERY_MERGE_PROVIDERS, "added": added}
        cache_changed = True

    if cache_changed:
        try:
            tmp = cache_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            os.replace(tmp, cache_path)
        except OSError:
            pass
    if total_added:
        console.print(f"[dim]Discovery: +{total_added} new free model(s) merged[/dim]")
    return total_added


# ---------------------------------------------------------------------------
# Provider cooldown + per-model broken-cooldown
# ---------------------------------------------------------------------------

_provider_pause_until = {}           # provider_key -> pause-until timestamp (ms)
_model_failures = defaultdict(int)   # (provider_key, model_id) -> consecutive failures
_model_last_fail = {}                # (provider_key, model_id) -> last failure ts (ms)
BROKEN_COOLDOWN_MS = [30_000, 60_000, 120_000, 300_000]  # 30s, 1m, 2m, 5m


def _record_failure(provider_key, model_id):
    k = (provider_key, model_id)
    _model_failures[k] += 1
    _model_last_fail[k] = time.time() * 1000


def _reset_failures(provider_key, model_id):
    k = (provider_key, model_id)
    _model_failures.pop(k, None)
    _model_last_fail.pop(k, None)


def _broken_cooldown_ms(n):
    n = max(1, int(n or 1))
    return BROKEN_COOLDOWN_MS[min(n - 1, len(BROKEN_COOLDOWN_MS) - 1)]


def _parse_retry_after_ms(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        secs = float(s)
        if secs >= 0:
            return int(round(secs * 1000))
    except ValueError:
        pass
    try:
        import email.utils
        dt = email.utils.parsedate_to_datetime(s)
        if dt is not None:
            delta = (dt.timestamp() * 1000) - (time.time() * 1000)
            if delta > 0:
                return int(delta)
    except Exception:
        pass
    return None


def _extract_retry_after_ms(headers, body_text):
    # aiohttp's CIMultiDict is case-insensitive, one lookup is enough
    v = headers.get("retry-after")
    if v is not None:
        ms = _parse_retry_after_ms(v)
        if ms and ms > 0:
            return ms
    # body "try again N seconds later" (OpenRouter style)
    if body_text:
        m = re.search(r"try again (\d+)\s*seconds? later", body_text, re.I)
        if m:
            try:
                secs = int(m.group(1))
                if secs > 0:
                    return secs * 1000
            except ValueError:
                pass
    return 0


def _is_provider_paused(provider_key, now=None):
    now = now if now is not None else time.time() * 1000
    until = _provider_pause_until.get(provider_key)
    if until is None:
        return False
    if now >= until:
        del _provider_pause_until[provider_key]
        return False
    return True


def _pause_provider(provider_key, ms, now=None):
    if not provider_key or not isinstance(ms, (int, float)) or ms <= 0:
        return False
    now = now if now is not None else time.time() * 1000
    until = now + ms
    if until > _provider_pause_until.get(provider_key, 0):
        _provider_pause_until[provider_key] = until
    return True


def _provider_pause_remaining(provider_key, now=None):
    now = now if now is not None else time.time() * 1000
    until = _provider_pause_until.get(provider_key)
    if not until:
        return 0
    rem = until - now
    if rem <= 0:
        del _provider_pause_until[provider_key]
        return 0
    return int(rem)


def _is_model_broken_cooldown(provider_key, model_id, now=None):
    now = now if now is not None else time.time() * 1000
    k = (provider_key, model_id)
    n = _model_failures.get(k, 0)
    if n == 0:
        return False, 0
    remaining = (_model_last_fail.get(k, 0) + _broken_cooldown_ms(n)) - now
    return (True, int(remaining)) if remaining > 0 else (False, 0)


# ---------------------------------------------------------------------------
# PROVIDERS  (model id, display name, tier, context)
# ---------------------------------------------------------------------------

PROVIDERS = {
    "nvidia": {
        "name": "NVIDIA NIM",
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "rpm": "40",
        "models": [
            ("z-ai/glm-5.3", "GLM-5-3", "S+", "1M"),
            ("moonshotai/kimi-k3", "Kimi K3", "S", "1M"),
            ("deepseek-ai/deepseek-v4-flash-0731", "DeepSeek V4 Flash 0731", "S+", "1M"),
            ("nvidia/nemotron-3.5-lightning-30b-a3b", "Nemotron 3.5 Lightning", "A+", "1M"),
            ("meta/muse-glimmer-30b", "Muse Glimmer 30B", "A", "131k"),
            ("nvidia/riva-translate-4b-instruct-v2", "Riva Translate 4B v2", "B+", "32k"),
            ("nvidia/ising-calibration-1.5-31b", "Ising Calibration 1.5 31B", "B+", "128k"),
            ("poolside/laguna-xs-2.1", "Laguna XS 2.1", "S+", "262k"),
            ("google/diffusiongemma-26b-a4b-it", "DiffusionGemma 26B", "A", "256k"),
            ("nvidia/nemotron-3-ultra-550b-a55b", "Nemotron 3 Ultra 550B", "S+", "1M"),
            ("nvidia/nemotron-3.5-content-safety", "Nemotron 3.5 Content Safety", "B", "32k"),
            ("nvidia/nemotron-3-super-120b-a12b", "Nemotron 3 Super 120B", "S", "1M"),
            ("mistralai/mistral-nemotron", "Mistral Nemotron", "A+", "128k"),
        ],
    },
    "groq": {
        "name": "Groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "rpm": "30",
        "models": [
            ("openai/gpt-oss-120b", "GPT OSS 120B", "S", "131k"),
            ("openai/gpt-oss-20b", "GPT OSS 20B", "A+", "131k"),
            ("groq/compound", "Groq Compound", "A", "131k"),
            ("groq/compound-mini", "Groq Compound Mini", "B+", "131k"),
            ("qwen/qwen3.8-27b", "Qwen3.8 27B", "A+", "131k"),
        ],
    },
    "openrouter": {
        "name": "OpenRouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "rpm": "20",
        "extra_headers": {"HTTP-Referer": "https://github.com/local/free-model-tester", "X-Title": "free-model-tester"},
        "models": [
            ("nvidia/nemotron-3-ultra-550b-a55b:free", "Nemotron 3 Ultra", "S+", "1M"),
            ("poolside/laguna-xs-2.1:free", "Laguna XS 2.1", "S+", "262k"),
            ("poolside/laguna-s-2.1:free", "Laguna S 2.1", "S+", "262k"),
            ("z-ai/glm-5.2:free", "GLM-5.2", "S+", "32k"),
            ("cohere/north-mini-code:free", "North Mini Code", "S", "256k"),
            ("nvidia/nemotron-3-super-120b-a12b:free", "Nemotron 3 Super", "S", "262k"),
            ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "Nemotron 3 Omni", "A+", "256k"),
            ("google/gemma-4-31b-it:free", "Gemma 4 31B", "A+", "262k"),
            ("google/gemma-4-26b-a4b-it:free", "Gemma 4 26B MoE", "A", "262k"),
            ("liquid/lfm-2.5-2.6b:free", "LFM2.5-2.6B", "C", "64k"),
            ("nvidia/nemotron-3.5-lightning:free", "Nemotron 3.5 Lightning", "B+", "1M"),
        ],
    },
    "mistral": {
        "name": "Mistral LP",
        "url": "https://api.mistral.ai/v1/chat/completions",
        "rpm": "2",
        "models": [
            ("mistral-small-latest", "Mistral Small", "A", "256k"),
            ("mistral-medium-latest", "Mistral Medium", "S+", "256k"),
            ("ministral-14b-2512", "Ministral 14B", "A", "256k"),
            ("ministral-8b-2512", "Ministral 8B", "A", "256k"),
            ("ministral-3b-2512", "Ministral 3B", "B+", "128k"),
        ],
    },
    "codestral": {
        "name": "Mistral Codestral",
        "url": "https://api.mistral.ai/v1/chat/completions",
        "rpm": "2",
        "models": [
            ("codestral-latest", "Codestral", "A", "256k"),
        ],
    },
    "googleai": {
        "name": "Google AI Studio",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "rpm": "15",
        "models": [
            ("gemini-3.8-flash", "Gemini 3.8 Flash", "S+", "1M"),
            ("gemini-3.7-flash", "Gemini 3.7 Flash", "S+", "1M"),
            ("gemini-3.6-flash", "Gemini 3.6 Flash", "S+", "1M"),
            ("gemini-3.5-flash", "Gemini 3.5 Flash", "S+", "1M"),
            ("gemini-3.5-flash-lite", "Gemini 3.5 Flash Lite", "S", "1M"),
            ("gemini-2.5-flash", "Gemini 2.5 Flash", "A+", "1M"),
        ],
    },
    "kilo": {
        "name": "Kilo",
        "url": "https://api.kilo.ai/api/gateway/chat/completions",
        "rpm": "--",
        "no_key": True,
        "models": [
            ("kilo-auto/free", "Kilo Auto Free", "A+", "256k"),
            ("thinkingmachines/inkling-small:free", "Inkling Small", "S+", "1M"),
            ("stepfun/step-3.7-flash:free", "Step 3.7 Flash", "A+", "262k"),
            ("poolside/laguna-s-2.1:free", "Laguna S 2.1", "A+", "262k"),
            ("poolside/laguna-xs-2.1:free", "Laguna XS 2.1", "S+", "262k"),
            ("nvidia/nemotron-3-ultra-550b-a55b:free", "Nemotron 3 Ultra", "A+", "1M"),
            ("cohere/north-mini-code:free", "North Mini Code", "A-", "256k"),
            ("nvidia/nemotron-3-super-120b-a12b:free", "Nemotron 3 Super", "A-", "262k"),
            ("openrouter/free", "OpenRouter Free Router", "B", "200k"),
            ("liquid/lfm-2.5-2.6b:free", "LFM2.5-2.6B", "C", "64k"),
        ],
    },
    "llm7": {
        "name": "LLM7",
        "url": "https://api.llm7.io/v1/chat/completions",
        "rpm": "--",
        "no_key": True,
        "models": [
            ("minimax-m2.7", "MiniMax M2.7", "S+", "180k"),
            ("mistral-Nemo-Instruct-2407", "Mistral Nemo 12B", "A-", "128k"),
            ("codestral-latest", "Codestral Latest", "A", "32k"),
            ("qwen3-4b", "Qwen3 4B", "A", "131k"),
        ],
    },
    "xkiro": {
        "name": "xKiro",
        "url": "https://api.xkiro.com/v1/chat/completions",
        "rpm": "--",
        "models": [
            ("openai/gpt-5.3-codex-spark", "Codex 5.3 Spark", "S+", "128k"),
            ("mistralai/mistral-large-2512", "Mistral Large 3", "S+", "256k"),
            ("mistralai/mistral-medium-3.5", "Mistral Medium 3.5", "S", "256k"),
            ("mistralai/mistral-small-2603", "Mistral Small 4", "S", "256k"),
            ("mistralai/codestral-2508", "Codestral", "S", "256k"),
            ("mistralai/devstral-medium", "Devstral 2", "S", "256k"),
            ("mistralai/ministral-14b", "Ministral 14B", "A", "256k"),
            ("mistralai/ministral-8b", "Ministral 8B", "A", "256k"),
            ("mistralai/ministral-3b", "Ministral 3B", "B+", "128k"),
            ("minimax/minimax-m3:free", "MiniMax M3 Free", "S+", "1M"),
            ("minimax/minimax-m2.7:free", "MiniMax M2.7 Free", "S+", "204k"),
            ("minimax/minimax-m2.7-highspeed:free", "MiniMax M2.7 Highspeed Free", "S+", "204k"),
            ("minimax/minimax-m2.5:free", "MiniMax M2.5 Free", "S", "204k"),
            ("minimax/minimax-m2.5-highspeed:free", "MiniMax M2.5 Highspeed Free", "S", "204k"),
            ("minimax/minimax-m2.1:free", "MiniMax M2.1 Free", "S", "204k"),
            ("minimax/minimax-m2.1-highspeed:free", "MiniMax M2.1 Highspeed Free", "S", "204k"),
            ("minimax/minimax-m2:free", "MiniMax M2 Free", "S", "204k"),
            ("deepseek/deepseek-v4.1-flash:free", "DeepSeek V4.1 Flash Free", "S+", "1M"),
            ("deepseek/deepseek-v4-pro", "DeepSeek V4 Pro", "S", "1M"),
            ("deepseek/deepseek-v4-flash", "DeepSeek V4 Flash", "S", "1M"),
            ("deepseek/deepseek-v3.2", "DeepSeek V3.2", "S", "131k"),
            ("deepseek/deepseek-chat-v3.1", "DeepSeek V3.1", "S", "163k"),
            ("sensenova/sensenova-6.8-flash-lite", "SenseNova 6.8 Flash Lite", "A+", "262k"),
            ("sensenova/sensenova-6.7-flash-lite", "SenseNova 6.7 Flash Lite", "A+", "262k"),
        ],
    },
    "pollinations": {
        "name": "Pollinations AI",
        "url": "https://gen.pollinations.ai/v1/chat/completions",
        "rpm": "--",
        "no_key": True,
        "models": [
            ("laguna", "Laguna XS.2", "S+", "1M"),
            ("minimax-m2.7", "MiniMax M2.7", "S+", "200k"),
            ("glm-5.3", "Z.ai GLM-5.3", "S+", "1M"),
            ("kimi", "Moonshot Kimi K2.6", "S+", "262k"),
            ("minimax", "MiniMax M3", "S+", "524k"),
            ("qwen-coder", "Qwen3 Coder", "S", "262k"),
            ("deepseek", "DeepSeek V3", "S", "1M"),
            ("kimi-code", "Kimi K2 Code", "S", "262k"),
            ("openai", "OpenAI GPT", "S", "400k"),
            ("gemma-4-31b", "Gemma 4 31B", "A+", "262k"),
            ("gpt-oss", "GPT OSS 20B", "A+", "131k"),
            ("qwen3.7-flash", "Qwen3.7 Flash", "A+", "1M"),
            ("nemotron-3.5-lightning", "Nemotron 3.5 Lightning", "B+", "262k"),
        ],
    },
    "cohere": {
        "name": "Cohere",
        "url": "https://api.cohere.com/compatibility/v1/chat/completions",
        "rpm": "2",
        "models": [
            ("command-r-plus-08-2024", "Command R+", "A", "128k"),
            ("command-r-08-2024", "Command R", "B+", "128k"),
            ("command-a-03-2025", "Command A", "A+", "256k"),
            ("north-mini-code", "North Mini Code", "S", "256k"),
            ("command-r7b-12-2024", "Command R7B", "B", "128k"),
            ("command-a-plus-05-2026", "Command A Plus", "S+", "128k"),
            ("command-a-reasoning-08-2025", "Command A Reasoning", "S", "256k"),
            ("c4ai-aya-expanse-32b", "Aya Expanse 32B", "A", "128k"),
            ("c4ai-aya-vision-32b", "Aya Vision 32B", "A", "128k"),
        ],
    },
}


# ---------------------------------------------------------------------------
# STATUS RATING
# ---------------------------------------------------------------------------

def rate_status(code, latency_ms):
    if code == "200":
        if latency_ms is None:
            return "OK", "green"
        if latency_ms < 200:
            return "PERFECT", "bold green"
        if latency_ms < 500:
            return "GOOD", "green"
        if latency_ms < 1000:
            return "USABLE", "yellow"
        return "SLOW", "red"
    if code in ("401", "403"):
        return "NEEDS KEY", "cyan"
    if code == "404":
        return "NOT FOUND", "red"
    if code == "429":
        return "RATE LIMIT", "magenta"
    if code == "000":
        return "TIMEOUT", "red"
    if code == "ERR":
        return "ERROR", "red"
    return f"HTTP {code}", "dim"


def sort_key(r):
    order = {"PERFECT": 0, "GOOD": 1, "USABLE": 2, "SLOW": 3,
             "NEEDS KEY": 4, "RATE LIMIT": 5, "NOT FOUND": 6,
             "TIMEOUT": 7, "ERROR": 8}
    label, _ = rate_status(r["code"], r["latency_ms"])
    return (order.get(label, 9), r["latency_ms"] if r["latency_ms"] is not None else 99999)


# ---------------------------------------------------------------------------
# PING
# ---------------------------------------------------------------------------

def build_payload(provider_info, model_id, deep=False):
    if provider_info.get("replicate_format"):
        return {"input": {"prompt": "Reply with exactly: OK" if deep else "hi", "max_new_tokens": 16 if deep else 1}}
    if provider_info.get("aihorde_format"):
        return {"prompt": "Reply with exactly: OK" if deep else "hi", "params": {"max_length": 32 if deep else 16, "max_context_length": 512}}
    if provider_info.get("strip_prefix"):
        model_id = model_id.replace(provider_info["strip_prefix"], "")
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with exactly: OK" if deep else "hi"}],
        "max_tokens": 16 if deep else 1,
    }


def extract_quota(headers):
    variants = [
        ("x-ratelimit-remaining", "x-ratelimit-limit"),
        ("x-ratelimit-remaining-requests", "x-ratelimit-limit-requests"),
        ("x-ratelimit-remaining-requests-day", "x-ratelimit-limit-requests-day"),
        ("x-ratelimit-remaining-tokens", "x-ratelimit-limit-tokens"),
        ("ratelimit-remaining", "ratelimit-limit"),
        ("ratelimit-remaining-requests", "ratelimit-limit-requests"),
    ]
    for rem_key, lim_key in variants:
        rem, lim = headers.get(rem_key), headers.get(lim_key)
        if rem is None or lim is None:
            continue
        try:
            lim_f = float(lim)
            if lim_f > 0:
                return max(0, min(100, round(float(rem) / lim_f * 100)))
        except (TypeError, ValueError):
            continue
    return None


def _base_headers(provider_info):
    h = {"Content-Type": "application/json", "User-Agent": "free-model-tester/1.0"}
    h.update(provider_info.get("extra_headers", {}))
    return h


def _provider_concurrency(provider_info, global_cap):
    """Cap parallel pings per provider using its published rpm, so a 2-rpm
    provider isn't hit with 40 simultaneous requests every round."""
    try:
        n = int(str(provider_info.get("rpm", "--")))
    except (TypeError, ValueError):
        return DEFAULT_PROVIDER_CONCURRENCY
    return max(2, min(global_cap, n))


async def ping_model(session, provider_key, provider_info, model, global_sem, prov_sem, timeout_s, jitter=True, deep=False):
    model_id, name, tier, ctx = model
    url = provider_info["url"]
    payload = build_payload(provider_info, model_id, deep=deep)

    result = {
        "provider_key": provider_key,
        "provider": provider_info["name"],
        "model": name,
        "model_id": model_id,
        "tier": tier,
        "ctx": ctx,
        "rpm": provider_info.get("rpm", "--"),
        "code": "ERR",
        "latency_ms": None,
        "quota": None,
        "error": None,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "auth": False,
        "key_used": None,
        "verified": False,
        "cooled": False,
        "deep": bool(deep),
    }

    # small stagger so a refresh doesn't stampede providers at t=0
    if jitter:
        await asyncio.sleep(random.uniform(0.05, 0.45))

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    keys = _get_keys(provider_key)

    is_cd, rem = _is_model_broken_cooldown(provider_key, model_id)
    if is_cd:
        result["code"] = "429"
        result["error"] = f"cooldown {rem // 1000}s (broken-cooldown)"
        result["latency_ms"] = 0
        result["cooled"] = True
        return result
    if _is_provider_paused(provider_key):
        rem = _provider_pause_remaining(provider_key)
        result["code"] = "429"
        result["error"] = f"provider paused {rem // 1000}s (Retry-After)"
        result["latency_ms"] = 0
        result["cooled"] = True
        return result

    max_tries = min(len(keys) if keys else 1, 8 if provider_key == "mistral" else 4)
    for attempt in range(max_tries):
        if keys:
            key, _total = _next_key(provider_key)
            headers = dict(_base_headers(provider_info))
            headers["Authorization"] = f"Bearer {key}"
            result["auth"] = True
            result["key_used"] = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
        else:
            headers = dict(_base_headers(provider_info))
            result["auth"] = False

        async with global_sem, prov_sem:
            t0 = time.perf_counter()
            try:
                async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
                    result["code"] = str(resp.status)
                    result["quota"] = extract_quota(resp.headers)
                    if resp.status >= 400:
                        body_text = await resp.text()
                        result["error"] = body_text[:250]
                        if resp.status == 429:
                            retry_ms = _extract_retry_after_ms(resp.headers, body_text)
                            _pause_provider(provider_key, retry_ms if retry_ms > 0 else
                                            (60_000 if provider_key == "mistral" else 300_000))
                            _record_failure(provider_key, model_id)
                            if attempt + 1 < max_tries:
                                continue
                        elif resp.status in (400, 402, 403, 404, 410, 500, 502, 503):
                            _record_failure(provider_key, model_id)
                        elif resp.status == 401 and keys and attempt + 1 < max_tries:
                            continue
                    else:
                        result["error"] = None
                        _reset_failures(provider_key, model_id)
                        if deep:
                            try:
                                d = await resp.json(content_type=None)
                                try:
                                    content = d["choices"][0]["message"]["content"] or ""
                                except Exception:
                                    content = json.dumps(d)
                                result["verified"] = "ok" in content.strip().lower()[:40]
                            except Exception:
                                result["verified"] = False
                            if not result["verified"]:
                                result["error"] = "deep-check failed: no OK token in response"
                    break
            except asyncio.TimeoutError:
                result["code"] = "000"
                result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
                result["error"] = "timeout"
                _record_failure(provider_key, model_id)
                _pause_provider(provider_key, 30_000)
                break
            except aiohttp.ClientError as e:
                result["code"] = "ERR"
                result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
                result["error"] = str(e)[:200]
                break
            except Exception as e:  # noqa: BLE001
                result["code"] = "ERR"
                result["error"] = str(e)[:200]
                break
    return result


async def test_all(provider_keys=None, concurrency=DEFAULT_CONCURRENCY,
                   timeout_s=DEFAULT_TIMEOUT, jitter=True, deep=False):
    chosen = {k: v for k, v in PROVIDERS.items() if not provider_keys or k in provider_keys}
    global_sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency, ssl=True)
    tasks = []
    async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
        await discover_models(session, provider_keys and set(provider_keys))
        for pk, info in chosen.items():
            prov_sem = asyncio.Semaphore(_provider_concurrency(info, concurrency))
            for model in info["models"]:
                if not _is_free_model(pk, model[0]):
                    continue
                tasks.append(ping_model(session, pk, info, model, global_sem, prov_sem, timeout_s, jitter, deep))
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]


# ---------------------------------------------------------------------------
# DISPLAY + OUTPUT
# ---------------------------------------------------------------------------

def build_table(results, last_refresh, next_refresh, refresh_count):
    table = Table(title="FREE MODEL TEST RESULTS", title_style="bold white on blue",
                  header_style="bold white", expand=False, show_lines=False, pad_edge=False)
    table.add_column("Provider", style="cyan", no_wrap=True, min_width=18)
    table.add_column("Model", style="white", no_wrap=True, min_width=24)
    table.add_column("Status", no_wrap=True, min_width=11)
    table.add_column("Latency", justify="right", style="yellow", no_wrap=True, min_width=8)
    table.add_column("Context", justify="right", style="blue", no_wrap=True, min_width=7)
    table.add_column("Tier", justify="center", style="magenta", no_wrap=True, min_width=4)
    table.add_column("Quota", justify="right", style="green", no_wrap=True, min_width=6)
    table.add_column("RPM", justify="right", style="dim", no_wrap=True, min_width=4)

    for r in sorted(results, key=sort_key):
        label, style = rate_status(r["code"], r["latency_ms"])
        lat = f"{r['latency_ms']}ms" if r["latency_ms"] is not None else "-"
        quota = f"{r['quota']}%" if r["quota"] is not None else "--"
        table.add_row(r["provider"], r["model"], Text(label, style=style),
                      lat, r["ctx"], r["tier"], quota, r["rpm"])

    counts = {}
    for r in results:
        label, _ = rate_status(r["code"], r["latency_ms"])
        counts[label] = counts.get(label, 0) + 1
    summary = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))

    footer = Text.from_markup(
        f"[dim]Total: {len(results)} models across {len(PROVIDERS)} providers[/dim]  |  "
        f"[dim]Refresh #{refresh_count}[/dim]\n"
        f"[bold]{summary}[/bold]\n"
        f"[dim]Last: {last_refresh}  |  Next: {next_refresh}  |  Ctrl+C to stop[/dim]"
    )
    return Panel(table, subtitle=footer, border_style="blue")


def save_results(results, prefix="results", interval=DEFAULT_INTERVAL):
    global _RESULTS_PREFIX
    _RESULTS_PREFIX = prefix
    json_path = _results_path()
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = json_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "count": len(results),
                "providers": len({r["provider_key"] for r in results}),
                "interval": interval,
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        os.replace(tmp, json_path)
    except OSError as e:
        console.print(f"[red]Could not save {json_path}: {e}[/red]")

    try:
        lines = [
            f"Free Model Tester - {datetime.now().isoformat(timespec='seconds')}",
            f"Total: {len(results)} models / {len(PROVIDERS)} providers",
            "",
            f"{'Provider':<20} {'Model':<30} {'Status':<12} {'Latency':>9} {'Ctx':>7} {'Tier':>5} {'Quota':>6}",
            "-" * 100,
        ]
        for r in sorted(results, key=sort_key):
            label, _ = rate_status(r["code"], r["latency_ms"])
            lat = f"{r['latency_ms']}ms" if r["latency_ms"] is not None else "-"
            quota = f"{r['quota']}%" if r["quota"] is not None else "--"
            lines.append(f"{r['provider']:<20} {r['model']:<30} {label:<12} {lat:>9} {r['ctx']:>7} {r['tier']:>5} {quota:>6}")
        # keep summary next to results.json, not cwd
        summary_path = json_path.with_name(json_path.stem + "_summary.txt")
        summary_tmp = summary_path.with_suffix(".txt.tmp")
        with open(summary_tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        os.replace(summary_tmp, summary_path)
    except OSError as e:
        console.print(f"[red]Could not save summary: {e}[/red]")


def append_history(results, out_prefix="results"):
    path = Path(out_prefix)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path = path.with_name(path.name + "_history.jsonl")
    try:
        if path.exists() and path.stat().st_size > 100 * 1024 * 1024:
            os.replace(path, str(path) + ".old")
        ts = time.time()
        with open(path, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps({
                    "ts": ts,
                    "provider": r["provider_key"],
                    "model": r["model_id"],
                    "code": r["code"],
                    "lat": r["latency_ms"],
                    "ok": r["code"] == "200",
                    "verified": bool(r.get("verified")),
                    "cooled": bool(r.get("cooled")),
                    "deep": bool(r.get("deep")),
                }) + "\n")
    except OSError as e:
        console.print(f"[red]Could not append history: {e}[/red]")


def history_stats(prefix="results", days=7):
    import statistics
    path = Path(prefix)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path = path.with_name(path.name + "_history.jsonl")
    if not path.exists():
        print(f"No history file at {path}. Run the tester first (without --once).")
        return
    cutoff = time.time() - days * 86400
    agg = {}
    total = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("ts", 0) < cutoff:
                continue
            if row.get("cooled"):
                continue
            key = (row.get("provider"), row.get("model"))
            s = agg.setdefault(key, {"n": 0, "ok": 0, "verified": 0, "lats": []})
            s["n"] += 1
            if row.get("ok"):
                s["ok"] += 1
            if row.get("verified"):
                s["verified"] += 1
            lat = row.get("lat")
            if isinstance(lat, (int, float)) and 0 < lat < 10000:
                s["lats"].append(lat)
            total += 1
    if not agg:
        print(f"No history entries in the last {days} day(s).")
        return
    rows = []
    for (prov, model), s in agg.items():
        uptime = s["ok"] / s["n"] * 100 if s["n"] else 0
        p50 = int(statistics.median(s["lats"])) if s["lats"] else None
        rows.append((uptime, prov, model, s["n"], s["ok"], s["verified"], p50))
    rows.sort(key=lambda r: (-r[0], str(r[1]), str(r[2])))
    print(f"History over last {days} day(s) — {total} samples, {len(agg)} (provider, model) pairs\n")
    print(f"{'UPTIME':>7}  {'PROVIDER':<18} {'MODEL':<34} {'N':>5} {'OK':>5} {'VERIF':>5} {'P50':>7}")
    print("-" * 92)
    for uptime, prov, model, n, ok, verif, p50 in rows:
        print(f"{uptime:6.1f}%  {str(prov):<18} {str(model):<34} {n:>5} {ok:>5} {verif:>5} "
              f"{(str(p50)+'ms') if p50 is not None else '--':>7}")


# ---------------------------------------------------------------------------
# LIVE DASHBOARD SERVER
# ---------------------------------------------------------------------------

def _results_path():
    p = Path(_RESULTS_PREFIX)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    if p.suffix != ".json":
        p = p.with_name(p.name + ".json")
    return p.resolve()


class _LiveHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory or str(WEB_DIR.parent), **kwargs)

    def log_message(self, fmt, *args):
        pass

    def _send_bytes(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self):
        route = urllib.parse.urlparse(self.path).path
        if route in ("/api/status", "/results.json", "/data/results.json", "/web/results.json"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.end_headers()
            return
        self.send_response(404)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_file(self, path, content_type):
        try:
            body = path.read_bytes()
        except OSError:
            self._send_bytes(b'{"error":"not found"}', "application/json; charset=utf-8", 404)
            return
        self._send_bytes(body, content_type)

    def do_GET(self):
        route = urllib.parse.urlparse(self.path).path
        rp = _results_path()

        if route in ("/", "/index.html", "/web", "/web/"):
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return

        if route in ("/results.json", "/data/results.json", "/web/results.json"):
            self._send_file(rp, "application/json; charset=utf-8")
            return

        if route == "/api/status":
            payload = {
                "ok": True,
                "server_time": datetime.now().isoformat(timespec="seconds"),
                "results_file": str(rp),
                "exists": rp.exists(),
            }
            try:
                data = json.loads(rp.read_text(encoding="utf-8"))
                payload.update({
                    "generated_at": data.get("generated_at"),
                    "count": data.get("count"),
                    "providers": data.get("providers"),
                    "interval": data.get("interval"),
                })
            except (OSError, ValueError) as e:
                payload["ok"] = False
                payload["error"] = str(e)
            self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
            return

        if route.startswith("/web/"):
            target = (WEB_DIR / route[len("/web/"):]).resolve()
            try:
                target.relative_to(WEB_DIR.resolve())
            except ValueError:
                self._send_bytes(b'{"error":"forbidden"}', "application/json; charset=utf-8", 403)
                return
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
                ctype += "; charset=utf-8"
            self._send_file(target, ctype)
            return

        self._send_bytes(b'{"error":"not found"}', "application/json; charset=utf-8", 404)


def start_web_server(port=None):
    if os.environ.get("RENDER"):
        host = os.environ.get("FREE_MODEL_TESTER_HOST") or "0.0.0.0"
        port = int(os.environ.get("PORT", "8765"))
    elif not WEB_DIR.exists():
        console.print(f"[yellow]No web/ folder at {WEB_DIR} - dashboard disabled.[/yellow]")
        return None
    else:
        host = os.environ.get("FREE_MODEL_TESTER_HOST") or "127.0.0.1"
        port = port or DEFAULT_PORT
    try:
        httpd = http.server.ThreadingHTTPServer((host, port), _LiveHandler)
    except OSError as e:
        console.print(f"[yellow]Port {port} unavailable ({e}) - dashboard disabled.[/yellow]")
        return None
    threading.Thread(target=httpd.serve_forever, name="dashboard", daemon=True).start()
    return f"http://{host}:{port}/"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

async def run(interval, once=False, timeout_s=DEFAULT_TIMEOUT, concurrency=DEFAULT_CONCURRENCY,
              provider_keys=None, out_prefix="results", jitter=True, deep=False):
    console.print(Panel.fit(
        f"[bold]Free Model Tester[/bold]\n"
        f"[dim]Pinging endpoints anonymously (keys used when present in env)[/dim]\n"
        f"[dim]{len(PROVIDERS)} providers / "
        f"{sum(len(p['models']) for p in PROVIDERS.values())} models[/dim]\n"
        f"[dim]Refresh interval: {interval}s  |  timeout: {timeout_s}s  |  concurrency: {concurrency}[/dim]\n"
        f"[dim]Dashboard: {_WEB_URL or 'disabled (--no-serve)'}[/dim]",
        border_style="blue",
    ))

    refresh_count = 0
    stop = asyncio.Event()

    def _stop(*_):
        stop.set()

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _stop)
            except (NotImplementedError, AttributeError):
                pass
    except Exception:
        pass

    while not stop.is_set():
        refresh_count += 1
        last_refresh = datetime.now().strftime("%H:%M:%S")
        t_round = time.perf_counter()
        console.print(f"[dim]Refreshing #{refresh_count} ...[/dim]")

        results = await test_all(provider_keys, concurrency, timeout_s, jitter, deep)
        save_results(results, out_prefix, interval)
        append_history(results, out_prefix)

        next_refresh = datetime.fromtimestamp(time.time() + interval).strftime("%H:%M:%S")
        console.print(f"[dim]Round #{refresh_count} done in {time.perf_counter() - t_round:.1f}s[/dim]")

        renderable = build_table(results, last_refresh, next_refresh, refresh_count)
        if once:
            console.print(renderable)
            break
        console.clear()
        console.print(renderable)

        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
        except KeyboardInterrupt:
            break

    console.print(f"\n[bold]Stopped.[/bold] Final results saved to {out_prefix}.json and {out_prefix}_summary.txt")


def parse_args(argv):
    p = argparse.ArgumentParser(description="Ping free LLM endpoints and record status/latency/quota.")
    p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help=f"seconds between rounds (default {DEFAULT_INTERVAL})")
    p.add_argument("--once", action="store_true", help="run a single round and exit")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"per-request timeout in seconds (default {DEFAULT_TIMEOUT})")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help=f"global concurrent requests (default {DEFAULT_CONCURRENCY})")
    p.add_argument("--providers", default="", help="comma-separated provider keys to test, e.g. nvidia,groq")
    p.add_argument("--output", default="results", help="output file prefix (default: results)")
    p.add_argument("--no-jitter", action="store_true", help="disable per-request startup stagger")
    p.add_argument("--list-providers", action="store_true", help="list provider keys and exit")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"dashboard server port (default {DEFAULT_PORT})")
    p.add_argument("--no-serve", dest="serve", action="store_false", help="do not start the built-in dashboard server")
    p.add_argument("--deep", action="store_true", help="require a real generated answer (not just a 200 handshake)")
    p.add_argument("--stats", action="store_true", help="print uptime stats from history and exit")
    p.add_argument("--days", type=int, default=7, help="days of history for --stats (default 7)")
    p.set_defaults(serve=True)
    return p.parse_args(argv)


def main():
    global _RESULTS_PREFIX, _WEB_URL
    args = parse_args(sys.argv[1:])
    if args.list_providers:
        for k, v in PROVIDERS.items():
            print(f"{k:<12} {v['name']:<22} {len(v['models'])} models  rpm={v.get('rpm','--')}")
        return
    if args.stats:
        history_stats(args.output, days=max(1, args.days))
        return
    interval = max(5, args.interval)
    _RESULTS_PREFIX = args.output
    provider_keys = {s.strip() for s in args.providers.split(",") if s.strip()} or None

    if args.serve and not args.once:
        _WEB_URL = start_web_server(args.port)
        if _WEB_URL:
            console.print(f"[green]Dashboard live at[/green] [bold]{_WEB_URL}[/bold] [dim](Ctrl+C stops both)[/dim]")

    try:
        asyncio.run(run(interval, once=args.once, timeout_s=args.timeout,
                        concurrency=args.concurrency, provider_keys=provider_keys,
                        out_prefix=args.output, jitter=not args.no_jitter, deep=args.deep))
    except KeyboardInterrupt:
        console.print("\n[bold]Stopped.[/bold]")


if __name__ == "__main__":
    main()
