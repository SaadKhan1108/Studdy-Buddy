"""AI providers: Google Gemini and Groq (OpenAI-compatible API).

Both share the same behaviour: request spacing (requests-per-minute), a disk cache for identical
requests, automatic backup model, and friendly error messages.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import time
from pathlib import Path

try:
    import httpx  # only needed for Groq
except ImportError:  # keeps Gemini working even if requirements were not re-installed
    httpx = None

DEFAULT_MODEL = "gemini-3.8-flash"
PROVIDERS = {
    "Google Gemini": {"model": DEFAULT_MODEL, "rpm": 5, "env": "GEMINI_API_KEY",
                      "url": "https://aistudio.google.com/apikey", "host": "aistudio.google.com/apikey"},
    "Groq": {"model": "openai/gpt-oss-120b", "rpm": 30, "env": "GROQ_API_KEY",
             "url": "https://console.groq.com/keys", "host": "console.groq.com/keys"},
}
CACHE_DB = Path("data/llm_cache.db")
_last_call = 0.0      # time of the previous API request (shared by the whole app)
_request_count = 0    # API requests sent since the app started


def requests_made() -> int:
    return _request_count


class LLMError(Exception):
    """Raised with a user-friendly message when the AI call fails."""


class _Fail(Exception):
    """Internal: an HTTP-level failure from a provider."""

    def __init__(self, code, message, retry_after=None):
        super().__init__(message)
        self.code, self.message = code, message
        self.retry_after = retry_after if retry_after is not None else parse_wait(message)


def parse_wait(text: str):
    """'Please try again in 7.66s' / 'retry in 2m59.5s' -> seconds (or None)."""
    m = re.search(r"(?:try again|retry)\s*(?:in|after)\s*((?:\d+(?:\.\d+)?[hms]\s*)+)", text or "", re.I)
    if not m:
        return None
    units = {"h": 3600, "m": 60, "s": 1}
    return sum(float(n) * units[u] for n, u in re.findall(r"(\d+(?:\.\d+)?)([hms])", m.group(1)))


def is_daily(text: str) -> bool:
    norm = re.sub(r"[\s_]", "", (text or "").lower())
    return any(k in norm for k in ("perday", "requestsperday", "tokensperday", "(rpd)", "(tpd)"))


def parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise LLMError("The AI returned an answer I couldn't read. Please try again.")


def _cache_get(key):
    try:
        with sqlite3.connect(CACHE_DB) as c:
            c.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, text TEXT)")
            row = c.execute("SELECT text FROM cache WHERE key=?", (key,)).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _cache_put(key, text):
    try:
        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(CACHE_DB) as c:
            c.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, text TEXT)")
            c.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, text))
    except Exception:
        pass


class BaseLLM:
    provider = "AI"
    # How much text the features may send per request (providers with small token limits use less)
    quiz_chars, notes_chars, notes_parts = 30000, 120000, 2

    def __init__(self, api_key, model, backup_model="", rpm=5, use_cache=True):
        self.api_key = (api_key or "").strip().strip('"').strip("'")
        if not self.api_key:
            raise LLMError(f"Add your free {self.provider} API key in the sidebar first.")
        self.model = model
        self.backup_model = (backup_model or "").strip()  # used automatically if the main model is rate-limited
        self.rpm = max(1, int(rpm))                        # calls are spaced to respect this limit
        self.use_cache = use_cache                         # identical requests are answered from disk

    def _pace(self):
        global _last_call, _request_count
        wait = _last_call + 60.0 / self.rpm - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        _request_count += 1

    def _call(self, prompt, system, as_json, temperature) -> str:  # implemented by providers
        raise NotImplementedError

    def _rate_msg(self, fail):
        if is_daily(fail.message):
            return (f"Daily free-tier limit reached for {self.provider} model '{self.model}'. It resets later "
                    "(check your usage page), or put a different model/provider in the sidebar.")
        return ("Free-tier rate limit reached (too many requests or tokens per minute). Wait about a minute "
                "and try again, or set a Backup model in the sidebar.")

    def generate(self, prompt: str, system: str | None = None, as_json: bool = False,
                 temperature: float = 0.3, tries: int = 3) -> str:
        key = hashlib.sha256(f"{self.provider}|{self.model}|{system}|{as_json}|{temperature}|{prompt}".encode()).hexdigest()
        if self.use_cache and (hit := _cache_get(key)) is not None:
            return hit
        attempt = 0
        while attempt < tries:
            attempt += 1
            try:
                self._pace()
                text = self._call(prompt, system, as_json, temperature) or ""
                if not text.strip():
                    raise LLMError("The AI returned an empty answer (it may have been filtered). Try rephrasing.")
                if self.use_cache:
                    _cache_put(key, text)
                return text
            except _Fail as f:
                code = f.code
                if code == 429:
                    if self.backup_model and self.model != self.backup_model:
                        self.model = self.backup_model  # switch models instead of waiting
                        attempt -= 1
                        continue
                    if (not is_daily(f.message) and f.retry_after is not None
                            and f.retry_after <= 65 and attempt < tries):
                        time.sleep(f.retry_after + 1)  # short per-minute limit: wait it out
                        continue
                    raise LLMError(self._rate_msg(f)) from f
                if code in (500, 502, 503) and attempt < tries:
                    time.sleep(2 ** attempt + random.random())
                    continue
                if code == 404:
                    raise LLMError(f"Model '{self.model}' is not available for your key. Change the "
                                   f"model name in the sidebar. Details: {f.message}") from f
                if code == 413:
                    raise LLMError("This request is larger than your plan allows per minute. Try a smaller "
                                   "topic, fewer questions, or another model.") from f
                if code in (400, 401, 403):
                    raise LLMError(f"{self.provider} rejected the request ({code}): {f.message} "
                                   "- check that the API key is correct and the model name is valid.") from f
                raise LLMError(f"{self.provider} API error ({code}): {f.message}") from f
            except LLMError:
                raise
            except Exception as e:  # network down, DNS, timeout, SDK problems
                if attempt < tries:
                    time.sleep(attempt + random.random())
                    continue
                raise LLMError(f"Couldn't reach {self.provider} ({type(e).__name__}: {e}). "
                               "Check your internet connection and try again.") from e
        raise LLMError("The AI request failed. Please try again.")

    def generate_json(self, prompt: str, system: str | None = None, temperature: float = 0.3, tries: int = 3):
        return parse_json(self.generate(prompt, system, as_json=True, temperature=temperature, tries=tries))


class GeminiLLM(BaseLLM):
    provider = "Gemini"

    def __init__(self, api_key, model=DEFAULT_MODEL, backup_model="", rpm=5, use_cache=True):
        super().__init__(api_key, model or DEFAULT_MODEL, backup_model, rpm, use_cache)
        from google import genai
        from google.genai import types

        # 60 s limit per request so the app never hangs silently
        self.client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=60_000))
        self.fast = True  # low "thinking" effort (much quicker); auto-disabled if the model rejects it

    def _config(self, system, as_json, temperature):
        from google.genai import types

        kw = dict(system_instruction=system, temperature=temperature,
                  response_mime_type="application/json" if as_json else None)
        if self.fast:
            try:
                kw["thinking_config"] = types.ThinkingConfig(thinking_level="low")
            except Exception:  # older SDK without this field
                pass
        return types.GenerateContentConfig(**kw)

    def _call(self, prompt, system, as_json, temperature):
        from google.genai import errors

        for _ in range(2):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=self._config(system, as_json, temperature))
                return resp.text or ""
            except errors.APIError as e:
                msg = str(getattr(e, "message", e))
                if getattr(e, "code", None) == 400 and self.fast and "think" in msg.lower():
                    self.fast = False  # this model doesn't accept the thinking setting: retry without it
                    continue
                raise _Fail(getattr(e, "code", None), msg) from e
        raise _Fail(400, "request rejected")


class GroqLLM(BaseLLM):
    """Groq's OpenAI-compatible chat API. Free-tier token-per-minute limits are small,
    so the features send less text per request than they do with Gemini."""
    provider = "Groq"
    quiz_chars, notes_chars, notes_parts = 9000, 12000, 4
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key, model="openai/gpt-oss-120b", backup_model="", rpm=30, use_cache=True):
        super().__init__(api_key, model or "openai/gpt-oss-120b", backup_model, rpm, use_cache)

    def _call(self, prompt, system, as_json, temperature):
        if httpx is None:
            raise LLMError("Groq support needs the 'httpx' package: run  pip install -r requirements.txt")
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_completion_tokens": 3000}
        if as_json:
            body["response_format"] = {"type": "json_object"}
        if "gpt-oss" in self.model:
            body["reasoning_effort"] = "low"  # faster, fewer tokens
        r = httpx.post(self.URL, headers={"Authorization": f"Bearer {self.api_key}"}, json=body, timeout=60)
        if r.status_code != 200:
            try:
                msg = r.json()["error"]["message"]
            except Exception:
                msg = r.text[:300]
            ra = r.headers.get("retry-after")
            raise _Fail(r.status_code, msg, float(ra) if ra and re.fullmatch(r"[\d.]+", ra) else None)
        return r.json()["choices"][0]["message"].get("content") or ""


LLM = GeminiLLM  # backwards-compatible name


def make_llm(provider, api_key, model="", backup="", rpm=None, use_cache=True):
    cls = GroqLLM if provider == "Groq" else GeminiLLM
    cfg = PROVIDERS["Groq" if provider == "Groq" else "Google Gemini"]
    return cls(api_key, model or cfg["model"], backup, rpm or cfg["rpm"], use_cache)
