import argparse
import asyncio
import base64
import datetime as dt
import json
import os
import random
import re
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import httpx
import pandas as pd
import yaml


# ----------------------------
# Utils
# ----------------------------

LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)
# For COT mode: extract from a dedicated final line.
# Accept both English and Chinese tags (Answer/答案) and both ':' / '：'.
FINAL_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:final\s*)?(?:answer|答案)\s*[:：]\s*([A-E](?:\s*,\s*[A-E])*)\s*$"
)

RETRIABLE_STATUS = {408, 429, 500, 502, 503, 504}

# Excel (openpyxl) cannot store certain control characters in worksheets.
# Mirror openpyxl's illegal-char policy:
# - illegal: 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F
# - allowed: tab(0x09), LF(0x0A), CR(0x0D)
_ILLEGAL_EXCEL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def _excel_safe_str(s: str) -> str:
    if not s:
        return ""
    return _ILLEGAL_EXCEL_CHARS_RE.sub("", s)

def _excel_safe_cell(v: Any) -> Any:
    """
    Convert a cell value into something safe for openpyxl.
    - Keeps numbers/bools/datetime-like as-is
    - Cleans illegal control chars from strings
    - Converts dict/list to JSON string then cleans
    """
    if v is None:
        return None
    try:
        if is_nan(v):
            return v
    except Exception:
        pass

    if isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, (dt.datetime, dt.date, dt.time)):
        return v
    # pandas Timestamp / NaT
    try:
        if isinstance(v, pd.Timestamp):
            return v
    except Exception:
        pass

    if isinstance(v, str):
        return _excel_safe_str(v)
    if isinstance(v, (dict, list)):
        try:
            return _excel_safe_str(json.dumps(v, ensure_ascii=False))
        except Exception:
            return _excel_safe_str(str(v))
    return _excel_safe_str(str(v))

def now_ms() -> int:
    return int(time.time() * 1000)

_DATA_URL_B64_RE = re.compile(
    r"data:(?:image|application|text)/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]{40,}",
    re.IGNORECASE,
)
# Generic "very long base64-like token" safeguard, to prevent sending huge blobs to judge.
_LONG_B64_TOKEN_RE = re.compile(r"\b[A-Za-z0-9+/]{200,}={0,2}\b")

def strip_base64_payloads(text: str) -> str:
    """
    Remove/shorten base64 payloads from text BEFORE sending to judge.
    This is best-effort and intentionally conservative.
    """
    if not text:
        return ""
    s = str(text)
    s = _DATA_URL_B64_RE.sub("data:<omitted>;base64,<omitted>", s)
    s = _LONG_B64_TOKEN_RE.sub("<base64_omitted>", s)
    return s

def is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and pd.isna(x)
    except Exception:
        return False

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def normalize_colname(s: str) -> str:
    return str(s).strip()

def load_options(options_cell: Any) -> List[str]:
    """
    Expect JSON array string like ["A: ...", "B: ...", ...]
    Robust parse with fallback.
    """
    if options_cell is None:
        return []
    if isinstance(options_cell, float) and pd.isna(options_cell):
        return []
    if isinstance(options_cell, list):
        return [str(x) for x in options_cell]

    s = str(options_cell).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass

    # fallback split
    parts = re.split(r"\s*\|\s*|\s*;\s*", s)
    return [p.strip() for p in parts if p.strip()]

def image_file_to_data_url(image_path: str) -> Optional[str]:
    if not image_path or not os.path.exists(image_path):
        return None

    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg"
    if ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    elif ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def resolve_image_path(images_root: str, img_value: Any) -> Optional[str]:
    """
    Resolve an image file path from a dataset cell value.
    Supports:
    - absolute paths
    - relative paths like "images/xxx.jpg"
    - bare filenames like "xxx.jpg" (will also try <images_root>/images/xxx.jpg)
    """
    if img_value is None or is_nan(img_value):
        return None
    img_rel = str(img_value).strip()
    if not img_rel or img_rel.lower() == "nan":
        return None

    if os.path.isabs(img_rel):
        return img_rel

    root = (images_root or "").strip()
    if not root:
        return img_rel

    p1 = os.path.join(root, img_rel)
    if os.path.exists(p1):
        return p1

    p2 = os.path.join(root, "images", img_rel)
    if os.path.exists(p2):
        return p2

    return p1

def extract_letters_csv(text: str) -> Optional[str]:
    """
    Extract multiple answers from model output.
    Return normalized CSV like 'A,B,C' (no spaces), unique letters in order of first appearance.
    If none found, return None.
    """
    if not text:
        return None
    letters = LETTER_RE.findall(text.upper())
    if not letters:
        return None

    seen = set()
    uniq = []
    for ch in letters:
        ch = ch.upper()
        if ch in ["A", "B", "C", "D", "E"] and ch not in seen:
            uniq.append(ch)
            seen.add(ch)

    return ",".join(uniq) if uniq else None

def normalize_csv_letters(s: str) -> str:
    """
    Normalize any text like 'A, B,C' or '答案是A和C' -> 'A,C'
    Removes spaces, keeps unique A-E in first-appearance order.
    """
    if not s:
        return ""
    s = str(s).upper().replace(" ", "")
    letters = re.findall(r"[A-E]", s)
    seen = set()
    uniq = []
    for ch in letters:
        if ch not in seen:
            uniq.append(ch)
            seen.add(ch)
    return ",".join(uniq)

def csv_to_set(s: str) -> set:
    if not s:
        return set()
    parts = [p for p in s.split(",") if p]
    return set(parts)

def extract_answer_from_text(text: str, cot_on: bool) -> Optional[str]:
    """
    If cot_on=True, prefer extracting from a dedicated final line like:
      Answer:A,B,C   (parser tolerates spaces around commas)
    If not found, fallback to last non-empty line, then fallback to global extraction.
    Return normalized CSV like 'A,B,C' or None.
    """
    if not text:
        return None

    if cot_on:
        # In COT mode we prefer explicit structured output.
        # 1) Try JSON: {"answer":"A,B"} or {"final_answer":"A,B"}
        try:
            candidate = (text or "").strip()
            if "```" in candidate:
                candidate = re.sub(r"(?s)^```(?:json)?\s*|\s*```$", "", candidate.strip()).strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    for k in ("answer", "final_answer", "final"):
                        v = obj.get(k)
                        if isinstance(v, str):
                            norm = normalize_csv_letters(v)
                            return norm if norm else None
        except Exception:
            pass

        # 2) Fallback: dedicated final line like Answer:A,B
        m = FINAL_ANSWER_RE.search(text)
        if m:
            norm = normalize_csv_letters(m.group(1))
            return norm if norm else None
        return None

    raw = extract_letters_csv(text)
    norm = normalize_csv_letters(raw or "")
    return norm if norm else None


def _strip_markdown_code_fences(s: str) -> str:
    s = (s or "").strip()
    if "```" not in s:
        return s
    # Remove outermost code fences if present
    return re.sub(r"(?s)^\s*```(?:json)?\s*|\s*```\s*$", "", s).strip()


def extract_first_json_object(s: str) -> Optional[str]:
    """
    Best-effort extraction of the first JSON object from text.
    We still require json.loads() to succeed afterwards.
    """
    if not s:
        return None
    candidate = _strip_markdown_code_fences(s).strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        return candidate[start : end + 1].strip()
    return None


def extract_first_json_object_strict(s: str) -> Optional[str]:
    """
    Streaming extraction of the first JSON object from text.
    More robust than using rfind('}') when extra text follows JSON.
    """
    if not s:
        return None
    candidate = _strip_markdown_code_fences(str(s)).strip()
    start = candidate.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return candidate[start : i + 1].strip()
    return None


def parse_answer_json(text: str, *, mcq: bool) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse answering-model output as a strict JSON object.
    Required: {"answer": ...}
    - MCQ: answer can be "A,B" or ["A","B"] (normalized to "A,B")
    - Freeform: answer can be string/number/boolean/list[string] (cast to string; list joined by newlines)
    """
    raw = (text or "").strip()
    candidate = extract_first_json_object_strict(raw) or extract_first_json_object(raw)
    if not candidate:
        return None, "no_json_object_found"
    try:
        obj = json.loads(candidate)
    except Exception as e:
        return None, f"json_parse_error: {type(e).__name__}: {e}"
    if not isinstance(obj, dict):
        return None, "json_not_object"
    if "answer" not in obj:
        return None, "missing_answer_field"

    ans = obj.get("answer")
    if mcq:
        if isinstance(ans, list):
            joined = ",".join([str(x) for x in ans if str(x).strip()])
            norm = normalize_csv_letters(joined)
        else:
            norm = normalize_csv_letters(str(ans or ""))
        if not norm:
            return None, "empty_or_unparseable_mcq_answer"
        obj["answer"] = norm
        return obj, None

    if ans is None:
        return None, "empty_answer"
    if isinstance(ans, list):
        parts = [str(x).strip() for x in ans if str(x).strip()]
        ans_s = "\n".join(parts).strip()
    else:
        ans_s = str(ans).strip()
    if not ans_s:
        return None, "empty_answer"
    obj["answer"] = ans_s
    return obj, None


def is_multiple_choice(question_type_raw: str, options: List[str]) -> bool:
    """
    Robust question type detection.
    - Prefer explicit Question_Type when provided
    - Fallback to presence of options (common when Question_Type is missing/dirty)
    """
    qt = str(question_type_raw or "").strip().lower()
    if qt:
        # English variants
        if qt in {"multiple choice", "mcq", "multi-choice", "multichoice", "single choice", "single-choice"}:
            return True
        if "multiple choice" in qt or (("choice" in qt) and ("freeform" not in qt) and ("fill" not in qt)):
            return True
        # Chinese variants (best-effort)
        if any(k in qt for k in ["选择题", "单选", "多选"]):
            return True

    # Fallback: if options exist, treat as MCQ.
    # This is intentionally permissive to avoid sending MCQ to freeform judging prompts.
    if isinstance(options, list) and len([o for o in options if str(o).strip()]) >= 2:
        return True
    return False


def _sanitize_call_for_logging(provider: str, call_obj: Optional[Dict[str, Any]], image_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Remove large base64 image payloads from request logs.
    Replace them with a compact reference (image_path) so jsonl stays readable.
    """
    if not call_obj:
        return call_obj

    # shallow copy top-level dict
    out = dict(call_obj)
    req = out.get("request")
    if not isinstance(req, dict):
        return out

    p = (provider or "").lower()
    req2 = dict(req)

    try:
        if p in {"openai", "openrouter"}:
            # messages: [{role, content}, ...]
            messages = req2.get("messages")
            if isinstance(messages, list):
                new_messages = []
                for msg in messages:
                    if not isinstance(msg, dict):
                        new_messages.append(msg)
                        continue
                    msg2 = dict(msg)
                    content = msg2.get("content")
                    # user content can be list of blocks
                    if isinstance(content, list):
                        new_blocks = []
                        for blk in content:
                            if isinstance(blk, dict) and blk.get("type") == "image_url":
                                blk2 = dict(blk)
                                img = blk2.get("image_url")
                                if isinstance(img, dict):
                                    img2 = dict(img)
                                    img2["url"] = image_path or "<image_omitted>"
                                    blk2["image_url"] = img2
                                new_blocks.append(blk2)
                            else:
                                new_blocks.append(blk)
                        msg2["content"] = new_blocks
                    new_messages.append(msg2)
                req2["messages"] = new_messages

        elif p == "gemini":
            # contents: [{role, parts:[{text}|{inline_data:{mime_type,data}}]}]
            contents = req2.get("contents")
            if isinstance(contents, list):
                new_contents = []
                for c in contents:
                    if not isinstance(c, dict):
                        new_contents.append(c)
                        continue
                    c2 = dict(c)
                    parts = c2.get("parts")
                    if isinstance(parts, list):
                        new_parts = []
                        for pt in parts:
                            if isinstance(pt, dict) and "inline_data" in pt and isinstance(pt.get("inline_data"), dict):
                                pt2 = dict(pt)
                                inline2 = dict(pt2["inline_data"])
                                inline2["data"] = image_path or "<image_omitted>"
                                pt2["inline_data"] = inline2
                                new_parts.append(pt2)
                            else:
                                new_parts.append(pt)
                        c2["parts"] = new_parts
                    new_contents.append(c2)
                req2["contents"] = new_contents

        elif p == "claude":
            # messages: [{role, content:[{type:text}|{type:image, source:{... data:...}}]}]
            messages = req2.get("messages")
            if isinstance(messages, list):
                new_messages = []
                for msg in messages:
                    if not isinstance(msg, dict):
                        new_messages.append(msg)
                        continue
                    msg2 = dict(msg)
                    content = msg2.get("content")
                    if isinstance(content, list):
                        new_blocks = []
                        for blk in content:
                            if isinstance(blk, dict) and blk.get("type") == "image":
                                blk2 = dict(blk)
                                src = blk2.get("source")
                                if isinstance(src, dict):
                                    src2 = dict(src)
                                    src2["data"] = image_path or "<image_omitted>"
                                    blk2["source"] = src2
                                new_blocks.append(blk2)
                            else:
                                new_blocks.append(blk)
                        msg2["content"] = new_blocks
                    new_messages.append(msg2)
                req2["messages"] = new_messages
    except Exception:
        # Best-effort sanitization only; never break logging due to unexpected schema.
        pass

    out["request"] = req2

    # Also sanitize RESPONSE (best-effort) to avoid huge unreadable blobs in jsonl logs
    try:
        resp = out.get("response")
        if isinstance(resp, dict):
            # deep-ish copy via JSON (safe for plain dict/list scalars)
            resp2 = json.loads(json.dumps(resp))
            choices = resp2.get("choices")
            if isinstance(choices, list):
                for ch in choices:
                    if not isinstance(ch, dict):
                        continue
                    msg = ch.get("message")
                    if isinstance(msg, dict):
                        rd = msg.get("reasoning_details")
                        if isinstance(rd, list):
                            new_rd = []
                            for it in rd:
                                if isinstance(it, dict) and it.get("type") == "reasoning.encrypted":
                                    it2 = dict(it)
                                    it2["data"] = "<omitted>"
                                    new_rd.append(it2)
                                else:
                                    new_rd.append(it)
                            msg["reasoning_details"] = new_rd
                        if isinstance(msg.get("reasoning"), str):
                            msg["reasoning"] = strip_base64_payloads(msg.get("reasoning"))
            out["response"] = resp2
    except Exception:
        pass
    return out


# ----------------------------
# Config
# ----------------------------

@dataclass
class ProviderConfig:
    provider: str  # openai | openrouter | gemini | claude
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 60.0
    temperature: float = 0.0
    top_p: float = 0.75
    max_tokens: int = 10000

@dataclass
class RunConfig:
    input_path: str
    sheet_name: Optional[str]
    images_root: str
    out_dir: str
    concurrency: int  # legacy: if set, used as default for both model/judge concurrency
    model_concurrency: int
    judge_concurrency: int
    max_retries: int
    retry_base_delay_s: float
    retry_max_delay_s: float
    skip_image_missing: bool = True
    limit: Optional[int] = None

    # NEW: cot switch
    cot: str = "off"  # on/off

    # NEW: answering-model JSON enforcement retries (content-level, not HTTP retries)
    answer_json_max_attempts: int = 3

    # NEW: majority vote (answering model called N times; take majority answer)
    majority_vote: int = 1

    # NEW: MCQ prompt hint about single-vs-multi answer (derived from gold Answer)
    # on/off (string for CLI/GUI consistency)
    mcq_cardinality_hint: str = "off"

    # NEW: reasoning effort (OpenRouter)
    # - off => do not send the parameter
    # - otherwise send reasoning.effort in the request
    model_reasoning_effort: str = "off"  # off/xhigh/high/medium/low/minimal/none
    judge_reasoning_effort: str = "off"  # off/xhigh/high/medium/low/minimal/none

    # OpenRouter cost tracking (best-effort, USD)
    cost_total_usd_model: float = 0.0
    cost_total_usd_judge: float = 0.0

    # vpn/proxy switch
    vpn: str = "off"       # on/off
    proxy: str = ""        # proxy url, e.g. http://127.0.0.1:7897

    # cancellation (best-effort; used by GUI)
    cancel_event: Optional[threading.Event] = None


# ----------------------------
# HTTP client & Retry
# ----------------------------

def make_async_client(run_cfg: RunConfig) -> httpx.AsyncClient:
    """
    vpn=off => direct (ignore env proxies)
    vpn=on  => use proxy (explicit --proxy, otherwise default localhost:7897)
    """
    if run_cfg.vpn == "off":
        return httpx.AsyncClient(trust_env=False)

    proxy = run_cfg.proxy.strip()
    if not proxy:
        proxy = "http://127.0.0.1:7897"  # default
        # socks5 example:
        # proxy = "socks5://127.0.0.1:7897"

    return httpx.AsyncClient(proxy=proxy, trust_env=False)

async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: Dict[str, str],
    json_payload: Dict[str, Any],
    timeout_s: float,
    max_retries: int,
    base_delay_s: float,
    max_delay_s: float,
) -> Tuple[Optional[httpx.Response], Optional[str], int]:
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = await client.request(
                method, url, headers=headers, json=json_payload, timeout=timeout_s
            )
            if resp.status_code in RETRIABLE_STATUS and attempt <= max_retries:
                delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
                delay *= (0.7 + 0.6 * random.random())
                await asyncio.sleep(delay)
                continue
            return resp, None, attempt
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            err = f"{type(e).__name__}: {e}"
            if attempt <= max_retries:
                delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
                delay *= (0.7 + 0.6 * random.random())
                await asyncio.sleep(delay)
                continue
            return None, err, attempt
        except Exception as e:
            return None, f"{type(e).__name__}: {e}", attempt


# ----------------------------
# Providers
# ----------------------------

class LLMProvider:
    def __init__(self, cfg: ProviderConfig, run_cfg: RunConfig):
        self.cfg = cfg
        self.run_cfg = run_cfg

    async def call(
        self,
        prompt: str,
        image_data_url: Optional[str] = None,
        *,
        label: str = "",
        role: str = "other",  # model|judge|other (used for cost buckets)
        reasoning_effort: str = "off",
    ) -> Dict[str, Any]:
        if self.run_cfg.cancel_event is not None and self.run_cfg.cancel_event.is_set():
            return {"request": None, "response": None, "error": "cancelled", "attempts": 0, "status": None}
        p = self.cfg.provider.lower()
        if p in {"openai", "openrouter"}:
            return await self._call_openai_chat_completions(
                prompt, image_data_url, label=label, role=role, reasoning_effort=reasoning_effort
            )
        elif p == "gemini":
            return await self._call_gemini(prompt, image_data_url)
        elif p == "claude":
            return await self._call_claude(prompt, image_data_url)
        else:
            raise ValueError(f"Unknown provider: {self.cfg.provider}")

    async def _call_openai_chat_completions(
        self,
        prompt: str,
        image_data_url: Optional[str],
        *,
        label: str = "",
        role: str = "other",
        reasoning_effort: str = "off",
    ) -> Dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        p = (self.cfg.provider or "").lower()
        if p == "openrouter":
            # Recommended by OpenRouter for attribution/analytics; safe defaults.
            headers.setdefault("HTTP-Referer", "https://localhost/")
            headers.setdefault("X-Title", "llm-evaluation")

        if image_data_url:
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        else:
            user_content = prompt

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": "You are a careful assistant. Follow instructions exactly."},
                {"role": "user", "content": user_content},
            ],
        }

        # OpenRouter: optional reasoning effort + require_parameters guard.
        if p == "openrouter":
            eff = (reasoning_effort or "off").strip().lower()
            if eff and eff != "off":
                payload["reasoning"] = {"effort": eff}
                payload["provider"] = {"require_parameters": True}

        async with make_async_client(self.run_cfg) as client:
            resp, err, attempts = await request_with_retry(
                client, "POST", url, headers, payload,
                self.cfg.timeout_s, self.run_cfg.max_retries,
                self.run_cfg.retry_base_delay_s, self.run_cfg.retry_max_delay_s
            )
            # Best-effort: OpenRouter cost query + fatal parameter guard
            cost_usd: Optional[float] = None
            cost_total_usd_model: Optional[float] = None
            cost_total_usd_judge: Optional[float] = None
            if p == "openrouter" and resp is not None:
                try:
                    # If require_parameters fails, OpenRouter should respond with a 4xx + error JSON.
                    if isinstance(resp.status_code, int) and resp.status_code >= 400:
                        try:
                            j = resp.json()
                        except Exception:
                            j = None
                        msg = None
                        if isinstance(j, dict):
                            msg = (j.get("error") or {}).get("message") if isinstance(j.get("error"), dict) else j.get("message")
                        if msg and ("require_parameters" in str(msg) or "reasoning" in str(msg) or "effort" in str(msg)):
                            if self.run_cfg.cancel_event is not None:
                                self.run_cfg.cancel_event.set()
                            print(
                                f"[OPENROUTER] Parameter not supported (reasoning.effort). "
                                f"Disable reasoning effort and re-run. error={msg}",
                                flush=True,
                            )
                            raise RuntimeError(
                                "OpenRouter rejected reasoning.effort with provider.require_parameters=true. "
                                "Disable reasoning effort (set to off) and re-run. "
                                f"Details: {msg}"
                            )
                except Exception:
                    pass

                try:
                    rj = self._resp_json(resp)
                    gen_id = rj.get("id") if isinstance(rj, dict) else None
                    if isinstance(gen_id, str) and gen_id:
                        cost_usd = await _openrouter_query_cost_usd(
                            client=client,
                            base_url=self.cfg.base_url,
                            api_key=self.cfg.api_key,
                            gen_id=gen_id,
                            run_cfg=self.run_cfg,
                        )
                        if isinstance(cost_usd, (int, float)):
                            # Best-effort: simple sum; concurrency ordering not guaranteed (acceptable per requirement).
                            if role == "model":
                                self.run_cfg.cost_total_usd_model = float(getattr(self.run_cfg, "cost_total_usd_model", 0.0) or 0.0) + float(cost_usd)
                            elif role == "judge":
                                self.run_cfg.cost_total_usd_judge = float(getattr(self.run_cfg, "cost_total_usd_judge", 0.0) or 0.0) + float(cost_usd)
                            cost_total_usd_model = float(getattr(self.run_cfg, "cost_total_usd_model", 0.0) or 0.0)
                            cost_total_usd_judge = float(getattr(self.run_cfg, "cost_total_usd_judge", 0.0) or 0.0)
                except Exception:
                    pass
        return {
            "request": payload,
            "response": self._resp_json(resp),
            "error": err,
            "attempts": attempts,
            "status": getattr(resp, "status_code", None),
            "cost_usd": cost_usd,
            "cost_total_usd_model": cost_total_usd_model,
            "cost_total_usd_judge": cost_total_usd_judge,
        }

    async def _call_gemini(self, prompt: str, image_data_url: Optional[str]) -> Dict[str, Any]:
        base = self.cfg.base_url.rstrip("/")
        url = f"{base}/v1beta/models/{self.cfg.model}:generateContent?key={self.cfg.api_key}"

        parts = [{"text": prompt}]
        if image_data_url:
            b64 = image_data_url.split("base64,", 1)[-1]
            mime = image_data_url.split(":", 1)[-1].split(";", 1)[0]
            parts.append({"inline_data": {"mime_type": mime, "data": b64}})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": self.cfg.temperature,
                "topP": self.cfg.top_p,
                "maxOutputTokens": self.cfg.max_tokens,
            },
        }

        async with make_async_client(self.run_cfg) as client:
            resp, err, attempts = await request_with_retry(
                client, "POST", url, {}, payload,
                self.cfg.timeout_s, self.run_cfg.max_retries,
                self.run_cfg.retry_base_delay_s, self.run_cfg.retry_max_delay_s
            )
        return {
            "request": payload,
            "response": self._resp_json(resp),
            "error": err,
            "attempts": attempts,
            "status": getattr(resp, "status_code", None),
        }

    async def _call_claude(self, prompt: str, image_data_url: Optional[str]) -> Dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + "/v1/messages"
        headers = {
            "x-api-key": self.cfg.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        content_blocks = [{"type": "text", "text": prompt}]
        if image_data_url:
            b64 = image_data_url.split("base64,", 1)[-1]
            mime = image_data_url.split(":", 1)[-1].split(";", 1)[0]
            content_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64}
            })

        payload = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "messages": [{"role": "user", "content": content_blocks}],
        }

        async with make_async_client(self.run_cfg) as client:
            resp, err, attempts = await request_with_retry(
                client, "POST", url, headers, payload,
                self.cfg.timeout_s, self.run_cfg.max_retries,
                self.run_cfg.retry_base_delay_s, self.run_cfg.retry_max_delay_s
            )
        return {
            "request": payload,
            "response": self._resp_json(resp),
            "error": err,
            "attempts": attempts,
            "status": getattr(resp, "status_code", None),
        }

    @staticmethod
    def _resp_json(resp: Optional[httpx.Response]) -> Any:
        if resp is None:
            return None
        try:
            return resp.json()
        except Exception:
            return {"raw_text": resp.text}


def normalize_api_protocol(name: str) -> str:
    """
    Normalize user-facing protocol names to internal provider ids.
    Supported (case-insensitive):
    - OpenAI -> openai
    - OpenRouter -> openrouter
    - Anthropic -> claude
    - Google -> gemini
    Also accepts legacy ids: openai/claude/gemini
    """
    s = (name or "").strip().lower()
    if s in {"openai"}:
        return "openai"
    if s in {"openrouter"}:
        return "openrouter"
    if s in {"anthropic", "claude"}:
        return "claude"
    if s in {"google", "gemini"}:
        return "gemini"
    return s


async def _openrouter_query_cost_usd(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    gen_id: str,
    run_cfg: RunConfig,
) -> Optional[float]:
    """
    Best-effort cost query for OpenRouter using generation id.
    Works only for provider=openrouter.
    """
    url = base_url.rstrip("/") + "/v1/generation"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = await client.get(url, params={"id": gen_id}, headers=headers, timeout=run_cfg.retry_max_delay_s or 20.0)
        if resp.status_code >= 400:
            return None
        j = resp.json()
        if not isinstance(j, dict):
            return None
        data = j.get("data")
        if isinstance(data, dict):
            # Try common locations
            for path in [
                ("total_cost",),
                ("usage", "total_cost"),
                ("usage", "total_cost_usd"),
                ("usage", "cost"),
            ]:
                cur: Any = data
                ok = True
                for k in path:
                    if isinstance(cur, dict) and k in cur:
                        cur = cur.get(k)
                    else:
                        ok = False
                        break
                if ok and isinstance(cur, (int, float)):
                    return float(cur)
        # fallback: search top-level keys
        for k in ("total_cost", "total_cost_usd", "cost"):
            v = j.get(k)
            if isinstance(v, (int, float)):
                return float(v)
    except Exception:
        return None
    return None


def extract_text_from_provider_response(provider: str, resp_json: Any) -> str:
    if not resp_json:
        return ""
    p = provider.lower()
    try:
        if p in {"openai", "openrouter"}:
            choices = resp_json.get("choices", [])
            if not choices:
                return ""
            content = choices[0]["message"]["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        texts.append(blk.get("text", ""))
                return "\n".join(texts).strip()
            return str(content)

        if p == "gemini":
            cands = resp_json.get("candidates", [])
            if not cands:
                return ""
            parts = cands[0].get("content", {}).get("parts", [])
            texts = [pt.get("text", "") for pt in parts if isinstance(pt, dict) and "text" in pt]
            return "\n".join(texts).strip()

        if p == "claude":
            blocks = resp_json.get("content", [])
            texts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
            return "\n".join(texts).strip()
    except Exception:
        return ""
    return ""


# ----------------------------
# Prompting
# ----------------------------

def build_mcq_prompt(question: str, options: List[str], cot_on: bool, *, multi_answer: Optional[bool] = None) -> str:
    opts = "\n".join(options)
    if multi_answer is True:
        cardinality_hint = "Important: The correct answer includes MORE THAN ONE option. Do NOT assume it is single-choice.\n"
    elif multi_answer is False:
        cardinality_hint = "Important: The correct answer is EXACTLY ONE option (single-choice).\n"
    else:
        cardinality_hint = ""

    if cot_on:
        return (
            "You will answer a multiple-choice question. Some questions may have multiple correct options.\n"
            + cardinality_hint +
            "You SHOULD output your chain-of-thought reasoning.\n"
            "Output format rules (STRICT):\n"
            "- Return ONLY a JSON object. No markdown. No extra text.\n"
            "- JSON schema:\n"
            '{ "answer": "A" | "A,B,C", "cot": string }\n'
            "- 'answer' must contain ONLY letters among A,B,C,D,E.\n"
            "- If multiple, separate by comma ',' with NO spaces (example: \"A,B,C\").\n"
            "- 'cot' is your step-by-step reasoning.\n"
            "Example:\n"
            '{ "answer": "B", "cot": "..." }\n\n'
            f"Question:\n{question}\n\n"
            f"Options:\n{opts}\n"
        )

    return (
        "You will answer a multiple-choice question. Some questions may have multiple correct options.\n"
        + cardinality_hint +
        "Do NOT output chain-of-thought. Provide only the final answer.\n"
        "Output format rules (STRICT):\n"
        "- Return ONLY a JSON object. No markdown. No extra text.\n"
        "- JSON schema:\n"
        '{ "answer": "A" | "A,B,C" }\n'
        "- 'answer' must contain ONLY letters among A,B,C,D,E.\n"
        "- If multiple, separate by comma ',' with NO spaces (example: \"A,B,C\").\n"
        "Example:\n"
        '{ "answer": "B" }\n\n'
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n"
    )


def build_freeform_answer_prompt(question: str, cot_on: bool) -> str:
    if cot_on:
        return (
            "Answer the following question.\n"
            "You SHOULD output your chain-of-thought reasoning.\n"
            "Output format rules (STRICT):\n"
            "- Return ONLY a JSON object. No markdown. No extra text.\n"
            "- JSON schema:\n"
            '{ "answer": string | number | boolean | list[string], "cot": string }\n'
            "- 'answer' must be the final result.\n"
            "- For fill-in with multiple blanks, you may output a list of strings.\n"
            "- 'cot' is your step-by-step reasoning.\n\n"
            f"Question:\n{question}\n"
        )

    return (
        "Answer the following question.\n"
        "Do NOT output chain-of-thought. Provide only the final answer.\n"
        "Output format rules (STRICT):\n"
        "- Return ONLY a JSON object. No markdown. No extra text.\n"
        "- JSON schema:\n"
        '{ "answer": string | number | boolean | list[string] }\n'
        "- 'answer' must be the final result only (concise, no derivations).\n"
        "- For fill-in with multiple blanks, you may output a list of strings.\n\n"
        f"Question:\n{question}\n"
    )


def build_answer_json_retry_prefix(attempt_idx: int, last_error: str) -> str:
    """
    Content-level retry prefix when the answering model fails to output valid JSON.
    attempt_idx is 2..N (human-friendly).
    """
    err = (last_error or "").strip()
    return (
        "IMPORTANT: Your previous response was NOT valid JSON and could not be parsed.\n"
        f"Failure reason: {err}\n"
        "You MUST respond with ONLY a single JSON object, matching the required schema.\n"
        "Do NOT include markdown fences. Do NOT include any extra text before or after the JSON.\n"
        f"(Retry attempt {attempt_idx})\n\n"
    )


def majority_vote_pick(answers: List[str]) -> Optional[str]:
    """
    Pick the majority answer from a list of answer strings.
    - Ignores empty/blank answers
    - In ties, returns the earliest-seen among the top frequency
    """
    cleaned: List[str] = []
    for a in answers or []:
        s = str(a or "").strip()
        if s:
            cleaned.append(s)
    if not cleaned:
        return None
    counts: Dict[str, int] = {}
    for s in cleaned:
        counts[s] = counts.get(s, 0) + 1
    best = max(counts.values())
    for s in cleaned:
        if counts.get(s, 0) == best:
            return s
    return cleaned[0]

def build_freeform_judge_prompt(question: str, model_answer: str, gold: str) -> str:
    return (
        "You are a strict evaluator. Decide whether the model answer should be counted as correct.\n"
        "First, extract the final answer/result from the model answer (keep it short). Do NOT include any chain-of-thought.\n"
        "Return ONLY a JSON object. No markdown, no extra text.\n"
        "The JSON schema is:\n"
        "{\n"
        '  "verdict": "correct" | "incorrect" | "unjudgeable",\n'
        '  "extracted_answer": string|null,\n'
        '  "reason": string\n'
        "}\n"
        "Example:\n"
        '{ "verdict": "correct", "extracted_answer": "3", "reason": "Matches the gold answer." }\n\n'
        f"Question:\n{question}\n\n"
        f"Gold Answer:\n{gold}\n\n"
        f"Model Answer:\n{model_answer}\n"
    )


def build_mcq_judge_prompt_minimal(model_answer: str, gold_letters_csv: str) -> str:
    """
    Judge prompt for MCQ WITHOUT question/options to save tokens.
    Decide correctness based on whether the set of chosen letters exactly matches the gold.
    """
    return (
        "You are a strict evaluator for a multiple-choice question. Some questions may have multiple correct options.\n"
        "Your task: decide whether the model answer is correct.\n"
        "Correctness rule:\n"
        "- Extract the option letters chosen by the model (A-E).\n"
        "- Treat answers as a SET (order-insensitive; duplicates ignored).\n"
        "- The answer is correct ONLY IF the extracted set exactly equals the gold set.\n\n"
        "Return ONLY a JSON object. No markdown, no extra text.\n"
        "The JSON schema is:\n"
        "{\n"
        '  "verdict": "correct" | "incorrect" | "unjudgeable",\n'
        '  "extracted_answer": string|null,\n'
        '  "reason": string\n'
        "}\n"
        'Notes:\n'
        '- If you can extract letters, set extracted_answer to a normalized CSV like "A" or "A,B,C" (no spaces).\n'
        '- If the model does not provide a usable answer, set verdict="unjudgeable".\n\n'
        f"Gold Answer (letters CSV):\n{gold_letters_csv}\n\n"
        f"Model Answer:\n{model_answer}\n"
    )


def build_mcq_judge_prompt_from_raw(model_output: str, gold_letters_csv: str) -> str:
    """
    Judge prompt for MCQ when the answering-model JSON parsing failed.
    The judge must extract option letters from the raw model output, then compare to gold set.
    """
    return (
        "You are a strict evaluator for a multiple-choice question. Some questions may have multiple correct options.\n"
        "Your tasks:\n"
        "1) Extract the option letters chosen by the model (A-E) from the MODEL OUTPUT.\n"
        "2) Treat answers as a SET (order-insensitive; duplicates ignored).\n"
        "3) The answer is correct ONLY IF the extracted set exactly equals the gold set.\n\n"
        "Return ONLY a JSON object. No markdown, no extra text.\n"
        "The JSON schema is:\n"
        "{\n"
        '  "verdict": "correct" | "incorrect" | "unjudgeable",\n'
        '  "extracted_answer": string|null,\n'
        '  "reason": string\n'
        "}\n"
        'Notes:\n'
        '- If you can extract letters, set extracted_answer to a normalized CSV like "A" or "A,B,C" (no spaces).\n'
        '- If the model does not provide a usable answer, set verdict="unjudgeable".\n\n'
        f"Gold Answer (letters CSV):\n{gold_letters_csv}\n\n"
        f"Model Output:\n{model_output}\n"
    )








def parse_judge_json(text: str) -> Dict[str, Any]:
    """
    Parse the judge's output and return a structured dictionary.
    """
    text = (text or "").strip()
    try:
        # Best-effort: judge may include extra text after JSON; extract the first JSON object.
        candidate = extract_first_json_object_strict(text) or extract_first_json_object(text)
        if not candidate:
            raise ValueError("no_json_object_found")
        judge_json = json.loads(candidate)

        # 解析 "verdict" 字段
        verdict = judge_json.get("verdict", "").lower()

        # Metric policy: unjudgeable counts as incorrect.
        judge_json["judge_correct"] = (verdict == "correct")

        return judge_json
    except Exception as e:
        # 如果解析失败，返回一个默认的结果
        return {
            "verdict": "unjudgeable",
            "extracted_answer": None,
            "reason": f"Failed to parse judge JSON: {str(e)}",
            "judge_correct": False,
        }


# ----------------------------
# Evaluation core
# ----------------------------

async def eval_one(
        row: Dict[str, Any],
        idx: int,
        total: int,
        model_provider: LLMProvider,
        judge_provider: Optional[LLMProvider],
        run_cfg: RunConfig,
        model_sem: asyncio.Semaphore,
        judge_sem: asyncio.Semaphore,
        write_lock: asyncio.Lock,
        results_path: str,
) -> Dict[str, Any]:
    if run_cfg.cancel_event is not None and run_cfg.cancel_event.is_set():
        out = {
            "id": str(row.get("id", "")).strip(),
            "idx": idx,
            "total": total,
            "skipped": True,
            "skip_reason": "cancelled",
            "gold_raw": None,
            "gold": None,
            "pred_raw": None,
            "pred": None,
            "rule_correct": None,
            "judge_correct": False,
            "latency_ms": 0,
            "image_path": None,
            "model_call": None,
            "judge_call": None,
        }
        async with write_lock:
            safe_jsonl_append(results_path, out)
        print(f"[{idx}/{total}] id={out.get('id')}  SKIP(cancelled)", flush=True)
        return out
    qid = str(row.get("id", "")).strip()
    question = str(row.get("Question", "")).strip()
    options = load_options(row.get("Options", ""))  # 获取选项
    question_type_raw = str(row.get("Question_Type", "")).strip()  # 获取题目类型（原始）
    mcq = is_multiple_choice(question_type_raw, options)

    gold_raw = str(row.get("Answer", "")).strip()
    gold_norm = normalize_csv_letters(gold_raw) if mcq else gold_raw
    multi_answer: Optional[bool] = None
    if mcq and (getattr(run_cfg, "mcq_cardinality_hint", "on") == "on"):
        try:
            # Hint the answering model about single-vs-multi answer, without revealing which options or how many.
            # Uses the gold answer as the source of truth.
            s = csv_to_set(gold_norm)
            if len(s) >= 2:
                multi_answer = True
            elif len(s) == 1:
                multi_answer = False
        except Exception:
            multi_answer = None

    def _pick_image_cell(r: Dict[str, Any]) -> Any:
        # Dataset may provide multiple image columns, e.g. Image / Image.1.
        for k in ("Image", "Image.1", "Image_1", "Image2", "Image_2"):
            v = r.get(k, None)
            if v is None or is_nan(v):
                continue
            s = str(v).strip()
            if s and s.lower() != "nan":
                return v
        return r.get("Image", None)

    img_cell = _pick_image_cell(row)
    # Image_Dependency: supports 0/1/2
    # - 0: no image dependency (or no image)
    # - 1: image may help, but do not skip if missing
    # - 2: image is required; skip if missing
    try:
        img_dep = int(row.get("Image_Dependency", 0) or 0)
    except Exception:
        img_dep = 0

    cot_on = (run_cfg.cot == "on")

    # Resolve image
    image_path = None
    image_data_url = None
    image_path = resolve_image_path(run_cfg.images_root, img_cell)
    if image_path and os.path.exists(image_path):
        image_data_url = image_file_to_data_url(image_path)

    # Skip if image required but missing (dependency level 2)
    if img_dep >= 2 and not image_data_url and run_cfg.skip_image_missing:
        img_preview = ""
        if img_cell is not None and not is_nan(img_cell):
            img_preview = str(img_cell).strip()
        out = {
            "id": qid,
            "idx": idx,
            "total": total,
            "skipped": True,
            "skip_reason": "image_required_but_missing",
            "gold_raw": gold_raw,
            "gold": gold_norm,
            "pred_raw": None,
            "pred": None,
            "rule_correct": None,
            "judge_correct": None,
            "latency_ms": 0,
            "image_path": image_path,
            "model_call": None,
            "judge_call": None,
        }
        async with write_lock:
            safe_jsonl_append(results_path, out)
        print(f"[{idx}/{total}] id={qid}  SKIP(image missing)  gold={gold_norm}  image={img_preview}", flush=True)
        return out

    base_prompt = (
        build_mcq_prompt(question, options, cot_on=cot_on, multi_answer=multi_answer)
        if mcq
        else build_freeform_answer_prompt(question, cot_on=cot_on)
    )

    async def _answer_once() -> Dict[str, Any]:
        """
        One answering-model attempt with JSON enforcement retries.
        Returns a dict containing:
        - answer (str|None)
        - answer_json_ok (bool)
        - answer_json_attempts (int)
        - answer_json_error (str|None)
        - model_text (str)
        - model_call_log (dict|None)
        - model_latency_ms (int)
        """
        model_call = None
        model_text = ""
        model_call_log = None
        answer_obj: Optional[Dict[str, Any]] = None
        answer_parse_error: Optional[str] = None
        answer_attempts = 0

        t0 = now_ms()
        for k in range(max(1, int(run_cfg.answer_json_max_attempts))):
            answer_attempts = k + 1
            prompt = base_prompt
            if k > 0:
                prompt = build_answer_json_retry_prefix(k + 1, answer_parse_error or "unknown") + base_prompt

            async with model_sem:
                model_call = await model_provider.call(
                    prompt=prompt,
                    image_data_url=image_data_url,
                    label=f"model id={qid} idx={idx}",
                    role="model",
                    reasoning_effort=getattr(run_cfg, "model_reasoning_effort", "off"),
                )

            model_text = extract_text_from_provider_response(
                model_provider.cfg.provider, (model_call or {}).get("response")
            )
            model_call_log = _sanitize_call_for_logging(model_provider.cfg.provider, model_call, image_path)

            answer_obj, answer_parse_error = parse_answer_json(model_text, mcq=mcq)
            if answer_obj is not None:
                answer_parse_error = None
                break

        t1 = now_ms()
        extracted: Optional[str] = None
        if answer_obj is not None:
            extracted = str(answer_obj.get("answer", "")).strip()
        return {
            "answer": extracted,
            "answer_json_ok": (answer_obj is not None),
            "answer_json_attempts": answer_attempts,
            "answer_json_error": answer_parse_error,
            "model_text": model_text,
            "model_call_log": model_call_log,
            "model_latency_ms": (t1 - t0),
        }

    # ---- Majority vote execution model (requested):
    # If majority_vote=N, run the "vote=1" path sequentially N times.
    # - Rounds 1..N-1: answering model only; judge fields are empty; print immediately.
    # - Round N: compute final pred from ALL answers so far, then call judge ONCE; print final with judge.
    vote_n = max(1, int(getattr(run_cfg, "majority_vote", 1) or 1))
    vote_runs: List[Dict[str, Any]] = []
    vote_answers: List[str] = []
    vote_t0 = now_ms()

    def _preview(s: str) -> str:
        s2 = (s or "").replace("\n", " ").strip()
        return (s2[:200] + "...") if len(s2) > 200 else s2

    final_out: Optional[Dict[str, Any]] = None

    for round_idx in range(1, vote_n + 1):
        if run_cfg.cancel_event is not None and run_cfg.cancel_event.is_set():
            cancelled_out = {
                "id": qid,
                "idx": idx,
                "total": total,
                "skipped": True,
                "skip_reason": "cancelled",
                "gold_raw": gold_raw,
                "gold": gold_norm,
                "pred_raw": None,
                "pred": None,
                "answer_json_ok": False,
                "answer_json_attempts": 0,
                "answer_json_error": "cancelled",
                "majority_vote_n": vote_n,
                "majority_vote_round": round_idx,
                "majority_vote_is_final": False,
                "majority_vote_answers": vote_answers,
                "rule_correct": None,
                "judge_correct": None,
                "latency_ms": 0,
                "judge_latency_ms": None,
                "total_latency_ms": 0,
                "question": question,
                "options": options,
                "question_type_raw": question_type_raw,
                "question_type_inferred": "Multiple Choice" if mcq else "Freeform",
                "image_path": image_path,
                "image_data_url": image_path if image_path else None,
                "cot": run_cfg.cot,
                "model_call": None,
                "model_text": "",
                "judge": None,
                "judge_call": None,
            }
            async with write_lock:
                safe_jsonl_append(results_path, cancelled_out)
            print(f"[{idx}/{total}] id={qid}  CANCELLED  round={round_idx}/{vote_n}", flush=True)
            return cancelled_out

        one = await _answer_once()
        vote_runs.append(one)
        if one.get("answer"):
            vote_answers.append(str(one.get("answer")))

        def _judge_detail(jb: Any) -> str:
            try:
                if not isinstance(jb, dict):
                    return ""
                jj = jb.get("judge_json")
                if not isinstance(jj, dict):
                    return ""
                verdict = str(jj.get("verdict", "") or "").strip()
                extracted_norm = str(jj.get("extracted_answer_normalized", "") or "").strip()
                extracted_raw = jj.get("extracted_answer", None)
                extracted_raw_s = str(extracted_raw).strip() if isinstance(extracted_raw, str) else ""
                reason = str(jj.get("reason", "") or "").strip()
                parts: List[str] = []
                if verdict:
                    parts.append(f"verdict={verdict}")
                if extracted_norm:
                    parts.append(f"extracted_norm={extracted_norm}")
                elif extracted_raw_s:
                    parts.append(f"extracted={extracted_raw_s}")
                if reason:
                    parts.append(f"reason={_preview(reason)}")
                return "  ".join(parts)
            except Exception:
                return ""

        # Rounds 1..N-1: print and log immediately with empty judge.
        if round_idx < vote_n:
            out_round = {
                "id": qid,
                "idx": idx,
                "total": total,
                "skipped": False,
                "skip_reason": None,
                "gold_raw": gold_raw,
                "gold": gold_norm,
                "pred_raw": one.get("answer"),
                "pred": one.get("answer"),
                "answer_json_ok": bool(one.get("answer_json_ok")),
                "answer_json_attempts": int(one.get("answer_json_attempts") or 0),
                "answer_json_error": one.get("answer_json_error"),
                "majority_vote_n": vote_n,
                "majority_vote_round": round_idx,
                "majority_vote_is_final": False,
                "majority_vote_answers": list(vote_answers),
                "rule_correct": None,
                "judge_correct": None,
                "latency_ms": int(one.get("model_latency_ms") or 0),
                "judge_latency_ms": None,
                "total_latency_ms": int(one.get("model_latency_ms") or 0),
                "question": question,
                "options": options,
                "question_type_raw": question_type_raw,
                "question_type_inferred": "Multiple Choice" if mcq else "Freeform",
                "image_path": image_path,
                "image_data_url": image_path if image_path else None,
                "cot": run_cfg.cot,
                "model_call": one.get("model_call_log"),
                "model_text": (one.get("model_text") or ""),
                "judge": None,
                "judge_call": None,
            }
            async with write_lock:
                safe_jsonl_append(results_path, out_round)
            # Cost info (OpenRouter only; USD). Accumulators are stored in run_cfg.
            mc = out_round.get("model_call") if isinstance(out_round, dict) else None
            m_cost = mc.get("cost_usd") if isinstance(mc, dict) else None
            m_total = float(getattr(run_cfg, "cost_total_usd_model", 0.0) or 0.0)
            j_total = float(getattr(run_cfg, "cost_total_usd_judge", 0.0) or 0.0)
            cost_part = ""
            if isinstance(m_cost, (int, float)):
                cost_part = f"  model_cost_usd={float(m_cost):.6f}  totals_usd(model={m_total:.6f},judge={j_total:.6f})"
            print(
                f"[{idx}/{total}] id={qid}  ROUND {round_idx}/{vote_n}  mcq={mcq}  gold={gold_norm}  pred={out_round.get('pred')}  judge_correct=  model_ms={out_round.get('latency_ms')}  answers_so_far={vote_answers}{cost_part}",
                flush=True,
            )
            print(f"   model_text: {_preview(out_round.get('model_text') or '')}", flush=True)
            print("   judge_text: ", flush=True)
            print("   judge_detail: ", flush=True)
            continue

        # Round N: compute final answer from ALL collected answers, then call judge ONCE.
        extracted_final_answer: Optional[str] = majority_vote_pick([str(x) for x in vote_answers if str(x).strip()])
        judge_block = None
        judge_correct: Optional[bool] = None
        judge_call_log = None
        judge_latency_ms: Optional[int] = None

        if judge_provider is not None:
            # Adaptive judging:
            # - If we have an extracted/voted final answer, judge on that short answer.
            # - If JSON parsing failed (no extracted answer), fall back to judging from the raw model output.
            judge_input_kind = "extracted"
            if mcq:
                if extracted_final_answer is not None:
                    judge_prompt = build_mcq_judge_prompt_minimal(extracted_final_answer, gold_norm)
                else:
                    judge_input_kind = "raw_model_text"
                    judge_prompt = build_mcq_judge_prompt_from_raw(strip_base64_payloads(one.get("model_text") or ""), gold_norm)
            else:
                q_clean = strip_base64_payloads(question)
                gold_clean = strip_base64_payloads(gold_raw)
                if extracted_final_answer is not None:
                    ans_clean = strip_base64_payloads(extracted_final_answer)
                else:
                    judge_input_kind = "raw_model_text"
                    ans_clean = strip_base64_payloads(one.get("model_text") or "")
                judge_prompt = build_freeform_judge_prompt(q_clean, ans_clean, gold_clean)

            async with judge_sem:
                jt0 = now_ms()
                judge_call = await judge_provider.call(
                    prompt=judge_prompt,
                    image_data_url=None,
                    label=f"judge id={qid} idx={idx}",
                    role="judge",
                    reasoning_effort=getattr(run_cfg, "judge_reasoning_effort", "off"),
                )
                jt1 = now_ms()

            judge_latency_ms = jt1 - jt0
            judge_text = extract_text_from_provider_response(judge_provider.cfg.provider, (judge_call or {}).get("response"))
            judge_call_log = _sanitize_call_for_logging(judge_provider.cfg.provider, judge_call, image_path=None)
            judge_json = parse_judge_json(judge_text)
            verdict = str(judge_json.get("verdict", "unjudgeable")).lower()
            judge_correct = (verdict == "correct")

            extracted = judge_json.get("extracted_answer", None)
            extracted_norm = normalize_csv_letters(extracted) if isinstance(extracted, str) else ""
            judge_json["extracted_answer_normalized"] = extracted_norm

            judge_block = {
                "judge_text": judge_text,
                "judge_json": judge_json,
                "judge_latency_ms": judge_latency_ms,
                "judge_input_kind": judge_input_kind,
            }
        else:
            # No judge result (e.g., answering JSON failed) => count as incorrect.
            judge_correct = False

        vote_t1 = now_ms()
        model_wall_ms = vote_t1 - vote_t0
        total_ms = model_wall_ms + (judge_latency_ms or 0)

        final_out = {
            "id": qid,
            "idx": idx,
            "total": total,
            "skipped": False,
            "skip_reason": None,
            "gold_raw": gold_raw,
            "gold": gold_norm,
            # Final pred is determined after collecting ALL N answers.
            "pred_raw": extracted_final_answer,
            "pred": extracted_final_answer,
            "answer_json_ok": bool(one.get("answer_json_ok")),
            "answer_json_attempts": int(one.get("answer_json_attempts") or 0),
            "answer_json_error": one.get("answer_json_error"),
            "majority_vote_n": vote_n,
            "majority_vote_round": round_idx,
            "majority_vote_is_final": True,
            "majority_vote_answers": list(vote_answers),
            "rule_correct": None,
            "judge_correct": judge_correct,
            # Keep similar meaning as before: model wall time across the whole vote loop.
            "latency_ms": model_wall_ms,
            "judge_latency_ms": judge_latency_ms,
            "total_latency_ms": total_ms,
            "question": question,
            "options": options,
            "question_type_raw": question_type_raw,
            "question_type_inferred": "Multiple Choice" if mcq else "Freeform",
            "image_path": image_path,
            "image_data_url": image_path if image_path else None,
            "cot": run_cfg.cot,
            # For this round's "vote=1 style" visibility, keep the last round's call/text.
            "model_call": one.get("model_call_log"),
            "model_text": (one.get("model_text") or ""),
            "judge": judge_block,
            "judge_call": judge_call_log,
        }

        async with write_lock:
            safe_jsonl_append(results_path, final_out)

        # Cost info (OpenRouter only; USD). Accumulators are stored in run_cfg.
        m_call = final_out.get("model_call") if isinstance(final_out, dict) else None
        j_call = final_out.get("judge_call") if isinstance(final_out, dict) else None
        m_cost = m_call.get("cost_usd") if isinstance(m_call, dict) else None
        j_cost = j_call.get("cost_usd") if isinstance(j_call, dict) else None
        m_total = float(getattr(run_cfg, "cost_total_usd_model", 0.0) or 0.0)
        j_total = float(getattr(run_cfg, "cost_total_usd_judge", 0.0) or 0.0)
        cost_part = ""
        if isinstance(m_cost, (int, float)) or isinstance(j_cost, (int, float)):
            cost_part = (
                f"  cost_usd(model={float(m_cost or 0.0):.6f},judge={float(j_cost or 0.0):.6f})"
                f"  totals_usd(model={m_total:.6f},judge={j_total:.6f})"
            )
        print(
            f"[{idx}/{total}] id={qid}  FINAL {round_idx}/{vote_n}  mcq={mcq}  gold={gold_norm}  pred={extracted_final_answer}  judge_correct={judge_correct}  model_ms={model_wall_ms}  judge_ms={(judge_latency_ms or 0)}  total_ms={total_ms}{cost_part}",
            flush=True,
        )
        print(f"   model_text: {_preview(final_out.get('model_text') or '')}", flush=True)
        jtxt = (judge_block.get("judge_text") if isinstance(judge_block, dict) else "") or ""
        print(f"   judge_text: {_preview(jtxt)}", flush=True)
        print(f"   judge_detail: {_judge_detail(judge_block)}", flush=True)
        break

    assert final_out is not None
    return final_out





async def run_eval(df: pd.DataFrame, model_cfg: ProviderConfig, judge_cfg: Optional[ProviderConfig],
             run_cfg: RunConfig) -> None:
    # out_dir is the base output directory; each run writes into a timestamp subfolder.
    ensure_dir(run_cfg.out_dir)

    # 确保文件名中的非法字符被替换
    model_name_safe = re.sub(r'[<>:"/\\|?*]', '_', model_cfg.model)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = os.path.join(run_cfg.out_dir, timestamp)
    ensure_dir(run_dir)
    results_path = os.path.join(run_dir, f"results_{timestamp}_{model_name_safe}.jsonl")  # Same file for all logs
    summary_path = os.path.join(run_dir, f"summary_{timestamp}_{model_name_safe}.json")

    if os.path.exists(results_path):
        os.remove(results_path)

    model_provider = LLMProvider(model_cfg, run_cfg)
    judge_provider = LLMProvider(judge_cfg, run_cfg) if judge_cfg else None

    sem = asyncio.Semaphore(run_cfg.concurrency)
    write_lock = asyncio.Lock()

    rows = df.to_dict(orient="records")
    if run_cfg.limit is not None:
        rows = rows[: run_cfg.limit]

    total_questions = len(rows)
    print(
        f"Total questions: {total_questions} | model_concurrency={run_cfg.model_concurrency} | judge_concurrency={run_cfg.judge_concurrency} | vpn={run_cfg.vpn} | cot={run_cfg.cot} | majority_vote={run_cfg.majority_vote}",
        flush=True,
    )

    model_sem = asyncio.Semaphore(run_cfg.model_concurrency)
    judge_sem = asyncio.Semaphore(run_cfg.judge_concurrency)

    # Create tasks so cancellation can stop in-flight work.
    tasks = [
        asyncio.create_task(
            eval_one(
                r, i + 1, total_questions,
                model_provider, judge_provider, run_cfg,
                model_sem, judge_sem,
                write_lock, results_path
            )
        )
        for i, r in enumerate(rows)
    ]

    results: List[Dict[str, Any]] = []
    pending = set(tasks)
    cancelled = False
    while pending:
        if run_cfg.cancel_event is not None and run_cfg.cancel_event.is_set():
            cancelled = True
            for t in pending:
                t.cancel()
            break
        done, pending = await asyncio.wait(pending, timeout=0.2, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            try:
                r = await t
                if isinstance(r, dict):
                    results.append(r)
            except asyncio.CancelledError:
                continue
            except Exception as e:
                # Keep going; record an error marker for summary.
                results.append({"skipped": True, "skip_reason": f"task_error:{type(e).__name__}:{e}"})

    if cancelled:
        # Drain cancellation to close sockets promptly.
        await asyncio.gather(*list(pending), return_exceptions=True)

    # Final correctness is ALWAYS determined by judge_correct for ALL question types.
    # Metric policy: unjudgeable counts as incorrect.
    # ============= METRICS (single accuracy) =============
    def _norm_group_value(v: Any) -> str:
        if v is None or is_nan(v):
            return "<missing>"
        s = str(v).strip()
        return s if s else "<missing>"

    def _fmt_pct(x: float) -> str:
        return f"{x * 100.0:.2f}%"

    def _group_breakdown(rows_in: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
        """
        Group-by breakdown over *non-skipped* rows.
        model_correct: True/False (unjudgeable already mapped to False)
        """
        buckets: Dict[str, Dict[str, int]] = {}
        for r in rows_in:
            if r.get("skipped"):
                continue
            g = _norm_group_value(r.get(key))
            b = buckets.setdefault(g, {"total": 0, "correct": 0, "incorrect": 0})
            b["total"] += 1
            mc = r.get("model_correct")
            if mc is True:
                b["correct"] += 1
            else:
                b["incorrect"] += 1

        out: Dict[str, Any] = {}
        for g, b in buckets.items():
            total = b["total"]
            acc = (b["correct"] / total) if total else 0.0
            out[g] = {
                **b,
                "accuracy": acc,
                "accuracy_str": _fmt_pct(acc),
            }
        return out

    # Align results back to question index (tasks complete out-of-order).
    result_by_idx: Dict[int, Dict[str, Any]] = {}
    for r in results:
        try:
            i = int(r.get("idx") or 0)
        except Exception:
            i = 0
        if i >= 1 and i not in result_by_idx:
            result_by_idx[i] = r

    # Build a metric-only view of rows (do not mutate the original dataset columns).
    metric_rows: List[Dict[str, Any]] = []
    model_correct_list: List[Optional[bool]] = []
    for i, row in enumerate(rows):
        res = result_by_idx.get(i + 1) or {}
        skipped = bool(res.get("skipped"))
        judge_correct = res.get("judge_correct")
        if skipped:
            mc_val: Optional[bool] = None
        else:
            # write-back: single source of truth is judge_correct (unjudgeable already mapped to False)
            mc_val = bool(judge_correct)

        model_correct_list.append(mc_val)
        metric = dict(row)
        metric["skipped"] = skipped
        metric["skip_reason"] = res.get("skip_reason")
        metric["question_type_inferred"] = res.get("question_type_inferred")
        metric["model_correct"] = mc_val
        metric_rows.append(metric)

    overall_total = sum(1 for r in metric_rows if not r.get("skipped"))
    overall_correct = sum(1 for r in metric_rows if (not r.get("skipped")) and r.get("model_correct") is True)
    overall_incorrect = overall_total - overall_correct
    overall_acc = (overall_correct / overall_total) if overall_total else 0.0

    overall = {
        "correct": overall_correct,
        "incorrect": overall_incorrect,
        "total": overall_total,
        "accuracy": overall_acc,
        "accuracy_str": _fmt_pct(overall_acc),
    }

    # Breakdowns (only if the corresponding column exists in the dataset)
    cols = set(df.columns)
    breakdowns: Dict[str, Any] = {
        "by_question_type_inferred": _group_breakdown(metric_rows, "question_type_inferred"),
    }
    if "Image_Dependency" in cols:
        breakdowns["by_image_dependency"] = _group_breakdown(metric_rows, "Image_Dependency")
    if "Image_Complexity" in cols:
        breakdowns["by_image_complexity"] = _group_breakdown(metric_rows, "Image_Complexity")
    if "Subfield" in cols:
        breakdowns["by_subfield"] = _group_breakdown(metric_rows, "Subfield")
    if "Academic_Level" in cols:
        breakdowns["by_academic_level"] = _group_breakdown(metric_rows, "Academic_Level")

    # Best-effort: field/domain and difficulty columns may vary by dataset.
    # We pick the first matching column name in a common candidate list.
    def _pick_col(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in cols:
                return c
        return None

    field_col = _pick_col(["Field", "Domain", "Subject", "领域", "学科", "科目"])
    if field_col:
        breakdowns["by_field"] = {"column": field_col, "groups": _group_breakdown(metric_rows, field_col)}

    difficulty_col = _pick_col(["Difficulty", "难度", "Level", "Difficult"])
    if difficulty_col:
        breakdowns["by_difficulty"] = {"column": difficulty_col, "groups": _group_breakdown(metric_rows, difficulty_col)}
    # =====================================================================

    def _print_breakdown(title: str, groups: Dict[str, Any]) -> None:
        try:
            items = list(groups.items())
            items.sort(key=lambda kv: int(kv[1].get("total", 0)), reverse=True)
            print(f"\n=== Breakdown: {title} ===", flush=True)
            for name, b in items:
                total = b.get("total", 0)
                correct = b.get("correct", 0)
                incorrect = b.get("incorrect", 0)
                acc_str = b.get("accuracy_str", "")
                print(f"- {name}: total={total} correct={correct} incorrect={incorrect} acc={acc_str}", flush=True)
        except Exception:
            return

    # Save results to a new Excel file.
    # Requirement: keep ALL original dataset columns; only add/update `model_correct`.
    output_df = df.copy()
    if run_cfg.limit is not None:
        output_df = output_df.head(int(run_cfg.limit))
    output_df["model_correct"] = model_correct_list
    # Robustness: clean illegal control characters that openpyxl rejects.
    for col in output_df.columns:
        if output_df[col].dtype == "object":
            output_df[col] = output_df[col].map(_excel_safe_cell)
    output_df.to_excel(os.path.join(run_dir, f"evaluated_{timestamp}_{model_name_safe}.xlsx"), index=False)

    total = len(results)
    skipped = sum(1 for r in results if r.get("skipped"))

    failed = 0
    for r in results:
        mc = r.get("model_call")
        if not mc:
            continue
        if mc.get("error"):
            failed += 1
            continue
        st = mc.get("status")
        if isinstance(st, int) and st >= 400:
            failed += 1

    latencies = [r.get("latency_ms", 0) for r in results if not r.get("skipped")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    summary = {
        "total": total,
        "skipped": skipped,
        "failed_calls": failed,
        "avg_latency_ms": avg_latency,
        "cancelled": bool(cancelled),
        "out_dir_base": run_cfg.out_dir,
        "run_dir": run_dir,

        # judge-only metrics (single source of truth)
        "overall": overall,
        "breakdowns": breakdowns,

        "timestamp_ms": now_ms(),
        "vpn": run_cfg.vpn,
        "proxy": run_cfg.proxy if run_cfg.vpn == "on" else "",
        "cot": run_cfg.cot,
        "answer_json_max_attempts": run_cfg.answer_json_max_attempts,
        "model": {
            "provider": model_cfg.provider,
            "base_url": model_cfg.base_url,
            "model": model_cfg.model,
        },
        "judge_model": None if not judge_cfg else {
            "provider": judge_cfg.provider,
            "base_url": judge_cfg.base_url,
            "model": judge_cfg.model,
        }
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    print("\n=== Scores ===", flush=True)
    print(f"Overall (accuracy): {overall.get('accuracy_str')}", flush=True)
    # Requested prints
    if isinstance(breakdowns.get("by_image_complexity"), dict):
        _print_breakdown("Image_Complexity", breakdowns["by_image_complexity"])
    if isinstance(breakdowns.get("by_subfield"), dict):
        _print_breakdown("Subfield", breakdowns["by_subfield"])
    if isinstance(breakdowns.get("by_academic_level"), dict):
        _print_breakdown("Academic_Level", breakdowns["by_academic_level"])

    print(f"\nSaved full archives to: {results_path}", flush=True)
    print(f"Saved summary to: {summary_path}", flush=True)


# ----------------------------
# CLI / Main
# ----------------------------

def build_provider_cfg(d: Dict[str, Any], prefix: str) -> ProviderConfig:
    provider = d[f"{prefix}_provider"]
    base_url = d[f"{prefix}_base_url"]
    api_key = d[f"{prefix}_api_key"]
    model = d[f"{prefix}_model"]
    timeout_s = float(d.get(f"{prefix}_timeout_s", 60.0))
    temperature = float(d.get(f"{prefix}_temperature", 0.0))
    max_tokens = int(d.get(f"{prefix}_max_tokens", 10000))
    return ProviderConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_s=timeout_s,
        temperature=temperature,
        max_tokens=max_tokens,
    )

def cli_main(argv: Optional[List[str]] = None, *, cancel_event: Optional[threading.Event] = None) -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, default="", help="Optional YAML config.")
    ap.add_argument(
        "--root",
        type=str,
        default="",
        help="Convenience: root directory containing dataset.xlsx and images/ (used when --input/--images-root are not provided).",
    )
    ap.add_argument("--input", type=str, required=False, default="", help="Path to .xlsx dataset.")
    ap.add_argument("--sheet", type=str, default="", help="Sheet name (default: first sheet).")
    ap.add_argument("--images-root", type=str, default="", help="Root directory where 'images/' folder lives.")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Base output directory. Each run writes into a timestamp subfolder under this dir (default: from YAML or 'out_run').",
    )
    ap.add_argument("--concurrency", type=int, default=None, help="(legacy) default for both model/judge concurrency")
    ap.add_argument("--model-concurrency", type=int, default=None, help="Max in-flight requests to the answering model")
    ap.add_argument("--judge-concurrency", type=int, default=None, help="Max in-flight requests to the judge model")
    ap.add_argument("--max-retries", type=int, default=None)
    ap.add_argument("--retry-base-delay-s", type=float, default=None)
    ap.add_argument("--retry-max-delay-s", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)

    # NEW: COT switch
    ap.add_argument(
        "--cot",
        type=str,
        choices=["on", "off"],
        default=None,
        help="COT mode: on=include chain-of-thought in JSON field 'cot'; off=answer-only (no chain-of-thought).",
    )

    ap.add_argument(
        "--answer-json-max-attempts",
        type=int,
        default=None,
        help="Max attempts to re-ask answering model when JSON parsing fails (content-level retries). Default: 3",
    )

    ap.add_argument(
        "--majority-vote",
        type=int,
        default=None,
        help="Call answering model N times per question and take majority answer (default: 1).",
    )

    ap.add_argument(
        "--mcq-cardinality-hint",
        type=str,
        choices=["on", "off"],
        default=None,
        help="MCQ prompt hint: tell the answering model whether the gold answer is single-choice or multi-choice (on/off). Default: on.",
    )

    ap.add_argument(
        "--model-reasoning-effort",
        type=str,
        choices=["off", "xhigh", "high", "medium", "low", "minimal", "none"],
        default=None,
        help="OpenRouter reasoning effort for the answering model. off => do not send reasoning.effort.",
    )
    ap.add_argument(
        "--judge-reasoning-effort",
        type=str,
        choices=["off", "xhigh", "high", "medium", "low", "minimal", "none"],
        default=None,
        help="OpenRouter reasoning effort for the judge model. off => do not send reasoning.effort.",
    )

    # VPN/proxy switch
    ap.add_argument("--vpn", type=str, choices=["on", "off"], default=None,
                    help="VPN mode switch: on=use proxy, off=direct")
    ap.add_argument("--proxy", type=str, default="",
                    help="Proxy URL, e.g. http://127.0.0.1:7897 or socks5://127.0.0.1:7897")

    # model (protocol/provider)
    ap.add_argument("--model-provider", "--api-protocol", dest="model_provider", type=str, default="")
    ap.add_argument("--model-base-url", type=str, default="")
    ap.add_argument("--model-api-key", type=str, default="")
    ap.add_argument("--model-name", type=str, default="")
    ap.add_argument("--model-timeout-s", type=float, default=None)
    ap.add_argument("--model-temperature", type=float, default=None)
    ap.add_argument("--model-top-p", type=float, default=None)
    ap.add_argument("--model-max-tokens", type=int, default=None)

    # judge (deprecated flag)
    ap.add_argument(
        "--judge-enable",
        action="store_true",
        help="DEPRECATED (no-op). Scoring always uses the judge model.",
    )
    ap.add_argument("--judge-provider", "--judge-api-protocol", dest="judge_provider", type=str, default="")
    ap.add_argument("--judge-base-url", type=str, default="")
    ap.add_argument("--judge-api-key", type=str, default="")
    ap.add_argument("--judge-name", type=str, default="")
    ap.add_argument("--judge-timeout-s", type=float, default=None)
    ap.add_argument("--judge-temperature", type=float, default=None)
    ap.add_argument("--judge-top-p", type=float, default=None)
    ap.add_argument("--judge-max-tokens", type=int, default=None)

    args = ap.parse_args(argv)

    cfg = load_yaml_config(args.config) if args.config else {}

    root_path = (args.root or cfg.get("root_path", "") or "").strip()

    input_path = args.input or cfg.get("input_path", "")
    if not input_path and root_path:
        input_path = os.path.join(root_path, "dataset.xlsx")
    if not input_path:
        raise ValueError("Missing --input (xlsx path) or input_path/root_path in YAML/CLI.")

    sheet_name = args.sheet or cfg.get("sheet_name", "")
    sheet_name = sheet_name if sheet_name else None

    images_root = args.images_root or cfg.get("images_root", "")
    if not images_root and root_path:
        # Convention: images live under <root>/images/
        # Keep images_root=<root> so dataset can store "images/xxx.jpg"
        images_root = root_path
    out_dir = (args.out_dir if args.out_dir is not None else cfg.get("out_dir", "out_run"))

    # model cfg
    model_dict = {
        "model_provider": normalize_api_protocol(args.model_provider or cfg.get("model", {}).get("provider", "openai")),
        "model_base_url": args.model_base_url or cfg.get("model", {}).get("base_url", "https://api.openai.com"),
        "model_api_key": args.model_api_key or cfg.get("model", {}).get("api_key", os.getenv("OPENAI_API_KEY", "")),
        "model_model": args.model_name or cfg.get("model", {}).get("model", ""),
        "model_timeout_s": (args.model_timeout_s if args.model_timeout_s is not None else cfg.get("model", {}).get("timeout_s", 60.0)),
        "model_temperature": (args.model_temperature if args.model_temperature is not None else cfg.get("model", {}).get("temperature", 0.0)),
        "model_top_p": (args.model_top_p if args.model_top_p is not None else cfg.get("model", {}).get("top_p", 0.75)),
        "model_max_tokens": (args.model_max_tokens if args.model_max_tokens is not None else cfg.get("model", {}).get("max_tokens", 10000)),
    }
    if not model_dict["model_model"]:
        raise ValueError("Missing model name: --model-name or model.model in YAML.")
    if not model_dict["model_api_key"] and model_dict["model_provider"].lower() != "gemini":
        raise ValueError("Missing model api key: --model-api-key or env/YAML.")

    # IMPORTANT: keep your original mapping for build_provider_cfg
    # build_provider_cfg expects keys: f"{prefix}_provider/base_url/api_key/model"
    model_dict_for_builder = {
        "model_provider": model_dict["model_provider"],
        "model_base_url": model_dict["model_base_url"],
        "model_api_key": model_dict["model_api_key"],
        "model_model": model_dict["model_model"],
        "model_timeout_s": model_dict["model_timeout_s"],
        "model_temperature": model_dict["model_temperature"],
        "model_top_p": model_dict["model_top_p"],
        "model_max_tokens": model_dict["model_max_tokens"],
    }

    # Fix: build_provider_cfg uses "{prefix}_model", not "{prefix}_model_model"
    # So we create a compatible dict:
    model_cfg = ProviderConfig(
        provider=model_dict_for_builder["model_provider"],
        base_url=model_dict_for_builder["model_base_url"],
        api_key=model_dict_for_builder["model_api_key"],
        model=model_dict_for_builder["model_model"],
        timeout_s=float(model_dict_for_builder.get("model_timeout_s", 60.0)),
        temperature=float(model_dict_for_builder.get("model_temperature", 0.0)),
        top_p=float(model_dict_for_builder.get("model_top_p", 0.75)),
        max_tokens=int(model_dict_for_builder.get("model_max_tokens", 10000)),
    )

    # judge cfg (ALWAYS ON): scoring always uses judge model now.
    judge_dict = {
        "judge_provider": normalize_api_protocol(args.judge_provider or cfg.get("judge", {}).get("provider", model_cfg.provider)),
        "judge_base_url": args.judge_base_url or cfg.get("judge", {}).get("base_url", model_cfg.base_url),
        "judge_api_key": (
            args.judge_api_key
            or cfg.get("judge", {}).get("api_key", "")
            or model_dict.get("model_api_key", "")
        ),
        "judge_model": args.judge_name or cfg.get("judge", {}).get("model", model_cfg.model),
        "judge_timeout_s": (args.judge_timeout_s if args.judge_timeout_s is not None else cfg.get("judge", {}).get("timeout_s", 60.0)),
        "judge_temperature": (
            args.judge_temperature if args.judge_temperature is not None else cfg.get("judge", {}).get("temperature", 0.0)
        ),
        "judge_top_p": (args.judge_top_p if args.judge_top_p is not None else cfg.get("judge", {}).get("top_p", 0.75)),
        "judge_max_tokens": (args.judge_max_tokens if args.judge_max_tokens is not None else cfg.get("judge", {}).get("max_tokens", 10000)),
    }
    if not judge_dict["judge_model"]:
        raise ValueError("Missing judge model name: --judge-name or judge.model in YAML (or model.model as fallback).")
    if not judge_dict["judge_api_key"] and judge_dict["judge_provider"].lower() != "gemini":
        raise ValueError("Missing judge api key: --judge-api-key or judge.api_key in YAML (or model api key fallback).")

    judge_cfg = ProviderConfig(
        provider=judge_dict["judge_provider"],
        base_url=judge_dict["judge_base_url"],
        api_key=judge_dict["judge_api_key"],
        model=judge_dict["judge_model"],
        timeout_s=float(judge_dict.get("judge_timeout_s", 60.0)),
        temperature=float(judge_dict.get("judge_temperature", 0.0)),
        top_p=float(judge_dict.get("judge_top_p", 0.75)),
        max_tokens=int(judge_dict.get("judge_max_tokens", 10000)),
    )

    # Concurrency: split quotas (pipeline). If not set, fall back to legacy --concurrency / YAML 'concurrency'.
    legacy_conc = (args.concurrency if args.concurrency is not None else cfg.get("concurrency", 8))
    model_conc = args.model_concurrency if args.model_concurrency is not None else int(cfg.get("model_concurrency", legacy_conc))
    judge_conc = args.judge_concurrency if args.judge_concurrency is not None else int(cfg.get("judge_concurrency", legacy_conc))

    run_cfg = RunConfig(
        input_path=input_path,
        sheet_name=sheet_name,
        images_root=images_root,
        out_dir=out_dir,
        concurrency=legacy_conc,
        model_concurrency=int(model_conc),
        judge_concurrency=int(judge_conc),
        max_retries=(args.max_retries if args.max_retries is not None else cfg.get("max_retries", 4)),
        retry_base_delay_s=(args.retry_base_delay_s if args.retry_base_delay_s is not None else cfg.get("retry_base_delay_s", 1.0)),
        retry_max_delay_s=(args.retry_max_delay_s if args.retry_max_delay_s is not None else cfg.get("retry_max_delay_s", 16.0)),
        skip_image_missing=True,
        limit=(args.limit if args.limit is not None else cfg.get("limit", None)),
        cot=(args.cot if args.cot is not None else cfg.get("cot", "off")),
        answer_json_max_attempts=int(
            args.answer_json_max_attempts
            if args.answer_json_max_attempts is not None
            else cfg.get("answer_json_max_attempts", 3)
        ),
        majority_vote=int(
            args.majority_vote
            if args.majority_vote is not None
            else cfg.get("majority_vote", 1)
        ),
        mcq_cardinality_hint=str(
            args.mcq_cardinality_hint
            if args.mcq_cardinality_hint is not None
            else cfg.get("mcq_cardinality_hint", "off")
        ),
        model_reasoning_effort=str(
            args.model_reasoning_effort
            if args.model_reasoning_effort is not None
            else cfg.get("model_reasoning_effort", cfg.get("reasoning_effort", "off"))
        ),
        judge_reasoning_effort=str(
            args.judge_reasoning_effort
            if args.judge_reasoning_effort is not None
            else cfg.get("judge_reasoning_effort", cfg.get("reasoning_effort", "off"))
        ),
        vpn=(args.vpn if args.vpn is not None else cfg.get("vpn", "off")),
        proxy=args.proxy or cfg.get("proxy", ""),
    )
    # Always have an event available so we can stop the run on fatal config/parameter issues.
    run_cfg.cancel_event = cancel_event if cancel_event is not None else threading.Event()

    # Read xlsx
    df = pd.read_excel(run_cfg.input_path, sheet_name=run_cfg.sheet_name, engine="openpyxl")
    if isinstance(df, dict):
        first_key = next(iter(df.keys()))
        df = df[first_key]
    df.columns = [normalize_colname(c) for c in df.columns]

    asyncio.run(run_eval(df, model_cfg, judge_cfg, run_cfg))

def main() -> None:
    cli_main(None, cancel_event=None)

if __name__ == "__main__":
    main()
