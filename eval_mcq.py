import argparse
import asyncio
import base64
import json
import os
import random
import re
import time
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

def now_ms() -> int:
    return int(time.time() * 1000)

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
        if p == "openai":
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
    return out


# ----------------------------
# Config
# ----------------------------

@dataclass
class ProviderConfig:
    provider: str  # openai | gemini | claude
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 512

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

    # vpn/proxy switch
    vpn: str = "off"       # on/off
    proxy: str = ""        # proxy url, e.g. http://127.0.0.1:7897


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

    async def call(self, prompt: str, image_data_url: Optional[str] = None) -> Dict[str, Any]:
        p = self.cfg.provider.lower()
        if p == "openai":
            return await self._call_openai_chat_completions(prompt, image_data_url)
        elif p == "gemini":
            return await self._call_gemini(prompt, image_data_url)
        elif p == "claude":
            return await self._call_claude(prompt, image_data_url)
        else:
            raise ValueError(f"Unknown provider: {self.cfg.provider}")

    async def _call_openai_chat_completions(self, prompt: str, image_data_url: Optional[str]) -> Dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}

        if image_data_url:
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        else:
            user_content = prompt

        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": "You are a careful assistant. Follow instructions exactly."},
                {"role": "user", "content": user_content},
            ],
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


def extract_text_from_provider_response(provider: str, resp_json: Any) -> str:
    if not resp_json:
        return ""
    p = provider.lower()
    try:
        if p == "openai":
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

def build_mcq_prompt(question: str, options: List[str], cot_on: bool) -> str:
    opts = "\n".join(options)

    if not cot_on:
        return (
            "You will answer a multiple-choice question. Some questions may have multiple correct options.\n"
            "Output format rules (STRICT):\n"
            "1) Output ONLY option letters among A,B,C,D,E.\n"
            "2) If multiple, separate by comma ',' with NO spaces. Example: A,B,C\n"
            "3) Output EXACTLY ONE LINE and NOTHING ELSE (no reasoning, no punctuation, no spaces).\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{opts}\n"
        )

    return (
        "You will answer a multiple-choice question. Some questions may have multiple correct options.\n"
        "You may think step-by-step internally, but you MUST NOT output chain-of-thought.\n"
        "Output format rules (COT mode, STRICT):\n"
        "- Return ONLY a JSON object. No markdown. No extra text.\n"
        "- JSON schema:\n"
        '{ "answer": "A" | "A,B,C", "brief_reason": string }\n'
        "- 'answer' must contain ONLY letters among A,B,C,D,E.\n"
        "- If multiple, separate by comma ',' with NO spaces (example: \"A,B,C\").\n"
        "- 'brief_reason' should be short (1-3 sentences).\n"
        "Example:\n"
        '{ "answer": "B", "brief_reason": "A 4g-gon is the standard fundamental polygon for genus g." }\n\n'
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n"
    )


def build_freeform_answer_prompt(question: str, cot_on: bool) -> str:
    if not cot_on:
        return f"Answer the following question:\n\n{question}\n\nProvide a complete and accurate answer:"
    return (
        "Answer the following question.\n"
        "You may think step-by-step internally, but you MUST NOT output chain-of-thought.\n"
        "Output format rules (COT mode, STRICT):\n"
        "- Return ONLY a JSON object. No markdown. No extra text.\n"
        "- JSON schema:\n"
        '{ "answer": string, "brief_reason": string }\n'
        "- 'answer' must be the final result only (concise, no derivations).\n"
        "- 'brief_reason' should be short (1-3 sentences).\n\n"
        f"Question:\n{question}\n"
    )

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


def build_mcq_judge_prompt(
    question: str,
    options: List[str],
    model_answer: str,
    gold_letters_csv: str,
) -> str:
    """
    Judge prompt for MCQ. The judge should decide correctness based on whether the set
    of chosen option letters exactly matches the gold (order-insensitive, duplicates ignored).
    """
    opts = "\n".join(options or [])
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
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n\n"
        f"Gold Answer (letters CSV):\n{gold_letters_csv}\n\n"
        f"Model Answer:\n{model_answer}\n"
    )








def parse_judge_json(text: str) -> Dict[str, Any]:
    """
    Parse the judge's output and return a structured dictionary.
    """
    text = (text or "").strip()
    try:
        # Best-effort: some models may wrap JSON with extra text/codefences.
        candidate = text
        if "```" in candidate:
            candidate = re.sub(r"(?s)^```(?:json)?\s*|\s*```$", "", candidate.strip()).strip()
        if not candidate.startswith("{"):
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = candidate[start : end + 1]

        judge_json = json.loads(candidate)

        # 解析 "verdict" 字段
        verdict = judge_json.get("verdict", "").lower()

        # 根据 "verdict" 判断结果
        if verdict == "correct":
            judge_json["judge_correct"] = True
        elif verdict == "incorrect":
            judge_json["judge_correct"] = False
        else:
            judge_json["judge_correct"] = None

        return judge_json
    except Exception as e:
        # 如果解析失败，返回一个默认的结果
        return {"verdict": "unjudgeable", "extracted_answer": None, "reason": f"Failed to parse judge JSON: {str(e)}"}


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
    qid = str(row.get("id", "")).strip()
    question = str(row.get("Question", "")).strip()
    options = load_options(row.get("Options", ""))  # 获取选项
    question_type_raw = str(row.get("Question_Type", "")).strip()  # 获取题目类型（原始）
    mcq = is_multiple_choice(question_type_raw, options)

    gold_raw = str(row.get("Answer", "")).strip()
    gold_norm = normalize_csv_letters(gold_raw) if mcq else gold_raw

    img_rel = str(row.get("Image", "") or "").strip()
    img_dep = int(row.get("Image_Dependency", 0) or 0)

    cot_on = (run_cfg.cot == "on")

    # Resolve image
    image_path = None
    image_data_url = None
    if img_rel:
        image_path = img_rel
        if run_cfg.images_root:
            image_path = os.path.join(run_cfg.images_root, img_rel) if not os.path.isabs(img_rel) else img_rel
        if os.path.exists(image_path):
            image_data_url = image_file_to_data_url(image_path)

    # Skip if image required but missing
    if img_dep == 1 and not image_data_url and run_cfg.skip_image_missing:
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
        print(f"[{idx}/{total}] id={qid}  SKIP(image missing)  gold={gold_norm}  image={img_rel}", flush=True)
        return out

    if mcq:
        prompt = build_mcq_prompt(question, options, cot_on=cot_on)
    else:
        prompt = build_freeform_answer_prompt(question, cot_on=cot_on)

    async with model_sem:
        t0 = now_ms()
        model_call = await model_provider.call(prompt=prompt, image_data_url=image_data_url)
        t1 = now_ms()

    model_text = extract_text_from_provider_response(model_provider.cfg.provider, model_call.get("response"))
    model_call_log = _sanitize_call_for_logging(model_provider.cfg.provider, model_call, image_path)

    # We do not rely on local extraction for scoring anymore.
    # Keep a best-effort extracted value for debugging only (may be None).
    pred_extracted_debug = extract_answer_from_text(model_text, cot_on=cot_on) if mcq else None
    rule_correct = None
    judge_call = None
    judge_block = None
    judge_correct = None
    judge_call_log = None

    # ALWAYS use judge model for scoring (all question types).
    if judge_provider is not None:
        judge_prompt = (
            build_mcq_judge_prompt(question, options, model_text, gold_norm)
            if mcq
            else build_freeform_judge_prompt(question, model_text, gold_raw)
        )
        async with judge_sem:
            jt0 = now_ms()
            judge_call = await judge_provider.call(prompt=judge_prompt, image_data_url=image_data_url)
            jt1 = now_ms()

        judge_text = extract_text_from_provider_response(judge_provider.cfg.provider, judge_call.get("response"))
        judge_call_log = _sanitize_call_for_logging(judge_provider.cfg.provider, judge_call, image_path)
        judge_json = parse_judge_json(judge_text)
        verdict = str(judge_json.get("verdict", "unjudgeable")).lower()

        if verdict == "correct":
            judge_correct = True
        elif verdict == "incorrect":
            judge_correct = False
        else:
            judge_correct = None

        extracted = judge_json.get("extracted_answer", None)
        extracted_norm = normalize_csv_letters(extracted) if isinstance(extracted, str) else ""
        judge_json["extracted_answer_normalized"] = extracted_norm

        judge_block = {
            "judge_text": judge_text,
            "judge_json": judge_json,
            "judge_latency_ms": jt1 - jt0,
        }

    # Prefer judge-extracted answer as the prediction shown in logs/Excel.
    judge_extracted_for_pred: Optional[str] = None
    if isinstance(judge_block, dict):
        jj = judge_block.get("judge_json") or {}
        if isinstance(jj, dict):
            if mcq:
                v = jj.get("extracted_answer_normalized")
                judge_extracted_for_pred = v if isinstance(v, str) and v else None
            else:
                v = jj.get("extracted_answer")
                judge_extracted_for_pred = v if isinstance(v, str) and v.strip() else None

    out = {
        "id": qid,
        "idx": idx,
        "total": total,
        "skipped": False,
        "skip_reason": None,
        "gold_raw": gold_raw,
        "gold": gold_norm,
        # Use judge-extracted answer as the canonical prediction.
        "pred_raw": judge_extracted_for_pred,
        "pred": judge_extracted_for_pred,
        # Keep debug extraction from model output for troubleshooting (not used for scoring).
        "pred_extracted_debug": pred_extracted_debug,
        "rule_correct": rule_correct,
        "judge_correct": judge_correct,
        "latency_ms": t1 - t0,
        "judge_latency_ms": (judge_block.get("judge_latency_ms") if isinstance(judge_block, dict) else None),
        "total_latency_ms": (t1 - t0) + (judge_block.get("judge_latency_ms") if isinstance(judge_block, dict) and isinstance(judge_block.get("judge_latency_ms"), int) else 0),
        "question": question,
        "options": options,
        "question_type_raw": question_type_raw,
        "question_type_inferred": "Multiple Choice" if mcq else "Freeform",
        "image_path": image_path,
        # Keep a compact image reference; do NOT log base64.
        "image_data_url": image_path if image_path else None,
        "cot": run_cfg.cot,
        # FULL ARCHIVE:
        "model_call": model_call_log,
        "model_text": model_text,
        "judge": judge_block,
        "judge_call": judge_call_log,
    }

    # IMPORTANT: Always append to the single shared results_path computed in run_eval().
    async with write_lock:
        safe_jsonl_append(results_path, out)

    # Print ONLY after both model + judge have completed for this question.
    # (Per user request: do not print immediately after model finishes.)
    model_preview = (model_text or "").replace("\n", " ").strip()
    if len(model_preview) > 200:
        model_preview = model_preview[:200] + "..."
    jtxt = (judge_block.get("judge_text") if isinstance(judge_block, dict) else "") or ""
    judge_preview = jtxt.replace("\n", " ").strip()
    if len(judge_preview) > 200:
        judge_preview = judge_preview[:200] + "..."

    print(
        f"[{idx}/{total}] id={qid}  DONE  mcq={mcq}  gold={gold_norm}  pred={judge_extracted_for_pred}  judge_correct={judge_correct}  model_ms={t1 - t0}  judge_ms={(out.get('judge_latency_ms') or 0)}  total_ms={out.get('total_latency_ms')}",
        flush=True
    )
    print(f"   model_text: {model_preview}", flush=True)
    print(f"   judge_text: {judge_preview}", flush=True)
    return out





async def run_eval(df: pd.DataFrame, model_cfg: ProviderConfig, judge_cfg: Optional[ProviderConfig],
             run_cfg: RunConfig) -> None:
    ensure_dir(run_cfg.out_dir)

    # 确保文件名中的非法字符被替换
    model_name_safe = re.sub(r'[<>:"/\\|?*]', '_', model_cfg.model)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    results_path = os.path.join(run_cfg.out_dir, f"results_{timestamp}_{model_name_safe}.jsonl")  # Same file for all logs
    summary_path = os.path.join(run_cfg.out_dir, f"summary_{timestamp}.json")

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
        f"Total questions: {total_questions} | model_concurrency={run_cfg.model_concurrency} | judge_concurrency={run_cfg.judge_concurrency} | vpn={run_cfg.vpn} | cot={run_cfg.cot}",
        flush=True
    )

    model_sem = asyncio.Semaphore(run_cfg.model_concurrency)
    judge_sem = asyncio.Semaphore(run_cfg.judge_concurrency)

    tasks = [
        eval_one(
            r, i + 1, total_questions,
            model_provider, judge_provider, run_cfg,
            model_sem, judge_sem,
            write_lock, results_path
        )
        for i, r in enumerate(rows)
    ]
    results = await asyncio.gather(*tasks)

    # Final correctness is ALWAYS determined by judge_correct for ALL question types.
    # ============= METRICS (judge-only) =============
    overall_total = 0
    overall_correct = 0
    judged_total = 0
    judged_correct = 0

    for result, row in zip(results, rows):
        if result.get("skipped"):
            row["model_correct"] = None
            continue

        final_correct = (result.get("judge_correct") is True)
        judged_total += 1
        if final_correct:
            judged_correct += 1
        # 写回表格：统一用 judge_correct（True/False/None）
        row["model_correct"] = result.get("judge_correct")

        overall_total += 1
        if final_correct:
            overall_correct += 1

    def _fmt_score(correct: int, total: int) -> str:
        pct = (correct / total * 100.0) if total else 0.0
        return f"{correct}/{total} ({pct:.2f}%)"

    overall_score_str = _fmt_score(overall_correct, overall_total)
    judged_score_str = _fmt_score(judged_correct, judged_total)
    # =====================================================================

    # Save results to a new Excel file
    output_df = pd.DataFrame(rows)
    output_df.to_excel(os.path.join(run_cfg.out_dir, f"evaluated_{timestamp}_{model_name_safe}.xlsx"), index=False)

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

        # NEW required metrics:
        "overall": {
            "correct": overall_correct,
            "total": overall_total,
            "score": overall_score_str,
        },
        "judged": {
            "correct": judged_correct,
            "total": judged_total,
            "score": judged_score_str,
        },

        "timestamp_ms": now_ms(),
        "vpn": run_cfg.vpn,
        "proxy": run_cfg.proxy if run_cfg.vpn == "on" else "",
        "cot": run_cfg.cot,
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
    print(f"Overall: {overall_score_str}", flush=True)
    print(f"Judged (all questions): {judged_score_str}", flush=True)

    print(f"\nSaved full archives to: {results_path}", flush=True)
    print(f"Saved summary to: {summary_path}", flush=True)


# ----------------------------
# CLI / Main
# ----------------------------

def build_provider_cfg(d: Dict[str, Any], prefix: str) -> ProviderConfig:
    provider = d[f"{prefix}_provider"]
    base_url = d[f"{prefix}_base_url"]
    api_key = d[f"{prefix}_api_key"]
    model = d[f"{prefix}__model"]
    timeout_s = float(d.get(f"{prefix}_timeout_s", 60.0))
    temperature = float(d.get(f"{prefix}_temperature", 0.0))
    max_tokens = int(d.get(f"{prefix}_max_tokens", 512))
    return ProviderConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_s=timeout_s,
        temperature=temperature,
        max_tokens=max_tokens,
    )

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, default="", help="Optional YAML config.")
    ap.add_argument("--input", type=str, required=False, default="", help="Path to .xlsx dataset.")
    ap.add_argument("--sheet", type=str, default="", help="Sheet name (default: first sheet).")
    ap.add_argument("--images-root", type=str, default="", help="Root directory where 'images/' folder lives.")
    ap.add_argument("--out-dir", type=str, default="out_eval", help="Output directory.")
    ap.add_argument("--concurrency", type=int, default=8, help="(legacy) used as default for both model/judge concurrency")
    ap.add_argument("--model-concurrency", type=int, default=None, help="Max in-flight requests to the answering model")
    ap.add_argument("--judge-concurrency", type=int, default=None, help="Max in-flight requests to the judge model")
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--retry-base-delay-s", type=float, default=1.0)
    ap.add_argument("--retry-max-delay-s", type=float, default=16.0)
    ap.add_argument("--limit", type=int, default=None)

    # NEW: COT switch
    ap.add_argument("--cot", type=str, choices=["on", "off"], default="off",
                    help="COT mode: on=allow reasoning but require last line Answer:A,B,C ; off=answer only")

    # VPN/proxy switch
    ap.add_argument("--vpn", type=str, choices=["on", "off"], default="off",
                    help="VPN mode switch: on=use proxy, off=direct")
    ap.add_argument("--proxy", type=str, default="",
                    help="Proxy URL, e.g. http://127.0.0.1:7897 or socks5://127.0.0.1:7897")

    # model
    ap.add_argument("--model-provider", type=str, default="")
    ap.add_argument("--model-base-url", type=str, default="")
    ap.add_argument("--model-api-key", type=str, default="")
    ap.add_argument("--model-name", type=str, default="")
    ap.add_argument("--model-timeout-s", type=float, default=60.0)
    ap.add_argument("--model-temperature", type=float, default=0.0)
    ap.add_argument("--model-max-tokens", type=int, default=256)

    # judge
    ap.add_argument("--judge-enable", action="store_true",
                    help="(legacy) Kept for backward compatibility; scoring always uses judge now.")
    ap.add_argument("--judge-provider", type=str, default="")
    ap.add_argument("--judge-base-url", type=str, default="")
    ap.add_argument("--judge-api-key", type=str, default="")
    ap.add_argument("--judge-name", type=str, default="")
    ap.add_argument("--judge-timeout-s", type=float, default=60.0)
    ap.add_argument("--judge-temperature", type=float, default=0.0)
    ap.add_argument("--judge-max-tokens", type=int, default=256)

    args = ap.parse_args()

    cfg = load_yaml_config(args.config) if args.config else {}

    input_path = args.input or cfg.get("input_path", "")
    if not input_path:
        raise ValueError("Missing --input (xlsx path) or input_path in YAML.")

    sheet_name = args.sheet or cfg.get("sheet_name", "")
    sheet_name = sheet_name if sheet_name else None

    images_root = args.images_root or cfg.get("images_root", "")
    out_dir = args.out_dir or cfg.get("out_dir", "out_eval")

    # model cfg
    model_dict = {
        "model_provider": args.model_provider or cfg.get("model", {}).get("provider", "openai"),
        "model_base_url": args.model_base_url or cfg.get("model", {}).get("base_url", "https://api.openai.com"),
        "model_api_key": args.model_api_key or cfg.get("model", {}).get("api_key", os.getenv("OPENAI_API_KEY", "")),
        "model_model": args.model_name or cfg.get("model", {}).get("model", ""),
        "model_timeout_s": args.model_timeout_s or cfg.get("model", {}).get("timeout_s", 60.0),
        "model_temperature": args.model_temperature if args.model_temperature is not None else cfg.get("model", {}).get("temperature", 0.0),
        "model_max_tokens": args.model_max_tokens or cfg.get("model", {}).get("max_tokens", 256),
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
        max_tokens=int(model_dict_for_builder.get("model_max_tokens", 256)),
    )

    # judge cfg (ALWAYS ON): scoring always uses judge model now.
    judge_dict = {
        "judge_provider": args.judge_provider or cfg.get("judge", {}).get("provider", model_cfg.provider),
        "judge_base_url": args.judge_base_url or cfg.get("judge", {}).get("base_url", model_cfg.base_url),
        "judge_api_key": (
            args.judge_api_key
            or cfg.get("judge", {}).get("api_key", "")
            or model_dict.get("model_api_key", "")
        ),
        "judge_model": args.judge_name or cfg.get("judge", {}).get("model", model_cfg.model),
        "judge_timeout_s": args.judge_timeout_s or cfg.get("judge", {}).get("timeout_s", 60.0),
        "judge_temperature": (
            args.judge_temperature if args.judge_temperature is not None else cfg.get("judge", {}).get("temperature", 0.0)
        ),
        "judge_max_tokens": args.judge_max_tokens or cfg.get("judge", {}).get("max_tokens", 256),
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
        max_tokens=int(judge_dict.get("judge_max_tokens", 256)),
    )

    # Concurrency: split quotas (pipeline). If not set, fall back to legacy --concurrency / YAML 'concurrency'.
    legacy_conc = args.concurrency or cfg.get("concurrency", 8)
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
        max_retries=args.max_retries or cfg.get("max_retries", 4),
        retry_base_delay_s=args.retry_base_delay_s or cfg.get("retry_base_delay_s", 1.0),
        retry_max_delay_s=args.retry_max_delay_s or cfg.get("retry_max_delay_s", 16.0),
        skip_image_missing=True,
        limit=args.limit or cfg.get("limit", None),
        cot=args.cot or cfg.get("cot", "off"),
        vpn=args.vpn or cfg.get("vpn", "off"),
        proxy=args.proxy or cfg.get("proxy", ""),
    )

    # Read xlsx
    df = pd.read_excel(run_cfg.input_path, sheet_name=run_cfg.sheet_name, engine="openpyxl")
    if isinstance(df, dict):
        first_key = next(iter(df.keys()))
        df = df[first_key]
    df.columns = [normalize_colname(c) for c in df.columns]

    asyncio.run(run_eval(df, model_cfg, judge_cfg, run_cfg))


if __name__ == "__main__":
    main()
