"""Evidence-grounded LLM explanations for anomalous log blocks."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _load_dotenv():
    """Load simple KEY=VALUE entries without adding a dependency."""
    env_path = __import__("pathlib").Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()
API_URL = os.getenv("LOGINSIGHT_LLM_BASE_URL", "https://api.openai.com/v1/responses")
MODEL = os.getenv("LOGINSIGHT_LLM_MODEL", "gpt-5-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


SYSTEM_PROMPT = (
    "You are a log incident analyst. Explain the detected anomaly using only "
    "the supplied evidence. Do not invent events, causes, services, or facts. "
    "If the evidence is insufficient, say so. Return concise JSON."
)


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "possible_cause": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["explanation", "possible_cause", "evidence"],
}


def _extract_output_text(payload):
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"]
    raise ValueError("The LLM response did not contain output text.")


def _compact_logs(raw_logs, limit=40):
    """Keep actual evidence bounded for free-tier token limits."""
    if len(raw_logs) <= limit:
        return raw_logs
    head = limit // 2
    tail = limit - head
    return raw_logs[:head] + [{"evidence_note": f"{len(raw_logs) - limit} middle log records omitted"}] + raw_logs[-tail:]


def _evidence_payload(block_id, anomaly_score, event_ids, raw_logs):
    return json.dumps({
        "block_id": block_id,
        "anomaly_score": anomaly_score,
        "event_ids": event_ids,
        "raw_logs": _compact_logs(raw_logs),
    }, ensure_ascii=False)


def _parse_gemini_response(response_payload):
    candidates = response_payload.get("candidates", [])
    if not candidates:
        raise ValueError("The Gemini response did not contain a candidate.")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if part.get("text"))
    if not text:
        raise ValueError("The Gemini response did not contain output text.")
    return json.loads(text)


def _explain_with_gemini(api_key, block_id, anomaly_score, event_ids, raw_logs):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{
            "text": _evidence_payload(block_id, anomaly_score, event_ids, raw_logs)
        }]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
            "temperature": 0.1,
        },
    }
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "LogInsight/0.1",
    }, method="POST")
    try:
        with urlopen(request, timeout=90) as response:
            return _parse_gemini_response(json.loads(response.read().decode("utf-8")))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed ({error.code}): {detail[:500]}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach the Gemini API: {error.reason}") from error


def explain_anomaly(block_id, anomaly_score, event_ids, raw_logs):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        explanation = _explain_with_gemini(
            gemini_api_key, block_id, anomaly_score, event_ids, raw_logs
        )
        if not isinstance(explanation.get("evidence"), list):
            raise ValueError("Gemini returned an invalid evidence list.")
        return explanation

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured on the backend.")

    payload = {
        "model": MODEL,
        "store": False,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": (
                SYSTEM_PROMPT
            )}]},
            {"role": "user", "content": [{"type": "input_text", "text": json.dumps({
                "block_id": block_id,
                "anomaly_score": anomaly_score,
                "event_ids": event_ids,
                "raw_logs": _compact_logs(raw_logs),
            }, ensure_ascii=False)}]},
        ],
        "text": {"format": {
            "type": "json_schema",
            "name": "log_anomaly_explanation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "possible_cause": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["explanation", "possible_cause", "evidence"],
                "additionalProperties": False,
            },
        }},
    }
    request = Request(API_URL, data=json.dumps(payload).encode("utf-8"), headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "LogInsight/0.1",
    }, method="POST")
    try:
        with urlopen(request, timeout=90) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API request failed ({error.code}): {detail[:500]}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach the LLM API: {error.reason}") from error

    explanation = json.loads(_extract_output_text(response_payload))
    if not isinstance(explanation.get("evidence"), list):
        raise ValueError("The LLM returned an invalid evidence list.")
    return explanation
