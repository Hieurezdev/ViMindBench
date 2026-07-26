"""OpenAI-compatible LLM adapter."""

import json
import os
from typing import Any, Dict
from openai import OpenAI


def request_json(prompt: str, *, max_tokens: int = 1800) -> Dict[str, Any]:
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    )
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
        max_tokens=max_tokens,
        timeout=120,
    )
    raw = (response.choices[0].message.content or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Model did not return JSON: {raw[:240]}")
    return json.loads(raw[start : end + 1])
