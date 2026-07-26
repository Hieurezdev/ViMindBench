"""OpenAI-compatible LLM adapter."""

import json
import os
from typing import Any, Dict
from openai import OpenAI


def _parse_json_object(raw: str) -> Dict[str, Any]:
    """Extract one JSON object while tolerating Markdown fences and prose."""
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError(f"Model did not return a JSON object: {cleaned[:240]}")
    value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("Model returned JSON but the top-level value is not an object")
    return value


def _repair_json(client: OpenAI, *, raw: str, model: str, max_tokens: int) -> str:
    """Ask once for a syntax-only repair when a local model emits invalid JSON."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Convert the following malformed response into one valid JSON object. "
                    "Preserve its intended fields and values where possible. Return JSON only; "
                    "do not explain or add Markdown.\n\n"
                    f"MALFORMED RESPONSE:\n{raw[:12000]}"
                ),
            }
        ],
        temperature=0,
        max_tokens=max_tokens,
        timeout=120,
    )
    return (response.choices[0].message.content or "").strip()


def _request_json(
    prompt: str,
    *,
    max_tokens: int,
    base_url: str | None,
    api_key: str,
    model: str,
) -> Dict[str, Any]:
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
        max_tokens=max_tokens,
        timeout=120,
    )
    raw = (response.choices[0].message.content or "").strip()
    try:
        return _parse_json_object(raw)
    except (json.JSONDecodeError, ValueError) as initial_error:
        repaired = _repair_json(client, raw=raw, model=model, max_tokens=max_tokens)
        try:
            return _parse_json_object(repaired)
        except (json.JSONDecodeError, ValueError) as repair_error:
            raise ValueError(
                "Model returned invalid JSON after one repair attempt; "
                f"initial={initial_error}; repair={repair_error}; raw={raw[:240]!r}"
            ) from repair_error


def request_json(prompt: str, *, max_tokens: int = 1800) -> Dict[str, Any]:
    """Call the primary LLM used by planning and MCQ generation."""
    return _request_json(
        prompt,
        max_tokens=max_tokens,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        model=os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
    )


def request_judge_json(prompt: str, *, max_tokens: int = 1800) -> Dict[str, Any]:
    """Call the optional judge-only endpoint, falling back to the primary LLM."""
    return _request_json(
        prompt,
        max_tokens=max_tokens,
        base_url=os.getenv("JUDGE_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("JUDGE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "EMPTY"),
        model=os.getenv("JUDGE_MODEL_NAME") or os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
    )


def request_insight_json(prompt: str, *, max_tokens: int = 900) -> Dict[str, Any]:
    """Call the optional A09 Notebook insight endpoint.

    It deliberately falls back to the primary model so the playbook-learning
    loop remains usable when an insight-only endpoint is not configured.
    """
    return _request_json(
        prompt,
        max_tokens=max_tokens,
        base_url=os.getenv("INSIGHT_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("INSIGHT_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "EMPTY"),
        model=os.getenv("INSIGHT_MODEL_NAME") or os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
    )
