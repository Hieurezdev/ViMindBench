"""OpenAI-compatible LLM adapter."""

import json
import logging
import os
from typing import Any, Dict
from openai import OpenAI

logger = logging.getLogger("mcq.llm_gateway")


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
    system_instruction: str | None = None,
    response_format: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.15,
        "max_tokens": max_tokens,
        "timeout": 120,
    }
    if response_format is not None:
        request_kwargs["response_format"] = response_format
    response = client.chat.completions.create(**request_kwargs)
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
                f"endpoint={base_url!r}; model={model!r}; initial={initial_error}; "
                f"repair={repair_error}; raw={raw[:240]!r}"
            ) from repair_error


def request_json(
    prompt: str,
    *,
    max_tokens: int = 1800,
    system_instruction: str | None = None,
    response_format: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Call the primary LLM used by planning and MCQ generation."""
    return _request_json(
        prompt,
        max_tokens=max_tokens,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        model=os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
        system_instruction=system_instruction,
        response_format=response_format,
    )


def request_judge_json(prompt: str, *, max_tokens: int = 1800) -> Dict[str, Any]:
    """Use the judge endpoint, then the primary endpoint if the judge is down."""
    primary_base_url = os.getenv("OPENAI_BASE_URL")
    primary_api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    primary_model = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    judge_base_url = os.getenv("JUDGE_OPENAI_BASE_URL") or primary_base_url
    judge_api_key = os.getenv("JUDGE_OPENAI_API_KEY") or primary_api_key
    judge_model = os.getenv("JUDGE_MODEL_NAME") or primary_model

    try:
        return _request_json(
            prompt,
            max_tokens=max_tokens,
            base_url=judge_base_url,
            api_key=judge_api_key,
            model=judge_model,
        )
    except Exception as judge_error:
        configured_separately = (
            judge_base_url != primary_base_url or judge_model != primary_model
        )
        if not configured_separately:
            raise
        logger.warning(
            "Judge endpoint failed (%s); retrying with the primary model endpoint.",
            type(judge_error).__name__,
        )
        return _request_json(
            prompt,
            max_tokens=max_tokens,
            base_url=primary_base_url,
            api_key=primary_api_key,
            model=primary_model,
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
