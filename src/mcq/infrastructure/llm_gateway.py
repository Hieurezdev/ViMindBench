"""OpenAI-compatible LLM adapter."""

import json
import os
from typing import Any, Dict
from openai import OpenAI


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
    import logging
    logger = logging.getLogger("mcq.llm_gateway")
    
    for attempt in range(3):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
            max_tokens=max_tokens,
            timeout=120,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            if attempt == 2:
                raise ValueError(f"Model did not return JSON: {raw[:240]}")
            logger.warning(f"Model did not return JSON format, retrying... (attempt {attempt + 1})")
            continue
        
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError as e:
            if attempt == 2:
                raise
            logger.warning(f"JSON decode error: {e}, retrying... (attempt {attempt + 1})")
            
    raise ValueError("Failed to get valid JSON after 3 attempts")


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
