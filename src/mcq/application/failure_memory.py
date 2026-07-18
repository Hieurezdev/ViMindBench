"""Judge failure memory with deterministic, offline lexical retrieval.

Only structured judge feedback is stored. Private model reasoning is never
persisted or injected into later prompts.
"""
import re
from typing import Any, Dict, List


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[\wÀ-ỹ]{3,}", text.lower()) if token}


def retrieve_similar_failures(memory: List[Dict[str, Any]], *, judge: str, level: str, question: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Return most relevant earlier failures, restricted to the same judge."""
    query_tokens = _tokens(question)
    scored = []
    for item in memory:
        if item.get("judge") != judge:
            continue
        candidate_tokens = _tokens(item.get("question", ""))
        overlap = len(query_tokens & candidate_tokens) / max(1, len(query_tokens | candidate_tokens))
        score = overlap + (0.15 if item.get("level") == level else 0.0)
        if score > 0:
            scored.append((score, item))
    return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:limit]]


def record_judge_failures(memory: List[Dict[str, Any]], *, reports: Dict[str, Dict[str, Any]], blueprint: Dict[str, Any], mcq: Dict[str, Any], iteration: int, max_entries: int = 500) -> List[Dict[str, Any]]:
    """Append failed judge reports in a compact, auditable representation."""
    updated = list(memory)
    for judge, report in reports.items():
        if report.get("passed", False):
            continue
        updated.append({
            "judge": judge,
            "level": blueprint.get("level", ""),
            "topic": blueprint.get("topic", ""),
            "question": mcq.get("question", ""),
            "issues": report.get("issues", ["failed"]),
            "feedback": report.get("feedback", ""),
            "iteration": iteration,
        })
    return updated[-max_entries:]


def prompt_context(failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expose only issue/feedback summaries suitable for a future judge prompt."""
    return [{"level": item.get("level"), "topic": item.get("topic"), "issues": item.get("issues", []), "feedback": item.get("feedback", "")} for item in failures]
