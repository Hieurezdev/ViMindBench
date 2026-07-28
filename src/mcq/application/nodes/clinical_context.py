"""Supplementary DSM-5 retrieval used only by A06 safety review."""

import builtins
from typing import Any, Dict
from ...domain import MCQState


def _query_text(value: Any) -> str:
    """Safely flatten imperfect model output for supplementary retrieval only."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(part for item in value if (part := _query_text(item)))
    if isinstance(value, dict):
        return " ".join(part for item in value.values() if (part := _query_text(item)))
    return ""


def dsm5_safety_context_node(state: MCQState) -> Dict[str, Any]:
    """Retrieve up to three DSM-5 chunks without changing A04 answer evidence."""
    if state.get("blueprint", {}).get("level") != "clinical_scenario":
        return {"dsm5_safety_docs": []}
    retriever = getattr(builtins, "RETRIEVER", None)
    if not retriever:
        return {"dsm5_safety_docs": []}
    mcq = state.get("mcq", {})
    query = " ".join(
        part
        for part in (
            state["blueprint"].get("topic", ""),
            _query_text(mcq.get("question", "")),
            _query_text(mcq.get("options", {})),
        )
        if isinstance(part, str) and part
    )
    try:
        return {
            "dsm5_safety_docs": retriever.search_dsm5(
                retriever.generate_embedding(query), k=3
            )
        }
    except Exception as exc:
        # A06 can still judge with the primary evidence; the outage is retained
        # as an audit signal rather than silently affecting answer grounding.
        return {
            "dsm5_safety_docs": [],
            "dsm5_safety_retrieval_error": type(exc).__name__,
        }
