"""Evidence normalization and tier policy."""

import os
from typing import Any, Dict, List


def document_tier(doc: Any) -> str:
    tier = str(
        (doc.metadata or {}).get("tier")
        or (doc.metadata or {}).get("source_tier")
        or ""
    ).lower()
    if tier in {"1", "tier 1", "tier_1", "tier1"}:
        return "Tier 1"
    if tier in {"2", "tier 2", "tier_2", "tier2"}:
        return "Tier 2"
    if tier in {"3", "tier 3", "tier_3", "tier3"}:
        return "Tier 3"
    if tier:
        return f"Unapproved tier ({tier})"
    return "Tier 2 (legacy-unclassified)"


def select_eligible_documents(candidates: List[Any], *, limit: int = 3) -> List[Any]:
    """Keep approved evidence in retrieval rank order, up to the requested limit."""
    allowed = {"Tier 1", "Tier 2"}
    if os.getenv("ALLOW_UNTIERED_EVIDENCE", "true").lower() in {"1", "true", "yes"}:
        allowed.add("Tier 2 (legacy-unclassified)")
    return [doc for doc in candidates if document_tier(doc) in allowed][:limit]


def evidence_refs(docs: List[Any]) -> List[Dict[str, Any]]:
    refs = []
    for doc in docs:
        meta = doc.metadata or {}
        refs.append(
            {
                "chunk_id": str(
                    meta.get("chunk_id") or meta.get("uuid") or meta.get("_id", "")
                ),
                "tier": document_tier(doc),
                "title": meta.get("title", ""),
                "score": meta.get("score", 0.0),
                "excerpt": doc.page_content[:1200],
            }
        )
    return refs
