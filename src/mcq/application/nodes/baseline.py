"""Nodes used by the controlled RQ2 baseline conditions."""

from typing import Any, Dict
from langchain_core.documents import Document

from ...domain import MCQState, RETRIEVAL_DEPTH_BY_DIFFICULTY


def direct_blueprint_node(state: MCQState) -> Dict[str, Any]:
    """Create a deterministic, no-LLM plan for the direct-generation control."""
    anchor = state["anchor"]
    difficulties = state.get("curriculum_difficulties", ["easy", "medium", "hard"])
    iteration = state.get("iteration_count", 0)
    difficulty = difficulties[iteration % len(difficulties)]
    levels = state.get("curriculum_levels", ["theory"])
    level = levels[iteration % len(levels)]
    title = str(anchor.get("title") or "tâm lý học")
    summary = str(anchor.get("summary") or anchor.get("content") or "")
    policy = RETRIEVAL_DEPTH_BY_DIFFICULTY[difficulty]
    return {
        "blueprint": {
            "topic": title[:180],
            "subtopic": "khái niệm và ứng dụng tâm lý học",
            "skill": "evidence_grounded_reasoning",
            "retrieval_query": f"{title} {summary}".strip(),
            "clinical_guardrail": "Ask for a safe educational or supportive next step; do not diagnose or prescribe.",
            "playbook_bullet_ids": [],
            "level": level,
            "difficulty": difficulty,
            "evidence_limit": policy["evidence_limit"],
            "min_evidence_refs": policy["min_evidence_refs"],
            "num_options": 4,
            "requires_emobench": False,
            "emobench": {"enabled": False},
        }
    }


def direct_context_node(state: MCQState) -> Dict[str, Any]:
    """Pass only the selected source chunk to A03; do not run vector retrieval."""
    anchor = state["anchor"]
    content = str(anchor.get("content") or anchor.get("summary") or anchor.get("title") or "")
    return {
        "evidence_docs": [
            Document(
                page_content=content,
                metadata={
                    "chunk_id": str(anchor.get("chunk_id") or "direct-anchor"),
                    "title": str(anchor.get("title") or ""),
                    "tier": anchor.get("tier"),
                    "score": 1.0,
                },
            )
        ]
    }


def baseline_accept_node(state: MCQState) -> Dict[str, Any]:
    """Publish unfiltered controls so external blind audit can compare methods."""
    return {"verdict": "verified", "quarantine_reason": []}
