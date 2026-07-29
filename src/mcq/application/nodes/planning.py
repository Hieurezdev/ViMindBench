"""A01 and A02 nodes."""

import builtins
import logging
from typing import Any, Dict
from ...domain import LEVELS, MCQState, RETRIEVAL_DEPTH_BY_DIFFICULTY
from ...infrastructure.evidence import select_eligible_documents
from ...infrastructure.llm_gateway import request_json
from ...infrastructure.mongo_anchor_repository import select_unused_anchor
from ..emobench import normalize_blueprint_emobench
from ..prompts import a01_curriculum

logger = logging.getLogger("mcq.planning")


def select_anchor_node(state: MCQState) -> Dict[str, Any]:
    anchor = select_unused_anchor(state.get("used_anchor_ids", []))
    if not anchor:
        return {"anchor": None}
    return {
        "anchor": anchor,
        "used_anchor_ids": [*state.get("used_anchor_ids", []), anchor["chunk_id"]],
    }


def _fallback_blueprint(anchor: Dict[str, Any], *, level: str, difficulty: str) -> Dict[str, Any]:
    """Keep retrieval running when an optional A01 chat response is empty."""
    title = str(anchor.get("title") or "tâm lý học")
    summary = str(anchor.get("summary") or "")
    return {
        "topic": title[:180],
        "subtopic": "khái niệm và ứng dụng tâm lý học",
        "skill": "evidence_grounded_reasoning",
        "retrieval_query": f"{title} {summary}".strip()[:1000],
        "clinical_guardrail": "Ask for a safe educational or supportive next step; do not diagnose or prescribe.",
        "playbook_bullet_ids": [],
    }


def curriculum_planner_node(state: MCQState) -> Dict[str, Any]:
    anchor = state["anchor"]
    levels = state.get("curriculum_levels", list(LEVELS))
    iteration = state.get("iteration_count", 0)
    level = levels[iteration % len(levels)]

    difficulties = state.get("curriculum_difficulties", ["easy", "medium", "hard"])
    difficulty = difficulties[(iteration // len(levels)) % len(difficulties)]

    try:
        blueprint = request_json(
            a01_curriculum.render(
                level=level,
                difficulty=difficulty,
                title=anchor["title"],
                summary=anchor["summary"],
                playbook=state.get("playbook", ""),
            ),
            max_tokens=700,
        )
        if not isinstance(blueprint, dict):
            raise ValueError("planner did not return an object")
    except Exception as exc:
        logger.warning("A01 fallback blueprint after %s", type(exc).__name__)
        blueprint = _fallback_blueprint(anchor, level=level, difficulty=difficulty)
    blueprint["level"] = level
    blueprint["difficulty"] = difficulty
    blueprint["evidence_limit"] = RETRIEVAL_DEPTH_BY_DIFFICULTY[difficulty]["evidence_limit"]
    blueprint["min_evidence_refs"] = RETRIEVAL_DEPTH_BY_DIFFICULTY[difficulty]["min_evidence_refs"]
    blueprint["num_options"] = 4
    blueprint["requires_emobench"] = level == "emotion"
    blueprint["emobench"] = normalize_blueprint_emobench(blueprint.get("emobench"), enabled=level == "emotion")
    blueprint["playbook_bullet_ids"] = blueprint.get("playbook_bullet_ids", [])
    return {"blueprint": blueprint}


def context_retriever_node(state: MCQState) -> Dict[str, Any]:
    retriever = getattr(builtins, "RETRIEVER", None)
    if not retriever:
        return {"evidence_docs": []}
    difficulty = state.get("blueprint", {}).get("difficulty", "medium")
    retrieval_depth = RETRIEVAL_DEPTH_BY_DIFFICULTY.get(
        difficulty, RETRIEVAL_DEPTH_BY_DIFFICULTY["medium"]
    )
    return {
        "evidence_docs": select_eligible_documents(
            retriever.search(
                state["blueprint"]["retrieval_query"],
                k=retrieval_depth["candidate_k"],
            ),
            limit=retrieval_depth["evidence_limit"],
        )
    }
