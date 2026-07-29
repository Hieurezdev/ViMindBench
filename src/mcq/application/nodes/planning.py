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

_REQUIRED_BLUEPRINT_FIELDS = (
    "topic",
    "subtopic",
    "skill",
    "retrieval_query",
    "clinical_guardrail",
    "playbook_bullet_ids",
)
_BLUEPRINT_WRAPPER_KEYS = (
    "curriculum_planning_blueprint",
    "blueprint",
    "planning_blueprint",
)


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


def _missing_blueprint_fields(blueprint: Dict[str, Any]) -> list[str]:
    """Return required A01 fields absent from a usable planner response."""
    missing = [
        field
        for field in _REQUIRED_BLUEPRINT_FIELDS[:-1]
        if not isinstance(blueprint.get(field), str) or not blueprint[field].strip()
    ]
    if not isinstance(blueprint.get("playbook_bullet_ids"), list):
        missing.append("playbook_bullet_ids")
    return missing


def _unwrap_blueprint(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    """Accept known model wrappers while preserving a strict inner schema."""
    for key in _BLUEPRINT_WRAPPER_KEYS:
        nested = blueprint.get(key)
        if isinstance(nested, dict):
            logger.info("A01 unwrapped planner response from key=%s", key)
            return nested
    return blueprint


def curriculum_planner_node(state: MCQState) -> Dict[str, Any]:
    anchor = state["anchor"]
    levels = state.get("curriculum_levels", list(LEVELS))
    iteration = state.get("iteration_count", 0)
    level = levels[iteration % len(levels)]

    difficulties = state.get("curriculum_difficulties", ["easy", "medium", "hard"])
    difficulty = difficulties[(iteration // len(levels)) % len(difficulties)]

    prompt = a01_curriculum.render(
        level=level,
        difficulty=difficulty,
        title=anchor["title"],
        summary=anchor["summary"],
        playbook=state.get("playbook", ""),
    )
    try:
        blueprint = request_json(
            prompt,
            max_tokens=1_000,
            system_instruction=a01_curriculum.SYSTEM_INSTRUCTION,
            response_format=a01_curriculum.response_schema(level=level, difficulty=difficulty),
        )
        if not isinstance(blueprint, dict):
            raise ValueError("planner did not return an object")
        blueprint = _unwrap_blueprint(blueprint)
    except Exception as exc:
        logger.warning("A01 fallback blueprint after %s", type(exc).__name__)
        blueprint = _fallback_blueprint(anchor, level=level, difficulty=difficulty)

    missing_fields = _missing_blueprint_fields(blueprint)
    if missing_fields:
        # A syntactically valid but incomplete object (often `{}` from an
        # OpenAI-compatible bridge) is not a usable plan. Retry once before
        # degrading to deterministic fields derived from the anchor.
        logger.warning(
            "A01 returned incomplete blueprint; keys=%s missing=%s; retrying once",
            sorted(str(key) for key in blueprint),
            missing_fields,
        )
        try:
            retried = request_json(
                f"{prompt}\n\nIMPORTANT: Your previous response was incomplete. "
                f"Return one JSON object only and include every required field: "
                f"{', '.join(_REQUIRED_BLUEPRINT_FIELDS)}. "
                "Use [] when no playbook bullet applies.",
                max_tokens=1_000,
                system_instruction=a01_curriculum.SYSTEM_INSTRUCTION,
                response_format=a01_curriculum.response_schema(level=level, difficulty=difficulty),
            )
            if not isinstance(retried, dict):
                raise ValueError("planner retry did not return an object")
            retried = _unwrap_blueprint(retried)
            retry_missing = _missing_blueprint_fields(retried)
            if retry_missing:
                logger.warning(
                    "A01 retry still incomplete; keys=%s missing=%s",
                    sorted(str(key) for key in retried),
                    retry_missing,
                )
            else:
                blueprint = retried
        except Exception as exc:
            logger.warning("A01 retry failed after %s", type(exc).__name__)

    fallback = _fallback_blueprint(anchor, level=level, difficulty=difficulty)
    for field in ("topic", "subtopic", "skill", "retrieval_query", "clinical_guardrail"):
        fallback_value = fallback[field]
        if not isinstance(blueprint.get(field), str) or not blueprint[field].strip():
            logger.warning("A01 missing/invalid %s; using fallback value", field)
            blueprint[field] = fallback_value
    if not isinstance(blueprint.get("playbook_bullet_ids"), list):
        logger.warning("A01 missing/invalid playbook_bullet_ids; using fallback value")
        blueprint["playbook_bullet_ids"] = fallback["playbook_bullet_ids"]
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
    query = state.get("blueprint", {}).get("retrieval_query")
    if not isinstance(query, str) or not query.strip():
        anchor = state.get("anchor", {}) or {}
        query = f"{anchor.get('title', '')} {anchor.get('summary', '')}".strip()
        logger.warning("A02 received no retrieval_query; using anchor fallback query")
    if not query:
        return {"evidence_docs": []}
    return {
        "evidence_docs": select_eligible_documents(
            retriever.search(
                query,
                k=retrieval_depth["candidate_k"],
            ),
            limit=retrieval_depth["evidence_limit"],
        )
    }
