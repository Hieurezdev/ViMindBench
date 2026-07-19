"""A01 and A02 nodes."""

import builtins
from typing import Any, Dict
from ...domain import LEVELS, MCQState
from ...infrastructure.evidence import select_eligible_documents
from ...infrastructure.llm_gateway import request_json
from ...infrastructure.mongo_anchor_repository import select_unused_anchor
from ..prompts import a01_curriculum


def select_anchor_node(state: MCQState) -> Dict[str, Any]:
    anchor = select_unused_anchor(state.get("used_anchor_ids", []))
    if not anchor:
        return {"anchor": None}
    return {
        "anchor": anchor,
        "used_anchor_ids": [*state.get("used_anchor_ids", []), anchor["chunk_id"]],
    }


def curriculum_planner_node(state: MCQState) -> Dict[str, Any]:
    anchor = state["anchor"]
    levels = state.get("curriculum_levels", list(LEVELS))
    iteration = state.get("iteration_count", 0)
    level = levels[iteration % len(levels)]
    
    difficulties = state.get("curriculum_difficulties", ["easy", "medium", "hard"])
    difficulty = difficulties[(iteration // len(levels)) % len(difficulties)]
    
    blueprint = request_json(
        a01_curriculum.render(
            level=level,
            difficulty=difficulty,
            title=anchor["title"],
            summary=anchor["summary"],
            playbook=state.get("playbook", "")[:7000],
        ),
        max_tokens=700,
    )
    blueprint["level"] = level
    blueprint["difficulty"] = difficulty
    blueprint["num_options"] = 4
    blueprint["requires_emobench"] = level == "emotion"
    blueprint["playbook_bullet_ids"] = blueprint.get("playbook_bullet_ids", [])
    return {"blueprint": blueprint}


def context_retriever_node(state: MCQState) -> Dict[str, Any]:
    retriever = getattr(builtins, "RETRIEVER", None)
    if not retriever:
        return {"evidence_docs": []}
    return {
        "evidence_docs": select_eligible_documents(
            retriever.search(state["blueprint"]["retrieval_query"], k=8)
        )
    }
