"""A03 MCQ generation node."""
from typing import Any, Dict
from ...domain import MCQState
from ...infrastructure.evidence import evidence_refs
from ...infrastructure.llm_gateway import request_json
from ..prompts import a03_mcq


def mcq_generator_node(state: MCQState) -> Dict[str, Any]:
    refs = evidence_refs(state.get("evidence_docs", []))
    if not refs:
        return {"mcq": {}}
    mcq = request_json(a03_mcq.render(blueprint=state["blueprint"], playbook=state.get("playbook", "")[:7000], evidence=refs))
    mcq["evidence_refs"] = mcq.get("evidence_refs") or [ref["chunk_id"] for ref in refs]
    return {"mcq": mcq}
