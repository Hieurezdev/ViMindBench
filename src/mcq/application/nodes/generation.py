"""A03 MCQ generation node."""
from typing import Any, Dict, List
from ...domain import MCQState
from ...infrastructure.evidence import evidence_refs
from ...infrastructure.llm_gateway import request_json
from ..prompts import a03_mcq


def _normalize_evidence_ref_ids(value: Any) -> List[str]:
    """Accept common LLM citation shapes, but keep state as chunk_id strings."""
    if not isinstance(value, list):
        return []
    ids: List[str] = []
    for item in value:
        chunk_id = item if isinstance(item, str) else item.get("chunk_id") if isinstance(item, dict) else None
        if isinstance(chunk_id, str) and chunk_id and chunk_id not in ids:
            ids.append(chunk_id)
    return ids


def mcq_generator_node(state: MCQState) -> Dict[str, Any]:
    refs = evidence_refs(state.get("evidence_docs", []))
    if not refs:
        return {"mcq": {}}
    mcq = request_json(a03_mcq.render(blueprint=state["blueprint"], playbook=state.get("playbook", "")[:7000], evidence=refs, judge_feedback=state.get("judge_feedback", [])))
    # A03 requests string IDs, but local models may return
    # [{"chunk_id": "..."}]. Normalize before the judge pipeline.
    mcq["evidence_refs"] = _normalize_evidence_ref_ids(mcq.get("evidence_refs")) or [ref["chunk_id"] for ref in refs]
    return {"mcq": mcq, "generation_attempt": state.get("generation_attempt", 0) + 1, "judge_reports": {}}


def prepare_regeneration_node(state: MCQState) -> Dict[str, Any]:
    """Carry only failed judge findings into the next bounded A03 attempt."""
    feedback = []
    for judge, report in state.get("judge_reports", {}).items():
        if not report.get("passed", False):
            feedback.append({"judge": judge, "issues": report.get("issues", ["failed"]), "feedback": report.get("feedback", "")})
    return {"judge_feedback": feedback, "dsm5_safety_docs": []}
