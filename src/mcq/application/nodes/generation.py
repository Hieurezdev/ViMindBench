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
    mcq = request_json(a03_mcq.render(blueprint=state["blueprint"], playbook=state.get("playbook", "")[:7000], evidence=refs, judge_feedback=state.get("judge_feedback", [])))
    mcq["evidence_refs"] = mcq.get("evidence_refs") or [ref["chunk_id"] for ref in refs]
    return {"mcq": mcq, "generation_attempt": state.get("generation_attempt", 0) + 1, "judge_reports": {}}


def prepare_regeneration_node(state: MCQState) -> Dict[str, Any]:
    """Carry only failed judge findings into the next bounded A03 attempt."""
    feedback = []
    for judge, report in state.get("judge_reports", {}).items():
        if not report.get("passed", False):
            feedback.append({"judge": judge, "issues": report.get("issues", ["failed"]), "feedback": report.get("feedback", "")})
    return {"judge_feedback": feedback, "dsm5_safety_docs": []}
