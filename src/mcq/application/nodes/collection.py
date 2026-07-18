"""A07 outcome collection and train/quarantine record serialization."""
import os
from datetime import datetime, timezone
from typing import Any, Dict, Set
from ...domain import MCQState
from ...infrastructure.evidence import evidence_refs


def collect_node(state: MCQState) -> Dict[str, Any]:
    blueprint, mcq, reports = state.get("blueprint", {}), state.get("mcq", {}), state.get("judge_reports", {})
    raw_cited = mcq.get("evidence_refs", [])
    cited: Set[str] = set(raw_cited) if isinstance(raw_cited, list) and all(isinstance(value, str) for value in raw_cited) else set()
    references = [{"chunk_id": ref["chunk_id"], "relation": "supports_answer", "support_strength": ref["score"]}
                  for ref in evidence_refs(state.get("evidence_docs", [])) if ref["chunk_id"] in cited]
    record = {"id": f"PSY-{state.get('iteration_count', 0) + 1:06d}", "question": mcq.get("question", ""),
        "options": mcq.get("options", {}), "answer": mcq.get("answer", ""), "evidence_refs": references,
        "distractor_analysis": mcq.get("distractor_analysis", {}),
        "reasoning": {"steps": mcq.get("audit_steps", []), "rationale_short": mcq.get("rationale_short", ""), "visibility": "internal_audit"},
        "metadata": {"topic": blueprint.get("topic", ""), "subtopic": blueprint.get("subtopic", ""), "question_type": blueprint.get("level", ""), "cognitive_skill": blueprint.get("skill", ""), "difficulty": blueprint.get("difficulty", "medium"), "language": "vi", "risk_tier": "B" if blueprint.get("level") == "clinical_scenario" else "A", "generation_type": "synthetic_grounded", "playbook_version": os.getenv("PLAYBOOK_VERSION", "v0.2")},
        "split": os.getenv("DATA_SPLIT", "train"),
        "validation": {"evidence_status": "pass" if reports.get("evidence", {}).get("passed") else "fail", "single_best_answer": "pass" if reports.get("single_answer", {}).get("passed") else "fail", "distractor_quality": "pass" if reports.get("single_answer", {}).get("passed") else "fail", "consistency_checked": bool(reports), "bias_checked": bool(reports.get("ei_safety_bias", {}).get("passed")), "safety_checked": bool(reports.get("ei_safety_bias", {}).get("passed")), "expert_verified": False, "reasoning_verified": "rule_verified" if state.get("verdict") == "verified" else "failed"},
        "_audit": {"created_at": datetime.now(timezone.utc).isoformat(), "anchor_id": state.get("anchor", {}).get("chunk_id"), "judge_reports": reports, "verdict": state.get("verdict"), "quarantine_reason": state.get("quarantine_reason", []), "playbook_delta": state.get("playbook_delta", [])}}
    key = "verified_outputs" if state.get("verdict") == "verified" else "quarantine_outputs"
    if key == "verified_outputs":
        record.pop("_audit", None)
    return {key: [*state.get(key, []), record], "iteration_count": state.get("iteration_count", 0) + 1,
            "anchor": None, "mcq": {}, "judge_reports": {}, "judge_feedback": [],
            "generation_attempt": 0, "dsm5_safety_docs": [], "quarantine_reason": []}
