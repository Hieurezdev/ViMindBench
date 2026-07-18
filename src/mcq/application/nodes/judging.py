"""A04–A07 quality-judging nodes."""
from typing import Any, Dict, Callable
from ...domain import MCQState
from ...infrastructure.evidence import evidence_refs
from ...infrastructure.llm_gateway import request_json
from ..failure_memory import prompt_context, retrieve_similar_failures
from ..prompts import a04_evidence_judge, a05_single_answer_judge, a06_safety_bias_judge


def _run_judge(state: MCQState, judge_name: str, prompt_factory: Callable[..., str], **extra: object) -> Dict[str, Any]:
    mcq, refs = state.get("mcq", {}), evidence_refs(state.get("evidence_docs", []))
    if not mcq or not refs:
        return {"passed": False, "issues": ["missing_mcq_or_evidence"], "severity": "blocking"}
    try:
        similar = retrieve_similar_failures(state.get("judge_failure_memory", []), judge=judge_name, level=state["blueprint"].get("level", ""), question=mcq.get("question", ""))
        return request_json(prompt_factory(blueprint=state["blueprint"], mcq=mcq, evidence=refs, past_failures=prompt_context(similar), **extra), max_tokens=700)
    except Exception as exc:
        return {"passed": False, "issues": [f"judge_error:{type(exc).__name__}"], "severity": "blocking"}


def _report(state: MCQState, name: str, report: Dict[str, Any]) -> Dict[str, Any]:
    return {"judge_reports": {**state.get("judge_reports", {}), name: report}}


def evidence_judge_node(state: MCQState) -> Dict[str, Any]:
    return _report(state, "evidence", _run_judge(state, "evidence", a04_evidence_judge.render))


def single_answer_judge_node(state: MCQState) -> Dict[str, Any]:
    return _report(state, "single_answer", _run_judge(state, "single_answer", a05_single_answer_judge.render))


def safety_bias_judge_node(state: MCQState) -> Dict[str, Any]:
    return _report(state, "ei_safety_bias", _run_judge(
        state, "ei_safety_bias", a06_safety_bias_judge.render,
        level=state["blueprint"].get("level", ""),
        dsm5_safety_context=evidence_refs(state.get("dsm5_safety_docs", [])),
    ))


def quality_gate_node(state: MCQState) -> Dict[str, Any]:
    reports, errors = state.get("judge_reports", {}), []
    options = state.get("mcq", {}).get("options", {})
    answer = state.get("mcq", {}).get("answer")
    if set(options) != {"A", "B", "C", "D"}: errors.append("invalid_options")
    if answer not in options: errors.append("invalid_answer")
    if answer and set(state.get("mcq", {}).get("distractor_analysis", {})) != (set(options) - {answer}): errors.append("incomplete_distractor_analysis")
    if not (1 <= len(state.get("evidence_docs", [])) <= 3): errors.append("evidence_count_not_1_to_3")
    available = {ref["chunk_id"] for ref in evidence_refs(state.get("evidence_docs", []))}
    cited = set(state.get("mcq", {}).get("evidence_refs", []))
    if not (1 <= len(cited) <= 3) or not cited.issubset(available): errors.append("invalid_evidence_refs")
    for judge, report in reports.items():
        if not report.get("passed", False): errors.extend(f"{judge}:{issue}" for issue in report.get("issues", ["failed"]))
    return {"verdict": "verified" if not errors else "quarantine", "quarantine_reason": errors}
