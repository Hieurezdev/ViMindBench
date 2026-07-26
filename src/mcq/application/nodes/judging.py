"""A04–A07 quality-judging nodes."""

import re
from typing import Any, Dict, Callable, Set
from ...domain import MCQState, RETRIEVAL_DEPTH_BY_DIFFICULTY
from ...infrastructure.evidence import evidence_refs
from ...infrastructure.llm_gateway import request_judge_json
from ..emobench import judge_context, validate_judge_report
from ..failure_memory import prompt_context, retrieve_similar_failures
from ..prompts import a04_evidence_judge, a05_single_answer_judge, a06_safety_bias_judge, a07_adversarial_solver

HAN_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _contains_han_script(value: Any) -> bool:
    """Detect Chinese Han characters in text exported by the MCQ generator."""
    if isinstance(value, str):
        return bool(HAN_CHARACTER.search(value))
    if isinstance(value, dict):
        return any(_contains_han_script(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_han_script(item) for item in value)
    return False


def _run_judge(
    state: MCQState,
    judge_name: str,
    prompt_factory: Callable[..., str],
    **extra: object,
) -> Dict[str, Any]:
    mcq, refs = state.get("mcq", {}), evidence_refs(state.get("evidence_docs", []))
    if not mcq or not refs:
        return {
            "passed": False,
            "issues": ["missing_mcq_or_evidence"],
            "severity": "blocking",
        }
    try:
        similar = retrieve_similar_failures(
            state.get("judge_failure_memory", []),
            judge=judge_name,
            level=state["blueprint"].get("level", ""),
            question=mcq.get("question", ""),
        )
        return request_judge_json(
            prompt_factory(
                blueprint=state["blueprint"],
                mcq=mcq,
                evidence=refs,
                past_failures=prompt_context(similar),
                **extra,
            ),
            max_tokens=700,
        )
    except Exception as exc:
        return {
            "passed": False,
            "issues": [f"judge_error:{type(exc).__name__}"],
            "severity": "blocking",
        }


def _report(state: MCQState, name: str, report: Dict[str, Any]) -> Dict[str, Any]:
    return {"judge_reports": {**state.get("judge_reports", {}), name: report}}


def evidence_judge_node(state: MCQState) -> Dict[str, Any]:
    return _report(
        state, "evidence", _run_judge(state, "evidence", a04_evidence_judge.render)
    )


def single_answer_judge_node(state: MCQState) -> Dict[str, Any]:
    return _report(
        state,
        "single_answer",
        _run_judge(state, "single_answer", a05_single_answer_judge.render),
    )


def safety_bias_judge_node(state: MCQState) -> Dict[str, Any]:
    report = _run_judge(
        state,
        "ei_safety_bias",
        a06_safety_bias_judge.render,
        level=state["blueprint"].get("level", ""),
        dsm5_safety_context=evidence_refs(state.get("dsm5_safety_docs", [])),
        emobench_context=judge_context(state["blueprint"]),
    )
    return _report(state, "ei_safety_bias", validate_judge_report(report, state["blueprint"]))


def adversarial_solver_node(state: MCQState) -> Dict[str, Any]:
    mcq = state.get("mcq", {})
    if not mcq:
        return _report(state, "adversarial_solver", {"passed": False, "issues": ["missing_mcq"]})

    try:
        solver_response = request_judge_json(
            a07_adversarial_solver.render(
                question=mcq.get("question", ""),
                options=mcq.get("options", {}),
                difficulty=state.get("blueprint", {}).get("difficulty", "medium")
            ),
            max_tokens=300,
        )

        selected_option = solver_response.get("selected_option")
        confidence = solver_response.get("confidence")
        actual_answer = mcq.get("answer")
        is_hard = state.get("blueprint", {}).get("difficulty") == "hard"

        passed = True
        issues = []
        feedback = ""

        if selected_option == actual_answer and confidence == "high" and is_hard:
            passed = False
            issues.append("adversarial:spurious_cues_found")
            feedback = "The solver identified the key with high confidence without evidence; rebalance option length, grammar, specificity, certainty, and qualification."

        return _report(state, "adversarial_solver", {
            "passed": passed,
            "issues": issues,
            "feedback": feedback
        })
    except Exception as exc:
        return _report(state, "adversarial_solver", {
            "passed": False,
            "issues": [f"solver_error:{type(exc).__name__}"],
        })


def quality_gate_node(state: MCQState) -> Dict[str, Any]:
    reports, errors = state.get("judge_reports", {}), []
    evidence_limit = RETRIEVAL_DEPTH_BY_DIFFICULTY.get(
        state.get("blueprint", {}).get("difficulty", "medium"),
        RETRIEVAL_DEPTH_BY_DIFFICULTY["medium"],
    )["evidence_limit"]
    options = state.get("mcq", {}).get("options", {})
    answer = state.get("mcq", {}).get("answer")
    if set(options) != {"A", "B", "C", "D"}:
        errors.append("invalid_options")
    if answer not in options:
        errors.append("invalid_answer")
    if answer and set(state.get("mcq", {}).get("distractor_analysis", {})) != (
        set(options) - {answer}
    ):
        errors.append("incomplete_distractor_analysis")
    language_fields = {
        key: state.get("mcq", {}).get(key)
        for key in ("question", "options", "rationale_short", "distractor_analysis", "audit_steps")
    }
    if _contains_han_script(language_fields):
        errors.append("contains_han_script")
    if not (1 <= len(state.get("evidence_docs", [])) <= evidence_limit):
        errors.append(f"evidence_count_not_1_to_{evidence_limit}")
    available = {
        ref["chunk_id"] for ref in evidence_refs(state.get("evidence_docs", []))
    }
    raw_cited = state.get("mcq", {}).get("evidence_refs", [])
    valid_cited_shape = isinstance(raw_cited, list) and all(
        isinstance(value, str) for value in raw_cited
    )
    cited: Set[str] = set(raw_cited) if valid_cited_shape else set()
    if (
        not valid_cited_shape
        or not (1 <= len(cited) <= evidence_limit)
        or not cited.issubset(available)
    ):
        errors.append("invalid_evidence_refs")
    for judge, report in reports.items():
        if not report.get("passed", False):
            errors.extend(
                f"{judge}:{issue}" for issue in report.get("issues", ["failed"])
            )
    return {
        "verdict": "verified" if not errors else "quarantine",
        "quarantine_reason": errors,
    }
