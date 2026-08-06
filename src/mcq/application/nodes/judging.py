"""A04–A07 quality-judging nodes."""

import os
import re
import logging
from typing import Any, Dict, Callable, Set
from ...domain import MCQState, RETRIEVAL_DEPTH_BY_DIFFICULTY
from ...infrastructure.evidence import evidence_refs
from ...infrastructure.llm_gateway import request_judge_json
from ..emobench import judge_context, validate_judge_report
from ..failure_memory import prompt_context, retrieve_similar_failures
from ..prompts import a04_evidence_judge, a05_single_answer_judge, a06_safety_bias_judge, a07_adversarial_solver

HAN_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
logger = logging.getLogger("mcq.judging")


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


def _evidence_report(state: MCQState) -> Dict[str, Any]:
    return _run_judge(state, "evidence", a04_evidence_judge.render)


def evidence_judge_node(state: MCQState) -> Dict[str, Any]:
    """Compatibility wrapper for callers that execute this judge in isolation."""
    return _report(state, "evidence", _evidence_report(state))


def evidence_judge_parallel_node(state: MCQState) -> Dict[str, Any]:
    """Parallel-safe A04 update; consolidation happens after the fan-in barrier."""
    return {"evidence_report": _evidence_report(state)}


def _validate_single_answer_report(report: Dict[str, Any], mcq: Dict[str, Any]) -> Dict[str, Any]:
    """Require an explicit key-correct / three-distractor-incorrect audit."""
    checked = dict(report)
    answer = mcq.get("answer")
    options = mcq.get("options", {})
    assessments = report.get("option_assessment")
    issues = list(report.get("issues", []))

    expected_keys = set(options) if isinstance(options, dict) else set()
    if not isinstance(assessments, dict) or set(assessments) != expected_keys:
        issues.append("option_assessment_incomplete")
    elif answer not in options or assessments.get(answer) != "correct":
        issues.append("declared_answer_not_judged_correct")
    elif any(
        assessments.get(option) != "incorrect"
        for option in options
        if option != answer
    ):
        issues.append("distractor_not_judged_incorrect")

    if issues:
        checked["passed"] = False
        checked["severity"] = "blocking"
    checked["issues"] = list(dict.fromkeys(issues))
    return checked


def _single_answer_report(state: MCQState) -> Dict[str, Any]:
    report = _run_judge(state, "single_answer", a05_single_answer_judge.render)
    return _validate_single_answer_report(report, state.get("mcq", {}))


def single_answer_judge_node(state: MCQState) -> Dict[str, Any]:
    return _report(state, "single_answer", _single_answer_report(state))


def single_answer_judge_parallel_node(state: MCQState) -> Dict[str, Any]:
    return {"single_answer_report": _single_answer_report(state)}


def _safety_bias_report(state: MCQState) -> Dict[str, Any]:
    report = _run_judge(
        state,
        "ei_safety_bias",
        a06_safety_bias_judge.render,
        level=state["blueprint"].get("level", ""),
        dsm5_safety_context=evidence_refs(state.get("dsm5_safety_docs", [])),
        emobench_context=judge_context(state["blueprint"]),
    )
    return validate_judge_report(report, state["blueprint"])


def safety_bias_judge_node(state: MCQState) -> Dict[str, Any]:
    return _report(state, "ei_safety_bias", _safety_bias_report(state))


def safety_bias_judge_parallel_node(state: MCQState) -> Dict[str, Any]:
    return {"ei_safety_bias_report": _safety_bias_report(state)}


def _adversarial_solver_report(state: MCQState) -> Dict[str, Any]:
    mcq = state.get("mcq", {})
    if not mcq:
        return {"passed": False, "issues": ["missing_mcq"]}

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
        cue_type = solver_response.get("surface_cue_type")
        cue_evidence = solver_response.get("surface_cue_evidence")
        actual_answer = mcq.get("answer")
        is_hard = state.get("blueprint", {}).get("difficulty") == "hard"

        passed = True
        issues = []
        feedback = ""

        concrete_cue = (
            isinstance(cue_type, str)
            and cue_type in {"length", "absolute_wording", "unique_qualification", "grammar", "detail_imbalance", "other"}
            and isinstance(cue_evidence, str)
            and bool(cue_evidence.strip())
        )
        if selected_option == actual_answer and confidence == "high" and is_hard and concrete_cue:
            passed = False
            issues.append("adversarial:spurious_cues_found")
            feedback = f"The solver identified a concrete {cue_type} cue: {cue_evidence.strip()}. Rebalance option length, grammar, specificity, certainty, and qualification."

        return {
            "passed": passed,
            "issues": issues,
            "feedback": feedback
        }
    except Exception as exc:
        return {
            "passed": False,
            "issues": [f"solver_error:{type(exc).__name__}"],
        }


def adversarial_solver_node(state: MCQState) -> Dict[str, Any]:
    return _report(state, "adversarial_solver", _adversarial_solver_report(state))


def adversarial_solver_parallel_node(state: MCQState) -> Dict[str, Any]:
    return {"adversarial_solver_report": _adversarial_solver_report(state)}


def consolidate_judge_reports_node(state: MCQState) -> Dict[str, Any]:
    """Fan-in reports written by independent parallel judge nodes."""
    reports = dict(state.get("judge_reports", {}))
    reports.update(
        {
            "evidence": state.get("evidence_report", {}),
            "single_answer": state.get("single_answer_report", {}),
            "ei_safety_bias": state.get("ei_safety_bias_report", {}),
            "adversarial_solver": state.get("adversarial_solver_report", {}),
        }
    )
    return {"judge_reports": reports}


def quality_gate_node(state: MCQState) -> Dict[str, Any]:
    reports, errors = state.get("judge_reports", {}), []
    evidence_policy = RETRIEVAL_DEPTH_BY_DIFFICULTY.get(
        state.get("blueprint", {}).get("difficulty", "medium"),
        RETRIEVAL_DEPTH_BY_DIFFICULTY["medium"],
    )
    evidence_limit = evidence_policy["evidence_limit"]
    min_evidence_refs = evidence_policy["min_evidence_refs"]
    options = state.get("mcq", {}).get("options", {})
    answer = state.get("mcq", {}).get("answer")
    valid_options_mapping = isinstance(options, dict) and set(options) == {
        "A",
        "B",
        "C",
        "D",
    }
    if not valid_options_mapping:
        errors.append("invalid_options")
    elif not all(isinstance(option, str) and option.strip() for option in options.values()):
        errors.append("invalid_option_text")
    if valid_options_mapping:
        if answer not in options:
            errors.append("invalid_answer")
        distractor_analysis = state.get("mcq", {}).get("distractor_analysis", {})
        if (
            answer
            and (
                not isinstance(distractor_analysis, dict)
                or set(distractor_analysis) != (set(options) - {answer})
            )
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
        or not (min_evidence_refs <= len(cited) <= evidence_limit)
        or not cited.issubset(available)
    ):
        errors.append("invalid_evidence_refs")
    adversarial_is_blocking = os.getenv(
        "A07_ADVERSARIAL_BLOCKING", "false"
    ).lower() in {"1", "true", "yes"}
    for judge, report in reports.items():
        if not report.get("passed", False):
            # A07 is a red-team signal by default. It often flags the very
            # evidence-based distinction that A05 requires for a valid MCQ.
            # Keep its full report in the audit trail, but require an explicit
            # opt-in before it can quarantine an otherwise verified item.
            if judge == "adversarial_solver" and not adversarial_is_blocking:
                logger.info(
                    "A07 adversarial finding retained as warning, not blocking: %s",
                    report.get("issues", ["failed"]),
                )
                continue
            errors.extend(
                f"{judge}:{issue}" for issue in report.get("issues", ["failed"])
            )
    return {
        "verdict": "verified" if not errors else "quarantine",
        "quarantine_reason": errors,
    }
