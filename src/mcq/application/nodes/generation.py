"""A03 MCQ generation node."""

import os
import re
from typing import Any, Dict, List
from ...domain import MCQState
from ...infrastructure.evidence import evidence_refs
from ...infrastructure.llm_gateway import request_json, request_judge_json
from ..prompts import a03_evidence_plan, a03_mcq, a03_preflight

HARD_EMPHATIC_WORDING = re.compile(
    r"\b(hoàn\s+toàn|tuyệt\s+đối|luôn\s+luôn|không\s+bao\s+giờ|duy\s+nhất|chắc\s+chắn|triệt\s+để|ngay\s+lập\s+tức|tự\s+ý|tất\s+cả)\b",
    re.IGNORECASE,
)
ANSWER_LABELS = ("A", "B", "C", "D")


def _normalize_evidence_ref_ids(value: Any) -> List[str]:
    """Accept common LLM citation shapes, but keep state as chunk_id strings."""
    if not isinstance(value, list):
        return []
    ids: List[str] = []
    for item in value:
        chunk_id = (
            item
            if isinstance(item, str)
            else item.get("chunk_id")
            if isinstance(item, dict)
            else None
        )
        if isinstance(chunk_id, str) and chunk_id and chunk_id not in ids:
            ids.append(chunk_id)
    return ids


def _required_answer_position(state: MCQState) -> str:
    """Cycle answer positions by public record ID to prevent positional bias."""
    record_number = state.get("next_record_id", state.get("iteration_count", 0) + 1)
    try:
        index = max(1, int(record_number)) - 1
    except (TypeError, ValueError):
        index = 0
    return ANSWER_LABELS[index % len(ANSWER_LABELS)]


def _normalize_and_validate_mcq(
    mcq: Any, *, refs: List[Dict[str, Any]], required_answer: str
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Normalize tolerated citation shapes and enforce A03's public contract.

    The full judges should assess psychology quality, not compensate for an
    incomplete JSON object from the generator.  This check is intentionally
    deterministic so malformed output is repaired before A04--A07 run.
    """
    if not isinstance(mcq, dict):
        return {}, {
            "passed": False,
            "issues": ["schema:not_json_object"],
            "feedback": "Return one JSON object, not a list, string, or wrapper.",
        }

    normalized = dict(mcq)
    normalized["evidence_refs"] = _normalize_evidence_ref_ids(
        normalized.get("evidence_refs")
    ) or [ref["chunk_id"] for ref in refs]
    options = normalized.get("options")
    answer = normalized.get("answer")
    analyses = normalized.get("distractor_analysis")
    issues: List[str] = []

    if not isinstance(normalized.get("question"), str) or not normalized["question"].strip():
        issues.append("schema:missing_question")
    if not isinstance(options, dict) or set(options) != {"A", "B", "C", "D"}:
        issues.append("schema:options_must_be_mapping_A_to_D")
    elif not all(isinstance(text, str) and text.strip() for text in options.values()):
        issues.append("schema:options_must_be_nonempty_strings")
    if not isinstance(answer, str) or answer not in {"A", "B", "C", "D"}:
        issues.append("schema:answer_must_be_A_to_D")
    elif answer != required_answer:
        issues.append(f"schema:answer_position_must_be_{required_answer}")
    elif isinstance(options, dict):
        expected_distractors = {"A", "B", "C", "D"} - {answer}
        if not isinstance(analyses, dict) or set(analyses) != expected_distractors:
            issues.append("schema:distractor_analysis_must_cover_exactly_three_wrong_options")
        elif not all(isinstance(text, str) and text.strip() for text in analyses.values()):
            issues.append("schema:distractor_analysis_must_be_nonempty_strings")
    if not isinstance(normalized.get("rationale_short"), str) or not normalized["rationale_short"].strip():
        issues.append("schema:missing_rationale_short")
    if not isinstance(normalized.get("audit_steps"), list) or not normalized["audit_steps"]:
        issues.append("schema:missing_audit_steps")

    if issues:
        return normalized, {
            "passed": False,
            "issues": issues,
            "feedback": (
                "Repair the JSON contract exactly. Return a non-empty Vietnamese question; "
                f"options as {{A,B,C,D}} strings; answer exactly {required_answer}; and "
                "distractor_analysis for exactly the other three letters."
            ),
        }
    return normalized, {"passed": True, "issues": [], "feedback": ""}


def _build_evidence_plan(blueprint: Dict[str, Any], refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep the key within explicitly extracted evidence claims when possible."""
    try:
        plan = _request_generation_json(
            blueprint,
            a03_evidence_plan.render(blueprint=blueprint, evidence=refs),
            max_tokens=700,
        )
        if isinstance(plan, dict):
            return plan
    except Exception:
        pass
    return {"supported_claims": [], "prohibited_inferences": []}


def _request_generation_json(
    blueprint: Dict[str, Any], prompt: str, *, max_tokens: int
) -> Dict[str, Any]:
    """Route hard-item generation to the stronger judge endpoint when enabled."""
    use_judge_for_hard = os.getenv("HARD_GENERATION_USE_JUDGE", "false").lower() in {"1", "true", "yes"}
    if blueprint.get("difficulty") == "hard" and use_judge_for_hard:
        # request_judge_json falls back to the primary generator endpoint if
        # the separate judge endpoint is unavailable.
        return request_judge_json(prompt, max_tokens=max_tokens)
    return request_json(prompt, max_tokens=max_tokens)


def _generate_mcq(
    *,
    blueprint: Dict[str, Any],
    playbook: str,
    refs: List[Dict[str, Any]],
    evidence_plan: Dict[str, Any],
    judge_feedback: List[Dict[str, Any]],
    required_answer: str,
) -> Dict[str, Any]:
    return _request_generation_json(
        blueprint,
        a03_mcq.render(
            blueprint=blueprint,
            playbook=playbook,
            evidence=refs,
            evidence_plan=evidence_plan,
            judge_feedback=judge_feedback,
            required_answer=required_answer,
        ),
        max_tokens=1800,
    )


def _hard_guard_report(blueprint: Dict[str, Any], mcq: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically reject common hard-item cues before LLM judging."""
    if blueprint.get("difficulty") != "hard" or os.getenv("A03_HARD_GUARD_ENABLED", "true").lower() not in {"1", "true", "yes"}:
        return {"passed": True, "issues": [], "feedback": ""}

    options = mcq.get("options")
    if not isinstance(options, dict) or set(options) != {"A", "B", "C", "D"} or not all(isinstance(text, str) and text.strip() for text in options.values()):
        return {
            "passed": False,
            "issues": ["hard_guard:invalid_option_shape"],
            "feedback": "Return four non-empty string options A–D in the same grammatical frame.",
        }

    issues: List[str] = []
    details: List[str] = []
    counts = {key: len(re.findall(r"\w+", text, re.UNICODE)) for key, text in options.items()}
    max_gap = int(os.getenv("A03_HARD_MAX_OPTION_WORD_GAP", "5"))
    if max(counts.values()) - min(counts.values()) > max_gap:
        issues.append("hard_guard:option_length_imbalance")
        details.append(f"word counts are {counts}; keep the gap at most {max_gap}")

    emphatic = {
        key: sorted({match.group(0).lower() for match in HARD_EMPHATIC_WORDING.finditer(text)})
        for key, text in options.items()
    }
    emphatic = {key: words for key, words in emphatic.items() if words}
    if emphatic:
        issues.append("hard_guard:emphatic_wording")
        details.append(f"remove emphatic wording from options: {emphatic}")

    if not issues:
        return {"passed": True, "issues": [], "feedback": ""}
    return {
        "passed": False,
        "issues": issues,
        "feedback": "Hard rewrite required: " + "; ".join(details) + ". Keep all options plausible near-misses of the same mechanism.",
    }


def _preflight_report(
    *, blueprint: Dict[str, Any], mcq: Dict[str, Any], refs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Use the judge model for one cheap repair opportunity before A04–A07."""
    hard_guard = _hard_guard_report(blueprint, mcq)
    if os.getenv("A03_PREFLIGHT_ENABLED", "true").lower() not in {"1", "true", "yes"}:
        return hard_guard
    try:
        report = request_judge_json(
            a03_preflight.render(blueprint=blueprint, mcq=mcq, evidence=refs),
            max_tokens=450,
        )
        if not isinstance(report, dict):
            report = {"passed": False, "issues": ["preflight_invalid_report"], "feedback": "Return a valid preflight JSON report."}
        if hard_guard.get("passed", False):
            return report
        return {
            "passed": False,
            "issues": [*hard_guard["issues"], *report.get("issues", [])],
            "feedback": " ".join(part for part in (hard_guard.get("feedback", ""), report.get("feedback", "")) if part),
        }
    except Exception:
        # The full A04–A07 chain remains authoritative if this optional early
        # review endpoint is unavailable.
        return hard_guard


def _generation_failure(state: MCQState, exc: Exception) -> Dict[str, Any]:
    """Turn malformed/refused generator output into a bounded workflow retry."""
    detail = str(exc).lower()
    issue = "generator_refusal" if "cannot fulfill" in detail else f"generator_error:{type(exc).__name__}"
    feedback = {
        "judge": "generator",
        "issues": [issue],
        "feedback": "Return only the requested Vietnamese MCQ JSON. Use the retrieved evidence and educational framing; do not add a refusal or prose.",
    }
    return {
        "mcq": {},
        "generation_attempt": state.get("generation_attempt", 0) + 1,
        "judge_reports": {
            "generation": {"passed": False, "issues": [issue], "severity": "blocking", "feedback": feedback["feedback"]}
        },
        "judge_feedback": [*state.get("judge_feedback", []), feedback],
    }


def mcq_generator_node(state: MCQState) -> Dict[str, Any]:
    refs = evidence_refs(state.get("evidence_docs", []))
    if not refs:
        return {"mcq": {}}

    full_playbook = state.get("playbook", "")
    blueprint = state["blueprint"]
    required_answer = _required_answer_position(state)
    method = state.get("experiment_method", "full")
    # Direct is the source-only control. It must not receive an additional
    # evidence-planning call or any LLM-as-judge preflight signal.
    evidence_plan = (
        {"supported_claims": [], "prohibited_inferences": []}
        if method == "direct"
        else _build_evidence_plan(blueprint, refs)
    )

    feedback = list(state.get("judge_feedback", []))
    try:
        mcq = _generate_mcq(
            blueprint=blueprint,
            playbook=full_playbook,
            refs=refs,
            evidence_plan=evidence_plan,
            judge_feedback=feedback,
            required_answer=required_answer,
        )
        mcq, schema_report = _normalize_and_validate_mcq(
            mcq, refs=refs, required_answer=required_answer
        )
        if not schema_report["passed"]:
            feedback.append(
                {
                    "judge": "a03_schema",
                    "issues": schema_report["issues"],
                    "feedback": schema_report["feedback"],
                }
            )
            mcq = _generate_mcq(
                blueprint=blueprint,
                playbook=full_playbook,
                refs=refs,
                evidence_plan=evidence_plan,
                judge_feedback=feedback,
                required_answer=required_answer,
            )
            mcq, schema_report = _normalize_and_validate_mcq(
                mcq, refs=refs, required_answer=required_answer
            )
            if not schema_report["passed"]:
                raise ValueError("A03 schema repair failed: " + ", ".join(schema_report["issues"]))
        preflight = (
            {"passed": True, "issues": [], "feedback": ""}
            if method in {"direct", "rag_only"}
            else _preflight_report(blueprint=blueprint, mcq=mcq, refs=refs)
        )
        if not preflight.get("passed", False):
            feedback.append(
                {
                    "judge": "a03_preflight",
                    "issues": preflight.get("issues", ["preflight_failed"]),
                    "feedback": preflight.get("feedback", "Repair the cited evidence and option balance."),
                }
            )
            mcq = _generate_mcq(
                blueprint=blueprint,
                playbook=full_playbook,
                refs=refs,
                evidence_plan=evidence_plan,
                judge_feedback=feedback,
                required_answer=required_answer,
            )
            mcq, schema_report = _normalize_and_validate_mcq(
                mcq, refs=refs, required_answer=required_answer
            )
            if not schema_report["passed"]:
                raise ValueError("A03 preflight repair broke schema: " + ", ".join(schema_report["issues"]))
    except Exception as exc:
        return _generation_failure(state, exc)
    return {
        "mcq": mcq,
        "generation_attempt": state.get("generation_attempt", 0) + 1,
        "judge_reports": {},
        "evidence_report": {},
        "single_answer_report": {},
        "ei_safety_bias_report": {},
        "adversarial_solver_report": {},
        "judge_feedback": feedback,
    }


def prepare_regeneration_node(state: MCQState) -> Dict[str, Any]:
    """Route item-local fixes to A03 and alignment failures to A01/A02."""
    feedback = []
    for judge, report in state.get("judge_reports", {}).items():
        if not report.get("passed", False):
            feedback.append(
                {
                    "judge": judge,
                    "issues": report.get("issues", ["failed"]),
                    "feedback": report.get("feedback", ""),
                }
            )
    issues = [
        str(issue)
        for item in feedback
        for issue in item.get("issues", [])
    ]
    replan_markers = (
        "blueprint_mismatch",
        "blueprint_topic_mismatch",
        "subtopic_mismatch",
        "subtopic_unsupported",
        "retrieval_query_mismatch",
        "evidence_ref_mismatch",
        "evidence_missing_primary_chunk",
        "keyed_option_unsupported",
        "unsupported_option_claims",
        "unsupported_key",
    )
    needs_replan = any(marker in issue for issue in issues for marker in replan_markers)
    return {
        "judge_feedback": feedback,
        "planning_feedback": feedback if needs_replan else [],
        "regeneration_route": "replan" if needs_replan else "generate",
        "dsm5_safety_docs": [],
    }
