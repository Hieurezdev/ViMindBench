"""A03 MCQ generation node."""

import os
import re
from typing import Any, Dict, List
from ...domain import MCQState
from ...infrastructure.evidence import evidence_refs
from ...infrastructure.llm_gateway import LLMOutputError, request_json, request_judge_json
from ..prompts import a03_evidence_plan, a03_mcq, a03_preflight
from ..option_quality import option_surface_report

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
    mcq: Any,
    *,
    refs: List[Dict[str, Any]],
    required_answer: str,
    blueprint: Dict[str, Any],
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
    if blueprint.get("level") == "clinical_scenario":
        for field in ("clinical_case", "case_summary"):
            if not isinstance(normalized.get(field), str) or not normalized[field].strip():
                issues.append(f"schema:missing_{field}")
        dsm5_ids = {ref["chunk_id"] for ref in refs if ref.get("source_kind") == "dsm5"}
        if dsm5_ids and not dsm5_ids.intersection(normalized["evidence_refs"]):
            issues.append("schema:missing_dsm5_citation")
        textbook_ids = {ref["chunk_id"] for ref in refs if ref.get("source_kind") == "textbook"}
        if textbook_ids and not textbook_ids.intersection(normalized["evidence_refs"]):
            issues.append("schema:missing_textbook_citation")

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
                + (
                    " Include non-empty clinical_case and case_summary, "
                    "and cite at least one supplied DSM-5 chunk and one textbook chunk when available."
                    if blueprint.get("level") == "clinical_scenario" else ""
                )
            ),
        }
    return normalized, {"passed": True, "issues": [], "feedback": ""}


def _build_evidence_plan(blueprint: Dict[str, Any], refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep the key within explicitly extracted evidence claims when possible."""
    try:
        plan = _request_generation_json(
            blueprint,
            a03_evidence_plan.render(blueprint=blueprint, evidence=refs),
            max_tokens=1800 if blueprint.get("level") == "clinical_scenario" else 700,
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
    previous_mcq: Dict[str, Any],
) -> Dict[str, Any]:
    max_tokens = 2600 if blueprint.get("level") == "clinical_scenario" else 1800
    if any("llm_output_truncated" in item.get("issues", []) for item in judge_feedback):
        max_tokens = max_tokens * 3 // 2
    return _request_generation_json(
        blueprint,
        a03_mcq.render(
            blueprint=blueprint,
            playbook=playbook,
            evidence=refs,
            evidence_plan=evidence_plan,
            judge_feedback=judge_feedback,
            required_answer=required_answer,
            previous_mcq=previous_mcq,
        ),
        max_tokens=max_tokens,
    )


def _hard_guard_report(blueprint: Dict[str, Any], mcq: Dict[str, Any]) -> Dict[str, Any]:
    """Catch gate-blocking cues at every difficulty, plus optional hard-item rules."""
    options = mcq.get("options")
    if not isinstance(options, dict) or set(options) != {"A", "B", "C", "D"} or not all(isinstance(text, str) and text.strip() for text in options.values()):
        return {
            "passed": False,
            "issues": ["hard_guard:invalid_option_shape"],
            "feedback": "Return four non-empty string options A–D in the same grammatical frame.",
        }

    is_hard = blueprint.get("difficulty") == "hard"
    max_gap = int(os.getenv("A07_HARD_MAX_OPTION_WORD_GAP" if is_hard else "A07_MAX_OPTION_WORD_GAP", "5" if is_hard else "8"))
    if is_hard and os.getenv("A03_HARD_GUARD_ENABLED", "true").lower() in {"1", "true", "yes"}:
        max_gap = min(max_gap, int(os.getenv("A03_HARD_MAX_OPTION_WORD_GAP", "5")))
    surface = option_surface_report(options, max_word_gap=max_gap)
    if not is_hard or os.getenv("A03_HARD_GUARD_ENABLED", "true").lower() not in {"1", "true", "yes"}:
        return surface
    issues = [issue.replace("surface_cue:", "hard_guard:") for issue in surface["issues"]]
    details = [surface["feedback"]] if surface["feedback"] else []
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
            a03_preflight.render(
                blueprint=blueprint, mcq=mcq,
                evidence=[ref for ref in refs if ref["chunk_id"] in mcq.get("evidence_refs", [])],
            ),
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
    issue = (
        exc.issue if isinstance(exc, LLMOutputError)
        else "generator_schema_repair_failed" if detail.startswith("a03 schema repair failed")
        else "generator_preflight_repair_broke_schema" if detail.startswith("a03 preflight repair broke schema")
        else "dsm5_evidence_missing" if "dsm5_evidence_missing" in detail
        else "textbook_evidence_missing" if "textbook_evidence_missing" in detail
        else "missing_retrieved_evidence" if "missing_retrieved_evidence" in detail
        else "generator_refusal" if "cannot fulfill" in detail
        else f"generator_error:{type(exc).__name__}"
    )
    feedback = {
        "judge": "generator",
        "issues": [issue],
        "feedback": (
            "Retrieve both eligible DSM-5 and Tier 1 textbook evidence before generating a clinical case."
            if issue in {"dsm5_evidence_missing", "textbook_evidence_missing"}
            else "Retrieve eligible evidence excerpts before generating the item."
            if issue == "missing_retrieved_evidence"
            else "The response reached the output token limit. Shorten the case, options and explanations; return one complete MCQ JSON. The next generation has a larger output budget."
            if issue == "llm_output_truncated"
            else "Repair the specific schema findings above without changing the valid clinical case. Return the complete MCQ JSON."
            if issue in {"generator_schema_repair_failed", "generator_preflight_repair_broke_schema"}
            else "Return only the requested Vietnamese MCQ JSON. Use the retrieved evidence and educational framing; do not add a refusal or prose."
        ),
    }
    return {
        "mcq": {},
        # Keep an invalid draft available for repair without sending it to judges.
        "generation_draft": state.get("generation_draft") or state.get("mcq", {}),
        "generation_attempt": state.get("generation_attempt", 0) + 1,
        "judge_reports": {
            "generation": {
                "passed": False, "issues": [issue], "severity": "blocking", "feedback": feedback["feedback"],
                "error_type": type(exc).__name__,
                "finish_reason": exc.finish_reason if isinstance(exc, LLMOutputError) else None,
            }
        },
        "judge_feedback": [*state.get("judge_feedback", []), feedback],
    }


def mcq_generator_node(state: MCQState) -> Dict[str, Any]:
    refs = evidence_refs(state.get("evidence_docs", []))
    blueprint = state["blueprint"]
    method = state.get("experiment_method", "full")
    if not refs:
        issue = (
            "dsm5_evidence_missing"
            if blueprint.get("level") == "clinical_scenario" and method != "direct"
            else "missing_retrieved_evidence"
        )
        return _generation_failure(state, ValueError(issue))

    full_playbook = state.get("playbook", "")
    required_answer = _required_answer_position(state)
    if blueprint.get("level") == "clinical_scenario" and method != "direct":
        if not any(ref.get("source_kind") == "dsm5" for ref in refs):
            return _generation_failure(state, ValueError("dsm5_evidence_missing"))
        if not any(ref.get("source_kind") == "textbook" for ref in refs):
            return _generation_failure(state, ValueError("textbook_evidence_missing"))
    # Direct is the source-only control. It must not receive an additional
    # evidence-planning call or any LLM-as-judge preflight signal.
    evidence_plan = (
        {"supported_claims": [], "prohibited_inferences": []}
        if method == "direct"
        else _build_evidence_plan(blueprint, refs)
    )

    feedback = list(state.get("judge_feedback", []))
    mcq = state.get("generation_draft") or state.get("mcq", {})
    try:
        mcq = _generate_mcq(
            blueprint=blueprint,
            playbook=full_playbook,
            refs=refs,
            evidence_plan=evidence_plan,
            judge_feedback=feedback,
            required_answer=required_answer,
            previous_mcq=mcq,
        )
        mcq, schema_report = _normalize_and_validate_mcq(
            mcq, refs=refs, required_answer=required_answer, blueprint=blueprint
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
                previous_mcq=mcq,
            )
            mcq, schema_report = _normalize_and_validate_mcq(
                mcq, refs=refs, required_answer=required_answer, blueprint=blueprint
            )
            if not schema_report["passed"]:
                feedback.append(
                    {
                        "judge": "a03_schema",
                        "issues": schema_report["issues"],
                        "feedback": schema_report["feedback"],
                    }
                )
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
                previous_mcq=mcq,
            )
            mcq, schema_report = _normalize_and_validate_mcq(
                mcq, refs=refs, required_answer=required_answer, blueprint=blueprint
            )
            if not schema_report["passed"]:
                feedback.append(
                    {
                        "judge": "a03_schema",
                        "issues": schema_report["issues"],
                        "feedback": schema_report["feedback"],
                    }
                )
                raise ValueError("A03 preflight repair broke schema: " + ", ".join(schema_report["issues"]))
    except Exception as exc:
        return _generation_failure(
            {**state, "generation_draft": mcq, "judge_feedback": feedback}, exc
        )
    return {
        "mcq": mcq,
        "generation_draft": {},
        "generation_attempt": state.get("generation_attempt", 0) + 1,
        "judge_reports": {},
        "evidence_report": {},
        "single_answer_report": {},
        "ei_safety_bias_report": {},
        "adversarial_solver_report": {},
        "judge_feedback": feedback,
    }


def prepare_regeneration_node(state: MCQState) -> Dict[str, Any]:
    """Repair the current item first; replan missing or persistently misaligned sources."""
    feedback: List[Dict[str, Any]] = []
    for judge, report in state.get("judge_reports", {}).items():
        if not report.get("passed", False):
            feedback.append(
                {
                    "judge": judge,
                    "issues": report.get("issues", ["failed"]),
                    "feedback": report.get("feedback", ""),
                }
            )
    reported_issues = {
        f"{item['judge']}:{issue}"
        for item in feedback
        for issue in item.get("issues", [])
    }
    gate_issues = [
        issue for issue in state.get("quarantine_reason", [])
        if issue not in reported_issues
    ]
    if gate_issues:
        options = state.get("mcq", {}).get("options", {})
        surface_feedback = ""
        if isinstance(options, dict) and set(options) == set(ANSWER_LABELS) and all(isinstance(text, str) for text in options.values()):
            surface_feedback = _hard_guard_report(state.get("blueprint", {}), state["mcq"])["feedback"]
        feedback.append(
            {
                "judge": "quality_gate",
                "issues": gate_issues,
                "feedback": "Repair these quality-gate failures while preserving the valid parts of the current item. " + surface_feedback,
            }
        )
    issues = [
        str(issue)
        for item in feedback
        for issue in item.get("issues", [])
    ]
    previous_feedback = state.get("judge_feedback", [])
    previous_issues = [
        str(issue) for item in previous_feedback for issue in item.get("issues", [])
    ]
    alignment_markers = (
        "blueprint_mismatch",
        "blueprint_topic_mismatch",
        "subtopic_mismatch",
        "subtopic_unsupported",
        "retrieval_query_mismatch",
        "evidence_missing_primary_chunk",
    )
    missing_sources = any(
        marker in issue for issue in issues
        for marker in ("dsm5_evidence_missing", "textbook_evidence_missing", "missing_retrieved_evidence")
    )
    repeated_alignment = (
        state.get("generation_attempt", 0) > 1
        and state.get("regeneration_route", "generate") == "generate"
        and any(
            any(marker in issue for issue in issues)
            and any(marker in issue for issue in previous_issues)
            for marker in alignment_markers
        )
    )
    needs_replan = missing_sources or repeated_alignment
    source_markers = ("irrelevant_textbook_citation", "irrelevant_dsm5_citation", "insufficient_evidence_support")
    needs_retrieval = (
        not needs_replan
        and bool(state.get("mcq"))
        and state.get("generation_attempt", 0) > 1
        and state.get("regeneration_route", "generate") == "generate"
        and any(
            any(marker in issue for issue in issues)
            and any(marker in issue for issue in previous_issues)
            for marker in source_markers
        )
    )
    accumulated_feedback = [item for item in previous_feedback if item not in feedback] + feedback
    route = "replan" if needs_replan else "retrieve" if needs_retrieval else "generate"
    update: Dict[str, Any] = {
        "judge_feedback": accumulated_feedback,
        "planning_feedback": feedback if needs_replan else [],
        "regeneration_route": route,
        "repair_history": [*state.get("repair_history", []), {
            "generation_attempt": state.get("generation_attempt", 0),
            "route": route,
            "feedback": feedback,
        }],
        "dsm5_safety_docs": [],
    }
    if needs_replan:
        # A new source pairing must not inherit a draft grounded in old evidence.
        update.update({"mcq": {}, "generation_draft": {}})
    elif needs_retrieval:
        # Search for evidence matching the repaired case, not just the broad anchor topic.
        blueprint = state["blueprint"]
        mcq = state["mcq"]
        query = " ".join(
            value for value in (blueprint.get("topic"), mcq.get("clinical_case"), mcq.get("question"))
            if isinstance(value, str) and value.strip()
        )[:1500]
        update["blueprint"] = {**blueprint, "retrieval_query": query}
    return update
