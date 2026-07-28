"""A03 MCQ generation node."""

import os
import random
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


def _sample_successful_strategies(playbook: str, sample_size: int = 3) -> str:
    """Randomly sample successful strategies to prevent mode collapse."""
    section_header = "## SUCCESSFUL STRATEGIES TO REPLICATE"
    if section_header not in playbook:
        return playbook

    parts = playbook.split(section_header)
    before_section = parts[0]
    success_section = parts[1]

    next_section_match = re.search(r'\n## ', success_section)
    if next_section_match:
        success_content = success_section[:next_section_match.start()]
        after_section = success_section[next_section_match.start():]
    else:
        success_content = success_section
        after_section = ""

    bullets = re.findall(r"^\[suc-[^\]]+\]\s+.*?(?=\n\[suc-|\Z)", success_content, re.MULTILINE | re.DOTALL)

    if len(bullets) <= sample_size:
        return playbook

    sampled_bullets = random.sample(bullets, sample_size)
    new_success_section = "\n" + "\n".join(b.strip() for b in sampled_bullets) + "\n"

    return before_section + section_header + new_success_section + after_section


def _build_evidence_plan(blueprint: Dict[str, Any], refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep the key within explicitly extracted evidence claims when possible."""
    try:
        plan = request_json(a03_evidence_plan.render(blueprint=blueprint, evidence=refs), max_tokens=700)
        if isinstance(plan, dict):
            return plan
    except Exception:
        pass
    return {"supported_claims": [], "prohibited_inferences": []}


def _generate_mcq(
    *,
    blueprint: Dict[str, Any],
    playbook: str,
    refs: List[Dict[str, Any]],
    evidence_plan: Dict[str, Any],
    judge_feedback: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return request_json(
        a03_mcq.render(
            blueprint=blueprint,
            playbook=playbook[:7000],
            evidence=refs,
            evidence_plan=evidence_plan,
            judge_feedback=judge_feedback,
        )
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
    filtered_playbook = _sample_successful_strategies(full_playbook, sample_size=3)
    blueprint = state["blueprint"]
    evidence_plan = _build_evidence_plan(blueprint, refs)

    feedback = list(state.get("judge_feedback", []))
    try:
        mcq = _generate_mcq(
            blueprint=blueprint,
            playbook=filtered_playbook,
            refs=refs,
            evidence_plan=evidence_plan,
            judge_feedback=feedback,
        )
        preflight = _preflight_report(blueprint=blueprint, mcq=mcq, refs=refs)
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
                playbook=filtered_playbook,
                refs=refs,
                evidence_plan=evidence_plan,
                judge_feedback=feedback,
            )
    except Exception as exc:
        return _generation_failure(state, exc)
    # A03 requests string IDs, but local models may return
    # [{"chunk_id": "..."}]. Normalize before the judge pipeline.
    mcq["evidence_refs"] = _normalize_evidence_ref_ids(mcq.get("evidence_refs")) or [
        ref["chunk_id"] for ref in refs
    ]
    return {
        "mcq": mcq,
        "generation_attempt": state.get("generation_attempt", 0) + 1,
        "judge_reports": {},
        "judge_feedback": feedback,
    }


def prepare_regeneration_node(state: MCQState) -> Dict[str, Any]:
    """Carry only failed judge findings into the next bounded A03 attempt."""
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
    return {"judge_feedback": feedback, "dsm5_safety_docs": []}
