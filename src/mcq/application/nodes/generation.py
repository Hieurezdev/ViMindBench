"""A03 MCQ generation node."""

import random
import re
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


def mcq_generator_node(state: MCQState) -> Dict[str, Any]:
    refs = evidence_refs(state.get("evidence_docs", []))
    if not refs:
        return {"mcq": {}}

    full_playbook = state.get("playbook", "")
    filtered_playbook = _sample_successful_strategies(full_playbook, sample_size=3)

    mcq = request_json(
        a03_mcq.render(
            blueprint=state["blueprint"],
            playbook=filtered_playbook[:7000],
            evidence=refs,
            judge_feedback=state.get("judge_feedback", []),
        )
    )
    # A03 requests string IDs, but local models may return
    # [{"chunk_id": "..."}]. Normalize before the judge pipeline.
    mcq["evidence_refs"] = _normalize_evidence_ref_ids(mcq.get("evidence_refs")) or [
        ref["chunk_id"] for ref in refs
    ]
    return {
        "mcq": mcq,
        "generation_attempt": state.get("generation_attempt", 0) + 1,
        "judge_reports": {},
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
