"""A08 Reflection and A09 Notebook (ACE playbook curation)."""

import builtins
import logging
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List
import json
from ...domain import MCQState
from ..failure_memory import record_judge_failures
from ...infrastructure.llm_gateway import request_insight_json
from ..prompts import a09_notebook

logger = logging.getLogger("mcq.learning")

ACE_BULLET = re.compile(
    r"^\[([^\]]+)\]\s+helpful=(\d+)\s+harmful=(\d+)\s+::\s+(.+)$", re.MULTILINE
)
SECTION_SLUGS = {
    "EVIDENCE & GROUNDING": "evi",
    "COMMON MISTAKES TO AVOID": "err",
    "SUCCESSFUL STRATEGIES TO REPLICATE": "suc"
}


def _section_bullet_contents(playbook: str, section: str) -> List[str]:
    """Return ACE bullet text from one Markdown section, without its metadata."""
    current_section = ""
    contents = []
    for line in playbook.splitlines():
        if line.startswith("## "):
            current_section = line[3:].strip()
            continue
        match = ACE_BULLET.match(line)
        if match and current_section == section:
            contents.append(match.group(4))
    return contents


def _cosine_similarity(left: List[float], right: List[float]) -> float | None:
    """Return cosine similarity, or None for unusable embedding vectors."""
    if not left or len(left) != len(right):
        return None
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return None
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _duplicates_common_mistake(playbook: str, content: str) -> bool:
    """Use the configured retriever embeddings to avoid duplicate error rules.

    The retriever is initialized once by the composition root and shares the
    embedding model/cache used by retrieval. If it is unavailable (notably in
    offline tests), preserve the existing curation behaviour instead of making
    learning fail closed.
    """
    existing = _section_bullet_contents(playbook, "COMMON MISTAKES TO AVOID")
    if not existing:
        return False

    retriever = getattr(builtins, "RETRIEVER", None)
    embed = getattr(retriever, "generate_embedding", None)
    if not callable(embed):
        logger.warning("Skipping playbook duplicate filter: embedding retriever unavailable")
        return False

    try:
        candidate = embed(content)
        threshold = float(os.getenv("PLAYBOOK_SIMILARITY_THRESHOLD", "0.8"))
        similarities = [
            similarity
            for rule in existing
            if (similarity := _cosine_similarity(candidate, embed(rule))) is not None
        ]
    except Exception as exc:
        logger.warning("Skipping playbook duplicate filter after embedding error: %s", exc)
        return False

    highest = max(similarities, default=None)
    if highest is not None and highest >= threshold:
        logger.info(
            "Skipped duplicate common-mistake rule: similarity=%.3f threshold=%.3f",
            highest,
            threshold,
        )
        return True
    return False


def _update_counters(playbook: str, bullet_ids: List[str], tag: str) -> str:
    lines = []
    for line in playbook.splitlines():
        match = ACE_BULLET.match(line)
        if match and match.group(1) in bullet_ids:
            bullet_id, helpful, harmful, content = match.groups()
            line = f"[{bullet_id}] helpful={int(helpful) + (tag == 'helpful')} harmful={int(harmful) + (tag == 'harmful')} :: {content}"
        lines.append(line)
    return "\n".join(lines)


def reflector_node(state: MCQState) -> Dict[str, Any]:
    playbook = _update_counters(
        state.get("playbook", ""),
        state.get("blueprint", {}).get("playbook_bullet_ids", []),
        "helpful" if state.get("verdict") == "verified" else "harmful",
    )
    judge_memory = record_judge_failures(
        state.get("judge_failure_memory", []),
        reports=state.get("judge_reports", {}),
        blueprint=state.get("blueprint", {}),
        mcq=state.get("mcq", {}),
        iteration=state.get("iteration_count", 0),
    )

    memory = list(state.get("failure_memory", []))

    if state.get("verdict") == "verified":
        if state.get("generation_attempt", 0) == 0:
            blueprint = state.get("blueprint", {})
            mcq = state.get("mcq", {})
            issue_type = "success:clinical_differential" if blueprint.get("level") == "clinical_scenario" else "success:high_quality_distractor"

            distractor_analysis = mcq.get("distractor_analysis", "")
            feedback_text = json.dumps(distractor_analysis, ensure_ascii=False) if isinstance(distractor_analysis, dict) else str(distractor_analysis)

            memory.append({
                "issue": issue_type,
                "level": blueprint.get("level"),
                "topic": blueprint.get("topic"),
                "feedback": feedback_text,
                "iteration": state.get("iteration_count", 0),
            })
            return {"playbook": playbook, "judge_failure_memory": judge_memory, "failure_memory": memory[-500:]}
        return {"playbook": playbook, "judge_failure_memory": judge_memory}

    for issue in state.get("quarantine_reason", []):
        memory.append(
            {
                "issue": issue,
                "level": state["blueprint"].get("level"),
                "topic": state["blueprint"].get("topic"),
                "iteration": state.get("iteration_count", 0),
            }
        )
    return {
        "playbook": playbook,
        "failure_memory": memory[-500:],
        "judge_failure_memory": judge_memory,
    }


def _notebook_rule(
    *,
    issue: str,
    occurrences: int,
    section: str,
    sample_feedback: str,
    fallback: str,
) -> str:
    """Ask the dedicated A09 endpoint only for a newly recurring pattern."""
    if not os.getenv("INSIGHT_OPENAI_BASE_URL"):
        return fallback
    try:
        response = request_insight_json(
            a09_notebook.render(
                issue=issue,
                occurrences=occurrences,
                section=section,
                sample_feedback=sample_feedback,
            )
        )
        rule = response.get("rule")
        if isinstance(rule, str) and 20 <= len(rule.strip()) <= 500:
            return rule.strip()
    except Exception:
        # Playbook curation must not stop the dataset run if the optional
        # insight model is unavailable or returns malformed JSON.
        pass
    return fallback


def playbook_curator_node(state: MCQState) -> Dict[str, Any]:
    counts = Counter(item["issue"] for item in state.get("failure_memory", []))
    recurring = [
        issue
        for issue, count in counts.items()
        if count >= int(os.getenv("PLAYBOOK_REPEAT_THRESHOLD", "3"))
    ]
    playbook, delta = state.get("playbook", ""), []

    for issue in recurring:
        if issue in playbook:
            continue

        is_success = issue.startswith("success:")
        sample_feedback = next(
            (
                item.get("feedback", "")
                for item in reversed(state.get("failure_memory", []))
                if item["issue"] == issue
            ),
            "",
        )
        if is_success:
            section = "SUCCESSFUL STRATEGIES TO REPLICATE"
            fallback = f"Replicate the successful pattern ({counts[issue]} occurrences): {issue}."
        else:
            section = (
                "EVIDENCE & GROUNDING"
                if issue.startswith("evidence") or "evidence:" in issue
                else "COMMON MISTAKES TO AVOID"
            )
            fallback = f"Before generation, prevent recurring failure ({counts[issue]} occurrences): {issue}."

        content = _notebook_rule(
            issue=issue,
            occurrences=counts[issue],
            section=section,
            sample_feedback=sample_feedback,
            fallback=fallback,
        )

        if section == "COMMON MISTAKES TO AVOID" and _duplicates_common_mistake(
            playbook, content
        ):
            continue

        next_id = 1 + max(
            (
                int(match.group(1).rsplit("-", 1)[-1])
                for match in ACE_BULLET.finditer(playbook)
                if match.group(1).startswith(SECTION_SLUGS[section])
            ),
            default=0,
        )
        bullet_id = f"{SECTION_SLUGS[section]}-{next_id:05d}"

        addition, header = (
            f"[{bullet_id}] helpful=0 harmful=0 :: {content}",
            f"## {section}",
        )

        header_pos = playbook.find(header)
        if header_pos < 0:
            playbook = playbook + "\n\n" + header + "\n" + addition
        else:
            after_section = playbook.find("\n## ", header_pos + len(header))
            if after_section < 0:
                playbook = playbook + "\n" + addition
            else:
                playbook = playbook[:after_section] + "\n" + addition + playbook[after_section:]

        delta.append(
            {
                "op": "ADD",
                "section": section,
                "bullet_id": bullet_id,
                "content": content,
                "support": counts[issue],
            }
        )
    return {"playbook": playbook, "playbook_delta": delta}
