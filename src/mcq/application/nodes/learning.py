"""A08 Reflection and A09 Notebook (ACE playbook curation)."""

import os
import re
from collections import Counter
from typing import Any, Dict, List
import json
from ...domain import MCQState
from ..failure_memory import record_judge_failures

ACE_BULLET = re.compile(
    r"^\[([^\]]+)\]\s+helpful=(\d+)\s+harmful=(\d+)\s+::\s+(.+)$", re.MULTILINE
)
SECTION_SLUGS = {
    "EVIDENCE & GROUNDING": "evi",
    "COMMON MISTAKES TO AVOID": "err",
    "SUCCESSFUL STRATEGIES TO REPLICATE": "suc"
}


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
        if is_success:
            section = "SUCCESSFUL STRATEGIES TO REPLICATE"
            sample_feedback = next((item.get("feedback", "") for item in reversed(state.get("failure_memory", [])) if item["issue"] == issue), "")
            content = f"Get inspiration from the following excellent thinking to apply: {sample_feedback}"
        else:
            section = (
                "EVIDENCE & GROUNDING"
                if issue.startswith("evidence") or "evidence:" in issue
                else "COMMON MISTAKES TO AVOID"
            )
            content = f"Before generation, prevent recurring failure ({counts[issue]} occurrences): {issue}."

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
