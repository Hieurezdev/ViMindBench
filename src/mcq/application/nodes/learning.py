"""A08 Reflection and A09 ACE playbook curation."""
import os
import re
from collections import Counter
from typing import Any, Dict, List
from ...domain import MCQState
from ..failure_memory import record_judge_failures

ACE_BULLET = re.compile(r"^\[([^\]]+)\]\s+helpful=(\d+)\s+harmful=(\d+)\s+::\s+(.+)$", re.MULTILINE)
SECTION_SLUGS = {"EVIDENCE & GROUNDING": "evi", "COMMON MISTAKES TO AVOID": "err"}


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
    playbook = _update_counters(state.get("playbook", ""), state.get("blueprint", {}).get("playbook_bullet_ids", []), "helpful" if state.get("verdict") == "verified" else "harmful")
    judge_memory = record_judge_failures(state.get("judge_failure_memory", []), reports=state.get("judge_reports", {}), blueprint=state.get("blueprint", {}), mcq=state.get("mcq", {}), iteration=state.get("iteration_count", 0))
    if state.get("verdict") == "verified":
        return {"playbook": playbook, "judge_failure_memory": judge_memory}
    memory = list(state.get("failure_memory", []))
    for issue in state.get("quarantine_reason", []):
        memory.append({"issue": issue, "level": state["blueprint"].get("level"), "topic": state["blueprint"].get("topic"), "iteration": state.get("iteration_count", 0)})
    return {"playbook": playbook, "failure_memory": memory[-500:], "judge_failure_memory": judge_memory}


def playbook_curator_node(state: MCQState) -> Dict[str, Any]:
    counts = Counter(item["issue"] for item in state.get("failure_memory", []))
    recurring = [issue for issue, count in counts.items() if count >= int(os.getenv("PLAYBOOK_REPEAT_THRESHOLD", "3"))]
    playbook, delta = state.get("playbook", ""), []
    for issue in recurring:
        if issue in playbook:
            continue
        section = "EVIDENCE & GROUNDING" if issue.startswith("evidence") or "evidence:" in issue else "COMMON MISTAKES TO AVOID"
        next_id = 1 + max((int(match.group(1).rsplit("-", 1)[-1]) for match in ACE_BULLET.finditer(playbook)), default=0)
        bullet_id = f"{SECTION_SLUGS[section]}-{next_id:05d}"
        content = f"Before generation, prevent recurring failure ({counts[issue]} occurrences): {issue}."
        addition, header = f"[{bullet_id}] helpful=0 harmful=0 :: {content}", f"## {section}"
        after_section = playbook.find("\n## ", playbook.find(header) + len(header))
        playbook = playbook + "\n" + addition if after_section < 0 else playbook[:after_section] + "\n" + addition + playbook[after_section:]
        delta.append({"op": "ADD", "section": section, "bullet_id": bullet_id, "content": content, "support": counts[issue]})
    return {"playbook": playbook, "playbook_delta": delta}
