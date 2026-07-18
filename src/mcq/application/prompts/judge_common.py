"""Shared rendering mechanics; individual rubric ownership stays per A04–A06 file."""
import json
from typing import Any, Dict, List


def render_judge(*, name: str, rubric: str, blueprint: Dict[str, Any], mcq: Dict[str, Any], evidence: List[Dict[str, Any]], past_failures: List[Dict[str, Any]]) -> str:
    return f"""You are {name}, an independent LLM judge for Vietnamese psychology MCQs.
{rubric}
Blueprint: {json.dumps(blueprint, ensure_ascii=False)}
MCQ: {json.dumps(mcq, ensure_ascii=False)}
Evidence: {json.dumps(evidence, ensure_ascii=False)}
Known past failure patterns for this judge: {json.dumps(past_failures, ensure_ascii=False)}
Use them only as a checklist against repeating a judging mistake. Evaluate the
current item independently; do not automatically fail an item because it merely
resembles an earlier failure.
Return JSON only: {{"passed":true|false,"issues":["short_machine_readable_issue"],
"severity":"none|warning|blocking","feedback":"short actionable feedback"}}."""
