"""EmoBench taxonomy adapter for A01 and the A06 LLM-as-judge.

The benchmark data are not answer evidence and are intentionally never retrieved
into MCQ generation. These stable task dimensions are adapted from EmoBench's
EU/EA taxonomy so the judge can emit auditable, machine-readable checks.
"""
from typing import Any, Dict, List

EU_CATEGORIES = ("complex_emotions", "emotional_cues", "personal_beliefs_experiences", "perspective_taking")
EA_RELATIONSHIPS = ("personal", "social")
EA_PROBLEM_OWNERS = ("self", "others")
EA_QUESTION_TYPES = ("response", "action")


def normalize_blueprint_emobench(value: Any, *, enabled: bool) -> Dict[str, Any]:
    """Keep an emotion blueprint inside the official EU/EA task space."""
    if not enabled:
        return {"enabled": False}
    source = value if isinstance(value, dict) else {}
    task = source.get("task") if source.get("task") in {"EU", "EA"} else "EU"
    if task == "EU":
        category = source.get("eu_category")
        return {"enabled": True, "task": "EU", "eu_category": category if category in EU_CATEGORIES else "complex_emotions"}
    return {
        "enabled": True,
        "task": "EA",
        "relationship_type": source.get("relationship_type") if source.get("relationship_type") in EA_RELATIONSHIPS else "personal",
        "problem_owner": source.get("problem_owner") if source.get("problem_owner") in EA_PROBLEM_OWNERS else "self",
        "question_type": source.get("question_type") if source.get("question_type") in EA_QUESTION_TYPES else "response",
    }


def judge_context(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    """Return the rubric and exact criterion keys A06 must score."""
    context = blueprint.get("emobench", {})
    if not isinstance(context, dict) or not context.get("enabled"):
        return {"enabled": False, "criteria": []}
    if context.get("task") == "EA":
        return {**context, "criteria": ["perspective_taking", "context_sensitive_response", "relationship_fit", "problem_owner_fit", "response_action_fit"]}
    return {**context, "task": "EU", "criteria": ["emotion_recognition", "emotion_cause_separation", "category_fit", "perspective_taking"]}


def validate_judge_report(report: Dict[str, Any], blueprint: Dict[str, Any]) -> Dict[str, Any]:
    """Reject incomplete or internally inconsistent structured EmoBench reports."""
    context = judge_context(blueprint)
    if not context["enabled"]:
        return report
    result = dict(report)
    issues: List[str] = list(result.get("issues", [])) if isinstance(result.get("issues", []), list) else ["invalid_judge_issues"]
    emobench = result.get("emobench")
    criteria = emobench.get("criteria") if isinstance(emobench, dict) else None
    if not isinstance(emobench, dict) or emobench.get("task") != context["task"] or not isinstance(criteria, dict):
        issues.append("emobench_incomplete_report")
    else:
        for name in context["criteria"]:
            verdict = criteria.get(name)
            if verdict not in {"pass", "fail"}:
                issues.append(f"emobench_missing_criterion:{name}")
            elif verdict == "fail":
                issues.append(f"emobench:{name}")
    if issues:
        result["passed"] = False
        result["severity"] = "blocking"
        result["issues"] = list(dict.fromkeys(issues))
    return result
