"""Prompt for A09 Notebook's recurring-error playbook insight."""


def render(*, issue: str, occurrences: int, section: str, sample_feedback: str) -> str:
    return f"""You are A09 Notebook for a Vietnamese psychology-MCQ quality pipeline.

A recurring pattern has been observed {occurrences} times.
Issue: {issue}
Target playbook section: {section}
Representative feedback: {sample_feedback[:1800] or "(no extra feedback recorded)"}

Write exactly one concise, actionable playbook rule that prevents this pattern
or replicates the successful pattern. The rule must be in English, must be
generalizable, must not invent clinical facts, must not mention this prompt,
and must not expose chain-of-thought. Return JSON only:
{{"rule": "one imperative rule, 20-500 characters"}}
"""
