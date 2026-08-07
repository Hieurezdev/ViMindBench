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


def render_merge(*, section: str, bullets: list[dict]) -> str:
    """Ask A09 to consolidate semantically overlapping ACE playbook rules."""
    return f"""You are A09 Notebook for a Vietnamese psychology-MCQ quality pipeline.

The following ACE playbook bullets in the same section are semantically
overlapping. Consolidate them into exactly one concise, actionable rule.

Section: {section}
Bullets to merge: {bullets}

Preserve every non-conflicting requirement. Do not mention bullet IDs, counts,
the merge process, prompts, sources, or chain-of-thought. Write in English,
remain generalizable, and do not invent clinical facts. Return JSON only:
{{"rule": "one imperative rule, 20-500 characters"}}
"""
