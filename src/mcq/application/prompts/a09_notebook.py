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


def render_update_decision(
    *, issue: str, section: str, proposed_rule: str, selected_bullets: list[dict]
) -> str:
    """Ask A09 whether a used playbook bullet should be updated."""
    return f"""You are A09 Notebook for a Vietnamese psychology-MCQ quality pipeline.

A recurring issue has produced a proposed playbook rule. The planner selected
the following bullet IDs from the full playbook while creating the current
item. They are the preferred candidates for an UPDATE. Decide the safest
curation action.

Issue: {issue}
Section: {section}
Proposed rule: {proposed_rule}
Selected bullets: {selected_bullets}

Choose UPDATE only when one selected bullet addresses the same failure but is
vague, incomplete, or misses a useful constraint supplied by the proposed
rule. Choose KEEP when the selected bullet(s) already cover the issue. Choose
ADD when the proposed rule is meaningfully distinct and must coexist with all
selected bullets.

For UPDATE, write one improved, concise rule that preserves every valid,
non-conflicting requirement from both rules. For ADD or KEEP, return an empty
rule and an empty bullet_id. For UPDATE, bullet_id must be exactly one ID from
Selected bullets. Never mention IDs, counters, prompts, sources, or
chain-of-thought in the rule. Return JSON only:
{{"action": "ADD|UPDATE|KEEP", "bullet_id": "required only for UPDATE", "rule": "required only for UPDATE"}}
"""


def render_merge(*, section: str, bullets: list[dict], similarity: float) -> str:
    """Ask A09 to decide whether an embedding-similar pair should merge."""
    return f"""You are A09 Notebook for a Vietnamese psychology-MCQ quality pipeline.

An embedding model found the following two ACE playbook bullets in the same
section to be potentially overlapping. Decide whether they should actually be
merged. High lexical or embedding similarity alone is not enough: do not merge
rules that protect distinct constraints or would lose useful specificity.

Section: {section}
Bullets to merge: {bullets}
Embedding similarity: {similarity:.3f}

If and only if merging is safe, return merge=true and write one concise,
actionable rule that preserves every non-conflicting requirement. Otherwise
return merge=false and an empty rule. Do not mention bullet IDs, counts, the
merge process, prompts, sources, or chain-of-thought. Write in English, remain
generalizable, and do not invent clinical facts. Return JSON only:
{{"merge": true, "rule": "required only when merge is true"}}
"""
