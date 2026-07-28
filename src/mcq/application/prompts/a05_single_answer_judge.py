"""A05 Single-Answer Judge prompt."""

from .judge_common import render_judge


def render(**kwargs: object) -> str:
    blueprint = kwargs.get("blueprint", {})
    hard_rule = ""
    medium_rule = ""
    if isinstance(blueprint, dict) and blueprint.get("difficulty") == "medium":
        medium_rule = " For difficulty=medium, require one near-miss distractor that is highly similar to the key in mechanism, topic, or wording but is demonstrably wrong on one precise distinction. The other two distractors may be more clearly wrong. Fail if the near-miss is also correct, equally defensible, or absent."
    if isinstance(blueprint, dict) and blueprint.get("difficulty") == "hard":
        hard_rule = " For difficulty=hard, fail if the key can be found through surface cues alone: it must not be uniquely longer, more nuanced, more qualified, more precise, or less absolute than every distractor."
    return render_judge(
        name="A05 Single-Answer Judge",
        rubric="Pass only if exactly one option is clearly best. Distractors must be plausible but demonstrably worse, mutually distinct, and must not reveal the key by length, absolutes, or wording." + medium_rule + hard_rule,
        **kwargs,
    )
