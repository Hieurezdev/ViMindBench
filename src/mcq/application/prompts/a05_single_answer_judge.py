"""A05 Single-Answer Judge prompt."""

from .judge_common import render_judge


def render(**kwargs: object) -> str:
    return render_judge(
        name="A05 Single-Answer Judge",
        rubric="Pass only if exactly one option is clearly best. Distractors must be plausible but demonstrably worse, mutually distinct, and must not reveal the key by length, absolutes, or wording.",
        **kwargs,
    )
