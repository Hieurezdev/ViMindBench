"""A04 Evidence Judge prompt."""

from .judge_common import render_judge


def render(**kwargs: object) -> str:
    return render_judge(
        name="A04 Evidence Judge",
        rubric="Pass only if the keyed option is directly supported by cited chunk(s). For every difficulty, also fail when a factual stem/vignette/example detail is not supported by one of the emitted evidence_refs chunks. Natural Vietnamese attribution is allowed, but fail raw IDs, bracketed citation markers, or a fabricated named expert. Fail if evidence merely relates to the topic or supports a different option.",
        **kwargs,
    )
