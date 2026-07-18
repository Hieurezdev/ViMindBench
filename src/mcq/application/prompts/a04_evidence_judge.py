"""A04 Evidence Judge prompt."""
from .judge_common import render_judge


def render(**kwargs: object) -> str:
    return render_judge(name="A04 Evidence Judge", rubric="Pass only if the keyed option is directly supported by cited chunk(s). Fail if evidence merely relates to the topic or supports a different option.", **kwargs)
