"""A05 Single-Answer Judge prompt."""

from .judge_common import render_judge


def render(**kwargs: object) -> str:
    blueprint = kwargs.get("blueprint", {})
    hard_rule = ""
    medium_rule = ""
    easy_rule = ""
    if isinstance(blueprint, dict) and blueprint.get("difficulty") == "easy":
        easy_rule = " For difficulty=easy, the three distractors may be less subtle than the key but must answer the same question. Do not pass three alternatives that merely contradict explicit case facts or name unrelated diagnoses; do not require hard-level synthesis."
    if isinstance(blueprint, dict) and blueprint.get("difficulty") == "medium":
        medium_rule = " For difficulty=medium, require one near-miss distractor that is highly similar to the key in mechanism, topic, or wording but is demonstrably wrong on one precise distinction. The other two distractors may be more clearly wrong. Fail if the near-miss is also correct, equally defensible, or absent."
    if isinstance(blueprint, dict) and blueprint.get("difficulty") == "hard":
        hard_rule = " For difficulty=hard, require all four options to address the same core mechanism or decision and to be plausible near-misses. Each wrong option must differ from the key by a small evidence-checkable distinction, while only the key is fully supported by the synthesis of at least two cited chunks. Fail if a distractor is unrelated, easy to eliminate, equally defensible, or if the key can be found through surface cues alone: it must not be uniquely longer, more nuanced, more qualified, more precise, or less absolute than every distractor."
    return render_judge(
        name="A05 Single-Answer Judge",
        rubric="Pass only if exactly one option is clearly best. Independently classify every option: the MCQ's declared answer must be correct, and each of the other exactly three options must be incorrect. Evaluate against cited excerpts and explicit case facts, not invented clinical criteria. A true alternative that answers the same question makes the key non-unique. Distractors must be plausible but demonstrably worse, mutually distinct, and must not reveal the key by length, absolutes, or wording." + easy_rule + medium_rule + hard_rule,
        result_schema_suffix=',"option_assessment":{"A":"correct|incorrect|ambiguous","B":"correct|incorrect|ambiguous","C":"correct|incorrect|ambiguous","D":"correct|incorrect|ambiguous"}',
        **kwargs,
    )
