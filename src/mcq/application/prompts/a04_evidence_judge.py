"""A04 Evidence Judge prompt."""

from .judge_common import render_judge


def render(**kwargs: object) -> str:
    blueprint = kwargs.get("blueprint")
    rubric = """Pass only if the keyed option applies claims directly supported by cited
chunks. Fail if evidence merely relates to the topic or supports a different
option. Natural Vietnamese attribution is allowed, including 'Theo DSM-5'; fail
raw IDs, bracketed citation markers or a fabricated named expert."""
    if isinstance(blueprint, dict) and blueprint.get("level") == "clinical_scenario":
        rubric += """
Evaluate a synthetic educational case, not a copied patient record. Patient
demographics, daily-life context, symptoms, history, duration, frequency and
explicit assessment findings may be invented. Do not fail solely because a
valid fictional patient detail does not occur verbatim in a source. Evaluate
whether the instantiated clinical pattern and interpretation agree with cited
theory, including symptom relationships, duration thresholds, impairment and
exclusions. A case value may differ from a criterion's minimum without being
wrong. Do not turn a source threshold into a requirement to copy that number.
Fail contradictory clinical reasoning, invented diagnostic rules or medical
effects, unsupported causal claims, or treating absent information as a confirmed
exclusion. A partial clinical presentation can be valid when the question does
not claim a full diagnosis. Fictional observations are premises; conclusions
must follow from those premises and cited clinical concepts.
Require case_summary to faithfully compress clinical_case, including its
fictional details, without adding facts, an inferred diagnosis, treatment or
the keyed answer. Verify the summary against the generated case rather than
against a source patient. Reject contradictions between the case and question.
Preserve the strength and scope of findings: a limited negative finding cannot
be summarized as ruling out all medical or substance causes. The question must
include enough case information to be answerable on its own.
Require at least one cited chunk of EACH supplied source_kind, dsm5 and textbook.
Each must support a clinical concept instantiated in the case or keyed reasoning;
they need not document the existence or exact details of the fictional patient.
A textbook supporting a relevant case concept need not separately support the
key for easy/medium; an unrelated citation still fails. For hard, require the
keyed synthesis to be supported by at least two distinct cited chunks.
Check source coverage separately from claim support. Mark a source_kind missing
ONLY when none of the cited IDs has that source_kind metadata. If the kind is
present but irrelevant, report insufficient support, not a missing source kind.
For easy/medium, evaluate each claim against the combined cited evidence:
DSM-5 may support the key while the textbook supports physical symptoms, course
or another clinical concept in the case. Do not require each claim or the key
to appear in BOTH source kinds. Read the actual excerpt before declaring a
claim absent; a truncated excerpt can still support claims within its visible text.
For every failure, identify the specific clinical claim or summary discrepancy
and the relevant evidence; do not assert missing source kinds without checking
the cited chunk IDs and their source_kind fields. This synthetic-case policy
overrides conflicting past failure patterns that prohibit valid fictional details.
"""
    else:
        rubric += """ For every difficulty, also fail when a factual stem/vignette/example
detail is not supported by one of the emitted evidence_refs chunks."""
    return render_judge(
        name="A04 Evidence Judge",
        rubric=rubric,
        **kwargs,
    )
