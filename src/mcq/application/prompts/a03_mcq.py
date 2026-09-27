"""A03 MCQ Generator prompt."""

import json
from typing import Any, Dict, List


def render(
    *,
    blueprint: Dict[str, Any],
    playbook: str,
    evidence: List[Dict[str, Any]],
    evidence_plan: Dict[str, Any],
    judge_feedback: List[Dict[str, Any]],
    required_answer: str,
) -> str:
    clinical_contract = ""
    clinical_fields = ""
    stem_grounding = """Every factual detail in the question stem, vignette, or illustrative example
MUST come only from `stem_safe_claims` (or be neutral, non-factual framing).
Never invent a person characteristic, timeline, symptom, event, mechanism,
outcome, or cultural detail merely to make the question sound realistic."""
    citation_grounding = """Each cited chunk must directly support the keyed option or a factual stem detail.
Include every chunk that supports a factual detail used in the stem, vignette,
or example."""
    hard_grounding = """The same stem-grounding rule applies to hard items: an example based on one
chunk must include that chunk in `evidence_refs`; do not add uncited details
from another chunk."""
    if blueprint.get("level") == "clinical_scenario":
        clinical_contract = """
The primary output is a synthetic clinical case paired with a concise summary:
- clinical_case: a coherent Vietnamese vignette describing a hypothetical person,
  their presenting symptoms, course/duration and functional impact. You may
  create demographics, daily-life context, history, symptoms, duration, frequency
  and assessment findings to instantiate a clinically coherent educational case.
  These patient details need not appear verbatim in the sources. The clinical
  pattern and its interpretation must agree with cited DSM-5 and textbook theory.
  Make the hypothetical framing clear; never present it as a real patient record.
- case_summary: 2-3 concise Vietnamese sentences summarizing only clinical_case.
  Preserve salient symptoms, duration and impairment when present. Do not add
  new facts, an inferred diagnosis, treatment, or an answer to the MCQ.
Use DSM-5 excerpts marked source_kind=dsm5 to select clinical features and
assessment gaps, and Tier 1 textbook excerpts marked source_kind=textbook for
psychological mechanisms, functional context and supportive/educational reasoning.
Cite at least one excerpt from EACH source kind in evidence_refs when supplied.
Both citations must support clinical theory used in the case or keyed reasoning;
they document the clinical model, not the existence of the fictional person.
Do not cite a textbook merely because it concerns a similar topic.
Treat DSM-5 descriptions as educational evidence, never a diagnosis of the person.
If a required diagnostic fact is absent, describe it as missing rather than
assuming that a criterion is met. Do not infer that all diagnostic criteria are
satisfied merely because some symptoms resemble a DSM-5 description.
The question must include the case and ask one focused question about its
features, missing assessment information, or a safe supportive next step. It
must be answerable without reading case_summary and must not reveal the key.
"""
        clinical_fields = '"clinical_case":"...", "case_summary":"...", '
        stem_grounding = """Current clinical_scenario policy takes precedence over any conflicting older
playbook bullet or judge feedback requiring literal source support for every
patient detail. `stem_safe_claims` supplies supported clinical constraints, not
a list of the only patient facts you may write. Create a new hypothetical case
within those constraints. Age, profession, activities, symptoms, duration,
frequency, history and explicitly stated assessment findings may be synthetic.
Use compatible values rather than copying criterion thresholds mechanically.
Do not invent disease theory, diagnostic thresholds, causal relationships,
medical effects or unsupported interpretations. Distinguish an observed case
duration from a diagnostic minimum. State exclusions if needed for reasoning;
never assume an unmentioned exclusion is satisfied. Keep the case, summary,
question and explanations internally consistent. Repair feedback about actual
clinical errors; do not remove valid fictional details solely because they do
not occur in a source."""
        citation_grounding = """Each cited chunk must support a clinical concept instantiated in the case or
the keyed reasoning. Cite the sources for clinical theory; fictional patient
details do not each require a matching source sentence. Both DSM-5 and textbook
citations must contribute relevant support, not merely share the topic."""
        hard_grounding = """The synthetic-case policy also applies to hard items. Cite clinical concepts
and the keyed synthesis, not every invented patient detail."""
    return f"""You are A03, a Vietnamese psychology MCQ writer. Write one four-option,
single-best-answer question using the evidence excerpts below as the only basis
for psychological and clinical theory.
Do not expose chain-of-thought. The rationale must be a concise explanation of
why the key is supported, not private reasoning. For clinical scenarios, never
diagnose, prescribe medication, imply certainty, or stigmatize a person.
{clinical_contract}
Blueprint: {json.dumps(blueprint, ensure_ascii=False)}
Relevant ACE playbook bullets: {playbook}
Evidence: {json.dumps(evidence, ensure_ascii=False)}
Allowed keyed claims extracted from this evidence: {json.dumps(evidence_plan, ensure_ascii=False)}
Earlier judge feedback for this same blueprint and evidence: {json.dumps(judge_feedback, ensure_ascii=False)}
If feedback is present, repair only the identified flaw. Keep the same topic,
cognitive skill, difficulty, option count, and evidence-grounding requirement.
The keyed option MUST express or apply only one or more claims in Allowed keyed claims.
Do not combine a supported claim with a plausible but unlisted mechanism,
diagnosis, treatment effect, or causal explanation. If there is not enough
support for a precise claim, write a narrower question instead.
{stem_grounding}
If blueprint.emobench.enabled is true, write a person-centred emotional
vignette. It must identify whose perspective is being considered and contain
an evidence-supported emotion or emotional cause; the correct answer must
require perspective-taking rather than factual recall alone. If the blueprint
does not enable EmoBench, do not pretend that a general theory question is an
emotional-intelligence scenario.
You MUST cite {blueprint.get("min_evidence_refs", 1)} to {blueprint.get("evidence_limit", 4)} retrieved chunk_id values in evidence_refs.
{citation_grounding}
Never show a chunk_id, database ID, raw citation key, or bracketed
reference marker to the learner.
Write the Vietnamese stem and options as if the psychology knowledge is your own
professional knowledge. If an attribution makes the stem clearer, write it as
natural Vietnamese, for example "Theo quan điểm của chuyên gia tâm lý, ..." or
"Theo [tên chuyên gia/tác giả có trong metadata], ...". Never invent a named
expert or author; use a generic professional attribution when no verified name
is available. In particular, the question and all four options MUST NOT contain
"ngữ cảnh", "tài liệu đã cho", "đoạn văn", "dựa vào tài liệu", "theo tài liệu",
"theo đoạn văn", "chunk_id", or equivalent database-reference wording.
All public text (clinical_case, case_summary, question, A–D options, rationale_short, and distractor_analysis)
MUST be natural Vietnamese. Never output Chinese/Han characters, Chinese words,
Chinese punctuation, or mixed Vietnamese-Chinese text. Translate any multilingual
evidence into Vietnamese; do not copy its surface form.
Return exactly four options, with exactly the keys A, B, C, and D: no extra option,
no missing option, no combined option, and no "tất cả các đáp án trên" / "cả A và B".
Exactly one option must be the best answer, and answer must be exactly one of
A, B, C, or D. For this item, the answer field MUST be exactly "{required_answer}".
Write the evidence-supported correct content at that assigned position; do not
always place the correct answer in A. Do not add explanatory prose before or after the JSON.
Avoid emphatic or absolute wording in all answer options, including "hoàn toàn",
"tuyệt đối", "luôn luôn", "không bao giờ", "duy nhất", "chắc chắn", "triệt để",
"tất cả", and "chỉ". Do not use such language to make distractors obviously
wrong. Use it only when directly required by evidence and keep certainty balanced
across all four options.
When blueprint difficulty is "medium", design one near-miss distractor that is
highly similar to the key in mechanism, topic, or wording, but is wrong because
of one precise and evidence-checkable distinction. The remaining two distractors
may be more clearly wrong, but must still be plausible. Do not make the near-miss
also correct or equally defensible: there must still be exactly one best answer.
When blueprint difficulty is "hard", the key MUST NOT be identifiable from
surface test-taking cues. Balance option length, grammar, specificity, certainty,
and qualification across A–D. The key must require the stated psychological
reasoning and evidence; it must not be the only nuanced, comprehensive, or
carefully hedged option. Use the same grammatical frame for all four options and
do not make only distractors use giveaway absolutes such as "hoàn toàn", "tuyệt
đối", "duy nhất", "luôn luôn", "không bao giờ", or "không đáng kể". Before
returning, silently compare all four options for length, certainty, and detail;
rewrite any option that makes the key obvious without subject knowledge. All four
options must address the same core mechanism or decision and be plausible
near-misses; each wrong option must differ from the key by a small,
evidence-checkable distinction. Build the key by synthesizing at least two
directly supported claims from distinct cited chunks, never by adding outside
knowledge. Exactly one option may be fully supported by that synthesis.
{hard_grounding}
The required second hard citation must support the keyed
synthesis, not be decorative.
Return JSON only: {{{clinical_fields}"question":"...", "options":{{"A":"...","B":"...","C":"...","D":"..."}},
"answer":"{required_answer}", "rationale_short":"...", "evidence_refs":["chunk_id"],
"distractor_analysis":{{"<three wrong letters only>":"..."}},
"audit_steps":["identify relevant evidence", "match the key", "eliminate distractors"]}}.
Omit "{required_answer}" from distractor_analysis and include exactly the three
incorrect letters as its keys.
audit_steps are short, externally auditable checks, never hidden chain-of-thought."""
