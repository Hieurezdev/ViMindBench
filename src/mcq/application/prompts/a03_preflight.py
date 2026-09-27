"""Fast preflight rubric before the full A04–A07 evaluation chain."""

import json
from typing import Any, Dict, List


def render(*, blueprint: Dict[str, Any], mcq: Dict[str, Any], evidence: List[Dict[str, Any]]) -> str:
    stem_policy = """Every factual detail in the question stem, vignette, or example is directly
   supported by a chunk_id included in the item's evidence_refs. Reject invented
   or uncited scenario details even when the key itself is supported."""
    if blueprint.get("level") == "clinical_scenario":
        stem_policy = """The synthetic case must be clinically coherent and its interpretation
   supported by cited DSM-5 and textbook concepts. Invented patient demographics,
   context, symptoms, history, duration, frequency and stated assessment findings
   are allowed without literal matches in the sources. Check clinical patterns,
   threshold interpretation and internal consistency; do not reject valid
   fictional details just because they are not source quotations. Fail invented
   disease theory, contradictory clinical reasoning or unsupported causal claims.
   Unmentioned exclusions are unknown, not evidence that an exclusion is met."""
    return f"""You are an A03 preflight critic for a Vietnamese psychology MCQ.
Check only these release blockers before the item reaches the full judge chain:
1. The declared answer is directly supported by its cited evidence, without an
   added mechanism or claim.
2. There is exactly one best answer and three incorrect distractors.
3. {stem_policy}
   Natural language attribution is allowed, but raw IDs,
   bracketed citation markers, and fabricated named experts are not.
4. For difficulty=hard, no answer is discoverable solely from length, absolute
   wording, unique nuance, or qualification; all four options are balanced,
   address the same mechanism, and are plausible near-misses. The key must cite
   at least two chunks and synthesize only directly supported claims.
5. For clinical_scenario, require a coherent clinical_case and concise
   case_summary. The summary must faithfully compress the generated case,
   including fictional patient details, without adding symptoms, duration, impairment,
   diagnosis, treatment, or conclusions absent from it. The question must remain
   answerable on its own and preserve the same facts as clinical_case. Require
   citations to both a supplied source_kind=dsm5 chunk and a
   source_kind=textbook chunk when available; each must support clinical theory
   used in the case or keyed reasoning. An unrelated textbook citation fails.
   For easy/medium, different sources may support different parts: DSM-5 may
   support the key while the textbook supports symptoms or course in the case.
   Do not require every claim or the key to be supported by BOTH source kinds.
   A source_kind is missing only if no cited ID has that source_kind metadata;
   insufficient claim support is a separate issue. Preserve the strength and
   scope of case statements in the summary: a limited negative finding cannot
   become an assertion that all possible causes have been excluded.
   Read the entire visible excerpt before marking a clinical claim unsupported.
   A cut-off ending does not invalidate support in earlier complete sentences.
   Never assume unseen text provides support; equally, do not declare a symptom
   absent merely because a later symptom list is truncated. Report the actual
   claim in the item, not a misquoted or substituted symptom.

Blueprint: {json.dumps(blueprint, ensure_ascii=False)}
MCQ: {json.dumps(mcq, ensure_ascii=False)}
Evidence: {json.dumps(evidence, ensure_ascii=False)}

Return JSON only: {{"passed":true|false,"issues":["unsupported_claim:<short>","unsupported_stem_detail:<short>","invalid_public_attribution:<short>","surface_cue:<short>","multiple_best_answers"],"feedback":"specific repair instruction"}}.
Do not use background knowledge. If uncertain, fail rather than invent support."""
