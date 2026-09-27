"""Evidence-first planning prompt for A03 generation."""

import json
from typing import Any, Dict, List


def render(*, blueprint: Dict[str, Any], evidence: List[Dict[str, Any]]) -> str:
    stem_policy = """`stem_safe_claims` covers only factual details that may appear in a question
stem, vignette, or example; do not treat plausible background details as safe."""
    if blueprint.get("level") == "clinical_scenario":
        stem_policy = """For clinical_scenario, `stem_safe_claims` records supported clinical theory and
constraints for constructing a new synthetic case, not a literal patient record.
Extract DSM-5 symptom patterns, duration thresholds, impairment and exclusions,
plus relevant textbook mechanisms or clinical context when explicitly supplied.
Keep source-specific frameworks clear; do not attribute a textbook's ICD criteria
to DSM-5. Separate diagnostic minimums from possible fictional case values.
The writer may invent patient demographics, context, history, symptoms, course,
frequency and stated assessment findings consistent with these constraints.
Do not prohibit synthetic details simply because no source patient has them.
Partial criteria do not establish a diagnosis; prohibit unsupported clinical
interpretations instead. case_summary must faithfully summarize the generated
case, including its fictional details, without introducing additional facts."""
    return f"""You are the evidence planner for a Vietnamese psychology MCQ.
Extract only claims that are directly stated or directly entailed by the supplied
evidence. Do not add therapy mechanisms, diagnoses, causal explanations, or
medical effects that are merely plausible background knowledge.

Blueprint: {json.dumps(blueprint, ensure_ascii=False)}
Evidence: {json.dumps(evidence, ensure_ascii=False)}

Return JSON only:
{{"supported_claims":[{{"claim":"Vietnamese claim safe for the keyed option","chunk_ids":["retrieved_chunk_id"]}}],"stem_safe_claims":[{{"claim":"Vietnamese factual detail safe for the question stem or example","chunk_ids":["retrieved_chunk_id"]}}],"prohibited_inferences":["claim not directly supported"]}}
Every chunk_id must be from the supplied evidence. Keep claims concise and do
not expose chain-of-thought.
{stem_policy}"""
