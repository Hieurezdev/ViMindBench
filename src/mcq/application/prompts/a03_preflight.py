"""Fast preflight rubric before the full A04–A07 evaluation chain."""

import json
from typing import Any, Dict, List


def render(*, blueprint: Dict[str, Any], mcq: Dict[str, Any], evidence: List[Dict[str, Any]]) -> str:
    return f"""You are an A03 preflight critic for a Vietnamese psychology MCQ.
Check only these release blockers before the item reaches the full judge chain:
1. The declared answer is directly supported by its cited evidence, without an
   added mechanism or claim.
2. There is exactly one best answer and three incorrect distractors.
3. For every difficulty, every factual detail in the question stem, vignette, or
   example is directly supported by a chunk_id included in the item's
   evidence_refs. Reject invented or uncited scenario details even when the key
   itself is supported. Natural language attribution is allowed, but raw IDs,
   bracketed citation markers, and fabricated named experts are not.
4. For difficulty=hard, no answer is discoverable solely from length, absolute
   wording, unique nuance, or qualification; all four options are balanced,
   address the same mechanism, and are plausible near-misses. The key must cite
   at least two chunks and synthesize only directly supported claims.

Blueprint: {json.dumps(blueprint, ensure_ascii=False)}
MCQ: {json.dumps(mcq, ensure_ascii=False)}
Evidence: {json.dumps(evidence, ensure_ascii=False)}

Return JSON only: {{"passed":true|false,"issues":["unsupported_claim:<short>","unsupported_stem_detail:<short>","invalid_public_attribution:<short>","surface_cue:<short>","multiple_best_answers"],"feedback":"specific repair instruction"}}.
Do not use background knowledge. If uncertain, fail rather than invent support."""
