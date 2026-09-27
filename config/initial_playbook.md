<!-- playbook_version: v0.2 -->

## STRATEGIES & INSIGHTS
[str-00001] helpful=0 harmful=0 :: Prioritize synthetic clinical cases and concise case summaries by default. Respect an explicitly selected curriculum level. Clinical cases require both retrieved DSM-5 and Tier 1 textbook evidence; do not substitute a theory question when clinical evidence is missing.

## EVIDENCE & GROUNDING
[evi-00002] helpful=0 harmful=0 :: Key an answer only when one or more retrieved Tier 1/2 chunks directly support it. Cite only retrieved chunk_id values; do not use background knowledge as evidence.
[evi-00003] helpful=0 harmful=0 :: A relevant chunk is insufficient: reject an item if the cited excerpt cannot distinguish the keyed option from its distractors.
[evi-00004] helpful=0 harmful=0 :: Every MCQ must export evidence_refs with chunk_id, relation=supports_answer, and support_strength, up to the difficulty policy limit: easy=1–2, medium=1–4, hard=2–6. Clinical scenarios may invent patient details consistent with cited DSM-5 and textbook theory; cite clinical concepts and reasoning rather than exact fictional patient details. For other levels, each factual stem detail must be supported by a cited chunk. A hard key must synthesize directly supported claims from at least two distinct cited chunks. Public wording must never show chunk_id or a database citation marker. Use natural attribution; name an expert only when present in verified metadata.

## QUESTION & DISTRACTOR DESIGN
[qad-00004] helpful=0 harmful=0 :: Write exactly four mutually distinct options, keyed exactly A, B, C, and D, with one and only one clearly best answer. Never add, omit, merge, or use meta-options such as "all of the above" or "A and B". Each distractor must be plausible yet demonstrably less appropriate from the evidence.
[qad-00005] helpful=0 harmful=0 :: Audit every wrong option briefly. Do not reveal the key through unusual length, precision, certainty, or qualification.
[qad-00006] helpful=0 harmful=0 :: The public MCQ must be written in natural Vietnamese only. Never generate Chinese/Han characters, Chinese words, Chinese punctuation, or mixed Vietnamese-Chinese text in the question, options, rationale, or distractor analysis. If a retrieved source is multilingual, translate its meaning into Vietnamese rather than copying its surface form.
[qad-00007] helpful=0 harmful=0 :: For difficulty=hard, all four options must address the same core mechanism or decision and be plausible near-misses. Keep them comparable in length, grammatical form, specificity, and hedging; do not make the key uniquely nuanced, qualified, comprehensive, or obviously less absolute. Each distractor must be wrong on a small evidence-checkable distinction; only the key may synthesize the directly supported claims from at least two cited chunks. A hard item must require the intended psychological reasoning and evidence, not superficial elimination.
[qad-00008] helpful=0 harmful=0 :: For difficulty=medium, create exactly one near-miss distractor that closely resembles the key in mechanism, topic, or wording, yet is wrong because of one precise, evidence-checkable distinction. The other two distractors may be more clearly wrong but remain plausible. Never allow the near-miss to be equally correct or defensible.
[qad-00009] helpful=0 harmful=0 :: Avoid emphatic or absolute wording in every answer option, including "hoàn toàn", "tuyệt đối", "luôn luôn", "không bao giờ", "duy nhất", "chắc chắn", "triệt để", "tất cả", and "chỉ". Such wording must not be used as a shortcut to make distractors obviously wrong. Use it only when directly required by evidence and apply the same certainty level comparably across all options.

## EMOTIONAL INTELLIGENCE & CONTEXT
[ei-00006] helpful=0 harmful=0 :: For emotion items, preserve the subject's perspective, distinguish emotion from cause, and choose context-sensitive empathetic responses rather than generic reassurance.

## CLINICAL SAFETY & BIAS
[cli-00007] helpful=0 harmful=0 :: Create new hypothetical clinical cases: demographics, context, symptoms, history, duration, frequency and assessment findings may be synthetic when clinically coherent. Ground clinical theory and interpretation in both DSM-5 and textbooks; do not invent diagnostic thresholds, causal claims or medical effects. case_summary must faithfully compress the generated case without adding facts, diagnosis, treatment or the key. Ask about features, missing assessment information or safe educational next steps. Partial criteria do not establish a diagnosis; unmentioned exclusions remain unknown. Do not prescribe medication, imply certainty or stigmatize.

## COMMON MISTAKES TO AVOID

## OTHERS
