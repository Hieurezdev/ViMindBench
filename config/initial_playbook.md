<!-- playbook_version: v0.2 -->

## STRATEGIES & INSIGHTS
[str-00001] helpful=0 harmful=0 :: Match the item level to the curriculum: theory, emotion, educational scenario, then clinical scenario. Do not force a clinical vignette when the evidence only supports a concept question.

## EVIDENCE & GROUNDING
[evi-00002] helpful=0 harmful=0 :: Key an answer only when one or more retrieved Tier 1/2 chunks directly support it. Cite only retrieved chunk_id values; do not use background knowledge as evidence.
[evi-00003] helpful=0 harmful=0 :: A relevant chunk is insufficient: reject an item if the cited excerpt cannot distinguish the keyed option from its distractors.
[evi-00004] helpful=0 harmful=0 :: Every MCQ must export one to three evidence_refs with chunk_id, relation=supports_answer, and support_strength. Keep citations in metadata, never in the question stem or option text. Write the question as recalled psychology knowledge; never mention or imply "ngữ cảnh", "tài liệu đã cho", "đoạn văn", a source, a citation, or supplied material.

## QUESTION & DISTRACTOR DESIGN
[qad-00004] helpful=0 harmful=0 :: Write exactly four mutually distinct options, keyed exactly A, B, C, and D, with one and only one clearly best answer. Never add, omit, merge, or use meta-options such as "all of the above" or "A and B". Each distractor must be plausible yet demonstrably less appropriate from the evidence.
[qad-00005] helpful=0 harmful=0 :: Audit every wrong option briefly. Do not reveal the key through unusual length, precision, certainty, or qualification.

## EMOTIONAL INTELLIGENCE & CONTEXT
[ei-00006] helpful=0 harmful=0 :: For emotion items, preserve the subject's perspective, distinguish emotion from cause, and choose context-sensitive empathetic responses rather than generic reassurance.

## CLINICAL SAFETY & BIAS
[cli-00007] helpful=0 harmful=0 :: For clinical scenarios, ask for the safest supportive or educational next step. Do not diagnose from sparse facts, prescribe medication, imply certainty, stigmatize, or ignore cultural context.

## COMMON MISTAKES TO AVOID

## OTHERS
