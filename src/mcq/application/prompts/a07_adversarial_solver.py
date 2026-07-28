"""A07 Adversarial Solver prompt."""

import json
from typing import Dict


def render(*, question: str, options: Dict[str, str], difficulty: str) -> str:
    options_text = json.dumps(options, ensure_ascii=False, indent=2)
    return f"""You are A07, a clever psychology student taking a multiple-choice exam.
Your task is to try to guess the correct answer WITHOUT seeing any study material.
Use your intuition, common medical knowledge, or test-taking strategies (e.g., picking the longest option, spotting absolute words like 'always', or finding the outlier).

Question (Target difficulty: {difficulty}):
{question}

Options:
{options_text}

Return JSON only: {{"selected_option": "A|B|C|D", "confidence": "high|low", "surface_cue_type":"none|length|absolute_wording|unique_qualification|grammar|detail_imbalance|other", "surface_cue_evidence":"short quoted or comparative cue", "reasoning": "..."}}.
Set confidence="high" because of bad item design only when you can name a concrete surface cue in the options. If your confidence comes from psychology knowledge, the question's meaning, or a genuine evidence-based distinction rather than a surface cue, set surface_cue_type="none" even if you select an answer. Do not invent a cue.
If the options are well-balanced and require specific knowledge, set surface_cue_type="none" and confidence="low" unless a real surface cue is present.
"""
