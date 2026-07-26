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

Return JSON only: {{"selected_option": "A|B|C|D", "confidence": "high|low", "reasoning": "..."}}
If you can confidently guess the correct answer due to obvious clues, spurious correlations, or bad distractor design, set confidence to "high".
If you are completely guessing randomly because the options are well-balanced and require specific knowledge, set confidence to "low".
"""
