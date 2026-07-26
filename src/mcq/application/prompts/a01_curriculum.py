"""A01 Curriculum Planner prompt."""

from typing import Any


def render(*, level: str, difficulty: str, title: str, summary: str, playbook: str) -> str:
    difficulty_guideline = {
        "easy": (
            "Recall and Comprehension (Bloom's Level 1-2): Focus on identifying foundational psychological terms, "
            "definitions, core principles, and prominent theories. The cognitive demand is direct knowledge retrieval."
        ),
        "medium": (
            "Application and Analysis (Bloom's Level 3-4): Focus on applying psychological concepts to straightforward, "
            "unambiguous case vignettes. The cognitive demand requires analyzing behaviors, identifying the most likely "
            "explanation, or connecting theory to practice."
        ),
        "hard": (
            "Synthesis and Evaluation (Bloom's Level 5-6): Focus on complex clinical/educational reasoning. The cognitive "
            "demand involves differential analysis (distinguishing between closely related phenomena or overlapping signs), "
            "evaluating multi-faceted scenarios with confounding variables, or prioritizing the most effective therapeutic/"
            "supportive intervention when multiple options seem plausible."
        )
    }.get(difficulty, "")

    return f"""You are A01, a Vietnamese psychology curriculum planner.
Create exactly one MCQ blueprint at level '{level}' and difficulty '{difficulty}' from the source below.
Difficulty definition for '{difficulty}': {difficulty_guideline}

Level order is theory → emotion → educational_scenario → clinical_scenario.
Clinical questions must ask for the safest educational/supportive next step,
not diagnosis, medication, prognosis, or emergency advice.
Source title: {title}
Source summary: {summary}
ACE playbook (choose only relevant bullet IDs): {playbook}
Return JSON only: {{"level":"{level}","topic":"...","subtopic":"...","skill":"...",
"difficulty":"{difficulty}","num_options":4,"retrieval_query":"...",
"requires_emobench":true|false,"emobench":{{"task":"EU|EA",
"eu_category":"complex_emotions|emotional_cues|personal_beliefs_experiences|perspective_taking|null",
"relationship_type":"personal|social|null","problem_owner":"self|others|null",
"question_type":"response|action|null}},"clinical_guardrail":"...","playbook_bullet_ids":["str-00001"]}}.
For level emotion, choose exactly one EmoBench task: EU for identifying an
emotion/cause, or EA for selecting an effective response/action. For every other
level, set emobench to null."""
