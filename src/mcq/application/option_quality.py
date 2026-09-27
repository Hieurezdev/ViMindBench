"""Shared deterministic surface-cue checks for generation and the quality gate."""

import re
from typing import TypedDict


ABSOLUTE_OPTION_WORDING = re.compile(
    r"\b(hoàn\s+toàn|tuyệt\s+đối|luôn\s+luôn|không\s+bao\s+giờ|duy\s+nhất|"
    r"chắc\s+chắn|triệt\s+để|tất\s+cả)\b",
    re.IGNORECASE,
)
META_OPTION_WORDING = re.compile(
    r"\b(tất\s+cả\s+(?:các\s+)?đáp\s+án\s+(?:trên|đúng)|"
    r"cả\s+[abcd]\s+(?:và|lẫn)\s+[abcd]|"
    r"[abcd]\s+(?:và|lẫn)\s+[abcd]\s+(?:đều\s+)?đúng)\b",
    re.IGNORECASE,
)


class OptionSurfaceReport(TypedDict):
    passed: bool
    issues: list[str]
    feedback: str
    word_counts: dict[str, int]


def option_surface_report(
    options: dict[str, str], *, max_word_gap: int
) -> OptionSurfaceReport:
    """Inspect already validated option strings without changing their content."""
    counts = {key: len(re.findall(r"\w+", text, re.UNICODE)) for key, text in options.items()}
    issues: list[str] = []
    details: list[str] = []
    if any(META_OPTION_WORDING.search(text) for text in options.values()):
        issues.append("surface_cue:meta_option")
        details.append("replace combined/all-of-the-above options with distinct plausible choices")
    if any(ABSOLUTE_OPTION_WORDING.search(text) for text in options.values()):
        issues.append("surface_cue:absolute_wording")
        details.append("remove giveaway absolutes and keep certainty balanced across A–D")
    if max(counts.values()) - min(counts.values()) > max_word_gap:
        issues.append("surface_cue:option_length_imbalance")
        details.append(
            f"word counts are {counts}; keep the gap at most {max_word_gap}; "
            "shorten long options or expand short options without changing their clinical meaning"
        )
    return {
        "passed": not issues,
        "issues": issues,
        "feedback": "; ".join(details),
        "word_counts": counts,
    }
