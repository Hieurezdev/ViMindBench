"""Domain state and constants; no OpenAI, MongoDB, or LangGraph wiring."""

from typing import Any, Dict, List, Optional, TypedDict
from langchain_core.documents import Document

LEVELS = ("theory", "emotion", "educational_scenario", "clinical_scenario")
RETRIEVAL_DEPTH_BY_DIFFICULTY = {
    "easy": {"candidate_k": 8, "evidence_limit": 2, "min_evidence_refs": 1},
    "medium": {"candidate_k": 16, "evidence_limit": 4, "min_evidence_refs": 1},
    "hard": {"candidate_k": 24, "evidence_limit": 6, "min_evidence_refs": 2},
}


class MCQState(TypedDict, total=False):
    iteration_count: int
    next_record_id: int
    max_iterations: int
    generation_attempt: int
    max_generation_retries: int
    output_path: str
    quarantine_path: str
    output_flush_interval: int
    learning_checkpoint_interval: int
    playbook_path: str
    failure_memory_path: str
    judge_memory_path: str
    verified_flushed_count: int
    quarantine_flushed_count: int
    learning_checkpoint_count: int
    curriculum_levels: List[str]
    curriculum_difficulties: List[str]
    used_anchor_ids: List[str]
    anchor: Optional[Dict[str, Any]]
    blueprint: Dict[str, Any]
    evidence_docs: List[Document]
    dsm5_safety_docs: List[Document]
    mcq: Dict[str, Any]
    judge_reports: Dict[str, Dict[str, Any]]
    evidence_report: Dict[str, Any]
    single_answer_report: Dict[str, Any]
    ei_safety_bias_report: Dict[str, Any]
    adversarial_solver_report: Dict[str, Any]
    judge_feedback: List[Dict[str, Any]]
    verdict: str
    quarantine_reason: List[str]
    verified_outputs: List[Dict[str, Any]]
    quarantine_outputs: List[Dict[str, Any]]
    failure_memory: List[Dict[str, Any]]
    judge_failure_memory: List[Dict[str, Any]]
    playbook: str
    playbook_delta: List[Dict[str, Any]]
    last_saved_count: int
