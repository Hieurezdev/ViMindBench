"""Offline unit tests for the evidence-gated MCQ pipeline.

No test in this module opens MongoDB or calls an LLM endpoint.
"""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from main import parse_levels
from src.mcq.application.nodes.collection import collect_node
from src.mcq.application.nodes.generation import _normalize_evidence_ref_ids
from src.mcq.application.nodes.judging import quality_gate_node
from src.mcq.application.nodes.learning import _update_counters, playbook_curator_node
from src.mcq.application.failure_memory import record_judge_failures, retrieve_similar_failures
from src.mcq.workflow import route_after_quality_gate
from src.mcq.application.nodes.persistence import flush_outputs_node
from src.mcq.infrastructure.evidence import document_tier, select_eligible_documents


def doc(chunk_id: str, tier: str | None, score: float = 0.9) -> SimpleNamespace:
    metadata = {"chunk_id": chunk_id, "score": score, "title": "Test evidence"}
    if tier is not None:
        metadata["tier"] = tier
    return SimpleNamespace(metadata=metadata, page_content="Evidence excerpt")


def passing_state() -> dict:
    return {
        "iteration_count": 0,
        "anchor": {"chunk_id": "anchor-1"},
        "blueprint": {"topic": "stress", "subtopic": "family", "level": "clinical_scenario", "skill": "causal_reasoning", "difficulty": "medium"},
        "evidence_docs": [doc("DSM5-DEP-014", "Tier 1", 0.92)],
        "mcq": {
            "question": "Một người ...?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": "B",
            "evidence_refs": ["DSM5-DEP-014"], "rationale_short": "B phù hợp với chunk.",
            "audit_steps": ["identify evidence", "match key", "eliminate distractors"],
            "distractor_analysis": {"A": "Không phù hợp", "C": "Đảo cơ chế", "D": "Thiếu dữ kiện"},
        },
        "judge_reports": {"evidence": {"passed": True}, "single_answer": {"passed": True}, "ei_safety_bias": {"passed": True}},
        "verified_outputs": [], "quarantine_outputs": [], "playbook_delta": [], "verdict": "verified", "quarantine_reason": [],
    }


class CurriculumLevelTests(unittest.TestCase):
    def test_default_levels_follow_curriculum_order(self) -> None:
        self.assertEqual(parse_levels(None), ["theory", "emotion", "educational_scenario", "clinical_scenario"])

    def test_levels_are_deduplicated_in_requested_order(self) -> None:
        self.assertEqual(parse_levels("clinical_scenario,theory,clinical_scenario"), ["clinical_scenario", "theory"])

    def test_invalid_level_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_levels("diagnosis")


class EvidencePolicyTests(unittest.TestCase):
    def test_explicit_tier_three_is_never_treated_as_legacy(self) -> None:
        self.assertEqual(document_tier(doc("t3", "Tier 3")), "Tier 3")

    def test_only_tier_one_and_two_are_accepted_in_strict_mode(self) -> None:
        with patch.dict(os.environ, {"ALLOW_UNTIERED_EVIDENCE": "false"}):
            eligible = select_eligible_documents([doc("t1", "Tier 1"), doc("t2", "Tier 2"), doc("t3", "Tier 3"), doc("legacy", None)])
        self.assertEqual([item.metadata["chunk_id"] for item in eligible], ["t1", "t2"])

    def test_legacy_chunk_requires_explicit_compatibility_switch(self) -> None:
        with patch.dict(os.environ, {"ALLOW_UNTIERED_EVIDENCE": "true"}):
            eligible = select_eligible_documents([doc("legacy", None)])
        self.assertEqual([item.metadata["chunk_id"] for item in eligible], ["legacy"])


class QualityAndPlaybookTests(unittest.TestCase):
    def test_generator_normalizes_object_evidence_references(self) -> None:
        self.assertEqual(
            _normalize_evidence_ref_ids([{"chunk_id": "chunk-1"}, "chunk-2", {"chunk_id": "chunk-1"}]),
            ["chunk-1", "chunk-2"],
        )

    def test_quality_gate_accepts_complete_grounded_item(self) -> None:
        state = passing_state()
        self.assertEqual(quality_gate_node(state)["verdict"], "verified")

    def test_quality_gate_quarantines_unknown_evidence_reference(self) -> None:
        state = passing_state()
        state["mcq"]["evidence_refs"] = ["not-retrieved"]
        result = quality_gate_node(state)
        self.assertEqual(result["verdict"], "quarantine")
        self.assertIn("invalid_evidence_refs", result["quarantine_reason"])

    def test_quality_gate_requires_at_most_three_citations(self) -> None:
        state = passing_state()
        state["evidence_docs"] = [doc(f"chunk-{index}", "Tier 1") for index in range(4)]
        state["mcq"]["evidence_refs"] = [f"chunk-{index}" for index in range(4)]
        result = quality_gate_node(state)
        self.assertEqual(result["verdict"], "quarantine")
        self.assertIn("invalid_evidence_refs", result["quarantine_reason"])

    def test_quality_gate_quarantines_object_evidence_references_without_crashing(self) -> None:
        state = passing_state()
        state["mcq"]["evidence_refs"] = [{"chunk_id": "DSM5-DEP-014"}]
        result = quality_gate_node(state)
        self.assertEqual(result["verdict"], "quarantine")
        self.assertIn("invalid_evidence_refs", result["quarantine_reason"])

    def test_reflector_counter_updates_only_selected_bullet(self) -> None:
        playbook = "## STRATEGIES & INSIGHTS\n[str-00001] helpful=0 harmful=0 :: Rule\n[str-00002] helpful=0 harmful=0 :: Other"
        updated = _update_counters(playbook, ["str-00001"], "helpful")
        self.assertIn("[str-00001] helpful=1 harmful=0", updated)
        self.assertIn("[str-00002] helpful=0 harmful=0", updated)

    def test_curator_adds_rule_only_after_repeat_threshold(self) -> None:
        state = {"playbook": "## EVIDENCE & GROUNDING\n\n## COMMON MISTAKES TO AVOID\n\n## OTHERS", "failure_memory": [{"issue": "evidence:unsupported_key"}] * 3}
        with patch.dict(os.environ, {"PLAYBOOK_REPEAT_THRESHOLD": "3"}):
            result = playbook_curator_node(state)
        self.assertEqual(len(result["playbook_delta"]), 1)
        self.assertIn("evidence:unsupported_key", result["playbook"])


class JudgeFailureMemoryTests(unittest.TestCase):
    def test_retrieval_is_scoped_to_the_same_judge(self) -> None:
        memory = [
            {"judge": "evidence", "level": "clinical_scenario", "question": "Triệu chứng trầm cảm kéo dài", "issues": ["unsupported_key"], "feedback": "Need direct support."},
            {"judge": "single_answer", "level": "clinical_scenario", "question": "Triệu chứng trầm cảm kéo dài", "issues": ["ambiguous"], "feedback": "Two answers fit."},
        ]
        result = retrieve_similar_failures(memory, judge="evidence", level="clinical_scenario", question="Triệu chứng trầm cảm kéo dài bao lâu?")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["judge"], "evidence")

    def test_only_failed_judges_are_recorded(self) -> None:
        reports = {"evidence": {"passed": False, "issues": ["unsupported_key"], "feedback": "Missing direct evidence."}, "single_answer": {"passed": True}}
        memory = record_judge_failures([], reports=reports, blueprint={"level": "theory", "topic": "memory"}, mcq={"question": "Câu hỏi?"}, iteration=2)
        self.assertEqual(memory[0]["judge"], "evidence")
        self.assertEqual(memory[0]["issues"], ["unsupported_key"])


class OutputSchemaTests(unittest.TestCase):
    def test_verified_record_matches_public_schema(self) -> None:
        output = collect_node(passing_state())["verified_outputs"][0]
        self.assertEqual(output["id"], "PSY-000001")
        self.assertEqual(set(output), {"id", "question", "options", "answer", "evidence_refs", "distractor_analysis", "reasoning", "metadata", "split", "validation"})
        self.assertEqual(output["validation"]["evidence_status"], "pass")
        self.assertNotIn("_audit", output)


class RegenerationRoutingTests(unittest.TestCase):
    def test_failed_first_attempt_retries(self) -> None:
        self.assertEqual(route_after_quality_gate({"verdict": "quarantine", "generation_attempt": 1, "max_generation_retries": 2}), "prepare_regeneration")

    def test_failed_item_quarantines_after_retry_budget(self) -> None:
        self.assertEqual(route_after_quality_gate({"verdict": "quarantine", "generation_attempt": 3, "max_generation_retries": 2}), "reflect")

    def test_verified_item_never_regenerates(self) -> None:
        self.assertEqual(route_after_quality_gate({"verdict": "verified", "generation_attempt": 1, "max_generation_retries": 2}), "reflect")


class OutputCheckpointTests(unittest.TestCase):
    def test_flushes_only_at_the_five_item_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "verified.jsonl")
            quarantine_path = os.path.join(directory, "quarantine.jsonl")
            state = {"iteration_count": 4, "output_flush_interval": 5, "output_path": output_path, "quarantine_path": quarantine_path, "verified_outputs": [{"id": "PSY-1"}], "quarantine_outputs": [{"id": "PSY-2"}], "verified_flushed_count": 0, "quarantine_flushed_count": 0}
            self.assertEqual(flush_outputs_node(state), {})
            state["iteration_count"] = 5
            result = flush_outputs_node(state)
            self.assertEqual(result, {"verified_flushed_count": 1, "quarantine_flushed_count": 1})
            with open(output_path, encoding="utf-8") as handle:
                self.assertEqual(len(handle.readlines()), 1)


if __name__ == "__main__":
    unittest.main()
