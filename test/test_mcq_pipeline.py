"""Offline unit tests for the evidence-gated MCQ pipeline.

No test in this module opens MongoDB or calls an LLM endpoint.
"""

import os
import builtins
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from main import parse_levels
from src.mcq.application.nodes.collection import collect_node
from src.mcq.application.nodes.generation import _normalize_evidence_ref_ids
from src.mcq.application.nodes.judging import quality_gate_node
from src.mcq.application.nodes.planning import context_retriever_node
from src.mcq.application.emobench import judge_context, normalize_blueprint_emobench, validate_judge_report
from src.mcq.application.nodes.learning import _update_counters, playbook_curator_node
from src.mcq.application.nodes import learning
from src.mcq.application.prompts import a03_mcq, a05_single_answer_judge
from src.mcq.application.failure_memory import (
    record_judge_failures,
    retrieve_similar_failures,
)
from src.mcq.workflow import route_after_quality_gate
from src.mcq.application.nodes.persistence import flush_outputs_node
from src.mcq.infrastructure.evidence import document_tier, select_eligible_documents
from src.mcq.infrastructure import llm_gateway


def doc(chunk_id: str, tier: str | None, score: float = 0.9) -> SimpleNamespace:
    metadata = {"chunk_id": chunk_id, "score": score, "title": "Test evidence"}
    if tier is not None:
        metadata["tier"] = tier
    return SimpleNamespace(metadata=metadata, page_content="Evidence excerpt")


def passing_state() -> dict:
    return {
        "iteration_count": 0,
        "anchor": {"chunk_id": "anchor-1"},
        "blueprint": {
            "topic": "stress",
            "subtopic": "family",
            "level": "clinical_scenario",
            "skill": "causal_reasoning",
            "difficulty": "medium",
        },
        "evidence_docs": [doc("DSM5-DEP-014", "Tier 1", 0.92)],
        "mcq": {
            "question": "Một người ...?",
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "answer": "B",
            "evidence_refs": ["DSM5-DEP-014"],
            "rationale_short": "B phù hợp với chunk.",
            "audit_steps": ["identify evidence", "match key", "eliminate distractors"],
            "distractor_analysis": {
                "A": "Không phù hợp",
                "C": "Đảo cơ chế",
                "D": "Thiếu dữ kiện",
            },
        },
        "judge_reports": {
            "evidence": {"passed": True},
            "single_answer": {"passed": True},
            "ei_safety_bias": {"passed": True},
        },
        "verified_outputs": [],
        "quarantine_outputs": [],
        "playbook_delta": [],
        "verdict": "verified",
        "quarantine_reason": [],
    }


class CurriculumLevelTests(unittest.TestCase):
    def test_default_levels_follow_curriculum_order(self) -> None:
        self.assertEqual(
            parse_levels(None),
            ["theory", "emotion", "educational_scenario", "clinical_scenario"],
        )

    def test_levels_are_deduplicated_in_requested_order(self) -> None:
        self.assertEqual(
            parse_levels("clinical_scenario,theory,clinical_scenario"),
            ["clinical_scenario", "theory"],
        )

    def test_invalid_level_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_levels("diagnosis")


class EvidencePolicyTests(unittest.TestCase):
    def test_explicit_tier_three_is_never_treated_as_legacy(self) -> None:
        self.assertEqual(document_tier(doc("t3", "Tier 3")), "Tier 3")

    def test_only_tier_one_and_two_are_accepted_in_strict_mode(self) -> None:
        with patch.dict(os.environ, {"ALLOW_UNTIERED_EVIDENCE": "false"}):
            eligible = select_eligible_documents(
                [
                    doc("t1", "Tier 1"),
                    doc("t2", "Tier 2"),
                    doc("t3", "Tier 3"),
                    doc("legacy", None),
                ]
            )
        self.assertEqual([item.metadata["chunk_id"] for item in eligible], ["t1", "t2"])

    def test_legacy_chunk_requires_explicit_compatibility_switch(self) -> None:
        with patch.dict(os.environ, {"ALLOW_UNTIERED_EVIDENCE": "true"}):
            eligible = select_eligible_documents([doc("legacy", None)])
        self.assertEqual([item.metadata["chunk_id"] for item in eligible], ["legacy"])

    def test_retrieval_depth_increases_with_difficulty(self) -> None:
        documents = [doc(f"chunk-{index}", "Tier 2") for index in range(6)]
        requested_k = []
        retriever = SimpleNamespace(search=lambda query, k: (requested_k.append(k), documents)[1])
        with patch.object(builtins, "RETRIEVER", retriever, create=True):
            medium = context_retriever_node({"blueprint": {"difficulty": "medium", "retrieval_query": "stress"}})
            hard = context_retriever_node({"blueprint": {"difficulty": "hard", "retrieval_query": "stress"}})
        self.assertEqual(requested_k, [16, 24])
        self.assertEqual(len(medium["evidence_docs"]), 4)
        self.assertEqual(len(hard["evidence_docs"]), 6)


class JudgeGatewayTests(unittest.TestCase):
    def test_medium_prompts_require_one_near_miss_distractor(self) -> None:
        blueprint = {"difficulty": "medium"}
        generator_prompt = a03_mcq.render(
            blueprint=blueprint,
            playbook="",
            evidence=[],
            judge_feedback=[],
        )
        judge_prompt = a05_single_answer_judge.render(
            blueprint=blueprint,
            mcq={},
            evidence=[],
            past_failures=[],
        )
        self.assertIn("one near-miss distractor", generator_prompt)
        self.assertIn("one near-miss distractor", judge_prompt)

    def test_hard_prompts_require_four_similar_options(self) -> None:
        blueprint = {"difficulty": "hard"}
        generator_prompt = a03_mcq.render(
            blueprint=blueprint,
            playbook="",
            evidence=[],
            judge_feedback=[],
        )
        judge_prompt = a05_single_answer_judge.render(
            blueprint=blueprint,
            mcq={},
            evidence=[],
            past_failures=[],
        )
        self.assertIn("All four options must be highly similar", generator_prompt)
        self.assertIn("all four options to be highly similar", judge_prompt)

    def test_json_parser_accepts_fenced_json_with_surrounding_prose(self) -> None:
        parsed = llm_gateway._parse_json_object(
            "Here is the result:\n```json\n{\"answer\": \"B\"}\n```\n"
        )
        self.assertEqual(parsed, {"answer": "B"})

    def test_judges_use_a_separate_configured_endpoint(self) -> None:
        environment = {
            "OPENAI_BASE_URL": "http://generator.test/v1",
            "OPENAI_API_KEY": "generator-key",
            "MODEL_NAME": "generator-model",
            "JUDGE_OPENAI_BASE_URL": "http://judge.test/v1",
            "JUDGE_OPENAI_API_KEY": "judge-key",
            "JUDGE_MODEL_NAME": "judge-model",
        }
        with patch.dict(os.environ, environment, clear=False), patch.object(
            llm_gateway, "_request_json", return_value={}
        ) as request:
            llm_gateway.request_judge_json("judge prompt", max_tokens=123)
        self.assertEqual(
            request.call_args.kwargs,
            {
                "max_tokens": 123,
                "base_url": "http://judge.test/v1",
                "api_key": "judge-key",
                "model": "judge-model",
            },
        )

    def test_insight_uses_a09_specific_endpoint(self) -> None:
        environment = {
            "OPENAI_BASE_URL": "http://generator.test/v1",
            "OPENAI_API_KEY": "generator-key",
            "MODEL_NAME": "generator-model",
            "INSIGHT_OPENAI_BASE_URL": "http://insight.test/v1",
            "INSIGHT_OPENAI_API_KEY": "insight-key",
            "INSIGHT_MODEL_NAME": "insight-model",
        }
        with patch.dict(os.environ, environment, clear=False), patch.object(
            llm_gateway, "_request_json", return_value={}
        ) as request:
            llm_gateway.request_insight_json("insight prompt", max_tokens=234)
        self.assertEqual(
            request.call_args.kwargs,
            {
                "max_tokens": 234,
                "base_url": "http://insight.test/v1",
                "api_key": "insight-key",
                "model": "insight-model",
            },
        )


class EmoBenchIntegrationTests(unittest.TestCase):
    def test_emotion_blueprint_is_normalized_to_an_eu_task(self) -> None:
        result = normalize_blueprint_emobench({"task": "EU", "eu_category": "emotional_cues"}, enabled=True)
        self.assertEqual(result, {"enabled": True, "task": "EU", "eu_category": "emotional_cues"})

    def test_ea_report_requires_every_task_criterion(self) -> None:
        blueprint = {"emobench": normalize_blueprint_emobench({"task": "EA", "relationship_type": "social", "problem_owner": "others", "question_type": "action"}, enabled=True)}
        context = judge_context(blueprint)
        report = {"passed": True, "issues": [], "emobench": {"task": "EA", "criteria": {name: "pass" for name in context["criteria"]}}}
        self.assertTrue(validate_judge_report(report, blueprint)["passed"])

        report["emobench"]["criteria"].pop("relationship_fit")
        checked = validate_judge_report(report, blueprint)
        self.assertFalse(checked["passed"])
        self.assertIn("emobench_missing_criterion:relationship_fit", checked["issues"])


class QualityAndPlaybookTests(unittest.TestCase):
    def test_generator_normalizes_object_evidence_references(self) -> None:
        self.assertEqual(
            _normalize_evidence_ref_ids(
                [{"chunk_id": "chunk-1"}, "chunk-2", {"chunk_id": "chunk-1"}]
            ),
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

    def test_quality_gate_quarantines_chinese_han_characters(self) -> None:
        state = passing_state()
        state["mcq"]["question"] = "Một người cảm thấy 焦虑 trong tình huống này?"
        result = quality_gate_node(state)
        self.assertEqual(result["verdict"], "quarantine")
        self.assertIn("contains_han_script", result["quarantine_reason"])

    def test_quality_gate_quarantines_hard_item_with_spurious_cues(self) -> None:
        state = passing_state()
        state["blueprint"]["difficulty"] = "hard"
        state["judge_reports"]["adversarial_solver"] = {
            "passed": False,
            "issues": ["adversarial:spurious_cues_found"],
        }
        result = quality_gate_node(state)
        self.assertEqual(result["verdict"], "quarantine")
        self.assertIn("adversarial_solver:adversarial:spurious_cues_found", result["quarantine_reason"])

    def test_quality_gate_requires_at_most_the_difficulty_limit_citations(self) -> None:
        state = passing_state()
        state["blueprint"]["difficulty"] = "medium"
        state["evidence_docs"] = [doc(f"chunk-{index}", "Tier 1") for index in range(5)]
        state["mcq"]["evidence_refs"] = [f"chunk-{index}" for index in range(5)]
        result = quality_gate_node(state)
        self.assertEqual(result["verdict"], "quarantine")
        self.assertIn("invalid_evidence_refs", result["quarantine_reason"])

    def test_quality_gate_quarantines_object_evidence_references_without_crashing(
        self,
    ) -> None:
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
        state = {
            "playbook": "## EVIDENCE & GROUNDING\n\n## COMMON MISTAKES TO AVOID\n\n## OTHERS",
            "failure_memory": [{"issue": "evidence:unsupported_key"}] * 3,
        }
        with patch.dict(
            os.environ,
            {"PLAYBOOK_REPEAT_THRESHOLD": "3", "INSIGHT_OPENAI_BASE_URL": ""},
        ):
            result = playbook_curator_node(state)
        self.assertEqual(len(result["playbook_delta"]), 1)
        self.assertIn("evidence:unsupported_key", result["playbook"])

    def test_curator_uses_insight_model_for_a_new_recurring_rule(self) -> None:
        state = {
            "playbook": "## EVIDENCE & GROUNDING\n\n## COMMON MISTAKES TO AVOID\n\n## OTHERS",
            "failure_memory": [{"issue": "evidence:unsupported_key", "feedback": "Answer key is not directly supported."}] * 3,
        }
        with patch.dict(
            os.environ,
            {"PLAYBOOK_REPEAT_THRESHOLD": "3", "INSIGHT_OPENAI_BASE_URL": "http://insight.test/v1"},
        ), patch.object(
            learning, "request_insight_json", return_value={"rule": "Require every answer key to be directly supported by its cited evidence."}
        ) as request:
            result = playbook_curator_node(state)
        self.assertEqual(request.call_count, 1)
        self.assertIn("Require every answer key", result["playbook"])


class JudgeFailureMemoryTests(unittest.TestCase):
    def test_retrieval_is_scoped_to_the_same_judge(self) -> None:
        memory = [
            {
                "judge": "evidence",
                "level": "clinical_scenario",
                "question": "Triệu chứng trầm cảm kéo dài",
                "issues": ["unsupported_key"],
                "feedback": "Need direct support.",
            },
            {
                "judge": "single_answer",
                "level": "clinical_scenario",
                "question": "Triệu chứng trầm cảm kéo dài",
                "issues": ["ambiguous"],
                "feedback": "Two answers fit.",
            },
        ]
        result = retrieve_similar_failures(
            memory,
            judge="evidence",
            level="clinical_scenario",
            question="Triệu chứng trầm cảm kéo dài bao lâu?",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["judge"], "evidence")

    def test_only_failed_judges_are_recorded(self) -> None:
        reports = {
            "evidence": {
                "passed": False,
                "issues": ["unsupported_key"],
                "feedback": "Missing direct evidence.",
            },
            "single_answer": {"passed": True},
        }
        memory = record_judge_failures(
            [],
            reports=reports,
            blueprint={"level": "theory", "topic": "memory"},
            mcq={"question": "Câu hỏi?"},
            iteration=2,
        )
        self.assertEqual(memory[0]["judge"], "evidence")
        self.assertEqual(memory[0]["issues"], ["unsupported_key"])


class OutputSchemaTests(unittest.TestCase):
    def test_verified_record_matches_public_schema(self) -> None:
        output = collect_node(passing_state())["verified_outputs"][0]
        self.assertEqual(output["id"], "PSY-000001")
        self.assertEqual(
            set(output),
            {
                "id",
                "question",
                "options",
                "answer",
                "evidence_refs",
                "distractor_analysis",
                "reasoning",
                "metadata",
                "split",
                "validation",
            },
        )
        self.assertEqual(output["validation"]["evidence_status"], "pass")
        self.assertNotIn("_audit", output)


class RegenerationRoutingTests(unittest.TestCase):
    def test_failed_first_attempt_retries(self) -> None:
        self.assertEqual(
            route_after_quality_gate(
                {
                    "verdict": "quarantine",
                    "generation_attempt": 1,
                    "max_generation_retries": 2,
                }
            ),
            "prepare_regeneration",
        )

    def test_failed_item_quarantines_after_retry_budget(self) -> None:
        self.assertEqual(
            route_after_quality_gate(
                {
                    "verdict": "quarantine",
                    "generation_attempt": 3,
                    "max_generation_retries": 2,
                }
            ),
            "reflect",
        )

    def test_verified_item_never_regenerates(self) -> None:
        self.assertEqual(
            route_after_quality_gate(
                {
                    "verdict": "verified",
                    "generation_attempt": 1,
                    "max_generation_retries": 2,
                }
            ),
            "reflect",
        )


class OutputCheckpointTests(unittest.TestCase):
    def test_flushes_only_at_the_five_item_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "verified.jsonl")
            quarantine_path = os.path.join(directory, "quarantine.jsonl")
            state = {
                "iteration_count": 4,
                "output_flush_interval": 5,
                "output_path": output_path,
                "quarantine_path": quarantine_path,
                "verified_outputs": [{"id": "PSY-1"}],
                "quarantine_outputs": [{"id": "PSY-2"}],
                "verified_flushed_count": 0,
                "quarantine_flushed_count": 0,
            }
            self.assertEqual(flush_outputs_node(state), {})
            state["iteration_count"] = 5
            result = flush_outputs_node(state)
            self.assertEqual(
                result, {"verified_flushed_count": 1, "quarantine_flushed_count": 1}
            )
            with open(output_path, encoding="utf-8") as handle:
                self.assertEqual(len(handle.readlines()), 1)


if __name__ == "__main__":
    unittest.main()
