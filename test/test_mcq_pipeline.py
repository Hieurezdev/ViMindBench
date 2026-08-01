"""Offline unit tests for the evidence-gated MCQ pipeline.

No test in this module opens MongoDB or calls an LLM endpoint.
"""

import json
import os
import builtins
import tempfile
import unittest
from collections import OrderedDict
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch

from main import load_next_record_id, parse_levels
from src.mcq.application.nodes.collection import collect_node
from src.mcq.application.nodes.generation import _hard_guard_report, _normalize_evidence_ref_ids, _request_generation_json, mcq_generator_node
from src.mcq.application.nodes import generation
from src.mcq.application.nodes.judging import _validate_single_answer_report, adversarial_solver_node, consolidate_judge_reports_node, quality_gate_node
from src.mcq.application.nodes import judging
from src.mcq.application.nodes.planning import context_retriever_node, curriculum_planner_node
from src.mcq.application.nodes import planning
from src.mcq.application.nodes.clinical_context import dsm5_safety_context_node
from src.mcq.application.emobench import judge_context, normalize_blueprint_emobench, validate_judge_report
from src.mcq.application.nodes.learning import _update_counters, playbook_curator_node
from src.mcq.application.nodes import learning
from src.mcq.application.prompts import a01_curriculum, a03_mcq, a05_single_answer_judge
from src.mcq.application.failure_memory import (
    record_judge_failures,
    retrieve_similar_failures,
)
from src.mcq.workflow import create_mcq_graph, route_after_judge_baseline, route_after_quality_gate
from src.mcq.application.nodes.baseline import direct_blueprint_node, direct_context_node
from src.mcq.application.nodes.persistence import flush_outputs_node
from src.mcq.infrastructure.evidence import document_tier, select_eligible_documents
from src.mcq.infrastructure import llm_gateway
from src.mcq.evaluation import evaluate_experiment, render_markdown
from src.retriever import MongoDBRetriever


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
    def test_a01_prompt_separates_reference_data_from_its_output_contract(self) -> None:
        prompt = a01_curriculum.render(
            level="theory",
            difficulty="easy",
            title="Tài liệu có thể chứa JSON",
            summary='{"question_vietnamese": "Không phải output A01"}',
            playbook="[str-00001] helpful=0 harmful=0 :: Quy tắc.",
        )
        self.assertIn("<SOURCE_DATA>", prompt)
        self.assertIn("<PLAYBOOK_DATA>", prompt)
        self.assertIn("Do not generate an MCQ", prompt)
        self.assertIn("OUTPUT CONTRACT", prompt)
        self.assertLess(prompt.rfind("OUTPUT CONTRACT"), prompt.rfind("Return the blueprint JSON now."))

    def test_a01_schema_requires_the_flat_blueprint_contract(self) -> None:
        schema = a01_curriculum.response_schema(level="theory", difficulty="hard")
        body = schema["json_schema"]["schema"]
        self.assertTrue(schema["json_schema"]["strict"])
        self.assertFalse(body["additionalProperties"])
        self.assertEqual(body["properties"]["level"]["const"], "theory")
        self.assertEqual(body["properties"]["difficulty"]["const"], "hard")
        self.assertIn("retrieval_query", body["required"])

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

    def test_planner_uses_fallback_blueprint_when_model_returns_empty(self) -> None:
        state = {
            "anchor": {"title": "Lo âu xã hội", "summary": "Né tránh các tình huống xã hội."},
            "curriculum_levels": ["theory"],
            "curriculum_difficulties": ["medium"],
            "iteration_count": 0,
            "playbook": "",
        }
        with patch.object(planning, "request_json", side_effect=ValueError("empty response")):
            result = curriculum_planner_node(state)
        self.assertEqual(result["blueprint"]["topic"], "Lo âu xã hội")
        self.assertEqual(result["blueprint"]["retrieval_query"], "Lo âu xã hội Né tránh các tình huống xã hội.")
        self.assertEqual(result["blueprint"]["difficulty"], "medium")

    def test_planner_receives_the_full_playbook(self) -> None:
        playbook = "x" * 7_000 + " FULL-PLAYBOOK-TAIL"
        state = {
            "anchor": {"title": "Lo âu", "summary": "Tóm tắt"},
            "curriculum_levels": ["theory"],
            "curriculum_difficulties": ["easy"],
            "iteration_count": 0,
            "playbook": playbook,
        }
        captured = []
        with patch.object(
            planning,
            "request_json",
            side_effect=lambda prompt, **_: captured.append(prompt) or {},
        ):
            curriculum_planner_node(state)
        self.assertIn("FULL-PLAYBOOK-TAIL", captured[0])

    def test_planner_fills_a_missing_retrieval_query(self) -> None:
        state = {
            "anchor": {"title": "Lo âu xã hội", "summary": "Né tránh xã hội."},
            "curriculum_levels": ["theory"],
            "curriculum_difficulties": ["easy"],
            "iteration_count": 0,
            "playbook": "",
        }
        with patch.object(planning, "request_json", return_value={"topic": "Lo âu"}):
            result = curriculum_planner_node(state)
        self.assertEqual(
            result["blueprint"]["retrieval_query"],
            "Lo âu xã hội Né tránh xã hội.",
        )

    def test_planner_retries_an_incomplete_blueprint_once(self) -> None:
        state = {
            "anchor": {"title": "Lo âu", "summary": "Tóm tắt"},
            "curriculum_levels": ["theory"],
            "curriculum_difficulties": ["easy"],
            "iteration_count": 0,
            "playbook": "",
        }
        complete = {
            "topic": "Lo âu",
            "subtopic": "Né tránh",
            "skill": "phân tích",
            "retrieval_query": "lo âu né tránh",
            "clinical_guardrail": "Không chẩn đoán.",
            "playbook_bullet_ids": [],
        }
        with patch.object(planning, "request_json", side_effect=[{}, complete]) as request:
            result = curriculum_planner_node(state)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result["blueprint"]["topic"], "Lo âu")
        self.assertEqual(result["blueprint"]["retrieval_query"], "lo âu né tránh")

    def test_planner_accepts_gemini_blueprint_wrapper(self) -> None:
        state = {
            "anchor": {"title": "Lo âu", "summary": "Tóm tắt"},
            "curriculum_levels": ["theory"],
            "curriculum_difficulties": ["easy"],
            "iteration_count": 0,
            "playbook": "",
        }
        inner = {
            "topic": "Lo âu",
            "subtopic": "Né tránh",
            "skill": "phân tích",
            "retrieval_query": "lo âu né tránh",
            "clinical_guardrail": "Không chẩn đoán.",
            "playbook_bullet_ids": [],
        }
        with patch.object(
            planning, "request_json", return_value={"curriculum_planning_blueprint": inner}
        ) as request:
            result = curriculum_planner_node(state)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(result["blueprint"]["topic"], "Lo âu")

    def test_planner_preserves_valid_playbook_bullet_ids(self) -> None:
        state = {
            "anchor": {"title": "Lo âu", "summary": "Tóm tắt"},
            "curriculum_levels": ["theory"],
            "curriculum_difficulties": ["easy"],
            "iteration_count": 0,
            "playbook": "",
        }
        response = {
            "topic": "Lo âu",
            "subtopic": "Né tránh",
            "skill": "so sánh",
            "retrieval_query": "lo âu né tránh",
            "clinical_guardrail": "Không chẩn đoán.",
            "playbook_bullet_ids": ["evi-00002"],
        }
        with patch.object(planning, "request_json", return_value=response):
            result = curriculum_planner_node(state)
        self.assertEqual(result["blueprint"]["playbook_bullet_ids"], ["evi-00002"])

    def test_retriever_falls_back_to_anchor_when_query_is_missing(self) -> None:
        requested_queries = []
        retriever = SimpleNamespace(
            search=lambda query, k: requested_queries.append((query, k)) or []
        )
        state = {
            "anchor": {"title": "Lo âu", "summary": "Né tránh"},
            "blueprint": {"difficulty": "easy"},
        }
        with patch.object(builtins, "RETRIEVER", retriever, create=True):
            result = context_retriever_node(state)
        self.assertEqual(result["evidence_docs"], [])
        self.assertEqual(requested_queries, [("Lo âu Né tránh", 8)])


class EvidencePolicyTests(unittest.TestCase):
    @staticmethod
    def _cache_ready_retriever() -> MongoDBRetriever:
        retriever = MongoDBRetriever.__new__(MongoDBRetriever)
        retriever.cache_size = 4
        retriever._cache_lock = RLock()
        retriever._embedding_cache = OrderedDict()
        retriever._search_cache = OrderedDict()
        retriever._dsm5_search_cache = OrderedDict()
        return retriever

    def test_embedding_cache_reuses_normalized_query(self) -> None:
        retriever = self._cache_ready_retriever()
        calls = []
        retriever._generate_embedding = lambda query: calls.append(query) or [0.1, 0.2]
        self.assertEqual(retriever.generate_embedding("Stress   học tập"), [0.1, 0.2])
        self.assertEqual(retriever.generate_embedding(" stress học tập "), [0.1, 0.2])
        self.assertEqual(calls, ["Stress   học tập"])

    def test_retrieval_cache_reuses_query_and_k(self) -> None:
        retriever = self._cache_ready_retriever()
        retriever.collection = object()
        retriever.use_local_embedding = True
        retriever.local_model = object()
        calls = []
        expected = [doc("cached", "Tier 1")]
        retriever._vector_search = lambda query, k: calls.append((query, k)) or expected
        self.assertEqual(retriever.search("lo âu", k=8), expected)
        self.assertEqual(retriever.search("  LO ÂU ", k=8), expected)
        self.assertEqual(calls, [("lo âu", 8)])

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

    def test_dsm5_query_tolerates_list_valued_option(self) -> None:
        queries = []
        retriever = SimpleNamespace(
            generate_embedding=lambda query: queries.append(query) or [0.1],
            search_dsm5=lambda embedding, k: [],
        )
        state = {
            "blueprint": {"level": "clinical_scenario", "topic": "stress"},
            "mcq": {
                "question": "Câu hỏi lâm sàng",
                "options": {"A": "a", "B": ["b", "bổ sung"], "C": "c", "D": "d"},
            },
        }
        with patch.object(builtins, "RETRIEVER", retriever, create=True):
            result = dsm5_safety_context_node(state)
        self.assertEqual(result["dsm5_safety_docs"], [])
        self.assertIn("bổ sung", queries[0])


class JudgeGatewayTests(unittest.TestCase):
    def test_parallel_judge_reports_are_consolidated_without_losing_generation_failures(self) -> None:
        result = consolidate_judge_reports_node(
            {
                "judge_reports": {"generation": {"passed": False}},
                "evidence_report": {"passed": True},
                "single_answer_report": {"passed": True},
                "ei_safety_bias_report": {"passed": True},
                "adversarial_solver_report": {"passed": True},
            }
        )
        self.assertEqual(set(result["judge_reports"]), {"generation", "evidence", "single_answer", "ei_safety_bias", "adversarial_solver"})

    def test_parallel_judge_graph_compiles(self) -> None:
        self.assertIsNotNone(create_mcq_graph())
        for method in ("direct", "rag_only", "rag_judges"):
            self.assertIsNotNone(create_mcq_graph(method))

    def test_direct_control_uses_anchor_without_retrieval(self) -> None:
        state = {
            "anchor": {"chunk_id": "anchor-1", "title": "Lo âu", "summary": "Né tránh"},
            "curriculum_levels": ["theory"],
            "curriculum_difficulties": ["easy"],
            "iteration_count": 0,
        }
        blueprint = direct_blueprint_node(state)["blueprint"]
        docs = direct_context_node({**state, "blueprint": blueprint})["evidence_docs"]
        self.assertEqual(blueprint["topic"], "Lo âu")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["chunk_id"], "anchor-1")

    def test_medium_prompts_require_one_near_miss_distractor(self) -> None:
        blueprint = {"difficulty": "medium"}
        generator_prompt = a03_mcq.render(
            blueprint=blueprint,
            playbook="",
            evidence=[],
            evidence_plan={},
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
        self.assertIn("Avoid emphatic or absolute wording", generator_prompt)

    def test_hard_prompts_require_similar_options_and_multiple_sources(self) -> None:
        blueprint = {"difficulty": "hard", "min_evidence_refs": 2, "evidence_limit": 6}
        generator_prompt = a03_mcq.render(
            blueprint=blueprint,
            playbook="",
            evidence=[],
            evidence_plan={},
            judge_feedback=[],
        )
        judge_prompt = a05_single_answer_judge.render(
            blueprint=blueprint,
            mcq={},
            evidence=[],
            past_failures=[],
        )
        self.assertIn("2 to 6", generator_prompt)
        self.assertIn("All four\noptions must address the same core mechanism", generator_prompt)
        self.assertIn("Every factual detail in the question stem", generator_prompt)
        self.assertIn("Theo quan điểm của chuyên gia tâm lý", generator_prompt)
        self.assertIn("at least two cited chunks", judge_prompt)

    def test_single_answer_audit_requires_key_and_three_incorrect_distractors(self) -> None:
        mcq = {"answer": "B", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}}
        passing = _validate_single_answer_report(
            {
                "passed": True,
                "issues": [],
                "option_assessment": {"A": "incorrect", "B": "correct", "C": "incorrect", "D": "incorrect"},
            },
            mcq,
        )
        self.assertTrue(passing["passed"])

        ambiguous = _validate_single_answer_report(
            {
                "passed": True,
                "issues": [],
                "option_assessment": {"A": "incorrect", "B": "correct", "C": "ambiguous", "D": "incorrect"},
            },
            mcq,
        )
        self.assertFalse(ambiguous["passed"])
        self.assertIn("distractor_not_judged_incorrect", ambiguous["issues"])

    def test_adversarial_solver_requires_a_concrete_surface_cue(self) -> None:
        state = {
            "blueprint": {"difficulty": "hard"},
            "mcq": {"question": "Q", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": "B"},
        }
        with patch.object(
            judging,
            "request_judge_json",
            return_value={"selected_option": "B", "confidence": "high", "surface_cue_type": "none", "surface_cue_evidence": ""},
        ):
            no_cue = adversarial_solver_node(state)
        self.assertTrue(no_cue["judge_reports"]["adversarial_solver"]["passed"])

        with patch.object(
            judging,
            "request_judge_json",
            return_value={"selected_option": "B", "confidence": "high", "surface_cue_type": "absolute_wording", "surface_cue_evidence": "A/C/D use only absolute wording"},
        ):
            cue = adversarial_solver_node(state)
        self.assertFalse(cue["judge_reports"]["adversarial_solver"]["passed"])

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

    def test_judge_endpoint_failure_falls_back_to_primary_model(self) -> None:
        environment = {
            "OPENAI_BASE_URL": "http://generator.test/v1",
            "OPENAI_API_KEY": "generator-key",
            "MODEL_NAME": "generator-model",
            "JUDGE_OPENAI_BASE_URL": "http://judge.test/v1",
            "JUDGE_OPENAI_API_KEY": "judge-key",
            "JUDGE_MODEL_NAME": "judge-model",
        }
        with patch.dict(os.environ, environment, clear=False), patch.object(
            llm_gateway,
            "_request_json",
            side_effect=[ConnectionError("judge unavailable"), {"passed": True}],
        ) as request:
            result = llm_gateway.request_judge_json("judge prompt", max_tokens=123)
        self.assertEqual(result, {"passed": True})
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[1].kwargs, {
            "max_tokens": 123,
            "base_url": "http://generator.test/v1",
            "api_key": "generator-key",
            "model": "generator-model",
        })

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
    def test_hard_generation_can_use_judge_endpoint(self) -> None:
        with patch.dict(os.environ, {"HARD_GENERATION_USE_JUDGE": "true"}), patch.object(
            generation, "request_judge_json", return_value={"question": "hard"}
        ) as judge, patch.object(generation, "request_json") as primary:
            result = _request_generation_json({"difficulty": "hard"}, "prompt", max_tokens=321)
        self.assertEqual(result, {"question": "hard"})
        judge.assert_called_once_with("prompt", max_tokens=321)
        primary.assert_not_called()

    def test_generator_normalizes_object_evidence_references(self) -> None:
        self.assertEqual(
            _normalize_evidence_ref_ids(
                [{"chunk_id": "chunk-1"}, "chunk-2", {"chunk_id": "chunk-1"}]
            ),
            ["chunk-1", "chunk-2"],
        )

    def test_hard_guard_rejects_emphatic_wording_and_length_imbalance(self) -> None:
        report = _hard_guard_report(
            {"difficulty": "hard"},
            {
                "options": {
                    "A": "Một lựa chọn có nhiều chi tiết để mô tả cơ chế tâm lý được đề cập.",
                    "B": "Hoàn toàn sai.",
                    "C": "Một lựa chọn khác.",
                    "D": "Một lựa chọn cuối.",
                }
            },
        )
        self.assertFalse(report["passed"])
        self.assertIn("hard_guard:emphatic_wording", report["issues"])
        self.assertIn("hard_guard:option_length_imbalance", report["issues"])

    def test_generator_repairs_once_after_preflight_failure(self) -> None:
        first = {"question": "Bản nháp", "options": {}, "evidence_refs": ["chunk-1"]}
        repaired = {"question": "Bản sửa", "options": {}, "evidence_refs": ["chunk-1"]}
        state = {
            "blueprint": {"difficulty": "hard", "evidence_limit": 2},
            "evidence_docs": [doc("chunk-1", "Tier 1")],
            "playbook": "",
            "judge_feedback": [],
            "generation_attempt": 0,
        }
        with patch.dict(os.environ, {"A03_PREFLIGHT_ENABLED": "true"}), patch.object(
            generation,
            "request_json",
            side_effect=[{"supported_claims": [], "prohibited_inferences": []}, first, repaired],
        ) as generate, patch.object(
            generation,
            "request_judge_json",
            return_value={"passed": False, "issues": ["surface_cue:absolute_distractors"], "feedback": "Balance the options."},
        ):
            result = mcq_generator_node(state)
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(result["mcq"]["question"], "Bản sửa")
        self.assertEqual(result["judge_feedback"][-1]["judge"], "a03_preflight")

    def test_generator_refusal_becomes_a_retryable_failure(self) -> None:
        state = {
            "blueprint": {"difficulty": "hard", "evidence_limit": 2},
            "evidence_docs": [doc("chunk-1", "Tier 1")],
            "playbook": "",
            "judge_feedback": [],
            "generation_attempt": 0,
        }
        with patch.object(
            generation,
            "request_json",
            side_effect=[{"supported_claims": []}, ValueError("Model did not return a JSON object: I cannot fulfill this request.")],
        ):
            result = mcq_generator_node(state)
        self.assertEqual(result["mcq"], {})
        self.assertEqual(result["judge_reports"]["generation"]["issues"], ["generator_refusal"])
        self.assertEqual(result["generation_attempt"], 1)

    def test_generator_receives_the_full_playbook(self) -> None:
        playbook = "## SUCCESSFUL STRATEGIES TO REPLICATE\n" + "x" * 200 + " FULL-PLAYBOOK-TAIL"
        state = {
            "blueprint": {"difficulty": "easy", "evidence_limit": 2},
            "evidence_docs": [doc("chunk-1", "Tier 1")],
            "playbook": playbook,
            "judge_feedback": [],
            "generation_attempt": 0,
        }
        captured = []
        with patch.object(generation, "_build_evidence_plan", return_value={}), patch.object(
            generation,
            "_generate_mcq",
            side_effect=lambda **kwargs: captured.append(kwargs["playbook"]) or {
                "question": "Câu hỏi",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "answer": "A",
                "evidence_refs": ["chunk-1"],
            },
        ), patch.object(generation, "_preflight_report", return_value={"passed": True}):
            mcq_generator_node(state)
        self.assertEqual(captured, [playbook])

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

    def test_quality_gate_quarantines_non_string_option(self) -> None:
        state = passing_state()
        state["mcq"]["options"]["B"] = ["b"]
        result = quality_gate_node(state)
        self.assertEqual(result["verdict"], "quarantine")
        self.assertIn("invalid_option_text", result["quarantine_reason"])

    def test_quality_gate_quarantines_list_of_option_objects_without_crashing(self) -> None:
        state = passing_state()
        state["mcq"]["options"] = [
            {"label": "A", "text": "a"},
            {"label": "B", "text": "b"},
        ]
        result = quality_gate_node(state)
        self.assertEqual(result["verdict"], "quarantine")
        self.assertIn("invalid_options", result["quarantine_reason"])

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

    def test_quality_gate_requires_two_citations_for_hard_items(self) -> None:
        state = passing_state()
        state["blueprint"]["difficulty"] = "hard"
        state["mcq"]["evidence_refs"] = ["DSM5-DEP-014"]
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

    def test_curator_skips_embedding_duplicate_common_mistake(self) -> None:
        state = {
            "playbook": (
                "## COMMON MISTAKES TO AVOID\n"
                "[err-00001] helpful=0 harmful=0 :: Avoid ambiguous answer keys.\n\n"
                "## OTHERS"
            ),
            "failure_memory": [{"issue": "single_answer:overlap"}] * 3,
        }
        retriever = SimpleNamespace(generate_embedding=lambda _: [1.0, 0.0])
        with patch.dict(
            os.environ,
            {"PLAYBOOK_REPEAT_THRESHOLD": "3", "PLAYBOOK_SIMILARITY_THRESHOLD": "0.8"},
        ), patch.object(learning, "_notebook_rule", return_value="Check answer ambiguity before generation."), patch.object(
            builtins, "RETRIEVER", retriever, create=True
        ):
            result = playbook_curator_node(state)
        self.assertEqual(result["playbook_delta"], [])
        self.assertNotIn("Check answer ambiguity", result["playbook"])


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

    def test_collection_uses_the_persisted_next_record_id(self) -> None:
        state = passing_state()
        state["next_record_id"] = 42
        result = collect_node(state)
        self.assertEqual(result["verified_outputs"][0]["id"], "PSY-000042")
        self.assertEqual(result["next_record_id"], 43)

    def test_control_output_is_labeled_and_not_claimed_as_judged(self) -> None:
        state = passing_state()
        state["experiment_method"] = "rag_only"
        state["judge_reports"] = {}
        output = collect_node(state)["verified_outputs"][0]
        self.assertEqual(output["metadata"]["experiment_method"], "rag_only")
        self.assertEqual(output["validation"]["evidence_status"], "not_run")


class RQ2EvaluationTests(unittest.TestCase):
    def test_rq2_metrics_keep_pipeline_and_expert_scores_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verified_path = os.path.join(directory, "full.jsonl")
            quarantine_path = os.path.join(directory, "full.quarantine.jsonl")
            verified = passing_state()
            verified_record = collect_node(verified)["verified_outputs"][0]
            with open(verified_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(verified_record) + "\n")
            quarantined = dict(verified_record)
            quarantined["id"] = "PSY-000002"
            with open(quarantine_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(quarantined) + "\n")

            report = evaluate_experiment(
                {"full": verified_path},
                {"full": quarantine_path},
                bootstrap_samples=100,
            )
        result = report["methods"]["full"]
        self.assertEqual(result["n_completed"], 2)
        self.assertEqual(result["internal_metrics"]["record_yield"]["value"], 0.5)
        self.assertIsNone(result["expert_audit_metrics"]["overall_publishable"]["value"])
        self.assertIn("| full |", render_markdown(report))


class RecordIdContinuationTests(unittest.TestCase):
    def test_next_id_uses_the_largest_id_across_verified_and_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verified_path = os.path.join(directory, "items.jsonl")
            quarantine_path = os.path.join(directory, "items.quarantine.jsonl")
            with open(verified_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "PSY-000007"}) + "\n")
                handle.write("not-json\n")
            with open(quarantine_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "PSY-000021"}) + "\n")
            self.assertEqual(load_next_record_id([verified_path, quarantine_path]), 22)


class RegenerationRoutingTests(unittest.TestCase):
    def test_judge_baseline_routes_to_collect_without_reflection(self) -> None:
        self.assertEqual(
            route_after_judge_baseline(
                {"verdict": "verified", "generation_attempt": 1, "max_generation_retries": 2}
            ),
            "collect",
        )
        self.assertEqual(
            route_after_judge_baseline(
                {"verdict": "quarantine", "generation_attempt": 3, "max_generation_retries": 2}
            ),
            "collect",
        )

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

    def test_checkpoints_learning_state_every_ten_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            playbook_path = os.path.join(directory, "run.playbook.md")
            failure_memory_path = os.path.join(directory, "run.failure_memory.json")
            judge_memory_path = os.path.join(
                directory, "run.judge_failure_memory.json"
            )
            result = flush_outputs_node(
                {
                    "iteration_count": 10,
                    "output_flush_interval": 100,
                    "learning_checkpoint_interval": 10,
                    "playbook_path": playbook_path,
                    "failure_memory_path": failure_memory_path,
                    "judge_memory_path": judge_memory_path,
                    "playbook": "# Checkpointed playbook\n",
                    "failure_memory": [{"issue": "evidence:unsupported_key"}],
                    "judge_failure_memory": [{"judge": "A04"}],
                }
            )
            self.assertEqual(result, {"learning_checkpoint_count": 1})
            with open(playbook_path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "# Checkpointed playbook\n")
            with open(failure_memory_path, encoding="utf-8") as handle:
                self.assertIn("unsupported_key", handle.read())
            with open(judge_memory_path, encoding="utf-8") as handle:
                self.assertIn('"A04"', handle.read())


if __name__ == "__main__":
    unittest.main()
