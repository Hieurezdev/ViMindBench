"""Composition root: LangGraph is kept outside domain and application layers."""

import logging
from typing import Any, Callable, Dict
from langgraph.graph import END, StateGraph
from .domain import EXPERIMENT_METHODS, MCQState
from .application.nodes.planning import (
    select_anchor_node,
    curriculum_planner_node,
    context_retriever_node,
)
from .application.nodes.generation import mcq_generator_node, prepare_regeneration_node
from .application.nodes.judging import (
    adversarial_solver_parallel_node,
    consolidate_judge_reports_node,
    evidence_judge_parallel_node,
    quality_gate_node,
    safety_bias_judge_parallel_node,
    single_answer_judge_parallel_node,
)
from .application.nodes.clinical_context import dsm5_safety_context_node
from .application.nodes.learning import reflector_node, playbook_curator_node
from .application.nodes.collection import collect_node
from .application.nodes.persistence import flush_outputs_node
from .application.nodes.baseline import (
    baseline_accept_node,
    direct_blueprint_node,
    direct_context_node,
)

logger = logging.getLogger("mcq.workflow")


def _trace_node(
    name: str, node: Callable[[MCQState], Dict[str, Any]]
) -> Callable[[MCQState], Dict[str, Any]]:
    """Log node boundaries without serializing prompts, secrets, or full evidence."""

    def traced(state: MCQState) -> Dict[str, Any]:
        logger.info(
            "[%s] start | iteration=%s attempt=%s",
            name,
            state.get("iteration_count", 0) + 1,
            state.get("generation_attempt", 0),
        )
        update = node(state)
        summary = []
        if "evidence_docs" in update:
            summary.append(f"evidence={len(update['evidence_docs'])}")
        if "dsm5_safety_docs" in update:
            summary.append(f"dsm5_safety={len(update['dsm5_safety_docs'])}")
        if "verdict" in update:
            summary.append(f"verdict={update['verdict']}")
        if "quarantine_reason" in update:
            summary.append(f"issues={len(update['quarantine_reason'])}")
        if "playbook_delta" in update:
            summary.append(f"playbook_delta={len(update['playbook_delta'])}")
        logger.info("[%s] done%s", name, f" | {', '.join(summary)}" if summary else "")
        return update

    return traced


def create_mcq_graph(experiment_method: str = "full"):
    """Build one controlled RQ2 condition.

    The controls intentionally keep raw output for blind external audit. Only
    ``full`` updates playbook/failure memory; this prevents treatment leakage.
    """
    if experiment_method not in EXPERIMENT_METHODS:
        raise ValueError(f"Unknown experiment method: {experiment_method}")
    graph = StateGraph(MCQState)
    nodes = (
        ("select_anchor", select_anchor_node),
        ("plan", curriculum_planner_node),
        ("direct_plan", direct_blueprint_node),
        ("retrieve", context_retriever_node),
        ("direct_context", direct_context_node),
        ("generate", mcq_generator_node),
        ("baseline_accept", baseline_accept_node),
        ("prepare_regeneration", prepare_regeneration_node),
        ("evidence_judge", evidence_judge_parallel_node),
        ("single_answer_judge", single_answer_judge_parallel_node),
        ("dsm5_safety_context", dsm5_safety_context_node),
        ("safety_bias_judge", safety_bias_judge_parallel_node),
        ("adversarial_solver", adversarial_solver_parallel_node),
        ("consolidate_judge_reports", consolidate_judge_reports_node),
        ("quality_gate", quality_gate_node),
        ("reflect", reflector_node),
        ("curate", playbook_curator_node),
        ("collect", collect_node),
        ("flush_outputs", flush_outputs_node),
    )
    for name, node in nodes:
        graph.add_node(name, _trace_node(name, node))
    graph.set_entry_point("select_anchor")
    first_node = "direct_plan" if experiment_method == "direct" else "plan"
    graph.add_conditional_edges(
        "select_anchor",
        lambda state: first_node if state.get("anchor") else END,
        {first_node: first_node, END: END},
    )
    common_edges = [("collect", "flush_outputs")]
    if experiment_method == "direct":
        common_edges += [
            ("direct_plan", "direct_context"),
            ("direct_context", "generate"),
            ("generate", "baseline_accept"),
            ("baseline_accept", "collect"),
        ]
    elif experiment_method == "rag_only":
        common_edges += [
            ("plan", "retrieve"),
            ("retrieve", "generate"),
            ("generate", "baseline_accept"),
            ("baseline_accept", "collect"),
        ]
    else:
        common_edges += [
            ("plan", "retrieve"),
            ("retrieve", "generate"),
            ("generate", "evidence_judge"),
            ("generate", "single_answer_judge"),
            ("generate", "dsm5_safety_context"),
            ("generate", "adversarial_solver"),
            ("dsm5_safety_context", "safety_bias_judge"),
            ("consolidate_judge_reports", "quality_gate"),
        ]
        if experiment_method == "full":
            common_edges += [("reflect", "curate"), ("curate", "collect")]
    for source, target in common_edges:
        graph.add_edge(source, target)
    if experiment_method in {"rag_judges", "full"}:
        graph.add_edge(
            ["evidence_judge", "single_answer_judge", "safety_bias_judge", "adversarial_solver"],
            "consolidate_judge_reports",
        )
        if experiment_method == "full":
            graph.add_conditional_edges(
                "quality_gate",
                route_after_quality_gate,
                {"prepare_regeneration": "prepare_regeneration", "reflect": "reflect"},
            )
        else:
            graph.add_conditional_edges(
                "quality_gate",
                route_after_judge_baseline,
                {"prepare_regeneration": "prepare_regeneration", "collect": "collect"},
            )
        graph.add_conditional_edges(
            "prepare_regeneration",
            route_after_prepare_regeneration,
            {"plan": "plan", "generate": "generate"},
        )
    graph.add_conditional_edges(
        "flush_outputs",
        lambda state: (
            "select_anchor"
            if state["iteration_count"] < state["max_iterations"]
            else END
        ),
        {"select_anchor": "select_anchor", END: END},
    )
    return graph.compile()


def route_after_quality_gate(state: MCQState) -> str:
    """Retry within a bounded budget; preparation selects the repair scope."""
    if state.get("verdict") == "verified":
        logger.info("[quality_gate] route=reflect | verified")
        return "reflect"
    retries_used = max(0, state.get("generation_attempt", 0) - 1)
    route = (
        "prepare_regeneration"
        if retries_used < state.get("max_generation_retries", 2)
        else "reflect"
    )
    logger.info(
        "[quality_gate] route=%s | retries_used=%s/%s",
        route,
        retries_used,
        state.get("max_generation_retries", 2),
    )
    return route


def route_after_prepare_regeneration(state: MCQState) -> str:
    """Re-plan/retrieve only for evidence or blueprint alignment failures."""
    route = "plan" if state.get("regeneration_route") == "replan" else "generate"
    logger.info("[prepare_regeneration] route=%s", route)
    return route


def route_after_judge_baseline(state: MCQState) -> str:
    """Retry judges baseline but never enter reflection/playbook treatment."""
    if state.get("verdict") == "verified":
        return "collect"
    retries_used = max(0, state.get("generation_attempt", 0) - 1)
    return (
        "prepare_regeneration"
        if retries_used < state.get("max_generation_retries", 2)
        else "collect"
    )
