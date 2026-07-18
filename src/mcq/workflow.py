"""Composition root: LangGraph is kept outside domain and application layers."""
import logging
from typing import Any, Callable, Dict
from langgraph.graph import END, StateGraph
from .domain import MCQState
from .application.nodes.planning import select_anchor_node, curriculum_planner_node, context_retriever_node
from .application.nodes.generation import mcq_generator_node, prepare_regeneration_node
from .application.nodes.judging import evidence_judge_node, single_answer_judge_node, safety_bias_judge_node, quality_gate_node
from .application.nodes.clinical_context import dsm5_safety_context_node
from .application.nodes.learning import reflector_node, playbook_curator_node
from .application.nodes.collection import collect_node
from .application.nodes.persistence import flush_outputs_node

logger = logging.getLogger("mcq.workflow")


def _trace_node(name: str, node: Callable[[MCQState], Dict[str, Any]]) -> Callable[[MCQState], Dict[str, Any]]:
    """Log node boundaries without serializing prompts, secrets, or full evidence."""
    def traced(state: MCQState) -> Dict[str, Any]:
        logger.info("[%s] start | iteration=%s attempt=%s", name, state.get("iteration_count", 0) + 1, state.get("generation_attempt", 0))
        update = node(state)
        summary = []
        if "evidence_docs" in update: summary.append(f"evidence={len(update['evidence_docs'])}")
        if "dsm5_safety_docs" in update: summary.append(f"dsm5_safety={len(update['dsm5_safety_docs'])}")
        if "verdict" in update: summary.append(f"verdict={update['verdict']}")
        if "quarantine_reason" in update: summary.append(f"issues={len(update['quarantine_reason'])}")
        if "playbook_delta" in update: summary.append(f"playbook_delta={len(update['playbook_delta'])}")
        logger.info("[%s] done%s", name, f" | {', '.join(summary)}" if summary else "")
        return update
    return traced


def create_mcq_graph():
    graph = StateGraph(MCQState)
    nodes = (("select_anchor", select_anchor_node), ("plan", curriculum_planner_node), ("retrieve", context_retriever_node),
             ("generate", mcq_generator_node), ("prepare_regeneration", prepare_regeneration_node), ("evidence_judge", evidence_judge_node), ("single_answer_judge", single_answer_judge_node),
             ("dsm5_safety_context", dsm5_safety_context_node), ("safety_bias_judge", safety_bias_judge_node), ("quality_gate", quality_gate_node), ("reflect", reflector_node),
             ("curate", playbook_curator_node), ("collect", collect_node), ("flush_outputs", flush_outputs_node))
    for name, node in nodes:
        graph.add_node(name, _trace_node(name, node))
    graph.set_entry_point("select_anchor")
    graph.add_conditional_edges("select_anchor", lambda state: "plan" if state.get("anchor") else END, {"plan": "plan", END: END})
    for source, target in (("plan", "retrieve"), ("retrieve", "generate"), ("generate", "evidence_judge"),
                           ("evidence_judge", "single_answer_judge"), ("single_answer_judge", "dsm5_safety_context"), ("dsm5_safety_context", "safety_bias_judge"),
                           ("safety_bias_judge", "quality_gate"), ("reflect", "curate"), ("curate", "collect"), ("collect", "flush_outputs")):
        graph.add_edge(source, target)
    graph.add_conditional_edges("quality_gate", route_after_quality_gate, {"prepare_regeneration": "prepare_regeneration", "reflect": "reflect"})
    graph.add_edge("prepare_regeneration", "generate")
    graph.add_conditional_edges("flush_outputs", lambda state: "select_anchor" if state["iteration_count"] < state["max_iterations"] else END, {"select_anchor": "select_anchor", END: END})
    return graph.compile()


def route_after_quality_gate(state: MCQState) -> str:
    """Retry the same blueprint/evidence before quarantining a failed item."""
    if state.get("verdict") == "verified":
        logger.info("[quality_gate] route=reflect | verified")
        return "reflect"
    retries_used = max(0, state.get("generation_attempt", 0) - 1)
    route = "prepare_regeneration" if retries_used < state.get("max_generation_retries", 2) else "reflect"
    logger.info("[quality_gate] route=%s | retries_used=%s/%s", route, retries_used, state.get("max_generation_retries", 2))
    return route
