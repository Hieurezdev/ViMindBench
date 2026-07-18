"""Composition root: LangGraph is kept outside domain and application layers."""
from langgraph.graph import END, StateGraph
from .domain import MCQState
from .application.nodes.planning import select_anchor_node, curriculum_planner_node, context_retriever_node
from .application.nodes.generation import mcq_generator_node, prepare_regeneration_node
from .application.nodes.judging import evidence_judge_node, single_answer_judge_node, safety_bias_judge_node, quality_gate_node
from .application.nodes.clinical_context import dsm5_safety_context_node
from .application.nodes.learning import reflector_node, playbook_curator_node
from .application.nodes.collection import collect_node


def create_mcq_graph():
    graph = StateGraph(MCQState)
    nodes = (("select_anchor", select_anchor_node), ("plan", curriculum_planner_node), ("retrieve", context_retriever_node),
             ("generate", mcq_generator_node), ("prepare_regeneration", prepare_regeneration_node), ("evidence_judge", evidence_judge_node), ("single_answer_judge", single_answer_judge_node),
             ("dsm5_safety_context", dsm5_safety_context_node), ("safety_bias_judge", safety_bias_judge_node), ("quality_gate", quality_gate_node), ("reflect", reflector_node),
             ("curate", playbook_curator_node), ("collect", collect_node))
    for name, node in nodes:
        graph.add_node(name, node)
    graph.set_entry_point("select_anchor")
    graph.add_conditional_edges("select_anchor", lambda state: "plan" if state.get("anchor") else END, {"plan": "plan", END: END})
    for source, target in (("plan", "retrieve"), ("retrieve", "generate"), ("generate", "evidence_judge"),
                           ("evidence_judge", "single_answer_judge"), ("single_answer_judge", "dsm5_safety_context"), ("dsm5_safety_context", "safety_bias_judge"),
                           ("safety_bias_judge", "quality_gate"), ("reflect", "curate"), ("curate", "collect")):
        graph.add_edge(source, target)
    graph.add_conditional_edges("quality_gate", route_after_quality_gate, {"prepare_regeneration": "prepare_regeneration", "reflect": "reflect"})
    graph.add_edge("prepare_regeneration", "generate")
    graph.add_conditional_edges("collect", lambda state: "select_anchor" if state["iteration_count"] < state["max_iterations"] else END, {"select_anchor": "select_anchor", END: END})
    return graph.compile()


def route_after_quality_gate(state: MCQState) -> str:
    """Retry the same blueprint/evidence before quarantining a failed item."""
    if state.get("verdict") == "verified":
        return "reflect"
    retries_used = max(0, state.get("generation_attempt", 0) - 1)
    return "prepare_regeneration" if retries_used < state.get("max_generation_retries", 2) else "reflect"
