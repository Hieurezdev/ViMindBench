
import random
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    first_filter_node,
    select_anchor_node,
    retrieve_node,
    retrieve_negative_node,
    generate_simple_qa_node,
    generate_reasoning_node,
    validate_qa_node,
    verify_grounding_node,
    parse_steps_node,
    verify_single_step_node,
    refine_single_step_node,
    check_more_questions_node,
    increment_step_node,
    check_format_node,
    format_output_node
)


def create_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("first_filter", first_filter_node)
    workflow.add_node("select_anchor", select_anchor_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("retrieve_negative", retrieve_negative_node)
    workflow.add_node("simple_qa", generate_simple_qa_node)
    workflow.add_node("generate_reasoning", generate_reasoning_node)
    workflow.add_node("validate_qa", validate_qa_node)
    workflow.add_node("verify_grounding", verify_grounding_node)
    workflow.add_node("parse_steps", parse_steps_node)
    workflow.add_node("verify_step", verify_single_step_node)
    workflow.add_node("refine_step", refine_single_step_node)
    workflow.add_node("check_more", check_more_questions_node)
    workflow.add_node("check_format", check_format_node)
    workflow.add_node("format_output", format_output_node)
    workflow.add_node("increment_step", increment_step_node)

    # Define Flow
    workflow.set_entry_point("first_filter")

    # First Filter → Select Anchor (pass-through; flow choice is already in state)
    def route_after_filter(state: AgentState):
        return "select_anchor"

    workflow.add_conditional_edges(
        "first_filter",
        route_after_filter,
        {"select_anchor": "select_anchor"}
    )

    # After anchor selection: simple_qa uses anchor directly, reasoning randomly picks retrieve mode
    def route_after_anchor(state: AgentState):
        if not state['is_reasoning_flow']:
            return "simple_qa"
        # Reasoning: randomly pick similar (retrieve) or opposing (retrieve_negative)
        return "retrieve" if random.random() < 0.5 else "retrieve_negative"

    workflow.add_conditional_edges(
        "select_anchor",
        route_after_anchor,
        {
            "retrieve": "retrieve",
            "retrieve_negative": "retrieve_negative",
            "simple_qa": "simple_qa"
        }
    )

    # Both retrieval paths lead to reasoning generator
    workflow.add_edge("retrieve", "generate_reasoning")
    workflow.add_edge("retrieve_negative", "generate_reasoning")

    # Both generation paths go through validate_qa
    workflow.add_edge("simple_qa", "validate_qa")
    workflow.add_edge("generate_reasoning", "validate_qa")

    # Routing sau validate_qa: Đẩy qua verify_grounding
    def route_after_validate(state: AgentState):
        passed = state.get('qa_validation_passed', False)
        attempts = state.get('qa_validation_attempts', 0)
        is_reasoning = state.get('is_reasoning_flow', False)

        if passed:
            # Nếu định dạng QA ok, đi kiểm tra tính factual
            return "verify_grounding"
        else:
            if attempts < 2:
                print(f"[validate_qa] Thử lại lần {attempts + 1}/2...")
                return "generate_reasoning" if is_reasoning else "simple_qa"
            else:
                print("[validate_qa] Hết lần thử. Bỏ qua câu hỏi này.")
                return "check_more"

    workflow.add_conditional_edges(
        "validate_qa",
        route_after_validate,
        {
            "verify_grounding": "verify_grounding",
            "check_more": "check_more",
            "generate_reasoning": "generate_reasoning",
            "simple_qa": "simple_qa"
        }
    )

    # Routing SAU verify_grounding
    def route_after_grounding(state: AgentState):
        passed = state.get('grounding_passed', False)
        attempts = state.get('grounding_attempts', 0)
        is_reasoning = state.get('is_reasoning_flow', False)

        if passed:
            if is_reasoning:
                return "parse_steps"
            else:
                return "format_output"
        else:
            if attempts < 2:
                print(f"[verify_grounding] Thử lại lần {attempts + 1}/2 do lệch tài liệu...")
                return "generate_reasoning" if is_reasoning else "simple_qa"
            else:
                print("[verify_grounding] Hết lần thử. Bỏ qua câu hỏi này.")
                return "check_more"

    workflow.add_conditional_edges(
        "verify_grounding",
        route_after_grounding,
        {
            "parse_steps": "parse_steps",
            "format_output": "format_output",
            "check_more": "check_more",
            "generate_reasoning": "generate_reasoning",
            "simple_qa": "simple_qa"
        }
    )

    # Parse Steps → Verify
    workflow.add_edge("parse_steps", "verify_step")

    # Conditional after single step verification
    def route_after_step_verification(state: AgentState):
        current_idx = state['current_step_index']
        retry_count = state.get('step_retry_count', 0)
        step_results = state.get('step_verification_results', [])
        reasoning_steps = state.get('reasoning_steps', [])

        if current_idx >= len(step_results) or current_idx >= len(reasoning_steps):
            return "check_format"

        if not step_results[current_idx]:
            if retry_count < 3:
                return "refine_step"

        if current_idx + 1 < len(state['reasoning_steps']):
            return "increment_step"
        else:
            return "check_format"

    workflow.add_conditional_edges(
        "verify_step",
        route_after_step_verification,
        {
            "refine_step": "refine_step",
            "increment_step": "increment_step",
            "check_format": "check_format"
        }
    )

    workflow.add_edge("refine_step", "verify_step")
    workflow.add_edge("increment_step", "verify_step")
    workflow.add_edge("check_format", "format_output")
    workflow.add_edge("format_output", "check_more")

    # Check if more questions are needed
    def route_after_check_more(state: AgentState):
        if state['iteration_count'] < state['max_iterations']:
            return "first_filter"
        else:
            return END

    workflow.add_conditional_edges(
        "check_more",
        route_after_check_more,
        {
            "first_filter": "first_filter",
            END: END
        }
    )

    return workflow.compile()
