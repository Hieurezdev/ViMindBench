
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    first_filter_node,
    select_anchor_node,
    retrieve_node,
    generate_simple_qa_node,
    generate_reasoning_node,
    parse_steps_node,
    verify_single_step_node,
    refine_single_step_node,
    check_more_questions_node,
    check_more_questions_node,
    increment_step_node,
    check_format_node
)

def create_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("first_filter", first_filter_node)
    workflow.add_node("select_anchor", select_anchor_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("simple_qa", generate_simple_qa_node)
    workflow.add_node("generate_reasoning", generate_reasoning_node)
    workflow.add_node("parse_steps", parse_steps_node)
    workflow.add_node("verify_step", verify_single_step_node)
    workflow.add_node("refine_step", refine_single_step_node)
    workflow.add_node("check_more", check_more_questions_node)
    workflow.add_node("check_format", check_format_node)
    
    # Define Flow
    workflow.set_entry_point("first_filter")
    
    # First Filter: Decide between Normal QA and Reasoning QA
    def route_after_filter(state: AgentState):
        if state['is_reasoning_flow']:
            return "select_anchor"  # Reasoning path
        else:
            return "select_anchor"  # Both paths need anchor selection
    
    workflow.add_conditional_edges(
        "first_filter",
        route_after_filter,
        {
            "select_anchor": "select_anchor"
        }
    )
    
    # After anchor selection, route based on flow
    def route_after_anchor(state: AgentState):
        if state['is_reasoning_flow']:
             return "retrieve"
        else:
             return "simple_qa"

    workflow.add_conditional_edges(
        "select_anchor",
        route_after_anchor,
        {
            "retrieve": "retrieve",
            "simple_qa": "simple_qa"
        }
    )
    
    # After retrieval, go to reasoning generator
    # (Simple QA is now routed directly from select_anchor, so retrieve only goes to reasoning)
    workflow.add_edge("retrieve", "generate_reasoning")
    
    # Simple QA goes directly to check_more
    workflow.add_edge("simple_qa", "check_more")
    
    # Reasoning path: Generate → Parse Steps → Verify
    workflow.add_edge("generate_reasoning", "parse_steps")
    workflow.add_edge("parse_steps", "verify_step")
    
    # Conditional after single step verification
    def route_after_step_verification(state: AgentState):
        current_idx = state['current_step_index']
        retry_count = state.get('step_retry_count', 0)
        
        # Check if current step passed
        if not state['step_verification_results'][current_idx]:
            # If failed, check retries
            if retry_count < 3:
                return "refine_step"  # Retry
            else:
                # Max retries reached, just accept it and move on
                # We need to artificially mark it as passed or just proceed logic
                # For simplicity, we proceed as if it passed (but result stays False in state if we want to log it)
                # Or we can treat it as 'done' for this step.
                pass
        
        # Step passed OR max retries reached
        if current_idx + 1 < len(state['reasoning_steps']):
            return "increment_step"  # Go to next step (via increment node)
        else:
            return "check_format"  # All steps verified, format check
    
    workflow.add_conditional_edges(
        "verify_step",
        route_after_step_verification,
        {
            "refine_step": "refine_step",
            "increment_step": "increment_step",
            "check_format": "check_format"
        }
    )
    
    # After refinement, update retry count
    workflow.add_edge("refine_step", "verify_step")
    
    # After incrementing step, verify the new step
    workflow.add_node("increment_step", increment_step_node)
    workflow.add_edge("increment_step", "verify_step")
    
    # After step verification loop, run format check
    workflow.add_edge("check_format", "check_more")
    
    # Check if more questions need to be generated
    def route_after_check_more(state: AgentState):
        if state['iteration_count'] < state['max_iterations']:
            return "first_filter"  # Generate another question
        else:
            return END  # Done
    
    workflow.add_conditional_edges(
        "check_more",
        route_after_check_more,
        {
            "first_filter": "first_filter",
            END: END
        }
    )
    
    return workflow.compile()
