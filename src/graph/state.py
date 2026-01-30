
from typing import List, Dict, Any, TypedDict, Optional
from langchain_core.documents import Document

class AgentState(TypedDict):
    # Input Data
    # all_chunks: List[Dict[str, Any]]  # Removed to optimize memory - use direct DB query
    used_anchor_ids: List[str]  # IDs of chunks already used as anchors
    
    # Current Iteration State
    iteration_count: int        # Number of QA pairs generated so far
    max_iterations: int         # Maximum QA pairs to generate
    
    # Flow Selection
    is_reasoning_flow: bool     # True = Reasoning QA, False = Simple QA
    
    # Anchor and Context
    anchor: Optional[Dict[str, Any]]      # Randomly selected anchor
    query: str                  # Query derived from anchor
    context_docs: List[Document]# Retrieved documents (3-5 related/opposing)
    
    # Outputs
    simple_qa: Optional[Dict[str, str]]     # Normal QA output
    reasoning_qa: Optional[Dict[str, str]]  # Reasoning QA output
    
    # Reasoning Step-by-Step Verification (New)
    reasoning_raw_output: str   # Raw LLM output with <think> tags
    reasoning_steps: List[str]  # Parsed individual steps
    current_step_index: int     # Current step being verified (0-indexed)
    step_retry_count: int       # Number of retries for current step
    step_verification_results: List[bool]  # Per-step verification results
    
    # Logs and Flags
    verification_passed: bool   # Overall verification flag
    reasoning_logs: List[str]   # Logs of verification/refinement
    final_output_ready: bool    # Ready to output
    
    # Output Collection
    all_outputs: List[Dict[str, Any]]  # Collected QA pairs across iterations
    last_saved_count: int               # Index of the last saved QA pair

