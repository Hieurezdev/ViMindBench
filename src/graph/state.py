
from typing import List, Dict, Any, TypedDict, Optional
from langchain_core.documents import Document

class AgentState(TypedDict):
    # Input Data
    used_anchor_ids: List[str]  # IDs of chunks already used as anchors
    
    # Current Iteration State
    iteration_count: int        # Number of QA pairs generated so far
    max_iterations: int         # Maximum QA pairs to generate
    
    # Flow Selection
    is_reasoning_flow: bool     # True = Reasoning QA, False = Simple QA
    
    # Anchor and Context
    anchor: Optional[Dict[str, Any]]      # Randomly selected anchor
    query: str                  # Query derived from anchor
    primary_retrieval_query: str # Query used for main retrieval
    negative_retrieval_query: str # Query used for opposing retrieval
    context_docs: List[Document]  # Retrieved documents (similar/opposing)
    negative_docs: List[Document] # Retrieved documents opposing the anchor
    
    # Outputs
    simple_qa: Optional[Dict[str, str]]     # Normal QA output
    reasoning_qa: Optional[Dict[str, str]]  # Reasoning QA output
    formatted_qa: Optional[Dict[str, Any]]  # Final formatted JSON output
    question_type_enum: str                 # Enum for question type
    
    # Reasoning Step-by-Step Verification
    reasoning_raw_output: str   # Raw LLM output with <think> tags
    reasoning_steps: List[str]  # Parsed individual steps
    current_step_index: int     # Current step being verified (0-indexed)
    step_retry_count: int       # Number of retries for current step
    step_verification_results: List[bool]  # Per-step verification results
    
    # Logs and Flags
    verification_passed: bool   # Overall verification flag
    reasoning_logs: List[str]   # Logs of verification/refinement
    format_check_passed: bool   # Flag for output format validation
    final_output_ready: bool    # Ready to output
    
    # QA Validation (kiểm tra câu hỏi trước khi vào step verification)
    qa_validation_passed: bool  # True nếu câu hỏi hợp lệ
    qa_validation_attempts: int # Số lần thử lại do validate fail
    
    # Grounding Validation (kiểm tra factual với reference)
    grounding_passed: bool      # True nếu match với tài liệu gốc
    grounding_attempts: int     # Số lần thử lại do sai khác tài liệu
    
    # Output Collection
    all_outputs: List[Dict[str, Any]]  # Collected QA pairs across iterations
    last_saved_count: int               # Index of the last saved QA pair
