
import os
import json
from dotenv import load_dotenv
from typing import List

from src.retriever import Retriever
from src.graph.workflow import create_graph

# Simple global context pattern for the retriever to be accessible by nodes
import builtins
builtins.RETRIEVER = None 

def main():
    load_dotenv()
    
    # 1. Initialize Retriever (just for connectivity check and context for nodes)
    # logic moved: we don't load all data anymore.
    print("Initializing pipeline...")
    
    # Set global retriever for nodes to access
    # Note: Retriever now connects to Mongo internally, no need to add_documents
    retriever = Retriever()
    builtins.RETRIEVER = retriever
    
    # 4. Initialize Graph
    app = create_graph()
    
    # 5. Run Pipeline with iterations
    num_qa_pairs = int(os.getenv("NUM_QA_PAIRS", "5000"))
    
    print(f"Running pipeline to generate {num_qa_pairs} QA pairs...")
    
    initial_state = {
        # "all_chunks": [], # Removed
        "iteration_count": 0,
        "max_iterations": num_qa_pairs,
        "is_reasoning_flow": True,
        "anchor": None,
        "query": "",
        "context_docs": [],
        "simple_qa": None,
        "reasoning_qa": None,
        "reasoning_raw_output": "",
        "reasoning_steps": [],
        "reasoning_steps": [],
        "current_step_index": 0,
        "step_retry_count": 0,
        "step_verification_results": [],
        "verification_passed": False,
        "reasoning_logs": [],
        "final_output_ready": False,
        "reasoning_logs": [],
        "final_output_ready": False,
        "all_outputs": [],
        "used_anchor_ids": [],
        "last_saved_count": 0
    }
    
    # Run the graph once - it will iterate internally
    final_state = app.invoke(initial_state)
    
    # 6. Save Output
    os.makedirs("data/output", exist_ok=True)
    output_path = "data/output/generated_reasoning_qa.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in final_state['all_outputs']:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Done. Generated {len(final_state['all_outputs'])} QA pairs.")
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
