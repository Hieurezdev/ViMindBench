
import os
import json
import builtins
import argparse
from dotenv import load_dotenv

# Simple global context pattern for the retriever to be accessible by nodes
builtins.RETRIEVER = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run psychology QA generation pipeline")
    parser.add_argument("--model_local", action="store_true", help="Use local LLM endpoint at http://localhost:8000/v1")
    parser.add_argument("--embedding_local", action="store_true", help="Use local sentence-transformers embedding model")
    parser.add_argument("--model_base_url", type=str, default=None, help="Override OPENAI_BASE_URL")
    parser.add_argument("--model_name", type=str, default=None, help="Override MODEL_NAME")
    parser.add_argument("--api_key", type=str, default=None, help="Override OPENAI_API_KEY")
    parser.add_argument("--embedding_model", type=str, default=None, help="Override EMBEDDING_MODEL")
    parser.add_argument("--embedding_base_url", type=str, default=None, help="Override EMBEDDING_BASE_URL")
    parser.add_argument("--num_qa_pairs", type=int, default=None, help="Override NUM_QA_PAIRS")
    parser.add_argument("--output_path", type=str, default=None, help="Override OUTPUT_PATH")
    return parser.parse_args()


def load_used_anchor_ids(output_files: list[str]) -> list[str]:
    """
    Scan existing output JSONL files and collect all anchor_ids already used,
    so the pipeline skips them on subsequent runs.
    """
    used = []
    for path in output_files:
        if not os.path.exists(path):
            continue
        print(f"Loading used anchor_ids from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    aid = entry.get("anchor_id")
                    if aid and aid not in used:
                        used.append(aid)
                except json.JSONDecodeError:
                    pass
        print(f"  → {len(used)} unique anchor_ids loaded so far.")
    return used


def main():
    load_dotenv()
    args = parse_args()

    # Apply CLI overrides before importing pipeline modules
    if args.model_local:
        os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        print("Using local model endpoint: http://localhost:8000/v1")

    if args.embedding_local:
        os.environ["USE_LOCAL_EMBEDDING"] = "true"
        print("Using local embedding model via sentence-transformers")

    if args.model_base_url:
        os.environ["OPENAI_BASE_URL"] = args.model_base_url

    if args.embedding_model:
        os.environ["EMBEDDING_MODEL"] = args.embedding_model

    if args.embedding_base_url:
        os.environ["EMBEDDING_BASE_URL"] = args.embedding_base_url

    if args.model_name:
        os.environ["MODEL_NAME"] = args.model_name

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    if args.num_qa_pairs is not None:
        os.environ["NUM_QA_PAIRS"] = str(args.num_qa_pairs)

    if args.output_path:
        os.environ["OUTPUT_PATH"] = args.output_path

    # If local LLM is requested, default to local embeddings as well unless explicitly disabled.
    if args.model_local and "USE_LOCAL_EMBEDDING" not in os.environ:
        os.environ["USE_LOCAL_EMBEDDING"] = "true"
        print("Auto-enabling local embedding because local model endpoint is used")

    from src.retriever import Retriever
    from src.graph.workflow import create_graph

    # ── 1. Retriever ──────────────────────────────────────────────────────
    print("Initializing pipeline...")
    retriever = Retriever()
    builtins.RETRIEVER = retriever

    # ── 2. Graph ──────────────────────────────────────────────────────────
    app = create_graph()

    # ── 3. Config ─────────────────────────────────────────────────────────
    num_qa_pairs = int(os.getenv("NUM_QA_PAIRS", "5000"))
    output_path = os.getenv(
        "OUTPUT_PATH",
        "data/output/generated_psychology_multiple_choice.jsonl"
    )

    print(f"Running pipeline to generate {num_qa_pairs} QA pairs...")

    # ── 4. Load already-used anchor_ids to avoid duplicates ───────────────
    existing_output_files = [
        output_path,
        "data/output/generated_psychology_multiple_choice-2.jsonl",
    ]
    used_anchor_ids = load_used_anchor_ids(existing_output_files)
    print(f"Total used anchor_ids (will be skipped): {len(used_anchor_ids)}")

    # ── 5. Initial State ──────────────────────────────────────────────────
    initial_state = {
        "iteration_count": 0,
        "max_iterations": num_qa_pairs,
        "is_reasoning_flow": True,
        "anchor": None,
        "query": "",
        "primary_retrieval_query": "",
        "negative_retrieval_query": "",
        "context_docs": [],
        "negative_docs": [],
        "simple_qa": None,
        "reasoning_qa": None,
        "reasoning_raw_output": "",
        "reasoning_steps": [],
        "current_step_index": 0,
        "step_retry_count": 0,
        "step_verification_results": [],
        "verification_passed": False,
        "reasoning_logs": [],
        "format_check_passed": True,
        "final_output_ready": False,
        "all_outputs": [],
        "used_anchor_ids": used_anchor_ids,
        "last_saved_count": 0,
        # QA Validation
        "qa_validation_passed": False,
        "qa_validation_attempts": 0,
        # Grounding Validation
        "grounding_passed": False,
        "grounding_attempts": 0,
    }

    # ── 6. Run ────────────────────────────────────────────────────────────
    final_state = app.invoke(initial_state)

    # ── 7. Final Save (remaining items not yet flushed by check_more) ─────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    last_saved = final_state.get("last_saved_count", 0)
    remaining_items = final_state["all_outputs"][last_saved:]

    if remaining_items:
        print(f"Saving final batch of {len(remaining_items)} items to {output_path}...")
        with open(output_path, "a", encoding="utf-8") as f:
            for entry in remaining_items:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Done. Generated {len(final_state['all_outputs'])} QA pairs in total.")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
