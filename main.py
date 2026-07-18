
import os
import json
import builtins
import argparse
from pathlib import Path
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
    parser.add_argument("--max_generation_retries", type=int, default=None, help="Override MAX_GENERATION_RETRIES (retries after initial generation)")
    parser.add_argument("--log_level", type=str, default=None, help="Override LOG_LEVEL: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--log_path", type=str, default=None, help="Optional run log path; defaults to <output>.run.log")
    parser.add_argument("--output_flush_interval", type=int, default=None, help="Override OUTPUT_FLUSH_INTERVAL; defaults to 5 completed items")
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated MCQ levels: theory,emotion,educational_scenario,clinical_scenario",
    )
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


def load_anchor_sidecar(path: str) -> list[str]:
    """Load opaque anchor IDs without adding provenance fields to train JSONL."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [str(value) for value in data if value]


def load_json_list(path: str) -> list[dict]:
    """Load an optional, non-training sidecar such as judge failure memory."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def load_playbook(persisted_path: str) -> str:
    """Prefer an accumulated playbook; otherwise start from the versioned seed."""
    seed_path = Path(__file__).parent / "config" / "initial_playbook.md"
    source = Path(persisted_path) if os.path.exists(persisted_path) else seed_path
    try:
        with open(source, "r", encoding="utf-8") as f:
            playbook = f.read().strip()
        if "## " not in playbook or "helpful=" not in playbook:
            raise ValueError("playbook must use ACE Markdown sections and bullets")
        print(f"Loaded ACE playbook: {source}")
        return playbook
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot load playbook from {source}: {exc}") from exc


def parse_levels(raw_levels: str | None) -> list[str]:
    """Validate the optional curriculum filter before starting external clients."""
    allowed = {"theory", "emotion", "educational_scenario", "clinical_scenario"}
    if not raw_levels:
        return ["theory", "emotion", "educational_scenario", "clinical_scenario"]
    levels = [level.strip() for level in raw_levels.split(",") if level.strip()]
    invalid = sorted(set(levels) - allowed)
    if not levels or invalid:
        raise ValueError(f"Invalid --levels value. Allowed: {', '.join(sorted(allowed))}; received: {raw_levels}")
    return list(dict.fromkeys(levels))


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

    if args.max_generation_retries is not None:
        if args.max_generation_retries < 0:
            raise SystemExit("error: --max_generation_retries must be >= 0")
        os.environ["MAX_GENERATION_RETRIES"] = str(args.max_generation_retries)

    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level
    if args.log_path:
        os.environ["LOG_PATH"] = args.log_path
    if args.output_flush_interval is not None:
        if args.output_flush_interval < 1:
            raise SystemExit("error: --output_flush_interval must be >= 1")
        os.environ["OUTPUT_FLUSH_INTERVAL"] = str(args.output_flush_interval)

    try:
        curriculum_levels = parse_levels(args.levels)
    except ValueError as exc:
        parser_error = argparse.ArgumentTypeError(str(exc))
        raise SystemExit(f"error: {parser_error}") from exc

    # If local LLM is requested, default to local embeddings as well unless explicitly disabled.
    if args.model_local and "USE_LOCAL_EMBEDDING" not in os.environ:
        os.environ["USE_LOCAL_EMBEDDING"] = "true"
        print("Auto-enabling local embedding because local model endpoint is used")

    output_path = os.getenv("OUTPUT_PATH", "data/output/generated_psychology_multiple_choice.jsonl")
    from src.mcq.infrastructure.observability import configure_logging
    logger = configure_logging(output_path=output_path, level=os.getenv("LOG_LEVEL", "INFO"), log_path=os.getenv("LOG_PATH"))
    logger.info("Run requested: levels=%s; target_attempts=%s; max_generation_retries=%s", ",".join(curriculum_levels), os.getenv("NUM_QA_PAIRS", "5000"), os.getenv("MAX_GENERATION_RETRIES", "2"))

    from src.retriever import Retriever
    from src.mcq.workflow import create_mcq_graph

    # ── 1. Retriever ──────────────────────────────────────────────────────
    print("Initializing pipeline...")
    retriever = Retriever()
    builtins.RETRIEVER = retriever

    # ── 2. Graph ──────────────────────────────────────────────────────────
    app = create_mcq_graph()

    # ── 3. Config ─────────────────────────────────────────────────────────
    num_qa_pairs = int(os.getenv("NUM_QA_PAIRS", "5000"))
    output_path = os.getenv("OUTPUT_PATH", "data/output/generated_psychology_multiple_choice.jsonl")

    print(f"Running pipeline to generate {num_qa_pairs} QA pairs...")
    print(f"Curriculum levels: {', '.join(curriculum_levels)}")

    # ── 4. Load already-used anchor_ids to avoid duplicates ───────────────
    existing_output_files = [
        output_path,
        "data/output/generated_psychology_multiple_choice-2.jsonl",
    ]
    anchor_sidecar = os.path.splitext(output_path)[0] + ".anchors.json"
    playbook_path = os.path.splitext(output_path)[0] + ".playbook.md"
    judge_memory_path = os.path.splitext(output_path)[0] + ".judge_failure_memory.json"
    quarantine_path = os.path.splitext(output_path)[0] + ".quarantine.jsonl"
    used_anchor_ids = list(set(load_used_anchor_ids(existing_output_files) + load_anchor_sidecar(anchor_sidecar)))
    print(f"Total used anchor_ids (will be skipped): {len(used_anchor_ids)}")

    # ── 5. Initial State ──────────────────────────────────────────────────
    initial_state = {
        "iteration_count": 0,
        "max_iterations": num_qa_pairs,
        "generation_attempt": 0,
        "max_generation_retries": int(os.getenv("MAX_GENERATION_RETRIES", "2")),
        "output_path": output_path,
        "quarantine_path": quarantine_path,
        "output_flush_interval": int(os.getenv("OUTPUT_FLUSH_INTERVAL", "5")),
        "verified_flushed_count": 0,
        "quarantine_flushed_count": 0,
        "curriculum_levels": curriculum_levels,
        "anchor": None,
        "blueprint": {},
        "evidence_docs": [],
        "dsm5_safety_docs": [],
        "mcq": {},
        "judge_reports": {},
        "judge_feedback": [],
        "verdict": "",
        "quarantine_reason": [],
        "verified_outputs": [],
        "quarantine_outputs": [],
        "failure_memory": [],
        "judge_failure_memory": load_json_list(judge_memory_path),
        "playbook": load_playbook(playbook_path),
        "playbook_delta": [],
        "used_anchor_ids": used_anchor_ids,
        "last_saved_count": 0,
    }

    # ── 6. Run ────────────────────────────────────────────────────────────
    final_state = app.invoke(initial_state, {"recursion_limit": 50000})

    # ── 7. Save verified data and a separate quarantine audit trail ───────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    verified_items = final_state.get("verified_outputs", [])
    quarantine_items = final_state.get("quarantine_outputs", [])
    from src.mcq.infrastructure.output_writer import append_jsonl
    final_verified = append_jsonl(output_path, verified_items[final_state.get("verified_flushed_count", 0):])
    final_quarantine = append_jsonl(quarantine_path, quarantine_items[final_state.get("quarantine_flushed_count", 0):])
    if final_verified or final_quarantine:
        logger.info("Final flush | verified=%s quarantine=%s", final_verified, final_quarantine)
    with open(playbook_path, "w", encoding="utf-8") as f:
        f.write(final_state.get("playbook", ""))
    with open(anchor_sidecar, "w", encoding="utf-8") as f:
        json.dump(final_state.get("used_anchor_ids", []), f, ensure_ascii=False, indent=2)
    with open(judge_memory_path, "w", encoding="utf-8") as f:
        json.dump(final_state.get("judge_failure_memory", []), f, ensure_ascii=False, indent=2)

    print(f"Done. Verified={len(verified_items)}, quarantined={len(quarantine_items)}.")
    print(f"Verified: {output_path}; quarantine: {quarantine_path}; playbook: {playbook_path}; judge memory: {judge_memory_path}")
    logger.info("Run complete: verified=%s; quarantined=%s; output=%s", len(verified_items), len(quarantine_items), output_path)


if __name__ == "__main__":
    main()
