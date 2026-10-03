#!/usr/bin/env python3
"""Run a single specific anchor ID through the MCQ pipeline for prompt evaluation."""

import argparse
import builtins
import os
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Test single anchor MCQ generation")
    parser.add_argument(
        "--anchor_id",
        type=str,
        default="ada27710-594d-4318-8266-e48a029ede36",
        help="Specific anchor chunk UUID from MongoDB",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/output/test_anchor_psy01.jsonl",
        help="Output path for the generated MCQ",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="theory",
        help="Curriculum level (theory, emotion, educational_scenario, clinical_scenario)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        help="Difficulty (easy, medium, hard)",
    )
    args = parser.parse_args()

    from src.retriever import Retriever
    from src.mcq.workflow import create_mcq_graph
    from pymongo import MongoClient

    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME", "Data")
    col_name = os.getenv("MONGO_COLLECTION_NAME", "mental")

    client = MongoClient(mongo_uri)
    col = client[db_name][col_name]
    doc = col.find_one({"uuid": args.anchor_id}) or col.find_one({"chunk_id": args.anchor_id})
    client.close()

    if not doc:
        print(f"Error: Anchor ID '{args.anchor_id}' not found in MongoDB collection '{db_name}.{col_name}'")
        return

    anchor = {
        "chunk_id": str(doc.get("chunk_id") or doc.get("uuid") or doc.get("_id")),
        "title": doc.get("title") or " / ".join(doc.get("headers", [])),
        "summary": doc.get("summary", ""),
        "content": doc.get("content", ""),
        "keywords": doc.get("keywords", []),
        "type": doc.get("type", ""),
        "tier": doc.get("tier") or doc.get("source_tier"),
    }
    print(f"✓ Loaded Anchor: [{anchor['chunk_id']}] {anchor['title']}")

    retriever = Retriever()
    builtins.RETRIEVER = retriever
    app = create_mcq_graph(experiment_method="full")

    quarantine_path = os.path.splitext(args.output_path)[0] + ".quarantine.jsonl"
    playbook_path = os.path.splitext(args.output_path)[0] + ".playbook.md"

    initial_state = {
        "iteration_count": 0,
        "max_iterations": 1,
        "generation_attempt": 0,
        "max_generation_retries": 2,
        "output_path": args.output_path,
        "quarantine_path": quarantine_path,
        "output_flush_interval": 1,
        "learning_checkpoint_interval": 10,
        "playbook_path": playbook_path,
        "playbook_usage_path": "",
        "failure_memory_path": os.path.splitext(args.output_path)[0] + ".failure_memory.json",
        "judge_memory_path": os.path.splitext(args.output_path)[0] + ".judge_failure_memory.json",
        "verified_flushed_count": 0,
        "quarantine_flushed_count": 0,
        "learning_checkpoint_count": 0,
        "next_record_id": 1,
        "curriculum_levels": [args.level],
        "curriculum_difficulties": [args.difficulty],
        "used_anchor_ids": [],
        "anchor": anchor,
        "blueprint": {},
        "evidence_docs": [],
        "dsm5_safety_docs": [],
        "mcq": {},
        "judge_reports": {},
        "judge_feedback": [],
        "planning_feedback": [],
        "regeneration_route": "generate",
        "verdict": "",
        "quarantine_reason": [],
        "verified_outputs": [],
        "quarantine_outputs": [],
        "failure_memory": [],
        "judge_failure_memory": [],
        "playbook": "",
        "playbook_usage": {},
        "playbook_delta": [],
        "experiment_method": "full",
        "last_saved_count": 0,
    }

    print("Running pipeline for the specific anchor...")
    final_state = app.invoke(initial_state, {"recursion_limit": 1000})

    verified = final_state.get("verified_outputs", [])
    quarantined = final_state.get("quarantine_outputs", [])

    print("\n" + "=" * 60)
    if verified:
        print(f"🎉 VERIFIED! Kết quả lưu tại: {args.output_path}")
        item = verified[-1]
        print(f"\nQuestion: {item.get('question')}")
        for k, v in item.get('options', {}).items():
            print(f"  {k}: {v}")
        print(f"Answer: {item.get('answer')}")
    elif quarantined:
        print(f"⚠️ QUARANTINED! Xem chi tiết tại: {quarantine_path}")
        item = quarantined[-1]
        print(f"\nQuestion: {item.get('question')}")
        for k, v in item.get('options', {}).items():
            print(f"  {k}: {v}")
        print(f"Answer: {item.get('answer')}")
        print(f"Issues: {item.get('_audit', {}).get('quarantine_reason')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
