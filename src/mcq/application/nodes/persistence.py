"""Periodic output and learning-state checkpoint node."""

import json
import logging
import os
import tempfile
from typing import Any, Dict
from ...domain import MCQState
from ...infrastructure.output_writer import append_jsonl

logger = logging.getLogger("mcq.persistence")


def _atomic_write_text(path: str, content: str) -> None:
    """Replace a sidecar atomically so an interrupted run keeps its last checkpoint."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".checkpoint-", suffix=".tmp", dir=directory, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _checkpoint_learning_state(state: MCQState) -> None:
    """Persist reusable learning state independently of JSONL output flushing."""
    _atomic_write_text(state["playbook_path"], state.get("playbook", ""))
    _atomic_write_text(
        state["failure_memory_path"],
        json.dumps(state.get("failure_memory", []), ensure_ascii=False, indent=2),
    )
    _atomic_write_text(
        state["judge_memory_path"],
        json.dumps(state.get("judge_failure_memory", []), ensure_ascii=False, indent=2),
    )


def flush_outputs_node(state: MCQState) -> Dict[str, Any]:
    """Flush output batches and periodically checkpoint reusable learning state."""
    interval = state.get("output_flush_interval", 5)
    learning_interval = state.get("learning_checkpoint_interval", 10)
    completed = state.get("iteration_count", 0)
    should_flush_output = completed > 0 and completed % interval == 0
    should_checkpoint_learning = completed > 0 and completed % learning_interval == 0
    if not should_flush_output and not should_checkpoint_learning:
        return {}

    update: Dict[str, Any] = {}
    if should_flush_output:
        verified_start = state.get("verified_flushed_count", 0)
        quarantine_start = state.get("quarantine_flushed_count", 0)
        verified = state.get("verified_outputs", [])[verified_start:]
        quarantined = state.get("quarantine_outputs", [])[quarantine_start:]
        wrote_verified = append_jsonl(state["output_path"], verified)
        wrote_quarantine = append_jsonl(state["quarantine_path"], quarantined)
        logger.info(
            "Output checkpoint at completed=%s | verified=%s quarantine=%s",
            completed,
            wrote_verified,
            wrote_quarantine,
        )
        update.update(
            {
                "verified_flushed_count": verified_start + wrote_verified,
                "quarantine_flushed_count": quarantine_start + wrote_quarantine,
            }
        )

    if should_checkpoint_learning:
        _checkpoint_learning_state(state)
        checkpoint_count = state.get("learning_checkpoint_count", 0) + 1
        logger.info(
            "Learning checkpoint at completed=%s | playbook=%s failure_memory=%s judge_memory=%s",
            completed,
            state["playbook_path"],
            state["failure_memory_path"],
            state["judge_memory_path"],
        )
        update["learning_checkpoint_count"] = checkpoint_count

    return update
