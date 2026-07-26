"""Periodic output checkpoint node."""

import logging
from typing import Any, Dict
from ...domain import MCQState
from ...infrastructure.output_writer import append_jsonl

logger = logging.getLogger("mcq.persistence")


def flush_outputs_node(state: MCQState) -> Dict[str, Any]:
    """Append unflushed records after each configured number of completed items."""
    interval = state.get("output_flush_interval", 5)
    completed = state.get("iteration_count", 0)
    if completed == 0 or completed % interval:
        return {}
    verified_start = state.get("verified_flushed_count", 0)
    quarantine_start = state.get("quarantine_flushed_count", 0)
    verified = state.get("verified_outputs", [])[verified_start:]
    quarantined = state.get("quarantine_outputs", [])[quarantine_start:]
    wrote_verified = append_jsonl(state["output_path"], verified)
    wrote_quarantine = append_jsonl(state["quarantine_path"], quarantined)
    logger.info(
        "Checkpoint flush at completed=%s | verified=%s quarantine=%s",
        completed,
        wrote_verified,
        wrote_quarantine,
    )
    return {
        "verified_flushed_count": verified_start + wrote_verified,
        "quarantine_flushed_count": quarantine_start + wrote_quarantine,
    }
