"""RQ2 metrics for controlled MCQ-generation experiments.

Run with::

    uv run python -m src.mcq.evaluation \
      --input direct=data/output/rq2_direct.jsonl \
      --input rag_only=data/output/rq2_rag_only.jsonl \
      --input rag_judges=data/output/rq2_rag_judges.jsonl \
      --input full=data/output/rq2_full.jsonl \
      --quarantine rag_judges=data/output/rq2_rag_judges.quarantine.jsonl \
      --quarantine full=data/output/rq2_full.quarantine.jsonl \
      --output data/output/rq2_metrics.json

The module deliberately distinguishes pipeline judge signals from optional
blinded human-expert labels. A judge pass is never reported as an expert pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PASS_VALUES = {"1", "true", "yes", "pass", "passed"}
AUDIT_FIELDS = (
    "overall_publishable",
    "key_correct",
    "single_best_answer",
    "evidence_supported",
    "distractors_plausible",
    "vi_language_quality",
    "ei_safety_bias",
)
JUDGE_FIELDS = ("evidence_status", "single_best_answer", "distractor_quality", "emobench_status")


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source = Path(path)
    if not source.exists():
        return records
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def _parse_method_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected METHOD=PATH")
    method, path = value.split("=", 1)
    if not method.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Expected non-empty METHOD=PATH")
    return method.strip(), path.strip()


def _is_positive(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in PASS_VALUES:
        return True
    if normalized in {"0", "false", "no", "fail", "failed"}:
        return False
    return None


def _rate_with_ci(values: Iterable[bool], *, bootstrap_samples: int, seed: int) -> dict[str, float | int | None]:
    observations = list(values)
    total = len(observations)
    if not total:
        return {"value": None, "n": 0, "ci95_low": None, "ci95_high": None}
    point = sum(observations) / total
    if total == 1:
        return {"value": point, "n": total, "ci95_low": point, "ci95_high": point}
    rng = random.Random(seed)
    samples = sorted(
        sum(observations[rng.randrange(total)] for _ in range(total)) / total
        for _ in range(bootstrap_samples)
    )
    low_index = math.floor(0.025 * (bootstrap_samples - 1))
    high_index = math.ceil(0.975 * (bootstrap_samples - 1))
    return {
        "value": point,
        "n": total,
        "ci95_low": samples[low_index],
        "ci95_high": samples[high_index],
    }


def _valid_mcq_schema(record: dict[str, Any]) -> bool:
    options = record.get("options")
    answer = record.get("answer")
    return (
        isinstance(record.get("question"), str)
        and bool(record["question"].strip())
        and isinstance(options, dict)
        and set(options) == {"A", "B", "C", "D"}
        and all(isinstance(option, str) and option.strip() for option in options.values())
        and answer in options
    )


def _judge_status(record: dict[str, Any], field: str) -> bool | None:
    status = record.get("validation", {}).get(field)
    if status == "not_run" or status is None:
        return None
    return status == "pass"


def _distribution(records: list[dict[str, Any]], metadata_key: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(record.get("metadata", {}).get(metadata_key, "unknown")) for record in records).items()
        )
    )


def load_expert_audit(path: str | None) -> dict[tuple[str, str], dict[str, bool]]:
    """Load blind-audit CSV keyed by `(method, id)`.

    Required columns: `method,id`; the remaining supported labels are listed
    in :data:`AUDIT_FIELDS` and accept pass/failed, true/false, or 1/0.
    """
    if not path:
        return {}
    labels: dict[tuple[str, str], dict[str, bool]] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"method", "id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Expert audit CSV requires columns: method,id")
        for row in reader:
            key = (str(row["method"]).strip(), str(row["id"]).strip())
            parsed = {
                field: value
                for field in AUDIT_FIELDS
                if (value := _is_positive(row.get(field))) is not None
            }
            if key[0] and key[1] and parsed:
                labels[key] = parsed
    return labels


def evaluate_method(
    method: str,
    verified: list[dict[str, Any]],
    quarantine: list[dict[str, Any]],
    *,
    expert_labels: dict[tuple[str, str], dict[str, bool]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    completed = [*verified, *quarantine]
    internal = {
        "record_yield": _rate_with_ci(
            [True] * len(verified) + [False] * len(quarantine),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "schema_valid_rate": _rate_with_ci(
            (_valid_mcq_schema(record) for record in completed),
            bootstrap_samples=bootstrap_samples,
            seed=seed + 1,
        ),
        "evidence_reference_rate": _rate_with_ci(
            (bool(record.get("evidence_refs")) for record in completed),
            bootstrap_samples=bootstrap_samples,
            seed=seed + 2,
        ),
    }
    for offset, field in enumerate(JUDGE_FIELDS, start=3):
        values = [value for record in completed if (value := _judge_status(record, field)) is not None]
        internal[field] = _rate_with_ci(values, bootstrap_samples=bootstrap_samples, seed=seed + offset)

    audited = [
        expert_labels[(method, str(record.get("id")))]
        for record in completed
        if (method, str(record.get("id"))) in expert_labels
    ]
    expert = {
        field: _rate_with_ci(
            (labels[field] for labels in audited if field in labels),
            bootstrap_samples=bootstrap_samples,
            seed=seed + 20 + index,
        )
        for index, field in enumerate(AUDIT_FIELDS)
    }
    return {
        "n_verified": len(verified),
        "n_quarantine": len(quarantine),
        "n_completed": len(completed),
        "n_expert_audited": len(audited),
        "internal_metrics": internal,
        "expert_audit_metrics": expert,
        "difficulty_distribution": _distribution(completed, "difficulty"),
        "question_type_distribution": _distribution(completed, "question_type"),
    }


def evaluate_experiment(
    inputs: dict[str, str],
    quarantines: dict[str, str],
    *,
    audit_csv: str | None = None,
    bootstrap_samples: int = 2_000,
    seed: int = 13,
) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be >= 100")
    labels = load_expert_audit(audit_csv)
    methods = {}
    for index, (method, verified_path) in enumerate(sorted(inputs.items())):
        methods[method] = evaluate_method(
            method,
            _read_jsonl(verified_path),
            _read_jsonl(quarantines[method]) if method in quarantines else [],
            expert_labels=labels,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index * 100,
        )
    return {
        "metric_version": "rq2-v1",
        "bootstrap_samples": bootstrap_samples,
        "methods": methods,
        "notes": {
            "record_yield": "verified records divided by verified plus quarantined completed records; it is not token cost or raw LLM-call yield.",
            "expert_audit": "Only populated from the optional blind expert-audit CSV; pipeline judge results are reported separately.",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact paper-ready table from an evaluation report."""
    rows = [
        "| Method | Completed | Record yield (95% CI) | Schema valid | Evidence refs | Expert pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, result in report["methods"].items():
        metrics = result["internal_metrics"]
        expert = result["expert_audit_metrics"]["overall_publishable"]

        def fmt(metric: dict[str, Any]) -> str:
            if metric["value"] is None:
                return "—"
            return f"{metric['value'] * 100:.1f}% [{metric['ci95_low'] * 100:.1f}, {metric['ci95_high'] * 100:.1f}]"

        rows.append(
            f"| {method} | {result['n_completed']} | {fmt(metrics['record_yield'])} | "
            f"{fmt(metrics['schema_valid_rate'])} | {fmt(metrics['evidence_reference_rate'])} | "
            f"{fmt(expert)} |"
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RQ2 metrics from pipeline JSONL outputs")
    parser.add_argument("--input", action="append", type=_parse_method_path, required=True, metavar="METHOD=PATH")
    parser.add_argument("--quarantine", action="append", type=_parse_method_path, default=[], metavar="METHOD=PATH")
    parser.add_argument("--audit_csv", help="Optional blind expert-audit CSV with method,id and audit labels")
    parser.add_argument("--output", required=True, help="Output metrics JSON path")
    parser.add_argument("--markdown_output", help="Optional Markdown table path")
    parser.add_argument("--bootstrap_samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    inputs, quarantines = dict(args.input), dict(args.quarantine)
    report = evaluate_experiment(
        inputs,
        quarantines,
        audit_csv=args.audit_csv,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output:
        markdown = Path(args.markdown_output)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote RQ2 metrics for {len(report['methods'])} methods to {output}")


if __name__ == "__main__":
    main()
