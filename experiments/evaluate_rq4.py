import argparse
import json
from pathlib import Path

def render_rq4_table(report_path: str, output_path: str | None = None) -> None:
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    
    rows = [
        "| Ablation Condition | Verified Rate (Yield) | Int: A04 Evidence | Int: A05 Single-Ans | Int: A06 Safety | Exp: Factual | Exp: Single-Ans | Exp: Safety |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    
    # Sort methods to have baseline first if present
    methods = sorted(report["methods"].items(), key=lambda x: (x[0] != "baseline", x[0]))
    
    for method, result in methods:
        metrics = result.get("internal_metrics", {})
        expert_metrics = result.get("expert_audit_metrics", {})
        
        def fmt(metric: dict) -> str:
            if not metric or metric.get("value") is None:
                return "_"
            return f"{metric['value'] * 100:.1f}%"

        rows.append(
            f"| {method} | {fmt(metrics.get('record_yield', {}))} | "
            f"{fmt(metrics.get('evidence_status', {}))} | "
            f"{fmt(metrics.get('single_best_answer', {}))} | "
            f"{fmt(metrics.get('emobench_status', {}))} | "
            f"{fmt(expert_metrics.get('evidence_supported', {}))} | "
            f"{fmt(expert_metrics.get('single_best_answer', {}))} | "
            f"{fmt(expert_metrics.get('ei_safety_bias', {}))} |"
        )
    
    markdown = "\n".join(rows) + "\n"
    
    print("\n--- RQ4 ABLATION TABLE ---\n")
    print(markdown)
    
    if output_path:
        Path(output_path).write_text(markdown, encoding="utf-8")
        print(f"\nSaved Markdown table to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Markdown table specifically for RQ4")
    parser.add_argument("--metrics_json", required=True, help="Path to the JSON output from evaluation.py")
    parser.add_argument("--output", help="Optional markdown output path")
    args = parser.parse_args()
    render_rq4_table(args.metrics_json, args.output)
