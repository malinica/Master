import csv
import json
import statistics
from pathlib import Path

NUMERIC_METRIC_FIELDS = [
    "memory_accuracy_strict", "behavioral_consistency",
    "false_memory_rate", "episodic_retrieval_accuracy",
    "avg_latency_seconds", "latency_overhead_vs_baseline_seconds",
]


def transcript_path(results_dir: Path, scenario_name: str, variant_key: str) -> Path:
    return results_dir / f"{scenario_name}__{variant_key}.json"


def write_run_transcript(results_dir: Path, scenario_name: str, variant_key: str, run_data: dict) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_path(results_dir, scenario_name, variant_key)
    path.write_text(json.dumps(run_data, indent=2, default=str), encoding="utf-8")
    return path


def write_summary_csv(results_dir: Path, rows: list[dict], filename: str = "summary.csv") -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def aggregate_summary_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["scenario"], row["variant"]), []).append(row)

    aggregated = []
    for (scenario, variant), group_rows in groups.items():
        agg_row = {
            "scenario": scenario,
            "variant": variant,
            "agent_variant": group_rows[0]["agent_variant"],
            "extraction_mode": group_rows[0]["extraction_mode"],
            "memory_types": group_rows[0]["memory_types"],
            "n_repeats": len(group_rows),
        }
        for field in NUMERIC_METRIC_FIELDS:
            values = [r[field] for r in group_rows if r.get(field) not in (None, "")]
            if not values:
                agg_row[f"{field}_mean"] = ""
                agg_row[f"{field}_stdev"] = ""
            else:
                agg_row[f"{field}_mean"] = round(statistics.mean(values), 3)
                agg_row[f"{field}_stdev"] = round(statistics.stdev(values), 3) if len(values) > 1 else 0.0
        aggregated.append(agg_row)
    return aggregated
