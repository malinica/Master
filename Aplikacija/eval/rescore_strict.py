"""Recomputes memory_accuracy_strict from saved JSON transcripts, without calling the model.

Usage:
  python -m eval.rescore_strict 20260918T051127Z
  python -m eval.rescore_strict 20260918T051127Z --csv   # also writes summary_<axis>_strict.csv
"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

from eval import metrics
from eval.scenarios import SCENARIOS, SCENARIOS_BY_NAME

RESULTS_ROOT = Path(__file__).parent / "results"


def rescore_transcript(transcript: dict) -> float:
    scenario = SCENARIOS_BY_NAME[transcript["scenario"]]
    stored_messages = [session["messages"] for session in transcript["sessions"]]
    if stored_messages != scenario.sessions:
        raise ValueError(
            f"{transcript['scenario']}: the messages in the transcript differ from "
            "eval/scenarios.py, so its expected_facts may not apply to these replies"
        )
    session_replies = [session["replies"] for session in transcript["sessions"]]
    return metrics.memory_accuracy_strict(scenario, session_replies)


def rescore_dir(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*_r*__*__*.json")):
        transcript = json.loads(path.read_text(encoding="utf-8"))
        if "sessions" not in transcript:
            continue
        rows.append({
            "axis": transcript["axis"],
            "scenario": transcript["scenario"],
            "variant": transcript["variant"],
            "repeat": int(re.search(r"_r(\d+)__", path.name).group(1)),
            "memory_accuracy": transcript["metrics"].get("memory_accuracy"),
            "memory_accuracy_strict": rescore_transcript(transcript),
        })
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["axis"], row["scenario"], row["variant"]), []).append(row)

    scenario_order = {s.name: i for i, s in enumerate(SCENARIOS)}

    def sort_key(key: tuple) -> tuple:
        axis, scenario, variant = key
        return axis, scenario_order.get(scenario, len(scenario_order)), variant != "baseline", variant

    aggregated = []
    for key in sorted(groups, key=sort_key):
        axis, scenario, variant = key
        group = groups[key]
        strict = [r["memory_accuracy_strict"] for r in group]
        original = [r["memory_accuracy"] for r in group if r["memory_accuracy"] is not None]
        aggregated.append({
            "axis": axis,
            "scenario": scenario,
            "variant": variant,
            "n_repeats": len(group),
            "memory_accuracy_mean": round(statistics.mean(original), 3) if original else "",
            "memory_accuracy_strict_mean": round(statistics.mean(strict), 3),
            "memory_accuracy_strict_stdev": round(statistics.stdev(strict), 3) if len(strict) > 1 else 0.0,
        })
    return aggregated


def write_csvs(results_dir: Path, aggregated: list[dict]) -> list[Path]:
    paths = []
    for axis in sorted({r["axis"] for r in aggregated}):
        axis_rows = [r for r in aggregated if r["axis"] == axis]
        path = results_dir / f"summary_{axis}_strict.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(axis_rows[0].keys()))
            writer.writeheader()
            writer.writerows(axis_rows)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="results folder name in eval/results (e.g. 20260918T051127Z) or a path")
    parser.add_argument("--csv", action="store_true", help="also write summary_<axis>_strict.csv into the folder")
    args = parser.parse_args()

    results_dir = Path(args.folder)
    if not results_dir.is_dir():
        results_dir = RESULTS_ROOT / args.folder
    if not results_dir.is_dir():
        raise SystemExit(f"No results folder {args.folder} (looked in {RESULTS_ROOT}).")

    aggregated = aggregate(rescore_dir(results_dir))
    if not aggregated:
        raise SystemExit(f"No transcripts found in {results_dir}.")

    print(f"{'axis':10} {'scenario':40} {'variant':26} {'n':>2}  {'original':>8}  strict (mean ± stdev)")
    for r in aggregated:
        original = f"{r['memory_accuracy_mean']:.2f}" if r["memory_accuracy_mean"] != "" else "—"
        print(
            f"{r['axis']:10} {r['scenario']:40} {r['variant']:26} {r['n_repeats']:>2}  {original:>8}  "
            f"{r['memory_accuracy_strict_mean']:.2f} ± {r['memory_accuracy_strict_stdev']:.2f}"
        )

    if args.csv:
        for path in write_csvs(results_dir, aggregated):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
