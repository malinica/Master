"""Splits the strict accuracy of semantic memory into write, search use and reply, from saved JSON transcripts.

Needs no database, API key or network. Only transcripts with turn_logs can be analysed.

Usage:
  python -m eval.memory_usage 20261012T055746Z
"""
import argparse
import json
import statistics
from pathlib import Path

from eval import metrics
from eval.scenarios import SCENARIOS, SCENARIOS_BY_NAME, Scenario

RESULTS_ROOT = Path(__file__).parent / "results"


def _groups(scenario: Scenario) -> list[list[str]]:
    idx = scenario.discriminative_groups
    if idx is None:
        idx = range(len(scenario.expected_facts))
    return [scenario.expected_facts[i] for i in idx]


def _share(groups: list[list[str]], text: str) -> float:
    # thousands separators are ignored, so "$200,000" matches a stored 200000.0
    if not groups:
        return 1.0
    text = text.lower().replace(",", "")
    return sum(any(alt.lower().replace(",", "") in text for alt in g) for g in groups) / len(groups)


def profile_write_share(scenario: Scenario, turn_logs: list[dict]) -> float | None:
    profiles = [
        tl["profile_after_write"] for tl in turn_logs
        if tl.get("session_index", 0) < scenario.check_from_session and "profile_after_write" in tl
    ]
    if not profiles:
        return None  # no semantic memory in this variant
    return _share(_groups(scenario), json.dumps(profiles[-1] or {}))


def _later_search_queries(scenario: Scenario, turn_logs: list[dict]) -> list[str]:
    return [
        call["args"].get("query", "")
        for tl in turn_logs if tl.get("session_index", 0) >= scenario.check_from_session
        for call in tl.get("tool_calls", []) if call["name"] == "real_estate_search"
    ]


def search_use_share(scenario: Scenario, turn_logs: list[dict]) -> float:
    return _share(_groups(scenario), " ".join(_later_search_queries(scenario, turn_logs)))


def analyse_dir(results_dir: Path) -> tuple[list[dict], dict]:
    rows = []
    tool_pattern: dict[tuple, int] = {}
    for path in sorted(results_dir.glob("*_r*__*__*.json")):
        t = json.loads(path.read_text(encoding="utf-8"))
        if "turn_logs" not in t:
            continue
        scenario = SCENARIOS_BY_NAME[t["scenario"]]
        logs = t["turn_logs"]
        replies = [s["replies"] for s in t["sessions"]]
        strict = metrics.memory_accuracy_strict(scenario, replies)
        rows.append({
            "axis": t["axis"], "scenario": t["scenario"], "variant": t["variant"],
            "write": profile_write_share(scenario, logs),
            "search_use": search_use_share(scenario, logs),
            "strict": strict,
            "reply_without_search": strict > 0 and not _later_search_queries(scenario, logs),
        })
        if t["variant"] in ("memory_hot_path", "memory_both"):
            for tl in logs:
                names = [c["name"] for c in tl.get("tool_calls", [])]
                if "real_estate_search" not in names:
                    continue
                key = (t["variant"], "update_user_profile" in names)
                total, logged = tool_pattern.get(key, (0, 0))
                tool_pattern[key] = (total + 1, logged + ("log_search_episode" in names))
    return rows, tool_pattern


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="results folder name in eval/results (e.g. 20261012T055746Z) or a path")
    args = parser.parse_args()
    results_dir = Path(args.folder)
    if not results_dir.is_dir():
        results_dir = RESULTS_ROOT / args.folder
    if not results_dir.is_dir():
        raise SystemExit(f"No results folder {args.folder} (looked in {RESULTS_ROOT}).")

    rows, tool_pattern = analyse_dir(results_dir)
    if not rows:
        raise SystemExit(f"No transcripts with turn_logs in {results_dir}.")

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["axis"], r["scenario"], r["variant"]), []).append(r)
    order = {s.name: i for i, s in enumerate(SCENARIOS)}

    def mean(values):
        values = [v for v in values if v is not None]
        return f"{statistics.mean(values):.2f}" if values else "  — "

    print(f"{'axis':10} {'scenario':40} {'variant':26} {'n':>2}  write  search_use  strict  reply_without_search")
    for key in sorted(groups, key=lambda k: (k[0], order.get(k[1], 99), k[2] != "baseline", k[2])):
        g = groups[key]
        print(
            f"{key[0]:10} {key[1]:40} {key[2]:26} {len(g):>2}  {mean(r['write'] for r in g):>5}  "
            f"{mean(r['search_use'] for r in g):>10}  {mean(r['strict'] for r in g):>6}  "
            f"{sum(r['reply_without_search'] for r in g):>20}"
        )

    print("\nTurns with a town search, hot-path and both modes (episode logged / turns):")
    for (variant, wrote_profile), (total, logged) in sorted(tool_pattern.items()):
        label = "with profile write" if wrote_profile else "without profile write"
        print(f"  {variant:16} {label:22} {logged:>3} / {total}")


if __name__ == "__main__":
    main()
