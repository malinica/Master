"""Comparison harness: every scenario on the mechanism and type axes, plus the D5 and D6 checks.

Usage:
  python -m eval.runner    # everything: all scenarios, both axes, D5 + D6, 5 repeats
  python -m eval.runner --scenarios budget_and_climate_preference --axis type --repeats 1
  python -m eval.runner --resume [FOLDER]    # continue an interrupted run
"""
import argparse
import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langmem.prompts.types import AnnotatedTrajectory

from db import threads_repo
from db.connection import STORE_INDEX_CONFIG, get_conn_string, windows_loop_factory
from db.eval_runs_repo import insert_eval_run
from db.users_repo import get_or_create_user
from eval import metrics
from eval.report import aggregate_summary_rows, transcript_path, write_run_transcript, write_summary_csv
from eval.scenarios import SCENARIOS, Scenario, resolve_scenarios
from graph.builder import build_graph
from memory.background import run_background_extraction
from memory.config import ALL_MEMORY_TYPES, ExtractionMode, MemoryType, SearchStrategy
from memory.episodic import asearch_episodes
from memory.procedural import aoptimize_procedural_prompt
from memory.reflection_executors import ReflectionExecutors
from memory.semantic import aget_current_profile
from realestate.service import RealEstateSearchService

RESULTS_ROOT = Path(__file__).parent / "results"

# "baseline" must come first: its latencies are the reference for the other variants.
MECHANISM_VARIANTS: dict[str, tuple[str, ExtractionMode | None, frozenset[MemoryType] | None]] = {
    "baseline": ("baseline", None, None),
    "memory_hot_path": ("memory", ExtractionMode.HOT_PATH, None),
    "memory_background": ("memory", ExtractionMode.BACKGROUND, None),
    "memory_both": ("memory", ExtractionMode.BOTH, None),
}

TYPE_VARIANTS: dict[str, tuple[str, ExtractionMode | None, frozenset[MemoryType] | None]] = {
    "baseline": ("baseline", None, None),
    "semantic_only_hot_path": ("memory", ExtractionMode.HOT_PATH, frozenset({MemoryType.SEMANTIC})),
    "semantic_only_background": ("memory", ExtractionMode.BACKGROUND, frozenset({MemoryType.SEMANTIC})),
    "semantic_only_both": ("memory", ExtractionMode.BOTH, frozenset({MemoryType.SEMANTIC})),
    "episodic_only_hot_path": ("memory", ExtractionMode.HOT_PATH, frozenset({MemoryType.EPISODIC})),
    "episodic_only_background": ("memory", ExtractionMode.BACKGROUND, frozenset({MemoryType.EPISODIC})),
    "episodic_only_both": ("memory", ExtractionMode.BOTH, frozenset({MemoryType.EPISODIC})),
}

AXES = {"mechanism": MECHANISM_VARIANTS, "type": TYPE_VARIANTS}

CONFIG_FILE = "run_config.json"
_QUOTA_ERROR_CODES = ("insufficient_quota", "billing_hard_limit_reached")


class QuotaExhausted(Exception):
    pass


def _is_quota_error(exc: BaseException) -> bool:
    seen, stack = set(), [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        if getattr(e, "code", None) in _QUOTA_ERROR_CODES or any(c in str(e) for c in _QUOTA_ERROR_CODES):
            return True
        stack += [e.__cause__, e.__context__, *getattr(e, "exceptions", ())]
    return False


def _summary_row(transcript: dict, repeat_index: int) -> dict:
    agent_variant, extraction_mode, memory_types = AXES[transcript["axis"]][transcript["variant"]]
    m = transcript["metrics"]

    def rounded(value):
        return round(value, 3) if value is not None else ""

    return {
        "scenario": transcript["scenario"],
        "variant": transcript["variant"],
        "repeat": repeat_index,
        "agent_variant": agent_variant,
        "extraction_mode": extraction_mode.value if extraction_mode else "",
        "memory_types": ",".join(sorted(t.value for t in memory_types)) if memory_types else "",
        "search_strategy": transcript["search_strategy"],
        "memory_accuracy_strict": rounded(m["memory_accuracy_strict"]),
        "behavioral_consistency": rounded(m["behavioral_consistency"]),
        "false_memory_rate": rounded(m["false_memory_rate"]),
        "episodic_retrieval_accuracy": rounded(m["episodic_retrieval_accuracy"]),
        "avg_latency_seconds": rounded(m["avg_latency_seconds"]),
        "latency_overhead_vs_baseline_seconds": rounded(m["latency_overhead_vs_baseline_seconds"]),
    }


def load_summary_rows(results_dir: Path) -> dict[str, list[dict]]:
    rows_by_axis: dict[str, list[dict]] = {}
    for path in sorted(results_dir.glob("*_r*__*__*.json")):
        transcript = json.loads(path.read_text(encoding="utf-8"))
        if "sessions" not in transcript:
            continue
        repeat_index = int(re.search(r"_r(\d+)__", path.name).group(1))
        rows_by_axis.setdefault(transcript["axis"], []).append(_summary_row(transcript, repeat_index))

    scenario_order = {s.name: i for i, s in enumerate(SCENARIOS)}
    for axis_label, rows in rows_by_axis.items():
        variant_order = list(AXES[axis_label])
        rows.sort(key=lambda r: (r["repeat"], scenario_order.get(r["scenario"], len(scenario_order)),
                                 variant_order.index(r["variant"])))
    return rows_by_axis


def write_summaries(results_dir: Path, repeats: int) -> None:
    for axis_label, rows in load_summary_rows(results_dir).items():
        write_summary_csv(results_dir, rows, filename=f"summary_{axis_label}.csv")
        if repeats > 1:
            write_summary_csv(
                results_dir, aggregate_summary_rows(rows), filename=f"summary_{axis_label}_aggregated.csv"
            )


def _extract_turn_log(all_messages: list, user_message: str) -> dict:
    start = 0
    for i in range(len(all_messages) - 1, -1, -1):
        msg = all_messages[i]
        if getattr(msg, "type", None) == "human" and msg.content == user_message:
            start = i
            break
    tool_calls, tool_results = [], []
    for msg in all_messages[start:]:
        if getattr(msg, "type", None) == "ai" and getattr(msg, "tool_calls", None):
            tool_calls.extend({"name": tc.get("name"), "args": tc.get("args")} for tc in msg.tool_calls)
        if getattr(msg, "type", None) == "tool":
            tool_results.append({
                "name": getattr(msg, "name", None),
                "status": getattr(msg, "status", None),
                "content": str(getattr(msg, "content", ""))[:500],
            })
    return {"tool_calls": tool_calls, "tool_results": tool_results}


async def run_turn(
    checkpointer,
    store: BaseStore,
    user_id: int,
    thread_id: str,
    message: str,
    agent_variant: str,
    extraction_mode: ExtractionMode | None,
    memory_types: frozenset[MemoryType] | None,
    search_service: RealEstateSearchService,
    search_strategy: SearchStrategy = SearchStrategy.SIMILARITY,
    executors: ReflectionExecutors | None = None,
    pending_bg: list | None = None,
    turn_log: dict | None = None,
) -> str:
    config = {"configurable": {"thread_id": thread_id}}

    if agent_variant != "memory":
        graph = build_graph(
            with_memory=False, checkpointer=checkpointer, store=None,
            user_id=user_id, search_service=search_service,
        )
        result = await graph.ainvoke({"messages": [{"role": "user", "content": message}]}, config=config)
        if turn_log is not None:
            turn_log.update(_extract_turn_log(result["messages"], message))
        return result["messages"][-1].content

    active_types = memory_types if memory_types is not None else ALL_MEMORY_TYPES
    graph = build_graph(
        with_memory=True, extraction_mode=extraction_mode, memory_types=active_types,
        search_strategy=search_strategy,
        checkpointer=checkpointer, store=store, user_id=user_id, search_service=search_service,
    )
    result = await graph.ainvoke({"messages": [{"role": "user", "content": message}]}, config=config)

    if turn_log is not None:
        turn_log.update(_extract_turn_log(result["messages"], message))
        if MemoryType.EPISODIC in active_types:
            found = await asearch_episodes(store, user_id, message, strategy=search_strategy)
            turn_log["episodes_found"] = [
                {
                    "query": it.value.get("content", {}).get("query"),
                    "summary": it.value.get("content", {}).get("summary"),
                    "score": it.score,
                }
                for it in found
            ]

    if extraction_mode in (ExtractionMode.BACKGROUND, ExtractionMode.BOTH):
        pending = await run_background_extraction(
            user_id, result["messages"], active_types, executors,
            extraction_mode=extraction_mode, thread_id=thread_id,
        )
        if pending_bg is not None:
            pending_bg.extend(pending)
        else:
            await asyncio.gather(*pending)

    if turn_log is not None and MemoryType.SEMANTIC in active_types:
        if pending_bg is None:
            _, profile = await aget_current_profile(store, user_id)
            turn_log["profile_after_write"] = profile

    return result["messages"][-1].content


async def run_one(
    scenario: Scenario,
    variant_key: str,
    variants: dict[str, tuple[str, ExtractionMode | None, frozenset[MemoryType] | None]],
    checkpointer,
    store: BaseStore,
    search_service: RealEstateSearchService,
    user_prefix: str,
    run_ts: str,
    axis_label: str,
    baseline_latencies: list[float],
    repeat_index: int = 0,
    search_strategy: SearchStrategy = SearchStrategy.SIMILARITY,
    executors: ReflectionExecutors | None = None,
) -> dict:
    agent_variant, extraction_mode, memory_types = variants[variant_key]
    active_types = memory_types if memory_types is not None else ALL_MEMORY_TYPES
    username = f"{user_prefix}_{axis_label}_r{repeat_index}_{run_ts}_{scenario.name}_{variant_key}"
    user = get_or_create_user(username)

    thread_ids: list[str] = []
    session_replies: list[list[str]] = []
    turn_latencies: list[float] = []
    turn_logs: list[dict] = []

    for session_index, session_messages in enumerate(scenario.sessions):
        thread = threads_repo.create_thread(
            user["id"],
            agent_variant=agent_variant,
            extraction_mode=extraction_mode.value if extraction_mode else None,
        )
        thread_ids.append(str(thread["thread_id"]))

        replies = []
        for message in session_messages:
            turn_log: dict = {"message": message, "session_index": session_index}
            turn_pending: list = []
            t0 = time.perf_counter()
            reply = await run_turn(
                checkpointer, store, user["id"], thread["thread_id"], message,
                agent_variant, extraction_mode, memory_types, search_service,
                search_strategy=search_strategy,
                executors=executors, pending_bg=turn_pending, turn_log=turn_log,
            )
            turn_latencies.append(time.perf_counter() - t0)
            replies.append(reply)

            if turn_pending:
                await asyncio.gather(*turn_pending)
            if agent_variant == "memory" and MemoryType.SEMANTIC in active_types and "profile_after_write" not in turn_log:
                _, profile = await aget_current_profile(store, user["id"])
                turn_log["profile_after_write"] = profile

            turn_log["reply"] = reply
            turn_logs.append(turn_log)
        session_replies.append(replies)

    accuracy_strict = metrics.memory_accuracy_strict(scenario, session_replies)
    consistency = metrics.behavioral_consistency(scenario, session_replies)
    false_mem = metrics.false_memory_rate(scenario, session_replies)
    episodic_retrieval = metrics.episodic_retrieval_accuracy(scenario, turn_logs)
    avg_latency = sum(turn_latencies) / len(turn_latencies) if turn_latencies else 0.0
    overhead = (
        None if agent_variant == "baseline"
        else metrics.latency_overhead_seconds(turn_latencies, baseline_latencies)
    )

    metrics_json = {
        "memory_accuracy_strict": accuracy_strict,
        "behavioral_consistency": consistency,
        "false_memory_rate": false_mem,
        "episodic_retrieval_accuracy": episodic_retrieval,
        "avg_latency_seconds": avg_latency,
        "per_turn_latency_seconds": turn_latencies,
        "latency_overhead_vs_baseline_seconds": overhead,
    }

    insert_eval_run(
        scenario_name=scenario.name,
        agent_variant=agent_variant,
        extraction_mode=extraction_mode.value if extraction_mode else None,
        user_id=user["id"],
        thread_ids=thread_ids,
        metrics=metrics_json,
    )

    transcript = {
        "axis": axis_label,
        "scenario": scenario.name,
        "variant": variant_key,
        "username": username,
        "user_id": user["id"],
        "thread_ids": thread_ids,
        "search_strategy": search_strategy.value,
        "sessions": [
            {"messages": msgs, "replies": reps}
            for msgs, reps in zip(scenario.sessions, session_replies)
        ],
        "turn_logs": turn_logs,
        "metrics": metrics_json,
    }

    print(
        f"[{axis_label}/r{repeat_index}/{scenario.name}/{variant_key}] "
        f"accuracy_strict={accuracy_strict:.2f} consistency={consistency:.2f} "
        f"avg_latency={avg_latency:.2f}s"
        + (f" overhead={overhead:+.2f}s" if overhead is not None else "")
        + (f" false_memory={false_mem:.2f}" if false_mem is not None else "")
        + (f" episodic_retrieval={episodic_retrieval:.2f}" if episodic_retrieval is not None else "")
    )

    return transcript


async def run_axis(
    axis_label: str,
    variants: dict[str, tuple[str, ExtractionMode | None, frozenset[MemoryType] | None]],
    scenario_names: str,
    variant_names: str,
    user_prefix: str,
    run_ts: str,
    search_service: RealEstateSearchService,
    checkpointer,
    store: BaseStore,
    results_dir: Path,
    repeat_index: int = 0,
    search_strategy: SearchStrategy = SearchStrategy.SIMILARITY,
    executors: ReflectionExecutors | None = None,
) -> None:
    scenarios = resolve_scenarios(scenario_names)
    variant_keys = (
        list(variants.keys()) if variant_names == "all"
        else [v.strip() for v in variant_names.split(",")]
    )
    for key in variant_keys:
        if key not in variants:
            raise ValueError(f"Unknown {axis_label} variant '{key}'. Known: {list(variants)}")
    variant_keys = [k for k in variants if k in variant_keys]

    skipped = 0
    for scenario in scenarios:
        baseline_latencies: list[float] = []
        run_name = f"{axis_label}_r{repeat_index}__{scenario.name}"
        for variant_key in variant_keys:
            finished = transcript_path(results_dir, run_name, variant_key)
            if finished.exists():
                skipped += 1
                if variant_key == "baseline":
                    baseline_latencies = json.loads(finished.read_text(encoding="utf-8"))["metrics"]["per_turn_latency_seconds"]
                continue

            # A malformed LLM tool call should skip only this (scenario, variant, repeat).
            try:
                transcript = await run_one(
                    scenario, variant_key, variants, checkpointer, store, search_service,
                    user_prefix, run_ts, axis_label, baseline_latencies, repeat_index,
                    search_strategy=search_strategy, executors=executors,
                )
            except Exception as e:
                if _is_quota_error(e):
                    raise QuotaExhausted from e
                print(
                    f"[{axis_label}/r{repeat_index}/{scenario.name}/{variant_key}] "
                    f"FAILED, skipping: {type(e).__name__}: {e}"
                )
                continue

            if variant_key == "baseline":
                baseline_latencies = transcript["metrics"]["per_turn_latency_seconds"]

            write_run_transcript(results_dir, run_name, variant_key, transcript)

    if skipped:
        print(f"[{axis_label}/r{repeat_index}] {skipped} run(s) already finished, skipped")


async def run_procedure_following_check(
    checkpointer, store: BaseStore, search_service: RealEstateSearchService, user_prefix: str, run_ts: str
) -> dict:
    username = f"{user_prefix}_procedure_{run_ts}"
    user = get_or_create_user(username)

    seed_thread = threads_repo.create_thread(
        user["id"], agent_variant="memory", extraction_mode=ExtractionMode.HOT_PATH.value
    )
    seed_message = (
        "Find me cheap sunny towns in Texas. Please always tell me how far the nearest "
        "hospital is from now on, every single time, even if I don't ask again."
    )
    await run_turn(
        checkpointer, store, user["id"], seed_thread["thread_id"], seed_message,
        "memory", ExtractionMode.HOT_PATH, ALL_MEMORY_TYPES, search_service,
    )

    baseline_graph = build_graph(
        with_memory=False, checkpointer=checkpointer, store=None,
        user_id=user["id"], search_service=search_service,
    )
    snapshot = await baseline_graph.aget_state({"configurable": {"thread_id": seed_thread["thread_id"]}})
    seed_messages = snapshot.values.get("messages", []) if snapshot else []
    if not seed_messages:
        return {"username": username, "examples": [], "procedure_adherence": None}

    await aoptimize_procedural_prompt(
        user["id"], [AnnotatedTrajectory(messages=seed_messages, feedback=None)], [str(seed_thread["thread_id"])]
    )

    test_cases = [("Find me warm affordable towns in Florida.", ["hospital"])]
    examples = []
    example_log = []
    for query, expected_markers in test_cases:
        thread = threads_repo.create_thread(user["id"], agent_variant="memory", extraction_mode=None)
        reply = await run_turn(
            checkpointer, store, user["id"], thread["thread_id"], query,
            "memory", None, frozenset({MemoryType.PROCEDURAL}), search_service,
        )
        examples.append((expected_markers, reply))
        example_log.append({"query": query, "expected_markers": expected_markers, "reply": reply})

    adherence = metrics.procedure_adherence(examples)
    result = {"username": username, "examples": example_log, "procedure_adherence": adherence}
    print(f"[procedure_following] adherence={adherence:.2f} over {len(examples)} example(s)")
    return result


# Old and new marker sets are disjoint words so "old value absent" stays unambiguous.
CONTRADICTION_CASES = [
    {
        "name": "budget_change",
        "old_message": "My budget for a new home is around $200,000.",
        "new_message": "Actually, my budget changed — it's now around $300,000.",
        "old_values": ["200,000", "200000", "$200k", "200k"],
        "new_values": ["300,000", "300000", "$300k", "300k"],
    },
    {
        "name": "preferred_state_change",
        "old_message": "I'm only interested in towns in Texas.",
        "new_message": "Actually, forget Texas — I only want towns in Vermont now.",
        "old_values": ["texas"],
        "new_values": ["vermont"],
    },
]
FOLLOWUP_QUERY = "What towns would you recommend for me?"


async def run_update_check(
    checkpointer, store: BaseStore, search_service: RealEstateSearchService, user_prefix: str, run_ts: str,
    repeats: int = 1,
) -> dict:
    results = []
    for case in CONTRADICTION_CASES:
        for session_mode in ("same_session", "new_session"):
            for repeat_index in range(repeats):
                username = f"{user_prefix}_update_{case['name']}_{session_mode}_r{repeat_index}_{run_ts}"
                user = get_or_create_user(username)
                thread = threads_repo.create_thread(
                    user["id"], agent_variant="memory", extraction_mode=ExtractionMode.HOT_PATH.value
                )

                for message in (case["old_message"], case["new_message"]):
                    await run_turn(
                        checkpointer, store, user["id"], thread["thread_id"], message,
                        "memory", ExtractionMode.HOT_PATH, frozenset({MemoryType.SEMANTIC}), search_service,
                    )

                followup_thread = thread
                if session_mode == "new_session":
                    followup_thread = threads_repo.create_thread(
                        user["id"], agent_variant="memory", extraction_mode=ExtractionMode.HOT_PATH.value
                    )

                followup_reply = await run_turn(
                    checkpointer, store, user["id"], followup_thread["thread_id"], FOLLOWUP_QUERY,
                    "memory", ExtractionMode.HOT_PATH, frozenset({MemoryType.SEMANTIC}), search_service,
                )

                accuracy = metrics.memory_update_accuracy(
                    old_values=case["old_values"], new_values=case["new_values"],
                    replies_after_update=[followup_reply],
                )
                results.append({
                    "case": case["name"], "session_mode": session_mode, "repeat": repeat_index,
                    "username": username, "followup_reply": followup_reply,
                    "memory_update_accuracy": accuracy,
                })
                print(
                    f"[update_check/{case['name']}/{session_mode}/r{repeat_index}] "
                    f"memory_update_accuracy={accuracy:.2f}"
                )

    summary: dict[str, float] = {}
    for r in results:
        key = f"{r['case']}/{r['session_mode']}"
        summary.setdefault(key, []).append(r["memory_update_accuracy"])
    summary = {k: sum(v) / len(v) for k, v in summary.items()}

    return {"results": results, "summary_by_case_and_session_mode": summary}


async def run_all(
    scenario_names: str, variant_names: str, user_prefix: str, llm_model: str, axis: str, repeats: int = 5,
    search_strategy: SearchStrategy = SearchStrategy.SIMILARITY, resume_dir: Path | None = None,
) -> Path:
    # Fresh usernames even when resuming: a half-finished run leaves its memory under the old one.
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = resume_dir or RESULTS_ROOT / run_ts
    if resume_dir is None:
        results_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "scenario_names": scenario_names, "variant_names": variant_names, "user_prefix": user_prefix,
            "llm_model": llm_model, "axis": axis, "repeats": repeats, "search_strategy": search_strategy.value,
        }
        (results_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")
    search_service = RealEstateSearchService("gradovi.db", use_llm=True, llm_model=llm_model)
    exhausted = False

    async with AsyncPostgresSaver.from_conn_string(get_conn_string()) as checkpointer, \
            AsyncPostgresStore.from_conn_string(get_conn_string(), index=STORE_INDEX_CONFIG) as store:

        # One set of executors for the whole run: each owns a worker thread and debounces per thread_id.
        executors = ReflectionExecutors(store, asyncio.get_running_loop())

        try:
            for repeat_index in range(repeats):
                if repeats > 1:
                    print(f"\n=== repeat {repeat_index + 1}/{repeats} ===")

                if axis in ("mechanism", "all"):
                    await run_axis(
                        "mechanism", MECHANISM_VARIANTS, scenario_names, variant_names,
                        user_prefix, run_ts, search_service, checkpointer, store, results_dir, repeat_index,
                        search_strategy=search_strategy, executors=executors,
                    )

                if axis in ("type", "all"):
                    await run_axis(
                        "type", TYPE_VARIANTS, scenario_names, variant_names,
                        user_prefix, run_ts, search_service, checkpointer, store, results_dir, repeat_index,
                        search_strategy=search_strategy, executors=executors,
                    )

            if axis == "all":
                procedure_file = results_dir / "procedure_following.json"
                if not procedure_file.exists():
                    try:
                        procedure_result = await run_procedure_following_check(
                            checkpointer, store, search_service, user_prefix, run_ts
                        )
                        procedure_file.write_text(json.dumps(procedure_result, indent=2, default=str), encoding="utf-8")
                    except Exception as e:
                        if _is_quota_error(e):
                            raise QuotaExhausted from e
                        print(f"[procedure_following] FAILED, skipping: {type(e).__name__}: {e}")

                update_file = results_dir / "update_proof_of_concept.json"
                if not update_file.exists():
                    try:
                        update_result = await run_update_check(
                            checkpointer, store, search_service, user_prefix, run_ts, repeats=repeats
                        )
                        update_file.write_text(json.dumps(update_result, indent=2, default=str), encoding="utf-8")
                    except Exception as e:
                        if _is_quota_error(e):
                            raise QuotaExhausted from e
                        print(f"[update_check] FAILED, skipping: {type(e).__name__}: {e}")
        except QuotaExhausted:
            exhausted = True

        write_summaries(results_dir, repeats)

        # Stop the executors' non-daemon threads; must run before `store` closes, since it
        # drains in-flight writes.
        await asyncio.get_running_loop().run_in_executor(None, executors.shutdown)

    if exhausted:
        print(
            f"\nOpenAI credit ran out. Every finished run is saved in {results_dir}.\n"
            f"After adding credit, continue where it stopped with:\n"
            f"  python -m eval.runner --resume {results_dir.name}"
        )
    else:
        print(f"\nDone. Results at {results_dir}")
    return results_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="all", help='"all" or comma-separated scenario names')
    parser.add_argument("--variants", default="all", help='"all" or comma-separated variant keys (per axis)')
    parser.add_argument("--axis", choices=["mechanism", "type", "all"], default="all")
    parser.add_argument("--user-prefix", default="eval", help="prefix for the throwaway eval usernames")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument(
        "--repeats", type=int, default=5,
        help="independent repeats per (scenario, variant), default 5, each with fresh users — "
             "a single LLM-driven run is a noisy sample, so >1 gives mean+stdev "
             "instead of one draw (writes an extra *_aggregated.csv)",
    )
    parser.add_argument(
        "--search-strategy", choices=[s.value for s in SearchStrategy],
        default=SearchStrategy.SIMILARITY.value,
        help="how episodic memories are ranked for retrieval — vector similarity "
             "(default), lexical keyword overlap, or a 50/50 hybrid of the two",
    )
    parser.add_argument(
        "--resume", nargs="?", const="latest", metavar="FOLDER",
        help="continue an interrupted run in eval/results/FOLDER (default: the latest one), "
             "with the settings it was started with; finished runs are skipped",
    )
    return parser.parse_args()


def _resolve_resume_dir(name: str) -> Path:
    if name == "latest":
        candidates = sorted(p for p in RESULTS_ROOT.iterdir() if (p / CONFIG_FILE).exists())
        if not candidates:
            raise SystemExit(f"No resumable run found: no folder in {RESULTS_ROOT} has {CONFIG_FILE}.")
        return candidates[-1]
    path = RESULTS_ROOT / name
    if not (path / CONFIG_FILE).exists():
        raise SystemExit(f"{path} has no {CONFIG_FILE}, so it was not started by a resumable runner.")
    return path


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.resume:
        resume_dir = _resolve_resume_dir(args.resume)
        config = json.loads((resume_dir / CONFIG_FILE).read_text(encoding="utf-8"))
        print(f"Resuming {resume_dir.name} with the settings it was started with.")
    else:
        resume_dir = None
        config = {
            "scenario_names": args.scenarios, "variant_names": args.variants, "user_prefix": args.user_prefix,
            "llm_model": args.llm_model, "axis": args.axis, "repeats": args.repeats,
            "search_strategy": args.search_strategy,
        }
    asyncio.run(
        run_all(**{**config, "search_strategy": SearchStrategy(config["search_strategy"])}, resume_dir=resume_dir),
        loop_factory=windows_loop_factory(),
    )


if __name__ == "__main__":
    main()
