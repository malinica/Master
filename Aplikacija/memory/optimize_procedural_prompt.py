import asyncio
import sys

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langmem.prompts.types import AnnotatedTrajectory

from db import threads_repo
from db.connection import get_conn_string, windows_loop_factory
from db.procedural_repo import get_latest_procedural_prompt, list_procedural_prompts
from db.users_repo import get_or_create_user
from graph.builder import build_graph
from graph.prompts import BASE_SYSTEM_PROMPT
from memory.procedural import aoptimize_procedural_prompt
from realestate.service import RealEstateSearchService


async def gather_trajectories(checkpointer, user_id: int, thread_ids: list[str]) -> list[AnnotatedTrajectory]:
    graph = build_graph(
        with_memory=False,
        checkpointer=checkpointer,
        store=None,
        user_id=user_id,
        search_service=RealEstateSearchService("gradovi.db", use_llm=False),
    )
    trajectories = []
    for thread_id in thread_ids:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(config)
        messages = snapshot.values.get("messages", []) if snapshot else []
        if messages:
            trajectories.append(AnnotatedTrajectory(messages=messages, feedback=None))
    return trajectories


async def main() -> None:
    load_dotenv()
    if len(sys.argv) < 2:
        print("Usage: python -m memory.optimize_procedural_prompt <username>")
        return
    username = sys.argv[1]
    user = get_or_create_user(username)

    memory_threads = [
        t for t in threads_repo.list_threads_for_user(user["id"]) if t["agent_variant"] == "memory"
    ]
    if not memory_threads:
        print(f"No memory-variant threads found for user '{username}' — nothing to optimize from.")
        return

    thread_ids = [str(t["thread_id"]) for t in memory_threads]
    print(f"Gathering trajectories from {len(thread_ids)} memory-variant thread(s)...")

    async with AsyncPostgresSaver.from_conn_string(get_conn_string()) as checkpointer:
        trajectories = await gather_trajectories(checkpointer, user["id"], thread_ids)

    if not trajectories:
        print("Threads found but no messages recovered — nothing to optimize from.")
        return
    print(f"Collected {len(trajectories)} trajectory(ies).")

    before = get_latest_procedural_prompt(user["id"])
    before_text = before["prompt_text"] if before else BASE_SYSTEM_PROMPT
    print(f"\n=== Prompt BEFORE (version {before['version'] if before else 0}) ===")
    print(before_text)

    result = await aoptimize_procedural_prompt(user["id"], trajectories, thread_ids)

    print(f"\n=== Prompt AFTER (version {result['version']}) ===")
    print(result["prompt_text"])

    print(f"\nAll versions stored for this user: {[p['version'] for p in list_procedural_prompts(user['id'])]}")


if __name__ == "__main__":
    asyncio.run(main(), loop_factory=windows_loop_factory())
