import asyncio

from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore
from langmem import create_memory_store_manager

from memory.config import ExtractionMode, MemoryType, SemanticBackgroundStrategy
from memory.namespaces import episodic_ns
from memory.schemas import SearchEpisode


def make_episodic_background_manager(store: BaseStore, user_id: int, llm_model: str = "gpt-4o-mini"):
    return create_memory_store_manager(
        ChatOpenAI(model=llm_model, temperature=0.1),
        schemas=[SearchEpisode],
        namespace=episodic_ns(user_id),
        enable_inserts=True,
        enable_deletes=False,
        store=store,
    )


async def run_background_extraction(
    user_id: int,
    messages: list,
    memory_types: frozenset[MemoryType],
    executors,
    semantic_strategy: SemanticBackgroundStrategy = SemanticBackgroundStrategy.FETCH_MERGE,
    extraction_mode: ExtractionMode | None = None,
    thread_id: str | None = None,
) -> list[asyncio.Future]:
    pending: list[asyncio.Future] = []
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None

    if MemoryType.SEMANTIC in memory_types:
        future = executors.semantic(user_id, semantic_strategy).submit(
            {"messages": messages}, config=config, thread_id=thread_id
        )
        pending.append(asyncio.wrap_future(future))

    if MemoryType.EPISODIC in memory_types and extraction_mode != ExtractionMode.BOTH:
        future = executors.episodic(user_id).submit(
            {"messages": messages}, config=config, thread_id=thread_id
        )
        pending.append(asyncio.wrap_future(future))

    return pending
