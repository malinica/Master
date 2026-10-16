from langgraph.store.base import BaseStore, SearchItem
from langmem import create_manage_memory_tool

from memory.config import SearchStrategy
from memory.namespaces import episodic_ns
from memory.schemas import SearchEpisode
from memory.stamped_store import KindStampingStore

_CANDIDATE_POOL_SIZE = 50

EPISODIC_TOOL_INSTRUCTIONS = (
    "Call this after you have shown the user real_estate_search results, to log what "
    "was searched for and found. Each call creates a NEW, independent record — unlike "
    "the user profile, this is a growing history, not a single document to update. "
    "Do not call action='update' or action='delete'. You MUST pass a `content` object "
    "with `query`, `summary`, and `top_result_towns` filled in — never omit content "
    "or leave it null."
)


def make_episodic_memory_tool(user_id: int, store: BaseStore):
    return create_manage_memory_tool(
        namespace=episodic_ns(user_id),
        schema=SearchEpisode,
        actions_permitted=("create",),
        store=KindStampingStore(store, SearchEpisode.__name__),
        instructions=EPISODIC_TOOL_INSTRUCTIONS,
        name="log_search_episode",
    )


def _episode_text(item: SearchItem) -> str:
    content = item.value.get("content", {})
    return f"{content.get('query', '')} {content.get('summary', '')}".lower()


def _keyword_overlap(query: str, item: SearchItem) -> float:
    query_words = set(query.lower().split())
    if not query_words:
        return 0.0
    episode_words = set(_episode_text(item).split())
    return len(query_words & episode_words) / len(query_words)


async def asearch_episodes(
    store: BaseStore,
    user_id: int,
    query: str,
    limit: int = 3,
    strategy: SearchStrategy = SearchStrategy.SIMILARITY,
) -> list[SearchItem]:
    namespace = episodic_ns(user_id)

    if strategy == SearchStrategy.SIMILARITY:
        return await store.asearch(namespace, query=query, limit=limit)

    candidates = await store.asearch(namespace, limit=_CANDIDATE_POOL_SIZE)
    if not candidates:
        return []

    if strategy == SearchStrategy.KEYWORD:
        ranked = sorted(candidates, key=lambda item: _keyword_overlap(query, item), reverse=True)
    else:  # HYBRID
        similarity_candidates = await store.asearch(namespace, query=query, limit=_CANDIDATE_POOL_SIZE)
        similarity_scores = {item.key: (item.score or 0.0) for item in similarity_candidates}
        ranked = sorted(
            candidates,
            key=lambda item: 0.5 * similarity_scores.get(item.key, 0.0) + 0.5 * _keyword_overlap(query, item),
            reverse=True,
        )
    return ranked[:limit]
