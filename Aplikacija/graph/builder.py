from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from graph.prompts import BASE_SYSTEM_PROMPT, build_memory_prompt
from graph.tools import make_current_weather_tool, make_real_estate_tool
from memory.config import ALL_MEMORY_TYPES, ExtractionMode, MemoryType, SearchStrategy
from memory.episodic import make_episodic_memory_tool
from memory.semantic import make_semantic_memory_tool
from realestate.service import RealEstateSearchService

ROLE_BY_MESSAGE_TYPE = {"human": "user", "ai": "assistant"}


def _on_update_user_profile_error(exc: Exception, request) -> str | None:
    if isinstance(exc, ValueError):
        return (
            "Profile not saved: action='update' needs the existing memory id. If CURRENT PROFILE "
            "shows no profile yet, call update_user_profile again with action='create' and no id; "
            "otherwise pass the id shown in CURRENT PROFILE."
        )
    return None


def messages_to_chat_display(messages) -> list[dict]:
    display = []
    for msg in messages:
        role = ROLE_BY_MESSAGE_TYPE.get(getattr(msg, "type", None))
        if role and msg.content:
            display.append({"role": role, "content": msg.content})
    return display


def build_graph(
    *,
    with_memory: bool,
    extraction_mode: ExtractionMode | None = None,
    memory_types: frozenset[MemoryType] | None = None,
    search_strategy: SearchStrategy = SearchStrategy.SIMILARITY,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore | None = None,
    user_id: int,
    search_service: RealEstateSearchService,
    llm_model: str = "gpt-4o-mini",
):
    tools = [make_real_estate_tool(search_service), make_current_weather_tool()]

    if not with_memory:
        graph = create_agent(
            model=ChatOpenAI(model=llm_model, temperature=0.2),
            tools=tools,
            system_prompt=BASE_SYSTEM_PROMPT,
            checkpointer=checkpointer,
            store=store,
        )
        return graph

    if store is None:
        raise ValueError("build_graph(with_memory=True, ...) requires a store")

    active_types = memory_types if memory_types is not None else ALL_MEMORY_TYPES

    if extraction_mode in (ExtractionMode.HOT_PATH, ExtractionMode.BOTH):
        if MemoryType.SEMANTIC in active_types:
            tools.append(make_semantic_memory_tool(user_id, store))
        if MemoryType.EPISODIC in active_types:
            tools.append(make_episodic_memory_tool(user_id, store))

    graph = create_agent(
        model=ChatOpenAI(model=llm_model, temperature=0.2),
        tools=tools,
        middleware=[
            build_memory_prompt(user_id, extraction_mode, active_types, search_strategy),
            ToolErrorMiddleware(_on_update_user_profile_error, tools=["update_user_profile"]),
        ],
        checkpointer=checkpointer,
        store=store,
    )
    return graph
