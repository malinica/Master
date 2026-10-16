import uuid
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore
from langmem import create_manage_memory_tool, create_memory_store_manager

from memory.namespaces import semantic_ns
from memory.schemas import UserProfile
from memory.stamped_store import KindStampingStore

SEMANTIC_TOOL_INSTRUCTIONS = (
    "This namespace holds exactly ONE user profile document, not a list of separate "
    "memories. The system prompt shows you the CURRENT PROFILE and its id (if one "
    "exists). If an id is shown, ALWAYS call this tool with action='update' and that "
    "same id — never create a second profile. Only use action='create' if the system "
    "prompt says no profile exists yet for this user. You MUST pass a `content` object "
    "with the relevant fields filled in (merging with what's already known) — never "
    "omit content or leave it null."
)


def make_semantic_memory_tool(user_id: int, store: BaseStore):
    return create_manage_memory_tool(
        namespace=semantic_ns(user_id),
        schema=UserProfile,
        actions_permitted=("create", "update"),
        store=KindStampingStore(store, UserProfile.__name__),
        instructions=SEMANTIC_TOOL_INSTRUCTIONS,
        name="update_user_profile",
    )


async def aget_current_profile(store: BaseStore, user_id: int) -> Tuple[Optional[str], Optional[dict]]:
    items = await store.asearch(semantic_ns(user_id), limit=1)
    if not items:
        return None, None
    item = items[0]
    return item.key, item.value.get("content")


def _messages_to_text(messages: list) -> str:
    lines = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if role in ("human", "ai") and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def aupdate_profile_from_messages(
    store: BaseStore,
    user_id: int,
    messages: list,
    llm_model: str = "gpt-4o-mini",
) -> UserProfile:
    profile_id, current_content = await aget_current_profile(store, user_id)

    extractor = ChatOpenAI(model=llm_model, temperature=0.1).with_structured_output(UserProfile)
    conversation_text = _messages_to_text(messages)

    prompt = (
        "Current known profile for this user (JSON, may be empty if none exists yet):\n"
        f"{current_content or {}}\n\n"
        "Conversation to learn from:\n"
        f"{conversation_text}\n\n"
        "Return the FULL updated profile: keep existing fields that are still valid, "
        "and merge in any new lasting preferences mentioned in the conversation above "
        "(budget, preferred states, climate, dislikes, hospital/school needs). Do not "
        "invent facts that weren't stated."
    )

    updated: UserProfile = await extractor.ainvoke(prompt)

    key = profile_id or str(uuid.uuid4())
    await store.aput(
        semantic_ns(user_id),
        key=key,
        value={"kind": UserProfile.__name__, "content": updated.model_dump()},
    )
    return updated


def make_semantic_store_manager(store: BaseStore, user_id: int, llm_model: str = "gpt-4o-mini"):
    return create_memory_store_manager(
        ChatOpenAI(model=llm_model, temperature=0.1),
        schemas=[UserProfile],
        namespace=semantic_ns(user_id),
        enable_inserts=True,
        enable_deletes=False,
        store=store,
    )
