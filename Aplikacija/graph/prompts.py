from langchain.agents.middleware import dynamic_prompt
from langgraph.config import get_store

from db.procedural_repo import get_latest_procedural_prompt
from memory.config import ExtractionMode, MemoryType, SearchStrategy
from memory.episodic import asearch_episodes
from memory.semantic import aget_current_profile

EPISODIC_RETRIEVAL_LIMIT = 3
_EPISODE_SUMMARY_CHAR_CAP = 220

SEMANTIC_READ_RULE = """
CURRENT PROFILE (shown below) holds what you already know about this user. Use it: when you search
for or recommend places, build the real_estate_search query from the profile preferences plus whatever
the user says now. Never ask the user to repeat something the profile already contains, and if some
preferences are missing, search with what you know instead of asking for the rest.
"""

SEMANTIC_TOOL_USAGE_HINT = """
When the user states a new lasting preference (budget, preferred states, climate, dislikes such as
humidity, whether they need schools or hospitals nearby), also call update_user_profile with the
complete merged profile in `content`, which must always be filled in. If CURRENT PROFILE shows an id,
use action='update' with that id; if it shows no profile yet, use action='create' without an id.
"""

EPISODIC_TOOL_USAGE_HINT = """
Call log_search_episode, in the SAME turn, right after calling real_estate_search —
this is not optional bookkeeping, it is a required step every single time you call
real_estate_search, with no exceptions. You MUST always fill in `content` with all of
its relevant fields populated based on the conversation — never call it with `content`
omitted or null.
"""

BASE_SYSTEM_PROMPT = """You are EstAIte, a helpful real estate search assistant.

You help users find US towns and cities to live in based on their preferences about
price, weather, temperature, hospitals, and schools. When the user describes what
they're looking for, call the real_estate_search tool with their request (in natural
language) and present the results back to them clearly. Do not fabricate town data —
always use the tool to look up real information.

You also have a current_weather tool that fetches today's actual live weather for a
specific US city from Open-Meteo. Only call it when the user explicitly asks about
current/today's/right-now weather for a place — not for the historical climate
matching that real_estate_search already handles.
"""


def load_base_prompt_text(user_id: int, with_memory: bool, memory_types: frozenset[MemoryType]) -> str:
    if not with_memory or MemoryType.PROCEDURAL not in memory_types:
        return BASE_SYSTEM_PROMPT

    latest = get_latest_procedural_prompt(user_id)
    return latest["prompt_text"] if latest else BASE_SYSTEM_PROMPT


def _last_human_message_text(messages: list) -> str | None:
    for message in reversed(messages):
        if getattr(message, "type", None) == "human" and message.content:
            return message.content
    return None


def _format_episodes_block(episodes: list) -> str:
    if not episodes:
        return "RELEVANT PAST SEARCHES: none found for this query."
    lines = ["RELEVANT PAST SEARCHES:"]
    for item in episodes:
        content = item.value.get("content", {})
        summary = (content.get("summary") or "")[:_EPISODE_SUMMARY_CHAR_CAP]
        towns = ", ".join(content.get("top_result_towns", []) or [])
        lines.append(f"- query: {content.get('query', '')!r} | summary: {summary} | towns: {towns}")
    return "\n".join(lines)


def build_memory_prompt(
    user_id: int,
    extraction_mode: ExtractionMode | None,
    memory_types: frozenset[MemoryType],
    search_strategy: SearchStrategy = SearchStrategy.SIMILARITY,
):
    base_text = load_base_prompt_text(user_id, with_memory=True, memory_types=memory_types)

    if MemoryType.SEMANTIC in memory_types:
        base_text = f"{base_text}\n{SEMANTIC_READ_RULE}"

    if extraction_mode in (ExtractionMode.HOT_PATH, ExtractionMode.BOTH):
        if MemoryType.SEMANTIC in memory_types:
            base_text = f"{base_text}\n{SEMANTIC_TOOL_USAGE_HINT}"
        if MemoryType.EPISODIC in memory_types:
            base_text = f"{base_text}\n{EPISODIC_TOOL_USAGE_HINT}"

    @dynamic_prompt
    async def prompt(request):
        text = base_text
        store = get_store()

        if MemoryType.SEMANTIC in memory_types:
            profile_id, profile = await aget_current_profile(store, user_id)
            if profile_id is None:
                profile_block = "CURRENT PROFILE: none yet for this user."
            else:
                profile_block = f"CURRENT PROFILE (id={profile_id}):\n{profile}"
            text = f"{text}\n\n{profile_block}"

        if MemoryType.EPISODIC in memory_types:
            query = _last_human_message_text(request.messages)
            episodes = (
                await asearch_episodes(
                    store, user_id, query, limit=EPISODIC_RETRIEVAL_LIMIT, strategy=search_strategy
                )
                if query
                else []
            )
            text = f"{text}\n\n{_format_episodes_block(episodes)}"

        return text

    return prompt
