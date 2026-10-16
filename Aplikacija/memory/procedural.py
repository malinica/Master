from langchain_openai import ChatOpenAI
from langmem import Prompt, create_prompt_optimizer
from langmem.prompts.types import AnnotatedTrajectory

from db.procedural_repo import get_latest_procedural_prompt, insert_procedural_prompt
from graph.prompts import BASE_SYSTEM_PROMPT

UPDATE_INSTRUCTIONS = (
    "Make minimal, targeted changes to how the assistant should search, log "
    "memories, and respond, based only on the observed conversations. It must "
    "remain a system prompt for a real estate search assistant that calls the "
    "real_estate_search tool — do not remove that instruction."
)
WHEN_TO_UPDATE = (
    "Whenever the trajectories reveal a recurring mistake (skipped tool calls, "
    "ignored user preferences, unhelpful answers) or a missed opportunity to "
    "serve the user better."
)


def make_prompt_optimizer(llm_model: str = "gpt-4o-mini"):
    return create_prompt_optimizer(
        ChatOpenAI(model=llm_model, temperature=0.0),
        kind="gradient",
    )


async def aoptimize_procedural_prompt(
    user_id: int,
    trajectories: list[AnnotatedTrajectory],
    source_thread_ids: list[str],
    llm_model: str = "gpt-4o-mini",
) -> dict:
    latest = get_latest_procedural_prompt(user_id)
    current_text = latest["prompt_text"] if latest else BASE_SYSTEM_PROMPT
    next_version = (latest["version"] + 1) if latest else 1

    optimizer = make_prompt_optimizer(llm_model)
    optimized_text = await optimizer.ainvoke(
        {
            "trajectories": trajectories,
            "prompt": Prompt(
                name="estaite_system_prompt",
                prompt=current_text,
                update_instructions=UPDATE_INSTRUCTIONS,
                when_to_update=WHEN_TO_UPDATE,
            ),
        }
    )

    return insert_procedural_prompt(
        user_id=user_id,
        version=next_version,
        prompt_text=optimized_text,
        source_thread_ids=source_thread_ids,
    )
