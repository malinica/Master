import gradio as gr

from db import threads_repo, users_repo
from db.users_repo import InvalidCredentials
from db.runtime import run_coroutine
from graph.builder import build_graph, messages_to_chat_display
from memory.background import run_background_extraction
from memory.config import ALL_MEMORY_TYPES, ExtractionMode, MemoryType, SearchStrategy
from memory.reflection_executors import ReflectionExecutors


def _parse_memory_types(memory_types_str: str | None) -> frozenset[MemoryType]:
    if not memory_types_str:
        return ALL_MEMORY_TYPES
    return frozenset(MemoryType(t) for t in memory_types_str.split(","))


class ChatHandlers:
    def __init__(
        self, search_service, checkpointer, store, loop,
        search_strategy: SearchStrategy = SearchStrategy.SIMILARITY,
    ):
        self.search_service = search_service
        self.checkpointer = checkpointer
        self.store = store
        self.search_strategy = search_strategy
        self.reflection_executors = ReflectionExecutors(store, loop)

    def login(self, username: str, password: str):
        username = (username or "").strip()
        password = password or ""
        if not username or not password:
            return (
                None,
                "Enter both username and password to continue.",
                gr.update(choices=[]),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        try:
            user = users_repo.authenticate_user(username, password)
            threads = threads_repo.list_threads_for_user(user["id"])
        except InvalidCredentials:
            return (
                None,
                f" Wrong password for '{username}'.",
                gr.update(choices=[]),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        except Exception as e:
            return (
                None,
                f" Could not reach the database: {e}",
                gr.update(choices=[]),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        choices = [(self._thread_label(t), t["thread_id"]) for t in threads]
        return (
            user["id"],
            f"Logged in as {user['display_name']}",
            gr.update(choices=choices, value=None),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    @staticmethod
    def _thread_label(thread: dict) -> str:
        title = thread.get("title") or "(untitled)"
        variant = thread["agent_variant"]
        if variant == "memory":
            types_suffix = f"[{thread['memory_types']}]" if thread.get("memory_types") else ""
            variant = f"memory:{thread['extraction_mode']}{types_suffix}"
        return f"{title} [{variant}] ({thread['updated_at'].strftime('%Y-%m-%d %H:%M')})"

    def logout(self, user_id):
        return (
            None,
            None,
            [],
            gr.update(choices=[], value=None),
            "<span style='color: var(--muted);'>Enter both username and password to start</span>",
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    def list_threads(self, user_id):
        if not user_id:
            return gr.update(choices=[])
        try:
            threads = threads_repo.list_threads_for_user(user_id)
        except Exception:
            return gr.update(choices=[])
        return gr.update(choices=[(self._thread_label(t), t["thread_id"]) for t in threads])

    async def _run_turn(
        self, user_id: int, thread_id: str, message: str, agent_variant: str,
        extraction_mode: str | None, memory_types: frozenset[MemoryType],
    ) -> str:
        if agent_variant != "memory":
            graph = build_graph(
                with_memory=False,
                checkpointer=self.checkpointer,
                store=None,
                user_id=user_id,
                search_service=self.search_service,
            )
            config = {"configurable": {"thread_id": thread_id}}
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": message}]}, config=config
            )
            return result["messages"][-1].content

        mode = ExtractionMode(extraction_mode)
        graph = build_graph(
            with_memory=True,
            extraction_mode=mode,
            memory_types=memory_types,
            search_strategy=self.search_strategy,
            checkpointer=self.checkpointer,
            store=self.store,
            user_id=user_id,
            search_service=self.search_service,
        )
        config = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": message}]}, config=config
        )
        if mode in (ExtractionMode.BACKGROUND, ExtractionMode.BOTH):
            # Fire-and-forget: unlike eval/runner.py, the UI doesn't wait for these writes.
            await run_background_extraction(
                user_id, result["messages"], memory_types, self.reflection_executors,
                extraction_mode=mode, thread_id=thread_id,
            )
        return result["messages"][-1].content

    def respond(self, user_message, chat_history, user_id, thread_id, agent_variant, extraction_mode, memory_types_choice):
        return run_coroutine(
            self._respond_async(user_message, chat_history, user_id, thread_id, agent_variant, extraction_mode, memory_types_choice),
        )

    async def _respond_async(self, user_message, chat_history, user_id, thread_id, agent_variant, extraction_mode, memory_types_choice):
        if not user_message.strip() or not user_id:
            return "", chat_history, thread_id

        try:
            if not thread_id:
                memory_types_str = ",".join(memory_types_choice) if memory_types_choice else None
                thread = threads_repo.create_thread(
                    user_id,
                    agent_variant=agent_variant,
                    extraction_mode=extraction_mode if agent_variant == "memory" else None,
                    memory_types=memory_types_str,
                )
                thread_id = thread["thread_id"]
                active_memory_types = _parse_memory_types(memory_types_str)
            else:
                thread = threads_repo.get_thread(thread_id)
                agent_variant = thread["agent_variant"]
                extraction_mode = thread["extraction_mode"]
                active_memory_types = _parse_memory_types(thread.get("memory_types"))
        except Exception as e:
            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": f" Could not reach the database: {e}"})
            return "", chat_history, thread_id

        try:
            assistant_text = await self._run_turn(
                user_id, thread_id, user_message, agent_variant, extraction_mode, active_memory_types
            )
        except Exception as e:
            assistant_text = f" Error while processing query: {str(e)}"

        is_first_message = len(chat_history or []) == 0
        try:
            threads_repo.touch_thread(thread_id, title=user_message[:60] if is_first_message else None)
        except Exception:
            pass

        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": assistant_text})

        return "", chat_history, thread_id

    def new_chat(self, user_id):
        return [], None, self.list_threads(user_id)

    def load_thread(self, thread_id, user_id):
        return run_coroutine(self._load_thread_async(thread_id, user_id))

    async def _load_thread_async(self, thread_id, user_id):
        if not thread_id or not user_id:
            return [], None

        try:
            graph = build_graph(
                with_memory=False,
                checkpointer=self.checkpointer,
                store=None,
                user_id=user_id,
                search_service=self.search_service,
            )
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = await graph.aget_state(config)
            messages = snapshot.values.get("messages", []) if snapshot else []
            return messages_to_chat_display(messages), thread_id
        except Exception:
            return [], thread_id

    def delete_thread(self, thread_id, user_id):
        if thread_id:
            try:
                threads_repo.delete_thread(thread_id)
            except Exception:
                pass
        return [], None, self.list_threads(user_id)

    def delete_all_threads(self, user_id):
        if not user_id:
            return [], None, gr.update(choices=[]), "No user logged in"
        try:
            count = threads_repo.delete_all_threads_for_user(user_id)
        except Exception as e:
            return [], None, self.list_threads(user_id), f" Could not delete sessions: {e}"
        return [], None, gr.update(choices=[]), f" Deleted {count} sessions"
