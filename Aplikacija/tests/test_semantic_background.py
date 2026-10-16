import asyncio
import concurrent.futures

from langgraph.store.memory import InMemoryStore

from memory import background, semantic
from memory.config import ExtractionMode, MemoryType, SemanticBackgroundStrategy
from memory.schemas import UserProfile


async def test_aupdate_profile_from_messages_writes_single_document(monkeypatch):
    store = InMemoryStore()
    user_id = 1
    profiles = [
        UserProfile(preferred_states=["TX"]),
        UserProfile(preferred_states=["TX", "FL"]),
    ]
    call_count = {"n": 0}

    class FakeExtractor:
        async def ainvoke(self, prompt):
            result = profiles[call_count["n"]]
            call_count["n"] += 1
            return result

    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        def with_structured_output(self, schema):
            return FakeExtractor()

    monkeypatch.setattr(semantic, "ChatOpenAI", FakeChatOpenAI)

    await semantic.aupdate_profile_from_messages(store, user_id, messages=[])
    await semantic.aupdate_profile_from_messages(store, user_id, messages=[])

    items = await store.asearch(semantic.semantic_ns(user_id))
    assert len(items) == 1
    assert items[0].value["content"]["preferred_states"] == ["TX", "FL"]


class FakeExecutors:

    def __init__(self):
        self.calls = []

    def semantic(self, user_id, strategy):
        return self._executor(("semantic", user_id, strategy))

    def episodic(self, user_id):
        return self._executor(("episodic", user_id))

    def _executor(self, key):
        calls = self.calls

        class Executor:
            def submit(self, payload, config=None, thread_id=None):
                calls.append((key, payload, thread_id))
                future = concurrent.futures.Future()
                future.set_result(None)
                return future

        return Executor()


async def test_run_background_extraction_defaults_to_fetch_merge():
    executors = FakeExecutors()

    pending = await background.run_background_extraction(
        user_id=1, messages=[], memory_types=frozenset({MemoryType.SEMANTIC}), executors=executors
    )
    await asyncio.gather(*pending)

    assert executors.calls == [(("semantic", 1, SemanticBackgroundStrategy.FETCH_MERGE), {"messages": []}, None)]


async def test_run_background_extraction_can_use_store_manager_strategy():
    executors = FakeExecutors()
    messages = [{"role": "user", "content": "hi"}]

    pending = await background.run_background_extraction(
        user_id=1, messages=messages, memory_types=frozenset({MemoryType.SEMANTIC}),
        executors=executors, semantic_strategy=SemanticBackgroundStrategy.STORE_MANAGER,
        thread_id="t1",
    )
    await asyncio.gather(*pending)

    assert executors.calls == [
        (("semantic", 1, SemanticBackgroundStrategy.STORE_MANAGER), {"messages": messages}, "t1")
    ]


async def test_run_background_extraction_both_mode_skips_episodic():
    both_types = frozenset({MemoryType.SEMANTIC, MemoryType.EPISODIC})

    for mode, expected in [
        (ExtractionMode.BACKGROUND, ["semantic", "episodic"]),
        (ExtractionMode.BOTH, ["semantic"]),
    ]:
        executors = FakeExecutors()
        pending = await background.run_background_extraction(
            user_id=1, messages=[], memory_types=both_types, executors=executors, extraction_mode=mode
        )
        await asyncio.gather(*pending)

        assert [key[0] for key, _, _ in executors.calls] == expected
