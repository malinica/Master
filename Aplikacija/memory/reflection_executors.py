import asyncio

from langmem import ReflectionExecutor

from memory.background import make_episodic_background_manager
from memory.config import SemanticBackgroundStrategy
from memory.namespaces import semantic_ns
from memory.semantic import aupdate_profile_from_messages, make_semantic_store_manager


class _FetchMergeReflector:

    def __init__(self, store, user_id: int, loop: asyncio.AbstractEventLoop):
        self.namespace = semantic_ns(user_id)
        self._store = store
        self._user_id = user_id
        self._loop = loop

    def invoke(self, payload: dict):
        future = asyncio.run_coroutine_threadsafe(
            aupdate_profile_from_messages(self._store, self._user_id, payload["messages"]),
            self._loop,
        )
        return future.result()


class ReflectionExecutors:

    def __init__(self, store, loop: asyncio.AbstractEventLoop):
        self._store = store
        self._loop = loop
        self._cache: dict[tuple, ReflectionExecutor] = {}

    def semantic(self, user_id: int, strategy: SemanticBackgroundStrategy) -> ReflectionExecutor:
        key = ("semantic", user_id, strategy)
        if key not in self._cache:
            reflector = (
                make_semantic_store_manager(self._store, user_id)
                if strategy == SemanticBackgroundStrategy.STORE_MANAGER
                else _FetchMergeReflector(self._store, user_id, self._loop)
            )
            self._cache[key] = ReflectionExecutor(reflector, store=self._store)
        return self._cache[key]

    def episodic(self, user_id: int) -> ReflectionExecutor:
        key = ("episodic", user_id)
        if key not in self._cache:
            reflector = make_episodic_background_manager(self._store, user_id)
            self._cache[key] = ReflectionExecutor(reflector, store=self._store)
        return self._cache[key]

    def shutdown(self, wait: bool = True) -> None:
        # non-daemon worker threads keep the process alive, so scripts that should exit on their own must call this
        for executor in self._cache.values():
            executor.shutdown(wait=wait)
