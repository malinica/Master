import asyncio
import threading

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from db.connection import windows_loop_factory

_loop: asyncio.AbstractEventLoop | None = None


def start_background_loop() -> asyncio.AbstractEventLoop:
    # store and checkpointer bind to the loop running at construction, so start this first
    global _loop
    loop_cls = windows_loop_factory()
    _loop = loop_cls() if loop_cls else asyncio.new_event_loop()
    threading.Thread(target=_loop.run_forever, daemon=True).start()
    return _loop


def run_coroutine(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


async def open_persistent_resources(conn_string: str, store_index_config: dict):
    # direct AsyncConnections instead of from_conn_string(): its context-manager
    # generator closes the connection once garbage-collected
    checkpointer_conn = await AsyncConnection.connect(
        conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
    )
    checkpointer = AsyncPostgresSaver(checkpointer_conn)

    store_conn = await AsyncConnection.connect(
        conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
    )
    store = AsyncPostgresStore(store_conn, index=store_index_config)

    return checkpointer, store
