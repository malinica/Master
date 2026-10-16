import asyncio
import os
import sys

STORE_INDEX_CONFIG = {
    "dims": 1536,
    "embed": "openai:text-embedding-3-small",
}


def windows_loop_factory():
    return asyncio.SelectorEventLoop if sys.platform == "win32" else None


def get_conn_string() -> str:
    conn_string = os.environ.get("PG_CONN_STRING")
    if conn_string:
        return conn_string

    user = os.environ.get("PG_USER", "sir_agent")
    password = os.environ.get("PG_PASSWORD", "sir_agent_pw")
    host = os.environ.get("PG_HOST", "localhost")
    port = os.environ.get("PG_PORT", "5435")
    db = os.environ.get("PG_DB", "sir_agent")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"
