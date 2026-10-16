import asyncio
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from db.connection import STORE_INDEX_CONFIG, get_conn_string, windows_loop_factory
from memory.schemas import SearchEpisode, UserProfile

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def run_custom_schema(conn_string: str) -> None:
    with psycopg.connect(conn_string, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("pgvector extension ready.")

    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with psycopg.connect(conn_string, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    print("Custom tables (users, threads, procedural_prompts, eval_runs) ready.")


async def run_langgraph_setup(conn_string: str) -> None:
    async with AsyncPostgresSaver.from_conn_string(conn_string) as saver:
        await saver.setup()
    print("LangGraph checkpointer tables ready.")

    async with AsyncPostgresStore.from_conn_string(conn_string, index=STORE_INDEX_CONFIG) as store:
        await store.setup()
    print("LangGraph store tables ready (with vector index).")


KIND_BY_NAMESPACE_SUFFIX = {
    "episodic_search": SearchEpisode.__name__,
    "semantic_profile": UserProfile.__name__,
}


def backfill_memory_kind(conn_string: str) -> None:
    with psycopg.connect(conn_string, autocommit=True) as conn:
        with conn.cursor() as cur:
            for suffix, kind in KIND_BY_NAMESPACE_SUFFIX.items():
                cur.execute(
                    "UPDATE store SET value = value || jsonb_build_object('kind', %s::text) "
                    "WHERE prefix LIKE %s AND value -> 'kind' IS NULL",
                    (kind, f"memories.%.{suffix}"),
                )
                print(f"Backfilled 'kind' on {cur.rowcount} {suffix} rows.")


def main() -> None:
    load_dotenv()
    conn_string = get_conn_string()
    run_custom_schema(conn_string)
    asyncio.run(run_langgraph_setup(conn_string), loop_factory=windows_loop_factory())
    backfill_memory_kind(conn_string)
    print("Migration complete.")


if __name__ == "__main__":
    main()
