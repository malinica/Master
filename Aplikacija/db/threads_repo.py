import uuid
from typing import Optional

import psycopg
from psycopg.rows import dict_row

from db.connection import get_conn_string


def _normalize(row: Optional[dict]) -> Optional[dict]:
    # psycopg returns uuid columns as uuid.UUID; callers (Gradio, LangGraph configs) expect str.
    if row is not None:
        row["thread_id"] = str(row["thread_id"])
    return row


def create_thread(
    user_id: int,
    agent_variant: str,
    extraction_mode: Optional[str] = None,
    memory_types: Optional[str] = None,
    title: Optional[str] = None,
) -> dict:
    thread_id = str(uuid.uuid4())
    with psycopg.connect(get_conn_string(), row_factory=dict_row, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO threads (thread_id, user_id, title, agent_variant, extraction_mode, memory_types) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "RETURNING thread_id, user_id, title, agent_variant, extraction_mode, memory_types, created_at, updated_at",
                (thread_id, user_id, title, agent_variant, extraction_mode, memory_types),
            )
            return _normalize(cur.fetchone())


def get_thread(thread_id: str) -> Optional[dict]:
    with psycopg.connect(get_conn_string(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM threads WHERE thread_id = %s", (thread_id,))
            return _normalize(cur.fetchone())


def list_threads_for_user(user_id: int) -> list[dict]:
    with psycopg.connect(get_conn_string(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM threads WHERE user_id = %s ORDER BY updated_at DESC",
                (user_id,),
            )
            return [_normalize(row) for row in cur.fetchall()]


def touch_thread(thread_id: str, title: Optional[str] = None) -> None:
    with psycopg.connect(get_conn_string(), autocommit=True) as conn:
        with conn.cursor() as cur:
            if title is not None:
                cur.execute(
                    "UPDATE threads SET updated_at = now(), title = %s WHERE thread_id = %s",
                    (title, thread_id),
                )
            else:
                cur.execute(
                    "UPDATE threads SET updated_at = now() WHERE thread_id = %s",
                    (thread_id,),
                )


def delete_thread(thread_id: str) -> None:
    with psycopg.connect(get_conn_string(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM threads WHERE thread_id = %s", (thread_id,))


def delete_all_threads_for_user(user_id: int) -> int:
    with psycopg.connect(get_conn_string(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM threads WHERE user_id = %s", (user_id,))
            return cur.rowcount
