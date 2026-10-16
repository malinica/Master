from typing import Optional

import psycopg
from psycopg.rows import dict_row

from db.connection import get_conn_string


def get_latest_procedural_prompt(user_id: int) -> Optional[dict]:
    with psycopg.connect(get_conn_string(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM procedural_prompts WHERE user_id = %s ORDER BY version DESC LIMIT 1",
                (user_id,),
            )
            return cur.fetchone()


def insert_procedural_prompt(
    user_id: int, version: int, prompt_text: str, source_thread_ids: list[str]
) -> dict:
    with psycopg.connect(get_conn_string(), row_factory=dict_row, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO procedural_prompts (user_id, version, prompt_text, source_thread_ids) "
                "VALUES (%s, %s, %s, %s) "
                "RETURNING id, user_id, version, prompt_text, source_thread_ids, created_at",
                (user_id, version, prompt_text, source_thread_ids),
            )
            return cur.fetchone()


def list_procedural_prompts(user_id: int) -> list[dict]:
    with psycopg.connect(get_conn_string(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM procedural_prompts WHERE user_id = %s ORDER BY version ASC",
                (user_id,),
            )
            return cur.fetchall()
