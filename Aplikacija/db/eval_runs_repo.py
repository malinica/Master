import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from db.connection import get_conn_string


def insert_eval_run(
    scenario_name: str,
    agent_variant: str,
    extraction_mode: str | None,
    user_id: int,
    thread_ids: list[str],
    metrics: dict,
) -> dict:
    with psycopg.connect(get_conn_string(), row_factory=dict_row, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO eval_runs "
                "(scenario_name, agent_variant, extraction_mode, user_id, thread_ids, metrics_json) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "RETURNING id, scenario_name, agent_variant, extraction_mode, user_id, thread_ids, "
                "metrics_json, run_at",
                (scenario_name, agent_variant, extraction_mode, user_id, thread_ids, Json(metrics)),
            )
            return cur.fetchone()
