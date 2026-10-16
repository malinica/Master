import sqlite3
from typing import List, Tuple


def state_like_clause(state: str) -> str:
    escaped = state.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_").replace("'", "''")
    return f"town_state LIKE '%\\_{escaped}' ESCAPE '\\'"


def run_query(db_path: str, sql_query: str) -> List[Tuple]:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql_query)
        rows = cur.fetchall()
        conn.close()
        return rows
    except sqlite3.Error as e:
        print(f" Database error: {e}\n  Query was: {sql_query}")
        return []
