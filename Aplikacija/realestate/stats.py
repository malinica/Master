import sqlite3
from typing import Dict, List, Optional

from realestate.db import state_like_clause
from realestate.models import ColumnStats

IMPORTANT_COLUMNS = [
    'number_of_real_estates', 'average_price',
    'hospital5km', 'hospital20km', 'hospital100km',
    'average_temperature', 'average_humidity', 'average_wind_speed',
    'average_sunny_days_per_year', 'average_rainy_days_per_year', 'average_snowy_days_per_year',
    'prekindergarten_count_10km', 'elementary_schools_count_10km', 'high_schools_count_10km',
    'prekindergarten_count_20km', 'elementary_schools_count_20km', 'high_schools_count_20km',
    'prekindergarten_count_100km', 'elementary_schools_count_100km', 'high_schools_count_100km'
]


def get_price_statistics(db_path: str, states: Optional[List[str]] = None) -> Dict[str, float]:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        base_query = """
            SELECT
                MIN(average_price) as min_price,
                MAX(average_price) as max_price,
                AVG(average_price) as avg_price,
                (SELECT average_price FROM statistics
                WHERE average_price IS NOT NULL {state_filter_sub}
                ORDER BY average_price LIMIT 1 OFFSET
                (SELECT COUNT(*) FROM statistics WHERE average_price IS NOT NULL {state_filter_sub}) / 4) as p25,
                (SELECT average_price FROM statistics
                WHERE average_price IS NOT NULL {state_filter_sub}
                ORDER BY average_price LIMIT 1 OFFSET
                (SELECT COUNT(*) FROM statistics WHERE average_price IS NOT NULL {state_filter_sub}) / 2) as p50,
                (SELECT average_price FROM statistics
                WHERE average_price IS NOT NULL {state_filter_sub}
                ORDER BY average_price LIMIT 1 OFFSET
                (SELECT COUNT(*) FROM statistics WHERE average_price IS NOT NULL {state_filter_sub}) * 3 / 4) as p75
            FROM statistics
            WHERE average_price IS NOT NULL {state_filter_main}
        """

        state_filter_main = ""
        state_filter_sub = ""

        if states:
            like_clauses = [state_like_clause(s) for s in states]
            state_filter_main = " AND (" + " OR ".join(like_clauses) + ")"
            state_filter_sub = state_filter_main

        query = base_query.format(state_filter_main=state_filter_main, state_filter_sub=state_filter_sub)

        cur.execute(query)
        row = cur.fetchone()
        conn.close()

        return {
            'min': row[0], 'max': row[1], 'avg': row[2],
            'p25': row[3], 'p50': row[4], 'p75': row[5]
        }

    except Exception as e:
        print(f"Database error: {e}")
        return {
            'min': 50000, 'max': 1000000, 'avg': 300000,
            'p25': 150000, 'p50': 250000, 'p75': 450000
        }


def get_all_column_statistics(db_path: str) -> Dict[str, ColumnStats]:
    stats = {}
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        for column in IMPORTANT_COLUMNS:
            cur.execute(f"""
                SELECT
                    MIN({column}) as min_val,
                    MAX({column}) as max_val,
                    AVG({column}) as avg_val
                FROM statistics
                WHERE {column} IS NOT NULL
            """)
            row = cur.fetchone()
            if row and row[0] is not None and row[1] is not None:
                stats[column] = ColumnStats(
                    min_val=float(row[0]),
                    max_val=float(row[1]),
                    avg_val=float(row[2]) if row[2] is not None else (float(row[0]) + float(row[1])) / 2
                )

        conn.close()
    except Exception as e:
        print(f" Error loading column statistics: {e}")
        for column in IMPORTANT_COLUMNS:
            stats[column] = ColumnStats(min_val=0, max_val=100, avg_val=50)

    return stats
