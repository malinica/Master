import math
from typing import List, Tuple

from realestate import db as db_module
from realestate import stats as stats_module
from realestate.models import QueryParameters

SELECT_COLUMNS = """town_state, number_of_real_estates, full_price, average_price,
            hospital5km, hospital20km, hospital100km, latitude, longitude,
            average_temperature, average_humidity, average_wind_speed,
            average_sunny_days_per_year, average_rainy_days_per_year, average_snowy_days_per_year,
            prekindergarten_count_10km, elementary_schools_count_10km, high_schools_count_10km,
            prekindergarten_count_20km, elementary_schools_count_20km, high_schools_count_20km,
            prekindergarten_count_100km, elementary_schools_count_100km, high_schools_count_100km"""


def _num(value) -> str:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"Non-finite numeric value in query parameters: {value!r}")
    return repr(v)


def build_sql_from_params(params: QueryParameters, db_path: str) -> Tuple[str, str, List[str]]:
    conditions = []
    order_direction = "DESC"
    order_column = "average_price"
    limit_clause = ""

    if params.states:
        state_conditions = [db_module.state_like_clause(state) for state in params.states]
        conditions.append(f"({' OR '.join(state_conditions)})")
        price_stats = stats_module.get_price_statistics(db_path, params.states)
    else:
        price_stats = stats_module.get_price_statistics(db_path, None)

    if params.price_min is not None and params.price_max is not None:
        conditions.append(f"average_price BETWEEN {_num(params.price_min)} AND {_num(params.price_max)}")
        price_range_ratio = params.price_max / params.price_min if params.price_min > 0 else 1
        if 1.4 <= price_range_ratio <= 1.6:
            target_price = (params.price_min + params.price_max) / 2
            order_column = f"ABS(average_price - {_num(target_price)})"
            order_direction = "ASC"
    elif params.price_min is not None:
        conditions.append(f"average_price >= {_num(params.price_min)}")
        if params.extreme_mode == "max":
            order_direction = "DESC"
    elif params.price_max is not None:
        conditions.append(f"average_price <= {_num(params.price_max)}")

    if params.temperature_min is not None and params.temperature_max is not None:
        conditions.append(f"average_temperature BETWEEN {_num(params.temperature_min)} AND {_num(params.temperature_max)}")
    elif params.temperature_min is not None:
        conditions.append(f"average_temperature >= {_num(params.temperature_min)}")
    elif params.temperature_max is not None:
        conditions.append(f"average_temperature <= {_num(params.temperature_max)}")

    if params.wind_speed_min is not None and params.wind_speed_max is not None:
        conditions.append(f"average_wind_speed BETWEEN {_num(params.wind_speed_min)} AND {_num(params.wind_speed_max)}")
    elif params.wind_speed_min is not None:
        conditions.append(f"average_wind_speed >= {_num(params.wind_speed_min)}")
    elif params.wind_speed_max is not None:
        conditions.append(f"average_wind_speed <= {_num(params.wind_speed_max)}")

    if params.humidity_min is not None and params.humidity_max is not None:
        conditions.append(f"average_humidity BETWEEN {_num(params.humidity_min)} AND {_num(params.humidity_max)}")
    elif params.humidity_min is not None:
        conditions.append(f"average_humidity >= {_num(params.humidity_min)}")
    elif params.humidity_max is not None:
        conditions.append(f"average_humidity <= {_num(params.humidity_max)}")

    if params.hospital_required:
        if params.hospital_distance == "nearby":
            conditions.append("hospital5km > 0")
        elif params.hospital_distance == "medium":
            conditions.append("hospital20km > 0")
        else:
            conditions.append("(hospital5km > 0 OR hospital20km > 0 OR hospital100km > 0)")

    if params.schools_required:
        if params.schools_distance == "nearby":
            conditions.append("""(prekindergarten_count_10km > 0 OR elementary_schools_count_10km > 0 OR high_schools_count_10km > 0)""")
        elif params.schools_distance == "medium":
            conditions.append("""(prekindergarten_count_20km > 0 OR elementary_schools_count_20km > 0 OR high_schools_count_20km > 0)""")
        elif params.schools_distance == "far":
            conditions.append("""(prekindergarten_count_100km > 0 OR elementary_schools_count_100km > 0 OR high_schools_count_100km > 0)""")
        else:
            conditions.append("""
                (
                    prekindergarten_count_10km > 0 OR elementary_schools_count_10km > 0 OR high_schools_count_10km > 0 OR
                    prekindergarten_count_20km > 0 OR elementary_schools_count_20km > 0 OR high_schools_count_20km > 0 OR
                    prekindergarten_count_100km > 0 OR elementary_schools_count_100km > 0 OR high_schools_count_100km > 0
                )
            """)

    wp = params.weather_preferences
    if wp:
        if isinstance(wp, str):
            wp = [wp.lower()]
        elif isinstance(wp, list):
            wp = [str(x).lower() for x in wp]
        else:
            wp = []

        if "sunny" in wp or "sunniest" in wp:
            order_column, order_direction = "average_sunny_days_per_year", "DESC"
            limit_clause = "LIMIT 5"
        elif "rainy" in wp or "rainiest" in wp:
            order_column, order_direction = "average_rainy_days_per_year", "DESC"
            limit_clause = "LIMIT 5"
        elif "snowy" in wp or "snowiest" in wp:
            order_column, order_direction = "average_snowy_days_per_year", "DESC"
            limit_clause = "LIMIT 5"

    if params.extreme_mode and not limit_clause:
        mode = params.extreme_mode.lower()

        temp_weight = params.weights.get('temperature', 0)
        price_weight = params.weights.get('price', 0)
        weather_weight = params.weights.get('weather', 0)

        if temp_weight >= 0.8:
            order_column = "average_temperature"
        elif price_weight >= 0.8:
            order_column = "average_price"
        elif weather_weight >= 0.8:
            if params.weather_preferences:
                wp_lower = params.weather_preferences[0].lower() if params.weather_preferences else ""
                if "snowy" in wp_lower:
                    order_column = "average_snowy_days_per_year"
                elif "rainy" in wp_lower:
                    order_column = "average_rainy_days_per_year"
                elif "sunny" in wp_lower:
                    order_column = "average_sunny_days_per_year"
                else:
                    order_column = "average_sunny_days_per_year"
            elif params.wind_speed_min is not None or params.wind_speed_max is not None or params.wind_preference:
                order_column = "average_wind_speed"
            elif params.humidity_min is not None or params.humidity_max is not None or params.humidity_preference:
                order_column = "average_humidity"
            else:
                order_column = "average_sunny_days_per_year"
        else:
            order_column = "average_temperature"

        if mode == "max":
            order_direction = "DESC"
            limit_clause = "LIMIT 5"
        elif mode == "min":
            order_direction = "ASC"
            limit_clause = "LIMIT 5"
        elif mode == "average":
            if price_weight >= 0.8:
                avg_val = price_stats.get('avg') or 300000
                order_column = f"ABS(average_price - {_num(avg_val)})"
                order_direction = "ASC"
                limit_clause = "LIMIT 5"
            elif temp_weight >= 0.8:
                column_stats = stats_module.get_all_column_statistics(db_path)
                temp_stats = column_stats.get('average_temperature')
                if temp_stats and temp_stats.avg_val is not None:
                    order_column = f"ABS(average_temperature - {_num(temp_stats.avg_val)})"
                    order_direction = "ASC"
                    limit_clause = "LIMIT 5"

    if not limit_clause:
        active_weights = [k for k, v in params.weights.items() if v > 0.5]

        if active_weights == ["hospital"]:
            order_column = """(COALESCE(hospital5km, 0) + COALESCE(hospital20km, 0) + COALESCE(hospital100km, 0))"""
            order_direction = "DESC"
            limit_clause = "LIMIT 10"
        elif active_weights == ["schools"] and params.schools_required:
            school_cols = []

            if params.schools_distance == "nearby" or "nearby" in str(params.schools_distance).lower():
                if "high" in params.school_types:
                    school_cols.append("high_schools_count_10km")
                elif "elementary" in params.school_types:
                    school_cols.append("elementary_schools_count_10km")
                elif "prekindergarten" in params.school_types:
                    school_cols.append("prekindergarten_count_10km")
                else:
                    school_cols = ["prekindergarten_count_10km", "elementary_schools_count_10km", "high_schools_count_10km"]

            elif params.schools_distance == "medium":
                if "high" in params.school_types:
                    school_cols.append("high_schools_count_20km")
                elif "elementary" in params.school_types:
                    school_cols.append("elementary_schools_count_20km")
                elif "prekindergarten" in params.school_types:
                    school_cols.append("prekindergarten_count_20km")
                else:
                    school_cols = ["prekindergarten_count_20km", "elementary_schools_count_20km", "high_schools_count_20km"]

            elif params.schools_distance == "far" or "100" in str(params.schools_distance):
                if "high" in params.school_types:
                    school_cols.append("high_schools_count_100km")
                elif "elementary" in params.school_types:
                    school_cols.append("elementary_schools_count_100km")
                elif "prekindergarten" in params.school_types:
                    school_cols.append("prekindergarten_count_100km")
                else:
                    school_cols = ["prekindergarten_count_100km", "elementary_schools_count_100km", "high_schools_count_100km"]
            else:
                school_cols = [
                    "prekindergarten_count_10km", "elementary_schools_count_10km", "high_schools_count_10km",
                    "prekindergarten_count_20km", "elementary_schools_count_20km", "high_schools_count_20km",
                    "prekindergarten_count_100km", "elementary_schools_count_100km", "high_schools_count_100km"
                ]

            if school_cols:
                order_column = f"({' + '.join([f'COALESCE({col}, 0)' for col in school_cols])})"
                order_direction = "DESC"
                limit_clause = "LIMIT 10"

    conditions = [c for c in conditions if c and c.strip()]
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    sql = f"""
    SELECT {SELECT_COLUMNS}
    FROM statistics
    WHERE {where_clause}
    ORDER BY {order_column} {order_direction} NULLS LAST
    {limit_clause};
    """

    return sql, order_direction, conditions


def build_select_query(where_clause: str, order_direction: str = "ASC") -> str:
    return f"""
    SELECT {SELECT_COLUMNS}
    FROM statistics
    WHERE {where_clause}
    ORDER BY average_price {order_direction} NULLS LAST;
    """


def apply_fallback_logic(
    conditions: List[str],
    params: QueryParameters,
    order_direction: str,
    user_specified_states: bool,
    db_path: str,
) -> Tuple[List[Tuple], str]:

    results = []
    fallback_msg = ""

    state_conditions = [c for c in conditions if "town_state LIKE" in c]
    non_state_conditions = [c for c in conditions if "town_state LIKE" not in c]

    if user_specified_states and not state_conditions:
        state_conditions = [db_module.state_like_clause(state) for state in params.states]

    if user_specified_states and len(non_state_conditions) > 0:
        relaxed_conditions = [c for c in non_state_conditions if "hospital" not in c.lower()]

        if len(relaxed_conditions) < len(non_state_conditions):
            all_relaxed = state_conditions + relaxed_conditions
            relaxed_where = " AND ".join(all_relaxed) if all_relaxed else " AND ".join(state_conditions)

            relaxed_sql = build_select_query(relaxed_where, order_direction)
            results = db_module.run_query(db_path, relaxed_sql)
            if results:
                fallback_msg = f"  No exact matches in {', '.join(params.states)}. Showing with relaxed criteria:"

    if not results and user_specified_states and any("average_price" in c for c in non_state_conditions):
        price_conditions = [c for c in non_state_conditions if "average_price" in c]
        simple_conditions = state_conditions + price_conditions
        simple_where = " AND ".join(simple_conditions)

        simple_sql = build_select_query(simple_where, order_direction)
        results = db_module.run_query(db_path, simple_sql)
        if results:
            fallback_msg = f"  Showing price-matched results in {', '.join(params.states)}:"

    if not results and user_specified_states:
        state_only_where = " AND ".join(state_conditions)
        state_only_sql = build_select_query(state_only_where, order_direction)
        results = db_module.run_query(db_path, state_only_sql)
        if results:
            fallback_msg = f"  No matches for your criteria. Showing all locations in {', '.join(params.states)}:"

    if not results and not user_specified_states and len(non_state_conditions) > 1:
        relaxed_conditions = [c for c in non_state_conditions if "hospital5km > 0" not in c]

        if len(relaxed_conditions) < len(non_state_conditions):
            relaxed_where = " AND ".join(relaxed_conditions) if relaxed_conditions else "1=1"
            relaxed_sql = build_select_query(relaxed_where, order_direction)
            results = db_module.run_query(db_path, relaxed_sql)
            if results:
                fallback_msg = "  Relaxed criteria:"

    if not results:
        fallback_sql = build_select_query("1=1", order_direction)
        results = db_module.run_query(db_path, fallback_sql)
        fallback_msg = "  No results for your query, showing best available options:"

    return results, fallback_msg
