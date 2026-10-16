from typing import Dict, List

from realestate.models import ColumnStats, PropertyResult, QueryParameters, QueryWeights


def normalize_value(column_stats: Dict[str, ColumnStats], column_name: str, value: float) -> float:
    if value is None:
        return 0.0

    if column_name not in column_stats:
        return 0.0

    stats = column_stats[column_name]

    if stats.max_val == stats.min_val:
        return 0.5

    normalized = (value - stats.min_val) / (stats.max_val - stats.min_val)

    return max(0.0, min(1.0, normalized))


def params_to_weights(params: QueryParameters) -> QueryWeights:
    return QueryWeights(
        price=params.weights.get('price', 0.2),
        hospital=params.weights.get('hospital', 0.2),
        temperature=params.weights.get('temperature', 0.2),
        weather=params.weights.get('weather', 0.2),
        schools=params.weights.get('schools', 0.2)
    )


def calculate_score(
    column_stats: Dict[str, ColumnStats],
    prop: PropertyResult,
    weights: QueryWeights,
    params: QueryParameters = None,
) -> float:
    score = 0.0
    params = params or QueryParameters()

    def is_relevant(cat: str, weight: float) -> bool:
        if weight > 0.25:
            return True
        if cat == "price":
            return any([params.price_min, params.price_max, params.price_range])
        if cat == "hospital":
            return params.hospital_required or params.hospital_distance
        if cat == "temperature":
            return any([params.temperature_preference, params.temperature_min, params.temperature_max])
        if cat == "weather":
            return bool(params.weather_preferences)
        if cat == "schools":
            return params.schools_required or params.schools_distance or params.school_types
        return False

    if prop.average_price and is_relevant("price", weights.price):
        price_norm = normalize_value(column_stats, 'average_price', prop.average_price)

        if params.price_range and params.price_range.lower() in ["expensive", "luxury", "najskuplje", "najskupljih"]:
            score += weights.price * price_norm * 3.0
        else:
            score += weights.price * (1 - price_norm) * 3.0

    if is_relevant("hospital", weights.hospital):
        hospital_score = 0.0
        if prop.hospital_5km:
            hospital_score += normalize_value(column_stats, 'hospital5km', prop.hospital_5km) * 3
        if prop.hospital_20km:
            hospital_score += normalize_value(column_stats, 'hospital20km', prop.hospital_20km) * 2
        if prop.hospital_100km:
            hospital_score += normalize_value(column_stats, 'hospital100km', prop.hospital_100km) * 1
        score += weights.hospital * min(3.0, hospital_score)

    if prop.temperature and is_relevant("temperature", weights.temperature):
        ideal_temp = 21.0
        stats = column_stats.get('average_temperature')
        if stats:
            temp_range = max(1.0, stats.max_val - stats.min_val)
            temp_deviation = abs(prop.temperature - ideal_temp) / temp_range
            temp_score = max(0.0, 1.0 - temp_deviation * 2)
            score += weights.temperature * temp_score * 2.0

    if is_relevant("weather", weights.weather):
        weather_score = 0.0

        wp = params.weather_preferences if params.weather_preferences else []
        if isinstance(wp, str):
            wp = [wp.lower()]
        elif isinstance(wp, list):
            wp = [str(x).lower() for x in wp]

        wants_rain = any(w in ["rainy", "rainiest", "rain"] for w in wp)
        wants_snow = any(w in ["snowy", "snowiest", "snow"] for w in wp)

        if prop.sunny_days:
            weather_score += normalize_value(column_stats, 'average_sunny_days_per_year', prop.sunny_days) * 0.5

        if prop.rainy_days:
            if wants_rain:
                weather_score += normalize_value(column_stats, 'average_rainy_days_per_year', prop.rainy_days) * 0.5
            else:
                weather_score += (1 - normalize_value(column_stats, 'average_rainy_days_per_year', prop.rainy_days)) * 0.3

        if prop.snowy_days:
            if wants_snow:
                weather_score += normalize_value(column_stats, 'average_snowy_days_per_year', prop.snowy_days) * 0.5
            else:
                weather_score += (1 - normalize_value(column_stats, 'average_snowy_days_per_year', prop.snowy_days)) * 0.3

        if prop.humidity:
            weather_score += (1 - normalize_value(column_stats, 'average_humidity', prop.humidity)) * 0.2
        if prop.wind_speed:
            weather_score += (1 - normalize_value(column_stats, 'average_wind_speed', prop.wind_speed)) * 0.2

        score += weights.weather * min(2.0, weather_score)

    if is_relevant("schools", weights.schools):
        schools_score = 0.0

        if prop.elementary_10km:
            schools_score += normalize_value(column_stats, 'elementary_schools_count_10km', prop.elementary_10km) * 1.0
        if prop.high_10km:
            schools_score += normalize_value(column_stats, 'high_schools_count_10km', prop.high_10km) * 0.8
        if prop.prekindergarten_10km:
            schools_score += normalize_value(column_stats, 'prekindergarten_count_10km', prop.prekindergarten_10km) * 0.6

        if prop.elementary_20km:
            schools_score += normalize_value(column_stats, 'elementary_schools_count_20km', prop.elementary_20km) * 0.6
        if prop.high_20km:
            schools_score += normalize_value(column_stats, 'high_schools_count_20km', prop.high_20km) * 0.5
        if prop.prekindergarten_20km:
            schools_score += normalize_value(column_stats, 'prekindergarten_count_20km', prop.prekindergarten_20km) * 0.4

        if prop.elementary_100km:
            schools_score += normalize_value(column_stats, 'elementary_schools_count_100km', prop.elementary_100km) * 0.3
        if prop.high_100km:
            schools_score += normalize_value(column_stats, 'high_schools_count_100km', prop.high_100km) * 0.2
        if prop.prekindergarten_100km:
            schools_score += normalize_value(column_stats, 'prekindergarten_count_100km', prop.prekindergarten_100km) * 0.1

        score += weights.schools * min(2.0, schools_score)

    return round(score, 3)


def sort_results_by_mode(properties: List[PropertyResult], params: QueryParameters, top_n: int) -> List[PropertyResult]:
    if not params.extreme_mode:
        properties.sort(key=lambda x: x.score, reverse=True)
        return properties[:top_n]

    mode = params.extreme_mode.lower()

    if mode == "max":
        properties.sort(key=lambda x: (-x.score, -(x.average_price or 0), -(x.temperature or 0)))
        return properties[:top_n]
    elif mode == "min":
        properties.sort(key=lambda x: (x.score, x.average_price or 0, x.temperature or 0))
        return properties[:top_n]
    elif mode == "average":
        if properties:
            temp_weight = params.weights.get('temperature', 0)
            price_weight = params.weights.get('price', 0)

            if temp_weight > price_weight:
                avg_temp = sum(p.temperature for p in properties if p.temperature) / len([p for p in properties if p.temperature])
                properties.sort(key=lambda x: abs(x.temperature - avg_temp) if x.temperature else 999)
            elif price_weight > temp_weight:
                avg_price = sum(p.average_price for p in properties if p.average_price) / len([p for p in properties if p.average_price])
                properties.sort(key=lambda x: abs(x.average_price - avg_price) if x.average_price else 999999)
            else:
                avg_score = sum(p.score for p in properties) / len(properties)
                properties.sort(key=lambda x: abs(x.score - avg_score))

            return properties[:top_n]
        else:
            return []
    else:
        properties.sort(key=lambda x: x.score, reverse=True)
        return properties[:top_n]
