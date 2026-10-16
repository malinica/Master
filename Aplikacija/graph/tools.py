import requests
from langchain_core.tools import tool

from realestate.service import RealEstateSearchService

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm",
}


def make_real_estate_tool(search_service: RealEstateSearchService):
    @tool
    def real_estate_search(query: str) -> str:
        """Search US towns/cities matching real-estate criteria described in `query`.

        Use this whenever the user asks about locations, prices, weather, hospitals,
        or schools for real estate purposes. Pass the user's request (or your own
        rephrasing of it, if you have combined it with remembered preferences) as
        the `query` string, in natural language.
        """
        return search_service.search(query, top_n=5, detailed=True)

    return real_estate_search


def _geocode_us_city(city: str, state: str = ""):
    geo_resp = requests.get(
        GEOCODING_URL,
        params={"name": city, "count": 5, "country": "US"},
        timeout=10,
    )
    geo_resp.raise_for_status()
    results = geo_resp.json().get("results") or []
    if not results:
        return None

    match = results[0]
    if state:
        for candidate in results:
            if candidate.get("admin1", "").lower() == state.lower():
                match = candidate
                break
    return match


def make_current_weather_tool():
    @tool
    def current_weather(city: str, state: str = "") -> str:
        """Look up today's actual current weather for a US city, live from Open-Meteo.

        Use this only when the user explicitly asks about *current*/*today's*/*right
        now* weather for a specific place — the historical climate averages used for
        matching towns already come from real_estate_search and don't need this tool.
        `city` is required, e.g. "Austin". `state` is optional, e.g. "Texas".
        """
        try:
            match = _geocode_us_city(city, state)
        except requests.RequestException as e:
            return f"Could not reach the geocoding service right now ({e})."
        if not match:
            return f"Could not find a US location named '{city}'."

        lat, lon = match["latitude"], match["longitude"]
        place = f"{match.get('name', city)}, {match.get('admin1', '')}"

        try:
            forecast_resp = requests.get(
                FORECAST_URL,
                params={"latitude": lat, "longitude": lon, "current_weather": "true"},
                timeout=10,
            )
            forecast_resp.raise_for_status()
            current = forecast_resp.json()["current_weather"]
        except (requests.RequestException, KeyError) as e:
            return f"Could not reach the weather service right now ({e})."

        condition = WEATHER_CODES.get(current["weathercode"], "unknown conditions")
        return (
            f"Current weather for {place}: {current['temperature']}°C, {condition}, "
            f"wind {current['windspeed']} km/h."
        )

    return current_weather
