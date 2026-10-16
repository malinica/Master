import requests

from graph import tools as tools_module
from graph.tools import make_current_weather_tool


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json_data


GEOCODING_RESPONSE = {
    "results": [
        {"name": "Austin", "admin1": "Texas", "latitude": 30.27, "longitude": -97.74},
        {"name": "Austin", "admin1": "Minnesota", "latitude": 43.67, "longitude": -92.97},
    ]
}

FORECAST_RESPONSE = {
    "current_weather": {
        "temperature": 25.3,
        "windspeed": 8.1,
        "weathercode": 1,
    }
}


def _fake_get_factory(forecast_response=FORECAST_RESPONSE):
    def fake_get(url, params=None, timeout=None):
        if url == tools_module.GEOCODING_URL:
            return FakeResponse(GEOCODING_RESPONSE)
        if url == tools_module.FORECAST_URL:
            return FakeResponse(forecast_response)
        raise AssertionError(f"unexpected URL requested: {url}")

    return fake_get


def test_current_weather_reports_forecast(monkeypatch):
    monkeypatch.setattr(tools_module.requests, "get", _fake_get_factory())

    weather_tool = make_current_weather_tool()
    result = weather_tool.invoke({"city": "Austin", "state": "Texas"})

    assert "Austin, Texas" in result
    assert "25.3" in result
    assert "mainly clear" in result


def test_current_weather_picks_matching_state(monkeypatch):
    monkeypatch.setattr(tools_module.requests, "get", _fake_get_factory())

    weather_tool = make_current_weather_tool()
    result = weather_tool.invoke({"city": "Austin", "state": "Minnesota"})

    assert "Austin, Minnesota" in result


def test_current_weather_unknown_city(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return FakeResponse({"results": []})

    monkeypatch.setattr(tools_module.requests, "get", fake_get)

    weather_tool = make_current_weather_tool()
    result = weather_tool.invoke({"city": "Nowheresville"})

    assert "Could not find" in result
