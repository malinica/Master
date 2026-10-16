from typing import List, Dict, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field


class QueryParameters(BaseModel):
    states: List[str] = Field(default_factory=list, description="List of states")
    cities: List[str] = Field(default_factory=list, description="List of cities")
    price_range: Optional[str] = Field(default=None, description="Keywords that describe price value")
    price_min: Optional[float] = Field(default=None, description="Minimum price")
    price_max: Optional[float] = Field(default=None, description="Maximum price")
    temperature_preference: Optional[str] = Field(default=None, description="Temperature")
    temperature_min: Optional[float] = Field(default=None, description="Minimum temperature")
    temperature_max: Optional[float] = Field(default=None, description="Maximum temperature")
    wind_preference: Optional[str] = Field(default=None, description="Wind speed preference like calm, breezy, windy")
    wind_speed_min: Optional[float] = Field(default=None, description="Minimum wind speed")
    wind_speed_max: Optional[float] = Field(default=None, description="Maximum wind speed")
    humidity_preference: Optional[str] = Field(default=None, description="Humidity preference like dry, humid, normal")
    humidity_min: Optional[float] = Field(default=None, description="Minimum humidity")
    humidity_max: Optional[float] = Field(default=None, description="Maximum humidity")
    hospital_required: bool = Field(default=False, description="Hospitals required")
    hospital_distance: Optional[str] = Field(default=None, description="Hospital distance")
    schools_required: bool = Field(default=False, description="Schools required")
    school_types: List[str] = Field(default_factory=list, description="School types")
    schools_distance: Optional[str] = Field(default=None, description="School distance")
    weather_preferences: List[str] = Field(default_factory=list, description="Weather keywords like sunny, snowy, rainy")
    min_properties: Optional[int] = Field(default=None, description="Min properties")
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            'price': 0.2,
            'hospital': 0.2,
            'temperature': 0.2,
            'weather': 0.2,
            'schools': 0.2
        },
        description="Weights for ranking"
    )
    extreme_mode: Optional[str] = Field(default=None, description="Indicates if query targets min or max value")


@dataclass
class PropertyResult:
    town_state: str
    number_of_real_estates: int
    full_price: float
    average_price: float
    hospital_5km: int
    hospital_20km: int
    hospital_100km: int
    latitude: float
    longitude: float
    temperature: float
    humidity: float
    wind_speed: float
    sunny_days: float
    rainy_days: float
    snowy_days: float
    prekindergarten_10km: int
    elementary_10km: int
    high_10km: int
    prekindergarten_20km: int
    elementary_20km: int
    high_20km: int
    prekindergarten_100km: int
    elementary_100km: int
    high_100km: int
    score: float = 0


@dataclass
class QueryWeights:
    price: float = 0.2
    hospital: float = 0.2
    temperature: float = 0.2
    weather: float = 0.2
    schools: float = 0.2


@dataclass
class ColumnStats:
    min_val: float
    max_val: float
    avg_val: float
