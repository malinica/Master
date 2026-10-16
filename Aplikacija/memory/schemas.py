from typing import List, Optional

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Standing preferences/facts about a user, learned across conversations (semantic memory)."""

    preferred_states: List[str] = Field(
        default_factory=list, description="US states the user has shown interest in"
    )
    price_min: Optional[float] = Field(default=None, description="Typical minimum budget in USD")
    price_max: Optional[float] = Field(default=None, description="Typical maximum budget in USD")
    temperature_preference: Optional[str] = Field(
        default=None, description="e.g. warm, cold, mild"
    )
    wants_hospitals_nearby: Optional[bool] = None
    wants_schools_nearby: Optional[bool] = None
    school_types: List[str] = Field(
        default_factory=list, description="e.g. elementary, high, prekindergarten"
    )
    dislikes: List[str] = Field(
        default_factory=list, description="things the user explicitly dislikes, e.g. humidity, snow"
    )
    notes: Optional[str] = Field(
        default=None, description="freeform notes that don't fit the fields above"
    )


class SearchEpisode(BaseModel):
    """One concrete past search event (episodic memory), distinct from UserProfile's standing facts."""

    query: str = Field(description="The user's original search request")
    summary: str = Field(description="Short summary of what was searched for and found")
    top_result_towns: List[str] = Field(
        default_factory=list, description="Top town/city names shown to the user, if any"
    )
