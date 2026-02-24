"""Pydantic schemas for ingestion input validation."""

from datetime import date

from pydantic import BaseModel, Field


class PJMFetchParams(BaseModel):
    """Parameters for PJM LMP data fetch. All required."""

    start_date: date = Field(..., description="Start date (inclusive)")
    end_date: date = Field(..., description="End date (inclusive)")
    node_or_zone: str = Field(
        ...,
        min_length=1,
        description="PJM node ID or zone identifier (e.g. 'PJM-RTO', '5123456789')",
    )


class WeatherFetchParams(BaseModel):
    """Parameters for Open-Meteo weather fetch. All required."""

    start_date: date = Field(..., description="Start date (inclusive)")
    end_date: date = Field(..., description="End date (inclusive)")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude (WGS84)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (WGS84)")
