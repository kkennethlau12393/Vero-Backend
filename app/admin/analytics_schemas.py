"""Pydantic models for admin analytics API."""
from __future__ import annotations

from pydantic import BaseModel


class GrowthDataPoint(BaseModel):
    date: str
    signups: int
    cumulative: int


class UserGrowthResponse(BaseModel):
    period: str
    data: list[GrowthDataPoint]


class RevenueDataPoint(BaseModel):
    date: str
    credits_used: float
    estimated_cost: float
    estimated_revenue: float


class RevenueResponse(BaseModel):
    period: str
    data: list[RevenueDataPoint]


class FeatureUsage(BaseModel):
    reason: str
    usage_count: int
    total_credits: float


class DailyCreditUsage(BaseModel):
    date: str
    reason: str
    credits: float


class CreditUsageResponse(BaseModel):
    period: str
    by_feature: list[FeatureUsage]
    daily: list[DailyCreditUsage]


class RetentionDataPoint(BaseModel):
    period: str
    total: int
    retained: int
    rate: float


class RetentionResponse(BaseModel):
    data: list[RetentionDataPoint]


class OverviewResponse(BaseModel):
    signups_today: int
    active_24h: int
    pro_subscribers: int
    credits_used_today: float
    credits_used_24h: float
    daily_burn_rate: float
    estimated_daily_revenue: float
