"""Pydantic models for admin Stripe API."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class StripeOverviewResponse(BaseModel):
    configured: bool
    total_subscriptions: int
    active_subscriptions: int
    past_due_subscriptions: int
    canceled_subscriptions: int
    trialing_subscriptions: int
    mrr: float
    arr: float
    total_revenue: float
    revenue_30d: float
    total_customers: int


class StripeSubscriptionSummary(BaseModel):
    subscription_id: str
    customer_id: str
    customer_email: str
    user_id: Optional[str] = None
    status: str
    price_display: str
    amount: float
    interval: str
    current_period_start: Optional[str] = None
    current_period_end: Optional[str] = None
    created: Optional[str] = None
    cancel_at_period_end: bool = False


class StripeSubscriptionListResponse(BaseModel):
    subscriptions: list[StripeSubscriptionSummary]
    has_more: bool
    total: int


class StripeInvoiceSummary(BaseModel):
    invoice_id: str
    number: Optional[str] = None
    customer_email: str
    status: Optional[str] = None
    amount_due: float
    amount_paid: float
    currency: str = "gbp"
    created: Optional[str] = None
    due_date: Optional[str] = None
    paid_at: Optional[str] = None
    hosted_invoice_url: Optional[str] = None
    pdf_url: Optional[str] = None


class StripeInvoiceListResponse(BaseModel):
    invoices: list[StripeInvoiceSummary]
    has_more: bool
    total: int


class RevenueDataPoint(BaseModel):
    date: str
    revenue: float


class StripeRevenueTimelineResponse(BaseModel):
    period: str
    data: list[RevenueDataPoint]


class SubscriptionItemDetail(BaseModel):
    price_id: Optional[str] = None
    product_id: Optional[str] = None
    amount: float
    interval: str
    quantity: int


class StripeSubscriptionDetailResponse(BaseModel):
    subscription_id: str
    customer_id: str
    customer_email: str
    status: str
    items: list[SubscriptionItemDetail]
    current_period_start: Optional[str] = None
    current_period_end: Optional[str] = None
    created: Optional[str] = None
    cancel_at_period_end: bool = False
    cancel_at: Optional[str] = None
    canceled_at: Optional[str] = None
    latest_invoice_status: Optional[str] = None
    latest_invoice_amount: float = 0
