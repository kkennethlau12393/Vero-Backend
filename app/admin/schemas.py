"""Pydantic models for admin API."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


class AdminMeResponse(BaseModel):
    is_admin: bool


class RecentSignup(BaseModel):
    email: str
    created_at: str


class AdminStatsResponse(BaseModel):
    total_users: int
    pro_subscribers: int
    mrr_estimate: float
    signups_7d: int
    active_7d: int
    active_30d: int
    total_credits_consumed_30d: float = 0
    estimated_cost_30d: float = 0
    estimated_margin_30d: float = 0
    recent_signups: list[RecentSignup]


class AdminUserSummary(BaseModel):
    user_id: str
    email: str
    name: Optional[str] = None
    plan: str
    credits_remaining: float
    subscription_status: str
    is_admin: bool
    created_at: Optional[str] = None
    last_sign_in_at: Optional[str] = None


class AdminUserListResponse(BaseModel):
    users: list[AdminUserSummary]
    total: int
    page: int
    per_page: int


class BillingDetail(BaseModel):
    plan: str
    credits_remaining: float
    credits_monthly: float
    subscription_status: str
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    current_period_end: Optional[str] = None
    admin_access_expires_at: Optional[str] = None


class WorkspaceSummary(BaseModel):
    workspace_id: str
    name: str
    created_at: Optional[str] = None


class TransactionSummary(BaseModel):
    id: str
    amount: float
    balance_after: float
    reason: str
    created_at: str


class AdminUserDetailResponse(BaseModel):
    user_id: str
    email: str
    name: Optional[str] = None
    plan: str
    credits_remaining: float
    subscription_status: str
    is_admin: bool
    created_at: Optional[str] = None
    last_sign_in_at: Optional[str] = None
    banned_until: Optional[str] = None
    billing: Optional[BillingDetail] = None
    workspaces: list[WorkspaceSummary] = []
    recent_transactions: list[TransactionSummary] = []


class SubscriptionActionRequest(BaseModel):
    action: str  # "grant_pro" | "revoke_to_free" | "grant_credits" | "grant_temporary_pro" | "process_refund" | "extend_billing"
    credits_amount: Optional[float] = None
    expires_at: Optional[str] = None
    refund_reason: Optional[str] = None
    new_period_end: Optional[str] = None


class AdminRoleRequest(BaseModel):
    is_admin: bool


class AuditLogEntry(BaseModel):
    id: int
    admin_email: Optional[str] = None
    action: str
    target_email: Optional[str] = None
    details: Any = {}
    created_at: str


class AuditLogResponse(BaseModel):
    entries: list[AuditLogEntry]
    total: int
    page: int
    per_page: int


class WorkspaceCostEntry(BaseModel):
    workspace_id: Optional[str] = None
    workspace_name: Optional[str] = None
    reason: str
    usage_count: int
    total_credits_used: float
    estimated_cost_gbp: float


class CostBreakdownResponse(BaseModel):
    entries: list[WorkspaceCostEntry]
    total_credits_used: float
    total_estimated_cost_gbp: float
    subscription_revenue_gbp: float
    margin_gbp: float


class MessageResponse(BaseModel):
    message: str


class BulkActionRequest(BaseModel):
    user_ids: list[str]  # list of user_id UUIDs
    action: str  # grant_pro, revoke_to_free, grant_credits, disable, enable
    credits_amount: Optional[float] = None  # for grant_credits


class BulkActionResult(BaseModel):
    user_id: str
    success: bool
    message: str


class BulkActionResponse(BaseModel):
    results: list[BulkActionResult]
    total: int
    succeeded: int
    failed: int
