"""Admin Stripe API — real Stripe data for dashboard analytics."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional
from uuid import UUID

import stripe
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.auth.admin import require_admin
from app.db import make_engine
from app.admin.admin_service import ensure_admin_tables
from app.admin.stripe_schemas import (
    StripeOverviewResponse,
    StripeSubscriptionListResponse,
    StripeInvoiceListResponse,
    StripeRevenueTimelineResponse,
    StripeSubscriptionDetailResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/admin/stripe", tags=["admin-stripe"])

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine()


def _stripe_configured() -> bool:
    return bool(stripe.api_key)


def _ts_to_iso(ts: int | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _cents_to_pounds(amount: int | None) -> float:
    if not amount:
        return 0.0
    return round(amount / 100.0, 2)


# ---------------------------------------------------------------------------
# GET /v1/admin/stripe/overview — real MRR, ARR, sub counts from Stripe
# ---------------------------------------------------------------------------

@router.get("/overview", response_model=StripeOverviewResponse)
def stripe_overview(
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    if not _stripe_configured():
        return StripeOverviewResponse(
            configured=False,
            total_subscriptions=0,
            active_subscriptions=0,
            past_due_subscriptions=0,
            canceled_subscriptions=0,
            trialing_subscriptions=0,
            mrr=0,
            arr=0,
            total_revenue=0,
            revenue_30d=0,
            total_customers=0,
        )

    # Count subscriptions by status
    active = stripe.Subscription.list(status="active", limit=1)
    past_due = stripe.Subscription.list(status="past_due", limit=1)
    canceled = stripe.Subscription.list(status="canceled", limit=1)
    trialing = stripe.Subscription.list(status="trialing", limit=1)

    active_count = active.total_count if hasattr(active, 'total_count') else len(active.data)
    past_due_count = past_due.total_count if hasattr(past_due, 'total_count') else len(past_due.data)
    canceled_count = canceled.total_count if hasattr(canceled, 'total_count') else len(canceled.data)
    trialing_count = trialing.total_count if hasattr(trialing, 'total_count') else len(trialing.data)

    # Calculate real MRR from active subscriptions
    mrr = 0.0
    all_active = stripe.Subscription.list(status="active", limit=100, expand=["data.items.data.price"])
    for sub in all_active.auto_paging_iter():
        for item in sub["items"]["data"]:
            price = item.get("price", {})
            amount = price.get("unit_amount", 0) or 0
            interval = price.get("recurring", {}).get("interval", "month")
            interval_count = price.get("recurring", {}).get("interval_count", 1)
            qty = item.get("quantity", 1)

            if interval == "year":
                mrr += (amount * qty) / (12 * interval_count)
            elif interval == "month":
                mrr += (amount * qty) / interval_count
            elif interval == "week":
                mrr += (amount * qty * 52) / (12 * interval_count)
            elif interval == "day":
                mrr += (amount * qty * 365) / (12 * interval_count)

    mrr_pounds = round(mrr / 100.0, 2)
    arr_pounds = round(mrr_pounds * 12, 2)

    # Total revenue (all time) and last 30 days
    total_revenue = 0.0
    revenue_30d = 0.0
    thirty_days_ago = int((datetime.now(timezone.utc).timestamp())) - (30 * 86400)

    charges = stripe.Charge.list(limit=100)
    for charge in charges.auto_paging_iter():
        if charge["status"] == "succeeded" and not charge.get("refunded"):
            total_revenue += charge["amount"]
            if charge["created"] >= thirty_days_ago:
                revenue_30d += charge["amount"]

    # Customer count
    customers = stripe.Customer.list(limit=1)
    customer_count = customers.total_count if hasattr(customers, 'total_count') else 0

    return StripeOverviewResponse(
        configured=True,
        total_subscriptions=active_count + past_due_count + canceled_count + trialing_count,
        active_subscriptions=active_count,
        past_due_subscriptions=past_due_count,
        canceled_subscriptions=canceled_count,
        trialing_subscriptions=trialing_count,
        mrr=mrr_pounds,
        arr=arr_pounds,
        total_revenue=_cents_to_pounds(int(total_revenue)),
        revenue_30d=_cents_to_pounds(int(revenue_30d)),
        total_customers=customer_count,
    )


# ---------------------------------------------------------------------------
# GET /v1/admin/stripe/subscriptions — list all subscriptions
# ---------------------------------------------------------------------------

@router.get("/subscriptions", response_model=StripeSubscriptionListResponse)
def list_subscriptions(
    status: Optional[str] = Query(None, regex="^(active|past_due|canceled|trialing|all)$"),
    limit: int = Query(25, ge=1, le=100),
    starting_after: Optional[str] = Query(None),
    engine: Engine = Depends(get_engine),
    admin_id: UUID = Depends(require_admin),
):
    if not _stripe_configured():
        return StripeSubscriptionListResponse(subscriptions=[], has_more=False, total=0)

    params: dict = {"limit": limit, "expand": ["data.customer", "data.items.data.price"]}
    if status and status != "all":
        params["status"] = status
    if starting_after:
        params["starting_after"] = starting_after

    result = stripe.Subscription.list(**params)

    # Map user emails from DB
    with engine.connect() as conn:
        ensure_admin_tables(conn)

        subs = []
        for sub in result.data:
            customer = sub.get("customer", {})
            customer_id = customer["id"] if isinstance(customer, dict) else customer
            customer_email = customer.get("email", "") if isinstance(customer, dict) else ""

            # Try to find user in our DB
            user_email = customer_email
            user_id = None
            if customer_id:
                row = conn.execute(
                    text("SELECT user_id FROM user_billing WHERE stripe_customer_id = :cid"),
                    {"cid": customer_id},
                ).mappings().first()
                if row:
                    user_id = str(row["user_id"])
                    user_row = conn.execute(
                        text("SELECT email FROM auth.users WHERE id = :uid"),
                        {"uid": row["user_id"]},
                    ).mappings().first()
                    if user_row:
                        user_email = user_row["email"]

            # Get price info
            price_info = ""
            amount = 0
            interval = "month"
            for item in sub["items"]["data"]:
                price = item.get("price", {})
                amount = price.get("unit_amount", 0) or 0
                interval = price.get("recurring", {}).get("interval", "month")
                price_info = f"£{amount/100:.2f}/{interval}"

            subs.append({
                "subscription_id": sub["id"],
                "customer_id": customer_id,
                "customer_email": user_email,
                "user_id": user_id,
                "status": sub["status"],
                "price_display": price_info,
                "amount": _cents_to_pounds(amount),
                "interval": interval,
                "current_period_start": _ts_to_iso(sub.get("current_period_start")),
                "current_period_end": _ts_to_iso(sub.get("current_period_end")),
                "created": _ts_to_iso(sub.get("created")),
                "cancel_at_period_end": sub.get("cancel_at_period_end", False),
            })

    return StripeSubscriptionListResponse(
        subscriptions=subs,
        has_more=result.has_more,
        total=len(subs),
    )


# ---------------------------------------------------------------------------
# GET /v1/admin/stripe/invoices — recent invoices
# ---------------------------------------------------------------------------

@router.get("/invoices", response_model=StripeInvoiceListResponse)
def list_invoices(
    limit: int = Query(25, ge=1, le=100),
    status: Optional[str] = Query(None, regex="^(draft|open|paid|uncollectible|void)$"),
    starting_after: Optional[str] = Query(None),
    admin_id: UUID = Depends(require_admin),
):
    if not _stripe_configured():
        return StripeInvoiceListResponse(invoices=[], has_more=False, total=0)

    params: dict = {"limit": limit, "expand": ["data.customer"]}
    if status:
        params["status"] = status
    if starting_after:
        params["starting_after"] = starting_after

    result = stripe.Invoice.list(**params)

    invoices = []
    for inv in result.data:
        customer = inv.get("customer", {})
        invoices.append({
            "invoice_id": inv["id"],
            "number": inv.get("number"),
            "customer_email": customer.get("email", "") if isinstance(customer, dict) else "",
            "status": inv["status"],
            "amount_due": _cents_to_pounds(inv.get("amount_due")),
            "amount_paid": _cents_to_pounds(inv.get("amount_paid")),
            "currency": inv.get("currency", "gbp"),
            "created": _ts_to_iso(inv.get("created")),
            "due_date": _ts_to_iso(inv.get("due_date")),
            "paid_at": _ts_to_iso(inv.get("status_transitions", {}).get("paid_at")),
            "hosted_invoice_url": inv.get("hosted_invoice_url"),
            "pdf_url": inv.get("invoice_pdf"),
        })

    return StripeInvoiceListResponse(
        invoices=invoices,
        has_more=result.has_more,
        total=len(invoices),
    )


# ---------------------------------------------------------------------------
# GET /v1/admin/stripe/revenue-timeline — revenue over time
# ---------------------------------------------------------------------------

@router.get("/revenue-timeline", response_model=StripeRevenueTimelineResponse)
def revenue_timeline(
    period: str = Query("30d", regex="^(7d|30d|90d|1y)$"),
    admin_id: UUID = Depends(require_admin),
):
    if not _stripe_configured():
        return StripeRevenueTimelineResponse(period=period, data=[])

    interval_map = {"7d": 7, "30d": 30, "90d": 90, "1y": 365}
    days = interval_map[period]
    start_ts = int(datetime.now(timezone.utc).timestamp()) - (days * 86400)

    # Get all successful charges in the period
    charges = stripe.Charge.list(
        limit=100,
        created={"gte": start_ts},
    )

    # Bucket by day
    daily: dict[str, float] = {}
    for charge in charges.auto_paging_iter():
        if charge["status"] == "succeeded" and not charge.get("refunded"):
            day = datetime.fromtimestamp(charge["created"], tz=timezone.utc).strftime("%Y-%m-%d")
            daily[day] = daily.get(day, 0) + charge["amount"]

    # Fill gaps
    data = []
    for i in range(days):
        day = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() - ((days - 1 - i) * 86400),
            tz=timezone.utc,
        ).strftime("%Y-%m-%d")
        data.append({
            "date": day,
            "revenue": _cents_to_pounds(int(daily.get(day, 0))),
        })

    return StripeRevenueTimelineResponse(period=period, data=data)


# ---------------------------------------------------------------------------
# GET /v1/admin/stripe/subscription/{id} — single subscription detail
# ---------------------------------------------------------------------------

@router.get("/subscription/{subscription_id}", response_model=StripeSubscriptionDetailResponse)
def subscription_detail(
    subscription_id: str,
    admin_id: UUID = Depends(require_admin),
):
    if not _stripe_configured():
        raise HTTPException(status_code=503, detail="Stripe not configured")

    try:
        sub = stripe.Subscription.retrieve(
            subscription_id,
            expand=["customer", "items.data.price", "latest_invoice"],
        )
    except stripe.InvalidRequestError:
        raise HTTPException(status_code=404, detail="Subscription not found")

    customer = sub.get("customer", {})
    latest_invoice = sub.get("latest_invoice", {})

    items = []
    for item in sub["items"]["data"]:
        price = item.get("price", {})
        items.append({
            "price_id": price.get("id"),
            "product_id": price.get("product"),
            "amount": _cents_to_pounds(price.get("unit_amount")),
            "interval": price.get("recurring", {}).get("interval", "month"),
            "quantity": item.get("quantity", 1),
        })

    return StripeSubscriptionDetailResponse(
        subscription_id=sub["id"],
        customer_id=customer["id"] if isinstance(customer, dict) else customer,
        customer_email=customer.get("email", "") if isinstance(customer, dict) else "",
        status=sub["status"],
        items=items,
        current_period_start=_ts_to_iso(sub.get("current_period_start")),
        current_period_end=_ts_to_iso(sub.get("current_period_end")),
        created=_ts_to_iso(sub.get("created")),
        cancel_at_period_end=sub.get("cancel_at_period_end", False),
        cancel_at=_ts_to_iso(sub.get("cancel_at")),
        canceled_at=_ts_to_iso(sub.get("canceled_at")),
        latest_invoice_status=latest_invoice.get("status") if isinstance(latest_invoice, dict) else None,
        latest_invoice_amount=_cents_to_pounds(latest_invoice.get("amount_paid")) if isinstance(latest_invoice, dict) else 0,
    )
