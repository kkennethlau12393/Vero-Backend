"""Billing API — Stripe checkout, webhook, status, portal."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional
from uuid import UUID

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.auth.jwt_user import get_current_user_id
from app.billing.credits import ensure_billing_tables
from app.db import make_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/billing", tags=["billing"])

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_PRO = os.environ.get("STRIPE_PRICE_PRO", "")

PRO_CREDITS = 200
FREE_CREDITS = 15


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_user(user_id: Optional[UUID]) -> UUID:
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id


def _get_or_create_billing_row(conn, user_id: UUID) -> dict:
    """Ensure a user_billing row exists and return it."""
    conn.execute(
        text("""
            INSERT INTO user_billing (user_id)
            VALUES (:uid)
            ON CONFLICT (user_id) DO NOTHING
        """),
        {"uid": user_id},
    )
    row = conn.execute(
        text("SELECT * FROM user_billing WHERE user_id = :uid"),
        {"uid": user_id},
    ).mappings().first()
    return dict(row)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CheckoutRequest(BaseModel):
    plan: str  # 'pro'
    source: str = "dashboard"


class CheckoutResponse(BaseModel):
    checkout_url: str


class BillingStatusResponse(BaseModel):
    plan: str
    credits_remaining: float
    credits_monthly_allowance: float
    subscription_status: str
    current_period_end: Optional[str]
    days_until_renewal: Optional[int]


class PortalResponse(BaseModel):
    portal_url: str


# ---------------------------------------------------------------------------
# Endpoint 1: POST /v1/billing/checkout
# ---------------------------------------------------------------------------

@router.post("/checkout", response_model=CheckoutResponse)
def create_checkout(
    req: CheckoutRequest,
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """Create a Stripe Checkout session for a subscription."""
    uid = _require_user(user_id)

    if req.plan != "pro":
        raise HTTPException(status_code=400, detail="Only 'pro' plan is available")

    if not STRIPE_PRICE_PRO:
        raise HTTPException(status_code=500, detail="Stripe price not configured")

    with engine.connect() as conn:
        ensure_billing_tables(conn)
        billing = _get_or_create_billing_row(conn, uid)

        # Look up user email
        user_row = conn.execute(
            text("SELECT email FROM auth.users WHERE id = :uid"),
            {"uid": uid},
        ).mappings().first()
        email = user_row["email"] if user_row else None

        # Get or create Stripe customer
        customer_id = billing.get("stripe_customer_id")
        if not customer_id:
            customer = stripe.Customer.create(
                email=email,
                metadata={"user_id": str(uid)},
            )
            customer_id = customer.id
            conn.execute(
                text("""
                    UPDATE user_billing
                    SET stripe_customer_id = :cid, updated_at = now()
                    WHERE user_id = :uid
                """),
                {"cid": customer_id, "uid": uid},
            )
            conn.commit()

    # Create checkout session
    cancel_path = "/settings" if req.source == "settings" else "/dashboard"
    session = stripe.checkout.Session.create(
        customer=customer_id,
        line_items=[{"price": STRIPE_PRICE_PRO, "quantity": 1}],
        mode="subscription",
        success_url="https://www.alexandrialabs.uk/dashboard?checkout=success",
        cancel_url=f"https://www.alexandrialabs.uk{cancel_path}?checkout=canceled",
        metadata={"user_id": str(uid)},
        subscription_data={"metadata": {"user_id": str(uid)}},
    )

    return CheckoutResponse(checkout_url=session.url)


# ---------------------------------------------------------------------------
# Endpoint 2: POST /v1/billing/webhook
# ---------------------------------------------------------------------------

@router.post("/webhook")
async def stripe_webhook(request: Request, engine: Engine = Depends(get_engine)):
    """Handle Stripe webhook events. No auth — verified by signature."""
    body = await request.body()
    sig = request.headers.get("stripe-signature", "")

    if not STRIPE_WEBHOOK_SECRET:
        logger.warning("STRIPE_WEBHOOK_SECRET not configured — ignoring webhook")
        return {"status": "ignored"}

    try:
        event = stripe.Webhook.construct_event(body, sig, STRIPE_WEBHOOK_SECRET)
    except stripe.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event["type"]
    data = event["data"]["object"]
    logger.info("Stripe webhook: %s", event_type)

    with engine.connect() as conn:
        ensure_billing_tables(conn)

        if event_type == "checkout.session.completed":
            _handle_checkout_completed(conn, data)
        elif event_type == "invoice.paid":
            _handle_invoice_paid(conn, data)
        elif event_type == "invoice.payment_failed":
            _handle_payment_failed(conn, data)
        elif event_type == "customer.subscription.updated":
            _handle_subscription_updated(conn, data)
        elif event_type == "customer.subscription.deleted":
            _handle_subscription_deleted(conn, data)
        else:
            logger.info("Unhandled Stripe event: %s", event_type)

    return {"status": "ok"}


def _handle_checkout_completed(conn, session: dict) -> None:
    """checkout.session.completed — activate subscription."""
    user_id = session.get("metadata", {}).get("user_id")
    if not user_id:
        logger.warning("checkout.session.completed without user_id in metadata")
        return

    subscription_id = session.get("subscription")
    customer_id = session.get("customer")

    # Fetch subscription details for period_end
    period_end = None
    if subscription_id:
        sub = stripe.Subscription.retrieve(subscription_id)
        if sub.get("current_period_end"):
            period_end = datetime.fromtimestamp(sub["current_period_end"], tz=timezone.utc)

    conn.execute(
        text("""
            INSERT INTO user_billing (user_id, plan, credits_remaining, credits_monthly,
                                       stripe_customer_id, stripe_subscription_id,
                                       subscription_status, current_period_end)
            VALUES (:uid, 'pro', :credits, :monthly, :cid, :sid, 'active', :period_end)
            ON CONFLICT (user_id) DO UPDATE SET
                plan = 'pro',
                credits_remaining = :credits,
                credits_monthly = :monthly,
                stripe_customer_id = COALESCE(EXCLUDED.stripe_customer_id, user_billing.stripe_customer_id),
                stripe_subscription_id = :sid,
                subscription_status = 'active',
                current_period_end = :period_end,
                updated_at = now()
        """),
        {
            "uid": user_id,
            "credits": PRO_CREDITS,
            "monthly": PRO_CREDITS,
            "cid": customer_id,
            "sid": subscription_id,
            "period_end": period_end,
        },
    )
    conn.commit()
    logger.info("Activated pro plan for user %s", user_id)


def _handle_invoice_paid(conn, invoice: dict) -> None:
    """invoice.paid — monthly renewal, reset credits."""
    subscription_id = invoice.get("subscription")
    if not subscription_id:
        return

    # Look up user by subscription_id
    row = conn.execute(
        text("SELECT user_id, credits_monthly FROM user_billing WHERE stripe_subscription_id = :sid"),
        {"sid": subscription_id},
    ).mappings().first()
    if not row:
        logger.warning("invoice.paid for unknown subscription %s", subscription_id)
        return

    user_id = row["user_id"]
    monthly = float(row["credits_monthly"])

    # Fetch updated period_end
    sub = stripe.Subscription.retrieve(subscription_id)
    period_end = datetime.fromtimestamp(sub["current_period_end"], tz=timezone.utc) if sub.get("current_period_end") else None

    conn.execute(
        text("""
            UPDATE user_billing
            SET credits_remaining = :monthly, current_period_end = :period_end, updated_at = now()
            WHERE user_id = :uid
        """),
        {"uid": user_id, "monthly": monthly, "period_end": period_end},
    )

    # Log the credit reset
    conn.execute(
        text("""
            INSERT INTO credit_transactions (user_id, amount, balance_after, reason, metadata)
            VALUES (:uid, :amount, :bal, 'subscription_renewal', '{}')
        """),
        {"uid": user_id, "amount": monthly, "bal": monthly},
    )
    conn.commit()
    logger.info("Renewed credits for user %s: %.0f", user_id, monthly)


def _handle_payment_failed(conn, invoice: dict) -> None:
    """invoice.payment_failed — mark subscription as past_due."""
    subscription_id = invoice.get("subscription")
    if not subscription_id:
        return

    conn.execute(
        text("""
            UPDATE user_billing
            SET subscription_status = 'past_due', updated_at = now()
            WHERE stripe_subscription_id = :sid
        """),
        {"sid": subscription_id},
    )
    conn.commit()
    logger.info("Marked subscription %s as past_due", subscription_id)


def _handle_subscription_updated(conn, subscription: dict) -> None:
    """customer.subscription.updated — handle plan changes."""
    subscription_id = subscription.get("id")
    status = subscription.get("status", "active")

    period_end = None
    if subscription.get("current_period_end"):
        period_end = datetime.fromtimestamp(subscription["current_period_end"], tz=timezone.utc)

    conn.execute(
        text("""
            UPDATE user_billing
            SET subscription_status = :status, current_period_end = :period_end, updated_at = now()
            WHERE stripe_subscription_id = :sid
        """),
        {"sid": subscription_id, "status": status, "period_end": period_end},
    )
    conn.commit()
    logger.info("Updated subscription %s status to %s", subscription_id, status)


def _handle_subscription_deleted(conn, subscription: dict) -> None:
    """customer.subscription.deleted — downgrade to free."""
    subscription_id = subscription.get("id")

    row = conn.execute(
        text("SELECT user_id, credits_remaining FROM user_billing WHERE stripe_subscription_id = :sid"),
        {"sid": subscription_id},
    ).mappings().first()
    if not row:
        return

    user_id = row["user_id"]
    # Cap credits at free tier allowance
    capped_credits = min(float(row["credits_remaining"]), FREE_CREDITS)

    conn.execute(
        text("""
            UPDATE user_billing
            SET plan = 'free',
                credits_remaining = :credits,
                credits_monthly = :free_monthly,
                subscription_status = 'canceled',
                stripe_subscription_id = NULL,
                current_period_end = NULL,
                updated_at = now()
            WHERE user_id = :uid
        """),
        {"uid": user_id, "credits": capped_credits, "free_monthly": FREE_CREDITS},
    )
    conn.commit()
    logger.info("Downgraded user %s to free plan", user_id)


# ---------------------------------------------------------------------------
# Endpoint 3: GET /v1/billing/status
# ---------------------------------------------------------------------------

@router.get("/status", response_model=BillingStatusResponse)
def get_billing_status(
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """Return current billing status for the authenticated user."""
    uid = _require_user(user_id)

    with engine.connect() as conn:
        ensure_billing_tables(conn)
        billing = _get_or_create_billing_row(conn, uid)
        conn.commit()

    period_end = billing.get("current_period_end")
    days_until = None
    period_end_str = None
    if period_end:
        period_end_str = period_end.isoformat()
        delta = period_end - datetime.now(timezone.utc)
        days_until = max(0, delta.days)

    return BillingStatusResponse(
        plan=billing["plan"],
        credits_remaining=float(billing["credits_remaining"]),
        credits_monthly_allowance=float(billing["credits_monthly"]),
        subscription_status=billing["subscription_status"],
        current_period_end=period_end_str,
        days_until_renewal=days_until,
    )


# ---------------------------------------------------------------------------
# Endpoint 4: POST /v1/billing/portal
# ---------------------------------------------------------------------------

@router.post("/portal", response_model=PortalResponse)
def create_portal_session(
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """Create a Stripe billing portal session for managing subscription."""
    uid = _require_user(user_id)

    with engine.connect() as conn:
        ensure_billing_tables(conn)
        billing = _get_or_create_billing_row(conn, uid)
        conn.commit()

    customer_id = billing.get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(status_code=400, detail="No active subscription found")

    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url="https://www.alexandrialabs.uk/dashboard",
    )

    return PortalResponse(portal_url=session.url)
