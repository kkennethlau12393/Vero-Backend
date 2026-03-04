"""Credit check and deduction for billing-gated features."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Table creation DDL — runs once per process via _ensure_tables().
_BILLING_DDL = """
CREATE TABLE IF NOT EXISTS public.user_billing (
    user_id              uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    plan                 text NOT NULL DEFAULT 'free',
    credits_remaining    numeric(10,2) NOT NULL DEFAULT 10,
    credits_monthly      numeric(10,2) NOT NULL DEFAULT 10,
    stripe_customer_id   text,
    stripe_subscription_id text,
    subscription_status  text NOT NULL DEFAULT 'none',
    current_period_end   timestamptz,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS public.credit_transactions (
    id            bigserial PRIMARY KEY,
    user_id       uuid NOT NULL REFERENCES auth.users(id),
    amount        numeric(10,2) NOT NULL,
    balance_after numeric(10,2) NOT NULL,
    reason        text NOT NULL,
    metadata      jsonb DEFAULT '{}',
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_credit_tx_user ON credit_transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_billing_stripe_customer ON user_billing(stripe_customer_id);
ALTER TABLE IF EXISTS public.user_billing
  ADD COLUMN IF NOT EXISTS workspaces_created_count INT NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS public.user_billing
  ADD COLUMN IF NOT EXISTS credits_period_start TIMESTAMPTZ DEFAULT now();
ALTER TABLE IF EXISTS public.user_billing
  ADD COLUMN IF NOT EXISTS billing_interval TEXT DEFAULT 'monthly';
"""

_tables_ensured = False


def ensure_billing_tables(conn) -> None:
    """Create billing tables if they don't exist (idempotent)."""
    global _tables_ensured
    if _tables_ensured:
        return
    conn.execute(text(_BILLING_DDL))
    conn.commit()
    _tables_ensured = True


def require_credits(
    conn,
    user_id: UUID,
    cost: float,
    reason: str,
    metadata: dict | None = None,
) -> None:
    """Check credits and deduct. Raises 402 if insufficient.

    Uses SELECT ... FOR UPDATE to prevent race conditions.
    Auto-creates a user_billing row (free plan, 15 credits) if none exists.
    """
    ensure_billing_tables(conn)

    # Upsert: ensure billing row exists for this user
    conn.execute(
        text("""
            INSERT INTO user_billing (user_id)
            VALUES (:uid)
            ON CONFLICT (user_id) DO NOTHING
        """),
        {"uid": user_id},
    )

    # Lock the row and read current balance
    row = conn.execute(
        text("""
            SELECT credits_remaining, credits_monthly,
                   credits_period_start, subscription_status, plan, billing_interval
            FROM user_billing
            WHERE user_id = :uid
            FOR UPDATE
        """),
        {"uid": user_id},
    ).mappings().first()

    # Lazy 30-day credit reset for active subscribers
    period_start = row["credits_period_start"]
    if (
        period_start is not None
        and row["billing_interval"] == "annual" and row["subscription_status"] == "active"
        and datetime.now(timezone.utc) > period_start + timedelta(days=30)
    ):
        monthly = float(row["credits_monthly"])
        conn.execute(
            text("""
                UPDATE user_billing
                SET credits_remaining = credits_monthly,
                    credits_period_start = now(),
                    updated_at = now()
                WHERE user_id = :uid
            """),
            {"uid": user_id},
        )
        conn.execute(
            text("""
                INSERT INTO credit_transactions
                (user_id, amount, balance_after, reason, metadata)
                VALUES (:uid, :monthly, :monthly, 'monthly_credit_reset', '{}')
            """),
            {"uid": user_id, "monthly": monthly},
        )
        balance = monthly
    else:
        balance = float(row["credits_remaining"])

    if balance < cost:
        conn.rollback()
        raise HTTPException(
            status_code=402,
            detail="insufficient_credits",
        )

    new_balance = balance - cost

    conn.execute(
        text("""
            UPDATE user_billing
            SET credits_remaining = :bal, updated_at = now()
            WHERE user_id = :uid
        """),
        {"uid": user_id, "bal": new_balance},
    )

    conn.execute(
        text("""
            INSERT INTO credit_transactions (user_id, amount, balance_after, reason, metadata)
            VALUES (:uid, :amount, :bal, :reason, :meta)
        """),
        {
            "uid": user_id,
            "amount": -cost,
            "bal": new_balance,
            "reason": reason,
            "meta": json.dumps(metadata or {}),
        },
    )

    conn.commit()
    logger.info("Deducted %.2f credits from user %s (%s). Balance: %.2f", cost, user_id, reason, new_balance)
