-- billing-bootstrap.sql
-- Creates billing tables for Stripe subscription management and credit tracking.
-- Safe to re-run (uses IF NOT EXISTS).

\set ON_ERROR_STOP on
BEGIN;

SET search_path = public;

-- Per-user billing state (plan, credits, Stripe IDs)
CREATE TABLE IF NOT EXISTS public.user_billing (
    user_id              uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    plan                 text NOT NULL DEFAULT 'free',
    credits_remaining    numeric(10,2) NOT NULL DEFAULT 15,
    credits_monthly      numeric(10,2) NOT NULL DEFAULT 15,
    stripe_customer_id   text,
    stripe_subscription_id text,
    subscription_status  text NOT NULL DEFAULT 'none',
    current_period_end   timestamptz,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now()
);

-- Audit log for every credit change
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

COMMIT;
