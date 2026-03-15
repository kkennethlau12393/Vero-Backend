-- Admin bootstrap SQL
-- Run once to set up admin infrastructure.

-- 1. Add is_admin column to public.users
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS is_admin boolean NOT NULL DEFAULT false;

-- 2. Admin audit log
CREATE TABLE IF NOT EXISTS public.admin_audit_log (
    id              bigserial PRIMARY KEY,
    admin_user_id   uuid NOT NULL REFERENCES auth.users(id),
    action          text NOT NULL,
    target_user_id  uuid REFERENCES auth.users(id),
    details         jsonb NOT NULL DEFAULT '{}',
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_admin_audit_created ON admin_audit_log(created_at DESC);

-- Bootstrap first admin:
-- UPDATE public.users SET is_admin = true WHERE user_id = '<your-user-uuid>';
