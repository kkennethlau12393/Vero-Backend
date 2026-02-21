-- migrate-multi-workspace.sql
-- Migrates from single-workspace-per-user to multi-workspace-per-user.
-- Safe to run multiple times (idempotent).
-- Run this BEFORE re-running user-bootstrap.sql on an existing database.

BEGIN;

-- 1. Drop the UNIQUE constraint on owner_user_id (allows multiple workspaces per user)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_schema = 'public' AND table_name = 'workspaces'
          AND constraint_type = 'UNIQUE'
          AND constraint_name = 'workspaces_owner_user_id_key'
    ) THEN
        ALTER TABLE public.workspaces DROP CONSTRAINT workspaces_owner_user_id_key;
        RAISE NOTICE 'Dropped UNIQUE constraint workspaces_owner_user_id_key';
    ELSE
        RAISE NOTICE 'UNIQUE constraint already removed or does not exist';
    END IF;
END $$;

-- 2. Drop the workspace_id column from users table (no longer 1:1)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'users' AND column_name = 'workspace_id'
    ) THEN
        ALTER TABLE public.users DROP COLUMN workspace_id;
        RAISE NOTICE 'Dropped workspace_id column from users table';
    ELSE
        RAISE NOTICE 'workspace_id column already removed from users table';
    END IF;
END $$;

-- 3. Drop the old current_workspace_id() function (no longer applicable)
DROP FUNCTION IF EXISTS public.current_workspace_id();

-- 4. Ensure the index on owner_user_id exists (non-unique, for lookups)
CREATE INDEX IF NOT EXISTS idx_workspaces_owner_user_id
    ON public.workspaces (owner_user_id);

COMMIT;
