-- user-bootstrap.sql
-- Supabase bootstrap for user/workspace ownership with strict RLS.
-- Current model: one user can own multiple workspaces.

\set ON_ERROR_STOP on
BEGIN;

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET search_path = public;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------
-- Local-dev auth compatibility
-- ---------------------------------------------------------------------
-- In Supabase, auth.users/auth.uid() already exist.
-- In local Postgres, create minimal equivalents so this bootstrap can run.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_namespace WHERE nspname = 'auth'
    ) THEN
        EXECUTE 'CREATE SCHEMA auth';
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS auth.users (
    id uuid PRIMARY KEY,
    email text,
    raw_user_meta_data jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_roles
        WHERE rolname = 'authenticated'
    ) THEN
        CREATE ROLE authenticated NOLOGIN;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'auth'
          AND p.proname = 'uid'
          AND pg_get_function_identity_arguments(p.oid) = ''
    ) THEN
        EXECUTE $sql$
            CREATE FUNCTION auth.uid()
            RETURNS uuid
            LANGUAGE sql
            STABLE
            AS $fn$
                SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid
            $fn$
        $sql$;
    END IF;
END $$;

-- ---------------------------------------------------------------------
-- Core tables
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.workspaces (
    workspace_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    workspace_name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT workspace_name_not_blank CHECK (length(btrim(workspace_name)) > 0)
);

CREATE TABLE IF NOT EXISTS public.users (
    user_id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    display_name text,
    first_name text,
    last_name text,
    avatar_url text,
    profile_picture_url text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- Drop legacy columns/constraints if migrating from single-workspace schema.
DO $$
BEGIN
    -- Drop the 1:1 workspace_id column from users if it exists
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'users' AND column_name = 'workspace_id'
    ) THEN
        ALTER TABLE public.users DROP COLUMN workspace_id;
    END IF;

    -- Drop the UNIQUE constraint on owner_user_id if it exists
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_schema = 'public' AND table_name = 'workspaces'
          AND constraint_type = 'UNIQUE'
          AND constraint_name LIKE '%owner_user_id%'
    ) THEN
        ALTER TABLE public.workspaces DROP CONSTRAINT workspaces_owner_user_id_key;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_workspaces_owner_user_id
    ON public.workspaces (owner_user_id);

-- ---------------------------------------------------------------------
-- Utility functions/triggers
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $fn$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$fn$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_users_set_updated_at'
    ) THEN
        CREATE TRIGGER trg_users_set_updated_at
            BEFORE UPDATE ON public.users
            FOR EACH ROW
            EXECUTE FUNCTION public.set_updated_at();
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_workspaces_set_updated_at'
    ) THEN
        CREATE TRIGGER trg_workspaces_set_updated_at
            BEFORE UPDATE ON public.workspaces
            FOR EACH ROW
            EXECUTE FUNCTION public.set_updated_at();
    END IF;
END $$;

-- Keeps app tables in sync with Supabase auth.users.
-- On new auth user:
--   1) create a default workspace for the user
--   2) ensure users profile row exists
CREATE OR REPLACE FUNCTION public.handle_new_auth_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $fn$
DECLARE
    v_workspace_id uuid;
    v_first_name text;
    v_last_name text;
    v_display_name text;
    v_workspace_name text;
    v_avatar_url text;
    v_profile_picture_url text;
BEGIN
    v_first_name := NULLIF(NEW.raw_user_meta_data ->> 'first_name', '');
    v_last_name := NULLIF(NEW.raw_user_meta_data ->> 'last_name', '');

    v_display_name := COALESCE(
        CASE
            WHEN v_first_name IS NOT NULL OR v_last_name IS NOT NULL
            THEN btrim(COALESCE(v_first_name, '') || ' ' || COALESCE(v_last_name, ''))
            ELSE NULL
        END,
        NEW.raw_user_meta_data ->> 'display_name',
        NEW.raw_user_meta_data ->> 'full_name',
        NEW.raw_user_meta_data ->> 'name',
        split_part(COALESCE(NEW.email, ''), '@', 1),
        'User'
    );

    v_avatar_url := COALESCE(
        NULLIF(NEW.raw_user_meta_data ->> 'avatar_url', ''),
        NULLIF(NEW.raw_user_meta_data ->> 'avatar', '')
    );

    v_profile_picture_url := COALESCE(
        NULLIF(NEW.raw_user_meta_data ->> 'profile_picture_url', ''),
        NULLIF(NEW.raw_user_meta_data ->> 'picture', ''),
        v_avatar_url
    );

    v_workspace_name := COALESCE(
        NEW.raw_user_meta_data ->> 'workspace_name',
        NULLIF(v_display_name, '') || '''s Workspace',
        'Workspace'
    );

    -- Create default workspace (no uniqueness conflict — user can have many)
    INSERT INTO public.workspaces (owner_user_id, workspace_name)
    VALUES (NEW.id, v_workspace_name)
    RETURNING workspace_id INTO v_workspace_id;

    -- Create user profile
    INSERT INTO public.users (
        user_id,
        display_name,
        first_name,
        last_name,
        avatar_url,
        profile_picture_url
    )
    VALUES (
        NEW.id,
        NULLIF(v_display_name, ''),
        v_first_name,
        v_last_name,
        v_avatar_url,
        v_profile_picture_url
    )
    ON CONFLICT (user_id) DO UPDATE
      SET display_name = COALESCE(public.users.display_name, EXCLUDED.display_name),
          first_name = COALESCE(public.users.first_name, EXCLUDED.first_name),
          last_name = COALESCE(public.users.last_name, EXCLUDED.last_name),
          avatar_url = COALESCE(public.users.avatar_url, EXCLUDED.avatar_url),
          profile_picture_url = COALESCE(public.users.profile_picture_url, EXCLUDED.profile_picture_url);

    RETURN NEW;
END;
$fn$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_trigger t
        JOIN pg_class c ON c.oid = t.tgrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE t.tgname = 'on_auth_user_created'
          AND n.nspname = 'auth'
          AND c.relname = 'users'
    ) THEN
        CREATE TRIGGER on_auth_user_created
            AFTER INSERT ON auth.users
            FOR EACH ROW
            EXECUTE FUNCTION public.handle_new_auth_user();
    END IF;
END $$;

-- Backfill: if auth users already exist, ensure app tables are synced.
DO $$
DECLARE
    r RECORD;
    v_first_name text;
    v_last_name text;
    v_display_name text;
    v_workspace_name text;
    v_avatar_url text;
    v_profile_picture_url text;
BEGIN
    FOR r IN
        SELECT au.id, au.email, au.raw_user_meta_data
        FROM auth.users au
        LEFT JOIN public.users u ON u.user_id = au.id
        WHERE u.user_id IS NULL
    LOOP
        v_first_name := NULLIF(r.raw_user_meta_data ->> 'first_name', '');
        v_last_name := NULLIF(r.raw_user_meta_data ->> 'last_name', '');

        v_display_name := COALESCE(
            CASE
                WHEN v_first_name IS NOT NULL OR v_last_name IS NOT NULL
                THEN btrim(COALESCE(v_first_name, '') || ' ' || COALESCE(v_last_name, ''))
                ELSE NULL
            END,
            r.raw_user_meta_data ->> 'display_name',
            r.raw_user_meta_data ->> 'full_name',
            r.raw_user_meta_data ->> 'name',
            split_part(COALESCE(r.email, ''), '@', 1),
            'User'
        );

        v_avatar_url := COALESCE(
            NULLIF(r.raw_user_meta_data ->> 'avatar_url', ''),
            NULLIF(r.raw_user_meta_data ->> 'avatar', '')
        );

        v_profile_picture_url := COALESCE(
            NULLIF(r.raw_user_meta_data ->> 'profile_picture_url', ''),
            NULLIF(r.raw_user_meta_data ->> 'picture', ''),
            v_avatar_url
        );

        v_workspace_name := COALESCE(
            r.raw_user_meta_data ->> 'workspace_name',
            NULLIF(v_display_name, '') || '''s Workspace',
            'Workspace'
        );

        -- Create default workspace for backfilled user
        INSERT INTO public.workspaces (owner_user_id, workspace_name)
        VALUES (r.id, v_workspace_name);

        INSERT INTO public.users (
            user_id,
            display_name,
            first_name,
            last_name,
            avatar_url,
            profile_picture_url
        )
        VALUES (
            r.id,
            NULLIF(v_display_name, ''),
            v_first_name,
            v_last_name,
            v_avatar_url,
            v_profile_picture_url
        )
        ON CONFLICT (user_id) DO UPDATE
          SET display_name = COALESCE(public.users.display_name, EXCLUDED.display_name),
              first_name = COALESCE(public.users.first_name, EXCLUDED.first_name),
              last_name = COALESCE(public.users.last_name, EXCLUDED.last_name),
              avatar_url = COALESCE(public.users.avatar_url, EXCLUDED.avatar_url),
              profile_picture_url = COALESCE(public.users.profile_picture_url, EXCLUDED.profile_picture_url);
    END LOOP;
END $$;

-- Create a new workspace for the signed-in user.
CREATE OR REPLACE FUNCTION public.create_workspace(
    p_workspace_name text
)
RETURNS uuid
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $fn$
DECLARE
    v_user_id uuid := auth.uid();
    v_workspace_id uuid;
BEGIN
    IF v_user_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    INSERT INTO public.workspaces (owner_user_id, workspace_name)
    VALUES (v_user_id, p_workspace_name)
    RETURNING workspace_id INTO v_workspace_id;

    RETURN v_workspace_id;
END;
$fn$;

REVOKE ALL ON FUNCTION public.create_workspace(text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.create_workspace(text) TO authenticated;

-- Bootstrap helper: ensure at least one workspace + profile exists for current auth user.
-- Returns the first workspace_id.
CREATE OR REPLACE FUNCTION public.bootstrap_workspace(
    p_workspace_name text,
    p_display_name text DEFAULT NULL
)
RETURNS uuid
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $fn$
DECLARE
    v_user_id uuid := auth.uid();
    v_workspace_id uuid;
BEGIN
    IF v_user_id IS NULL THEN
        RAISE EXCEPTION 'Not authenticated';
    END IF;

    -- Return existing workspace if user already has one
    SELECT workspace_id
      INTO v_workspace_id
      FROM public.workspaces
     WHERE owner_user_id = v_user_id
     ORDER BY created_at ASC
     LIMIT 1;

    IF v_workspace_id IS NOT NULL THEN
        RETURN v_workspace_id;
    END IF;

    -- Create first workspace
    INSERT INTO public.workspaces (owner_user_id, workspace_name)
    VALUES (v_user_id, p_workspace_name)
    RETURNING workspace_id INTO v_workspace_id;

    -- Create user profile
    INSERT INTO public.users (user_id, display_name)
    VALUES (v_user_id, p_display_name)
    ON CONFLICT (user_id) DO UPDATE
      SET display_name = COALESCE(public.users.display_name, EXCLUDED.display_name);

    RETURN v_workspace_id;
END;
$fn$;

REVOKE ALL ON FUNCTION public.bootstrap_workspace(text, text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.bootstrap_workspace(text, text) TO authenticated;

-- ---------------------------------------------------------------------
-- Row Level Security
-- ---------------------------------------------------------------------
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.workspaces ENABLE ROW LEVEL SECURITY;

-- users: each authenticated user can only see/manage their own row.
DROP POLICY IF EXISTS users_select_own ON public.users;
CREATE POLICY users_select_own
ON public.users
FOR SELECT
TO authenticated
USING ((SELECT auth.uid()) IS NOT NULL AND user_id = (SELECT auth.uid()));

DROP POLICY IF EXISTS users_insert_own ON public.users;
CREATE POLICY users_insert_own
ON public.users
FOR INSERT
TO authenticated
WITH CHECK (
    (SELECT auth.uid()) IS NOT NULL
    AND user_id = (SELECT auth.uid())
);

DROP POLICY IF EXISTS users_update_own ON public.users;
CREATE POLICY users_update_own
ON public.users
FOR UPDATE
TO authenticated
USING ((SELECT auth.uid()) IS NOT NULL AND user_id = (SELECT auth.uid()))
WITH CHECK ((SELECT auth.uid()) IS NOT NULL AND user_id = (SELECT auth.uid()));

DROP POLICY IF EXISTS users_delete_own ON public.users;
CREATE POLICY users_delete_own
ON public.users
FOR DELETE
TO authenticated
USING ((SELECT auth.uid()) IS NOT NULL AND user_id = (SELECT auth.uid()));

-- workspaces: owner can read and manage only their own workspaces.
DROP POLICY IF EXISTS workspaces_select_own ON public.workspaces;
CREATE POLICY workspaces_select_own
ON public.workspaces
FOR SELECT
TO authenticated
USING ((SELECT auth.uid()) IS NOT NULL AND owner_user_id = (SELECT auth.uid()));

DROP POLICY IF EXISTS workspaces_insert_own ON public.workspaces;
CREATE POLICY workspaces_insert_own
ON public.workspaces
FOR INSERT
TO authenticated
WITH CHECK ((SELECT auth.uid()) IS NOT NULL AND owner_user_id = (SELECT auth.uid()));

DROP POLICY IF EXISTS workspaces_update_own ON public.workspaces;
CREATE POLICY workspaces_update_own
ON public.workspaces
FOR UPDATE
TO authenticated
USING ((SELECT auth.uid()) IS NOT NULL AND owner_user_id = (SELECT auth.uid()))
WITH CHECK ((SELECT auth.uid()) IS NOT NULL AND owner_user_id = (SELECT auth.uid()));

DROP POLICY IF EXISTS workspaces_delete_own ON public.workspaces;
CREATE POLICY workspaces_delete_own
ON public.workspaces
FOR DELETE
TO authenticated
USING ((SELECT auth.uid()) IS NOT NULL AND owner_user_id = (SELECT auth.uid()));

-- Explicit grants for Supabase client roles.
GRANT SELECT, INSERT, UPDATE, DELETE ON public.users TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.workspaces TO authenticated;

COMMIT;
