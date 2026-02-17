-- db/bootstrap.sql (idempotent for feature_2 production ranking + maps)
\set ON_ERROR_STOP on
BEGIN;

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

-- ---------------------------------------------------------------------
-- Extensions (safe)
-- ---------------------------------------------------------------------

-- pgcrypto is optional; used for gen_random_uuid() if you want it.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- pgvector is required for embeddings. If it's not installed in the Postgres
-- image, this will fail with a clear message.
DO $$
BEGIN
  BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
  EXCEPTION
    WHEN undefined_file THEN
      RAISE EXCEPTION
        'pgvector extension is not installed in this Postgres instance. Use a pgvector-enabled image (e.g. pgvector/pgvector:pg16) or install the extension on the server.';
  END;
END $$;

-- ---------------------------------------------------------------------
-- Candidate sets and items (v1 + prod)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.candidate_sets (
    candidate_set_id uuid PRIMARY KEY,
    tenant_id uuid NOT NULL,
    workspace_id uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    seed_type text,
    seed_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    params_hash text NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_candidate_sets_tenant_params_hash
    ON public.candidate_sets (tenant_id, params_hash);

CREATE INDEX IF NOT EXISTS idx_candidate_sets_tenant
    ON public.candidate_sets (tenant_id);

CREATE INDEX IF NOT EXISTS idx_candidate_sets_workspace
    ON public.candidate_sets (workspace_id);

CREATE INDEX IF NOT EXISTS idx_candidate_sets_tenant_seed_type_created
    ON public.candidate_sets (tenant_id, seed_type, created_at DESC);

CREATE TABLE IF NOT EXISTS public.candidate_set_items (
    candidate_set_id uuid NOT NULL,
    work_id text NOT NULL,
    provenance_json jsonb NOT NULL DEFAULT '[]'::jsonb,
    PRIMARY KEY (candidate_set_id, work_id)
);

CREATE INDEX IF NOT EXISTS idx_candidate_set_items_candidate_set
    ON public.candidate_set_items (candidate_set_id);

CREATE INDEX IF NOT EXISTS idx_candidate_set_items_work
    ON public.candidate_set_items (work_id);

DO $$ BEGIN
  ALTER TABLE public.candidate_set_items
    ADD CONSTRAINT candidate_set_items_candidate_set_id_fkey
    FOREIGN KEY (candidate_set_id) REFERENCES public.candidate_sets(candidate_set_id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ---------------------------------------------------------------------
-- Graph drafts (maps pipeline)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.graph_drafts (
    graph_draft_id uuid PRIMARY KEY,
    tenant_id uuid NOT NULL,
    workspace_id uuid,
    candidate_set_id uuid,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_graph_drafts_workspace_created
    ON public.graph_drafts (workspace_id, created_at DESC);

CREATE TABLE IF NOT EXISTS public.graph_draft_nodes (
    graph_draft_id uuid NOT NULL,
    work_id text NOT NULL,
    PRIMARY KEY (graph_draft_id, work_id)
);

CREATE TABLE IF NOT EXISTS public.graph_draft_edges (
    graph_draft_id uuid NOT NULL,
    from_work_id text NOT NULL,
    to_work_id text NOT NULL,
    PRIMARY KEY (graph_draft_id, from_work_id, to_work_id)
);

DO $$ BEGIN
  ALTER TABLE public.graph_drafts
    ADD CONSTRAINT graph_drafts_candidate_set_id_fkey
    FOREIGN KEY (candidate_set_id) REFERENCES public.candidate_sets(candidate_set_id) ON DELETE SET NULL;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.graph_draft_nodes
    ADD CONSTRAINT graph_draft_nodes_graph_draft_id_fkey
    FOREIGN KEY (graph_draft_id) REFERENCES public.graph_drafts(graph_draft_id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.graph_draft_edges
    ADD CONSTRAINT graph_draft_edges_graph_draft_id_fkey
    FOREIGN KEY (graph_draft_id) REFERENCES public.graph_drafts(graph_draft_id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ---------------------------------------------------------------------
-- Maps + nodes + edges
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.maps (
    map_id uuid PRIMARY KEY,
    tenant_id uuid NOT NULL,
    workspace_id uuid,
    graph_draft_id uuid NOT NULL,
    default_grouping text NOT NULL CHECK (default_grouping = ANY (ARRAY['topic','subfield'])),
    allowed_groupings jsonb NOT NULL,
    stats_json jsonb NOT NULL,
    params_json jsonb NOT NULL,
    params_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    version integer
);

CREATE UNIQUE INDEX IF NOT EXISTS maps_tenant_params_hash_uniq
    ON public.maps (tenant_id, params_hash);

CREATE INDEX IF NOT EXISTS maps_tenant_created_desc_idx
    ON public.maps (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS maps_workspace_created_desc_idx
    ON public.maps (workspace_id, created_at DESC);

CREATE INDEX IF NOT EXISTS maps_tenant_graph_draft_idx
    ON public.maps (tenant_id, graph_draft_id);

CREATE TABLE IF NOT EXISTS public.map_nodes (
    map_id uuid NOT NULL,
    work_id text NOT NULL,
    best_topic_id text,
    best_subfield_id text,
    topic_score numeric,
    connector_score numeric,
    x numeric,
    y numeric,
    work_preview_json jsonb,
    PRIMARY KEY (map_id, work_id)
);

CREATE INDEX IF NOT EXISTS map_nodes_map_connector_desc_idx
    ON public.map_nodes (map_id, connector_score DESC);
CREATE INDEX IF NOT EXISTS map_nodes_map_id_idx
    ON public.map_nodes (map_id);
CREATE INDEX IF NOT EXISTS map_nodes_map_subfield_idx
    ON public.map_nodes (map_id, best_subfield_id);
CREATE INDEX IF NOT EXISTS map_nodes_map_topic_idx
    ON public.map_nodes (map_id, best_topic_id);

CREATE TABLE IF NOT EXISTS public.map_edges (
    map_id uuid NOT NULL,
    from_work_id text NOT NULL,
    to_work_id text NOT NULL,
    PRIMARY KEY (map_id, from_work_id, to_work_id)
);

CREATE INDEX IF NOT EXISTS map_edges_map_from_idx
    ON public.map_edges (map_id, from_work_id);
CREATE INDEX IF NOT EXISTS map_edges_map_to_idx
    ON public.map_edges (map_id, to_work_id);

DO $$ BEGIN
  ALTER TABLE public.maps
    ADD CONSTRAINT maps_graph_draft_id_fkey
    FOREIGN KEY (graph_draft_id) REFERENCES public.graph_drafts(graph_draft_id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.map_nodes
    ADD CONSTRAINT map_nodes_map_id_fkey
    FOREIGN KEY (map_id) REFERENCES public.maps(map_id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.map_edges
    ADD CONSTRAINT map_edges_map_id_fkey
    FOREIGN KEY (map_id) REFERENCES public.maps(map_id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Updated_at trigger (idempotent)
CREATE OR REPLACE FUNCTION public.set_maps_updated_at()
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
    SELECT 1
    FROM pg_trigger
    WHERE tgname = 'trg_maps_set_updated_at'
  ) THEN
    CREATE TRIGGER trg_maps_set_updated_at
      BEFORE UPDATE ON public.maps
      FOR EACH ROW
      EXECUTE FUNCTION public.set_maps_updated_at();
  END IF;
END $$;

-- ---------------------------------------------------------------------
-- Topics hierarchy
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.openalex_topics (
    topic_id text PRIMARY KEY,
    subfield_id text NOT NULL,
    field_id text NOT NULL
);

-- Add display_name column for topic lookup (for "Research this topic" feature)
ALTER TABLE public.openalex_topics
  ADD COLUMN IF NOT EXISTS display_name text;

CREATE INDEX IF NOT EXISTS idx_openalex_topics_subfield
    ON public.openalex_topics (subfield_id);
CREATE INDEX IF NOT EXISTS idx_openalex_topics_field
    ON public.openalex_topics (field_id);

-- ---------------------------------------------------------------------
-- Ranking jobs + results
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.rank_jobs (
    rank_job_id uuid PRIMARY KEY,
    tenant_id uuid NOT NULL,
    workspace_id uuid,
    rank_type text NOT NULL,
    candidate_set_id uuid NOT NULL,
    context_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    filters_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    rank_params_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    params_hash text NOT NULL,
    status text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz,
    error_json jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_rank_jobs_tenant_rank_type_params_hash_uq
    ON public.rank_jobs (tenant_id, rank_type, params_hash);

CREATE INDEX IF NOT EXISTS idx_rank_jobs_tenant_candidate_set
    ON public.rank_jobs (tenant_id, candidate_set_id);

CREATE INDEX IF NOT EXISTS idx_rank_jobs_workspace_status_created
    ON public.rank_jobs (workspace_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_rank_jobs_tenant_status_created
    ON public.rank_jobs (tenant_id, status, created_at DESC);

DO $$ BEGIN
  ALTER TABLE public.rank_jobs
    ADD CONSTRAINT rank_jobs_status_check
    CHECK (status = ANY (ARRAY['pending','running','completed','failed']));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.rank_jobs
    ADD CONSTRAINT rank_jobs_candidate_set_id_fkey
    FOREIGN KEY (candidate_set_id) REFERENCES public.candidate_sets(candidate_set_id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS public.rank_results (
    rank_job_id uuid NOT NULL,
    rank_index integer NOT NULL,
    work_id text NOT NULL,
    score double precision NOT NULL,
    score_breakdown_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    reasons_json jsonb NOT NULL DEFAULT '[]'::jsonb,
    work_preview_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    provenance_json jsonb NOT NULL DEFAULT '[]'::jsonb,
    PRIMARY KEY (rank_job_id, rank_index)
);

CREATE INDEX IF NOT EXISTS idx_rank_results_rank_job
    ON public.rank_results (rank_job_id);

CREATE INDEX IF NOT EXISTS idx_rank_results_rank_job_work
    ON public.rank_results (rank_job_id, work_id);

CREATE INDEX IF NOT EXISTS idx_rank_results_rank_job_score_desc
    ON public.rank_results (rank_job_id, score DESC);

DO $$ BEGIN
  ALTER TABLE public.rank_results
    ADD CONSTRAINT rank_results_rank_job_id_fkey
    FOREIGN KEY (rank_job_id) REFERENCES public.rank_jobs(rank_job_id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ---------------------------------------------------------------------
-- Works + hybrid retrieval fields
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.works (
    work_id text PRIMARY KEY,
    title text,
    year integer,
    cited_by_count integer,
    authors_json jsonb,
    venue text,
    primary_topic_id text,
    primary_topic_score numeric,
    topics_json jsonb,
    is_retracted boolean DEFAULT false,
    abstract text
);

-- Add/ensure new columns on existing works table
ALTER TABLE public.works
  ADD COLUMN IF NOT EXISTS abstract text;

-- Paper category for ranking (foundational, methodological, applied, implementation, handbook)
ALTER TABLE public.works
  ADD COLUMN IF NOT EXISTS category text,
  ADD COLUMN IF NOT EXISTS category_confidence double precision;

-- Referenced works for Feature 3 novelty assessment
ALTER TABLE public.works
  ADD COLUMN IF NOT EXISTS referenced_works_json jsonb;

-- External IDs for cross-source matching
ALTER TABLE public.works
  ADD COLUMN IF NOT EXISTS doi text,
  ADD COLUMN IF NOT EXISTS arxiv_id text;

-- Track abstract provenance for Feature 3 enrichment
ALTER TABLE public.works
  ADD COLUMN IF NOT EXISTS abstract_source text,
  ADD COLUMN IF NOT EXISTS abstract_validated_at timestamptz;

-- Track topic provenance for Feature 3 inference
ALTER TABLE public.works
  ADD COLUMN IF NOT EXISTS topic_source text,
  ADD COLUMN IF NOT EXISTS topic_inferred_at timestamptz;

-- Cached landmark papers for Feature 3 grounding supplement (global cache)
-- Used by novelty assessment and temporal timeline features
ALTER TABLE public.works
  ADD COLUMN IF NOT EXISTS landmark_works_json jsonb;

-- Cached citing papers for per-node timeline (Feature 3)
ALTER TABLE public.works
  ADD COLUMN IF NOT EXISTS citing_works_json jsonb;

-- Open access fields (populated from OpenAlex)
ALTER TABLE public.works
  ADD COLUMN IF NOT EXISTS is_open_access boolean,
  ADD COLUMN IF NOT EXISTS oa_status text,      -- gold, green, hybrid, bronze, closed
  ADD COLUMN IF NOT EXISTS oa_pdf_url text;

-- Indexes for external ID lookups
CREATE INDEX IF NOT EXISTS idx_works_doi ON public.works (doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_works_arxiv_id ON public.works (arxiv_id) WHERE arxiv_id IS NOT NULL;

-- search_tsv: try to add generated column; if an older schema already has a
-- non-generated search_tsv, we won't replace it (avoids breaking).
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema='public'
      AND table_name='works'
      AND column_name='search_tsv'
  ) THEN
    EXECUTE $sql$
      ALTER TABLE public.works
      ADD COLUMN search_tsv tsvector
      GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title,'')), 'A') ||
        setweight(to_tsvector('english', coalesce(abstract,'')), 'B')
      ) STORED
    $sql$;
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_works_search_tsv
    ON public.works USING gin (search_tsv);

CREATE INDEX IF NOT EXISTS idx_works_year
    ON public.works (year);

-- ---------------------------------------------------------------------
-- Work embeddings (semantic retrieval)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.work_embeddings (
    work_id text PRIMARY KEY REFERENCES public.works (work_id) ON DELETE CASCADE,
    embedding vector(1536) NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- ivfflat index requires ANALYZE and suitable settings; safe to create anyway.
CREATE INDEX IF NOT EXISTS idx_work_embeddings_ivfflat
  ON public.work_embeddings USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- ---------------------------------------------------------------------
-- Query expansion cache (LLM-generated expansion terms)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.query_expansion_cache (
    query_hash text PRIMARY KEY,
    query_text text NOT NULL,
    expansion_terms jsonb NOT NULL,
    model_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_expansion_cache_created
    ON public.query_expansion_cache (created_at DESC);

-- ---------------------------------------------------------------------
-- LLM relevance score cache (per paper per query)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.llm_relevance_cache (
    paper_id text NOT NULL,
    query_hash text NOT NULL,
    relevance_score double precision NOT NULL,
    paper_type text DEFAULT 'other',
    model_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (paper_id, query_hash)
);

CREATE INDEX IF NOT EXISTS idx_llm_relevance_cache_query
    ON public.llm_relevance_cache (query_hash);

CREATE INDEX IF NOT EXISTS idx_llm_relevance_cache_created
    ON public.llm_relevance_cache (created_at DESC);

-- ---------------------------------------------------------------------
-- Query classification cache (LLM-based query type detection)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.query_classification_cache (
    query_hash text PRIMARY KEY,
    query_text text NOT NULL,
    query_type text NOT NULL,
    confidence double precision NOT NULL,
    model_version text NOT NULL,
    is_emerging_field boolean DEFAULT false,
    field_emergence_year integer,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_classification_cache_created
    ON public.query_classification_cache (created_at DESC);

-- Migration: Add emerging field columns if they don't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'query_classification_cache' AND column_name = 'is_emerging_field') THEN
        ALTER TABLE public.query_classification_cache ADD COLUMN is_emerging_field boolean DEFAULT false;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'query_classification_cache' AND column_name = 'field_emergence_year') THEN
        ALTER TABLE public.query_classification_cache ADD COLUMN field_emergence_year integer;
    END IF;
END$$;

-- Migration: Add query specificity columns for temporal map flow
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'query_classification_cache' AND column_name = 'query_specificity') THEN
        ALTER TABLE public.query_classification_cache ADD COLUMN query_specificity text DEFAULT 'broad';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'query_classification_cache' AND column_name = 'detected_topic_id') THEN
        ALTER TABLE public.query_classification_cache ADD COLUMN detected_topic_id text;
    END IF;
END$$;

-- ---------------------------------------------------------------------
-- Paper categories (populated by external categorization model)
-- This ranking feature reads from it but doesn't write to it
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.paper_categories (
    paper_id text NOT NULL,
    query_hash text NOT NULL,
    category text NOT NULL,
    confidence double precision NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (paper_id, query_hash)
);

CREATE INDEX IF NOT EXISTS idx_paper_categories_query
    ON public.paper_categories (query_hash);

CREATE INDEX IF NOT EXISTS idx_paper_categories_created
    ON public.paper_categories (created_at DESC);

-- ---------------------------------------------------------------------
-- Node details cache (Feature 3 - LLM-generated paper summaries)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.node_details_cache (
    work_id text PRIMARY KEY,
    summary text,
    keywords jsonb,
    novelty_assessment jsonb,
    model_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_node_details_cache_created
    ON public.node_details_cache (created_at DESC);

-- Migration: Add assessment_unavailable_reason for honest handling of missing data
ALTER TABLE public.node_details_cache
    ADD COLUMN IF NOT EXISTS assessment_unavailable_reason text;

-- ==========================================================================
-- Feature 4: Methodology Comparison
-- ==========================================================================

-- Methodology fingerprint cache (per paper, reusable across comparisons)
CREATE TABLE IF NOT EXISTS public.methodology_fingerprint_cache (
    work_id text PRIMARY KEY,
    fingerprint_json jsonb NOT NULL,
    source_quality text NOT NULL,
    model_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_methodology_fingerprint_created
    ON public.methodology_fingerprint_cache (created_at DESC);

-- Comparison result cache (per set of papers)
CREATE TABLE IF NOT EXISTS public.methodology_comparison_cache (
    comparison_hash text PRIMARY KEY,
    tenant_id uuid NOT NULL,
    work_ids text[] NOT NULL,
    result_json jsonb NOT NULL,
    model_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_methodology_comparison_created
    ON public.methodology_comparison_cache (created_at DESC);

-- Paper full text cache (from Semantic Scholar)
CREATE TABLE IF NOT EXISTS public.paper_full_text_cache (
    work_id text PRIMARY KEY,
    s2_paper_id text,
    methods_text text,
    full_text_available boolean NOT NULL DEFAULT false,
    fetched_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_paper_full_text_fetched
    ON public.paper_full_text_cache (fetched_at DESC);

-- ==========================================================================
-- Tenant settings (institutional access configuration)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS public.tenant_settings (
    tenant_id uuid PRIMARY KEY,
    institutional_proxy_prefix text,  -- e.g. 'https://proxy.university.edu/login?url='
    libkey_api_key text,              -- Optional Third Iron LibKey API key
    libkey_library_id text,           -- Optional LibKey library ID
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- ==========================================================================
-- Feature 5: Research Gap Analysis
-- ==========================================================================

-- Feature usage tracking (for coverage calculation)
CREATE TABLE IF NOT EXISTS public.gap_feature_usage (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    map_id uuid NOT NULL REFERENCES public.maps(map_id) ON DELETE CASCADE,
    feature_type text NOT NULL,
    work_id text,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_gap_feature_usage_unique
    ON public.gap_feature_usage (map_id, feature_type, COALESCE(work_id, ''));

CREATE INDEX IF NOT EXISTS idx_gap_feature_usage_map
    ON public.gap_feature_usage (map_id);

CREATE INDEX IF NOT EXISTS idx_gap_feature_usage_map_type
    ON public.gap_feature_usage (map_id, feature_type);

-- Gap analysis results
CREATE TABLE IF NOT EXISTS public.gap_analysis_results (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    map_id uuid NOT NULL REFERENCES public.maps(map_id) ON DELETE CASCADE,
    tenant_id uuid NOT NULL,
    workspace_id uuid,
    gaps jsonb NOT NULL DEFAULT '[]'::jsonb,
    data_sources_used text[] NOT NULL DEFAULT '{}',
    coverage_pct double precision NOT NULL,
    total_candidates_detected integer NOT NULL DEFAULT 0,
    candidates_validated integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_gap_analysis_results_map
    ON public.gap_analysis_results (map_id);

CREATE INDEX IF NOT EXISTS idx_gap_analysis_results_tenant
    ON public.gap_analysis_results (tenant_id);

CREATE INDEX IF NOT EXISTS idx_gap_analysis_results_workspace_created
    ON public.gap_analysis_results (workspace_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_gap_analysis_results_created
    ON public.gap_analysis_results (created_at DESC);

-- Gap analysis jobs (async processing)
CREATE TABLE IF NOT EXISTS public.gap_analysis_jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    map_id uuid NOT NULL REFERENCES public.maps(map_id) ON DELETE CASCADE,
    tenant_id uuid NOT NULL,
    workspace_id uuid,
    status text NOT NULL DEFAULT 'pending',
    progress double precision NOT NULL DEFAULT 0,
    error_message text,
    result_id uuid REFERENCES public.gap_analysis_results(id) ON DELETE SET NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_gap_analysis_jobs_map
    ON public.gap_analysis_jobs (map_id);

CREATE INDEX IF NOT EXISTS idx_gap_analysis_jobs_tenant
    ON public.gap_analysis_jobs (tenant_id);

CREATE INDEX IF NOT EXISTS idx_gap_analysis_jobs_workspace_status
    ON public.gap_analysis_jobs (workspace_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_gap_analysis_jobs_status
    ON public.gap_analysis_jobs (status);

DO $$ BEGIN
  ALTER TABLE public.gap_analysis_jobs
    ADD CONSTRAINT gap_analysis_jobs_status_check
    CHECK (status = ANY (ARRAY['pending', 'detecting', 'validating', 'complete', 'failed']));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Gap candidate cache (intermediate results from detection)
CREATE TABLE IF NOT EXISTS public.gap_candidates (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id uuid NOT NULL REFERENCES public.gap_analysis_jobs(id) ON DELETE CASCADE,
    gap_type text NOT NULL,
    candidate_data jsonb NOT NULL,
    detection_score double precision NOT NULL,
    validation_status text DEFAULT 'pending',
    validation_result jsonb,
    confidence double precision,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_gap_candidates_job
    ON public.gap_candidates (job_id);

CREATE INDEX IF NOT EXISTS idx_gap_candidates_job_type
    ON public.gap_candidates (job_id, gap_type);

CREATE INDEX IF NOT EXISTS idx_gap_candidates_score
    ON public.gap_candidates (job_id, detection_score DESC);

-- ---------------------------------------------------------------------
-- Workspace metadata/index tables
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.workspace_artifacts (
    workspace_artifact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id uuid NOT NULL,
    artifact_type text NOT NULL CHECK (artifact_type = ANY (ARRAY[
        'map','rank_job','graph_draft','gap_analysis_job','gap_analysis_result'
    ])),
    artifact_id uuid NOT NULL,
    title text,
    metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    archived_at timestamptz
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_workspace_artifacts_workspace_type_artifact
    ON public.workspace_artifacts (workspace_id, artifact_type, artifact_id);

CREATE INDEX IF NOT EXISTS idx_workspace_artifacts_workspace_created
    ON public.workspace_artifacts (workspace_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_workspace_artifacts_workspace_type_created
    ON public.workspace_artifacts (workspace_id, artifact_type, created_at DESC);

CREATE TABLE IF NOT EXISTS public.workspace_collections (
    workspace_collection_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id uuid NOT NULL,
    collection_name text NOT NULL,
    description text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    archived_at timestamptz,
    CONSTRAINT workspace_collection_name_not_blank CHECK (length(btrim(collection_name)) > 0)
);

CREATE INDEX IF NOT EXISTS idx_workspace_collections_workspace_created
    ON public.workspace_collections (workspace_id, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS uq_workspace_collections_workspace_name
    ON public.workspace_collections (workspace_id, collection_name);

CREATE TABLE IF NOT EXISTS public.workspace_collection_items (
    workspace_collection_id uuid NOT NULL,
    workspace_id uuid NOT NULL,
    work_id text NOT NULL,
    source_artifact_type text,
    source_artifact_id uuid,
    note text,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_collection_id, work_id)
);

CREATE INDEX IF NOT EXISTS idx_workspace_collection_items_workspace_created
    ON public.workspace_collection_items (workspace_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_workspace_collection_items_work_id
    ON public.workspace_collection_items (work_id);

DO $$ BEGIN
  ALTER TABLE public.workspace_collection_items
    ADD CONSTRAINT workspace_collection_items_source_artifact_type_check
    CHECK (
      source_artifact_type IS NULL
      OR source_artifact_type = ANY (ARRAY[
        'map','rank_job','graph_draft','gap_analysis_job','gap_analysis_result'
      ])
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Keep workspace_collection_items.workspace_id aligned with its parent collection.
CREATE OR REPLACE FUNCTION public.workspace_collection_items_sync_workspace_id()
RETURNS trigger
LANGUAGE plpgsql
AS $fn$
DECLARE
  v_workspace_id uuid;
BEGIN
  SELECT workspace_id
    INTO v_workspace_id
    FROM public.workspace_collections
   WHERE workspace_collection_id = NEW.workspace_collection_id;

  IF v_workspace_id IS NULL THEN
    RAISE EXCEPTION 'workspace_collection_id % not found', NEW.workspace_collection_id;
  END IF;

  NEW.workspace_id := v_workspace_id;
  RETURN NEW;
END;
$fn$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_trigger
    WHERE tgname = 'trg_workspace_collection_items_sync_workspace_id'
  ) THEN
    CREATE TRIGGER trg_workspace_collection_items_sync_workspace_id
      BEFORE INSERT OR UPDATE ON public.workspace_collection_items
      FOR EACH ROW
      EXECUTE FUNCTION public.workspace_collection_items_sync_workspace_id();
  END IF;
END $$;

CREATE OR REPLACE FUNCTION public.workspace_collections_set_updated_at()
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
    SELECT 1
    FROM pg_trigger
    WHERE tgname = 'trg_workspace_collections_set_updated_at'
  ) THEN
    CREATE TRIGGER trg_workspace_collections_set_updated_at
      BEFORE UPDATE ON public.workspace_collections
      FOR EACH ROW
      EXECUTE FUNCTION public.workspace_collections_set_updated_at();
  END IF;
END $$;

-- Optional mapping table for gradual migration from tenant_id to workspace_id.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'workspaces') THEN
    EXECUTE '
      CREATE TABLE IF NOT EXISTS public.tenant_workspace_map (
        tenant_id uuid PRIMARY KEY,
        workspace_id uuid NOT NULL REFERENCES public.workspaces(workspace_id) ON DELETE CASCADE,
        created_at timestamptz NOT NULL DEFAULT now()
      )';
    EXECUTE '
      CREATE UNIQUE INDEX IF NOT EXISTS uq_tenant_workspace_map_workspace
      ON public.tenant_workspace_map (workspace_id)';
  END IF;
END $$;

DO $$ BEGIN
  ALTER TABLE public.candidate_sets
    ADD COLUMN IF NOT EXISTS workspace_id uuid;
EXCEPTION WHEN duplicate_column THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.graph_drafts
    ADD COLUMN IF NOT EXISTS workspace_id uuid;
EXCEPTION WHEN duplicate_column THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.maps
    ADD COLUMN IF NOT EXISTS workspace_id uuid;
EXCEPTION WHEN duplicate_column THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.rank_jobs
    ADD COLUMN IF NOT EXISTS workspace_id uuid;
EXCEPTION WHEN duplicate_column THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.gap_analysis_results
    ADD COLUMN IF NOT EXISTS workspace_id uuid;
EXCEPTION WHEN duplicate_column THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.gap_analysis_jobs
    ADD COLUMN IF NOT EXISTS workspace_id uuid;
EXCEPTION WHEN duplicate_column THEN NULL; END $$;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'workspaces') THEN
    BEGIN
      ALTER TABLE public.candidate_sets
        ADD CONSTRAINT candidate_sets_workspace_id_fkey
        FOREIGN KEY (workspace_id) REFERENCES public.workspaces(workspace_id) ON DELETE CASCADE;
    EXCEPTION WHEN duplicate_object THEN NULL; END;

    BEGIN
      ALTER TABLE public.graph_drafts
        ADD CONSTRAINT graph_drafts_workspace_id_fkey
        FOREIGN KEY (workspace_id) REFERENCES public.workspaces(workspace_id) ON DELETE CASCADE;
    EXCEPTION WHEN duplicate_object THEN NULL; END;

    BEGIN
      ALTER TABLE public.maps
        ADD CONSTRAINT maps_workspace_id_fkey
        FOREIGN KEY (workspace_id) REFERENCES public.workspaces(workspace_id) ON DELETE CASCADE;
    EXCEPTION WHEN duplicate_object THEN NULL; END;

    BEGIN
      ALTER TABLE public.rank_jobs
        ADD CONSTRAINT rank_jobs_workspace_id_fkey
        FOREIGN KEY (workspace_id) REFERENCES public.workspaces(workspace_id) ON DELETE CASCADE;
    EXCEPTION WHEN duplicate_object THEN NULL; END;

    BEGIN
      ALTER TABLE public.gap_analysis_results
        ADD CONSTRAINT gap_analysis_results_workspace_id_fkey
        FOREIGN KEY (workspace_id) REFERENCES public.workspaces(workspace_id) ON DELETE CASCADE;
    EXCEPTION WHEN duplicate_object THEN NULL; END;

    BEGIN
      ALTER TABLE public.gap_analysis_jobs
        ADD CONSTRAINT gap_analysis_jobs_workspace_id_fkey
        FOREIGN KEY (workspace_id) REFERENCES public.workspaces(workspace_id) ON DELETE CASCADE;
    EXCEPTION WHEN duplicate_object THEN NULL; END;

    BEGIN
      ALTER TABLE public.workspace_artifacts
        ADD CONSTRAINT workspace_artifacts_workspace_id_fkey
        FOREIGN KEY (workspace_id) REFERENCES public.workspaces(workspace_id) ON DELETE CASCADE;
    EXCEPTION WHEN duplicate_object THEN NULL; END;

    BEGIN
      ALTER TABLE public.workspace_collections
        ADD CONSTRAINT workspace_collections_workspace_id_fkey
        FOREIGN KEY (workspace_id) REFERENCES public.workspaces(workspace_id) ON DELETE CASCADE;
    EXCEPTION WHEN duplicate_object THEN NULL; END;

    BEGIN
      ALTER TABLE public.workspace_collection_items
        ADD CONSTRAINT workspace_collection_items_collection_fkey
        FOREIGN KEY (workspace_collection_id) REFERENCES public.workspace_collections(workspace_collection_id) ON DELETE CASCADE;
    EXCEPTION WHEN duplicate_object THEN NULL; END;

    BEGIN
      ALTER TABLE public.workspace_collection_items
        ADD CONSTRAINT workspace_collection_items_workspace_id_fkey
        FOREIGN KEY (workspace_id) REFERENCES public.workspaces(workspace_id) ON DELETE CASCADE;
    EXCEPTION WHEN duplicate_object THEN NULL; END;
  END IF;
END $$;

-- ---------------------------------------------------------------------
-- Naming consistency pass (idempotent)
-- Convention: <table>_<column>_fkey for foreign key constraints.
-- ---------------------------------------------------------------------
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_constraint c
    JOIN pg_class t ON t.oid = c.conrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    WHERE n.nspname = 'public'
      AND t.relname = 'workspace_collection_items'
      AND c.conname = 'workspace_collection_items_collection_fkey'
  )
  AND NOT EXISTS (
    SELECT 1 FROM pg_constraint c
    JOIN pg_class t ON t.oid = c.conrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    WHERE n.nspname = 'public'
      AND t.relname = 'workspace_collection_items'
      AND c.conname = 'workspace_collection_items_workspace_collection_id_fkey'
  ) THEN
    ALTER TABLE public.workspace_collection_items
      RENAME CONSTRAINT workspace_collection_items_collection_fkey
      TO workspace_collection_items_workspace_collection_id_fkey;
  END IF;
END $$;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'methodology_comparison_cache'
      AND column_name = 'tenant_id'
      AND data_type <> 'uuid'
  ) THEN
    IF NOT EXISTS (
      SELECT 1
      FROM public.methodology_comparison_cache
      WHERE tenant_id IS NOT NULL
        AND tenant_id !~* '^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    ) THEN
      ALTER TABLE public.methodology_comparison_cache
        ALTER COLUMN tenant_id TYPE uuid
        USING tenant_id::uuid;
    END IF;
  END IF;
END $$;

-- ---------------------------------------------------------------------
-- Workspace backfill helpers
-- ---------------------------------------------------------------------
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'tenant_workspace_map') THEN
    UPDATE public.candidate_sets cs
       SET workspace_id = twm.workspace_id
      FROM public.tenant_workspace_map twm
     WHERE cs.workspace_id IS NULL
       AND cs.tenant_id = twm.tenant_id;

    UPDATE public.graph_drafts gd
       SET workspace_id = twm.workspace_id
      FROM public.tenant_workspace_map twm
     WHERE gd.workspace_id IS NULL
       AND gd.tenant_id = twm.tenant_id;

    UPDATE public.maps m
       SET workspace_id = twm.workspace_id
      FROM public.tenant_workspace_map twm
     WHERE m.workspace_id IS NULL
       AND m.tenant_id = twm.tenant_id;

    UPDATE public.rank_jobs rj
       SET workspace_id = twm.workspace_id
      FROM public.tenant_workspace_map twm
     WHERE rj.workspace_id IS NULL
       AND rj.tenant_id = twm.tenant_id;

    UPDATE public.gap_analysis_results gr
       SET workspace_id = twm.workspace_id
      FROM public.tenant_workspace_map twm
     WHERE gr.workspace_id IS NULL
       AND gr.tenant_id = twm.tenant_id;

    UPDATE public.gap_analysis_jobs gj
       SET workspace_id = twm.workspace_id
      FROM public.tenant_workspace_map twm
     WHERE gj.workspace_id IS NULL
       AND gj.tenant_id = twm.tenant_id;
  END IF;

  -- Relationship-derived fallback for rows that still have NULL workspace_id.
  UPDATE public.maps m
     SET workspace_id = gd.workspace_id
    FROM public.graph_drafts gd
   WHERE m.workspace_id IS NULL
     AND m.graph_draft_id = gd.graph_draft_id
     AND gd.workspace_id IS NOT NULL;

  UPDATE public.graph_drafts gd
     SET workspace_id = m.workspace_id
    FROM public.maps m
   WHERE gd.workspace_id IS NULL
     AND m.graph_draft_id = gd.graph_draft_id
     AND m.workspace_id IS NOT NULL;

  UPDATE public.rank_jobs rj
     SET workspace_id = cs.workspace_id
    FROM public.candidate_sets cs
   WHERE rj.workspace_id IS NULL
     AND rj.candidate_set_id = cs.candidate_set_id
     AND cs.workspace_id IS NOT NULL;

  UPDATE public.candidate_sets cs
     SET workspace_id = rj.workspace_id
    FROM public.rank_jobs rj
   WHERE cs.workspace_id IS NULL
     AND rj.candidate_set_id = cs.candidate_set_id
     AND rj.workspace_id IS NOT NULL;

  UPDATE public.gap_analysis_results gr
     SET workspace_id = m.workspace_id
    FROM public.maps m
   WHERE gr.workspace_id IS NULL
     AND gr.map_id = m.map_id
     AND m.workspace_id IS NOT NULL;

  UPDATE public.gap_analysis_jobs gj
     SET workspace_id = m.workspace_id
    FROM public.maps m
   WHERE gj.workspace_id IS NULL
     AND gj.map_id = m.map_id
     AND m.workspace_id IS NOT NULL;
END $$;

-- ---------------------------------------------------------------------
-- Workspace artifact auto-sync
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.workspace_artifacts_upsert(
  p_workspace_id uuid,
  p_artifact_type text,
  p_artifact_id uuid,
  p_title text DEFAULT NULL,
  p_metadata_json jsonb DEFAULT '{}'::jsonb
)
RETURNS void
LANGUAGE sql
AS $fn$
  INSERT INTO public.workspace_artifacts (
    workspace_id,
    artifact_type,
    artifact_id,
    title,
    metadata_json
  )
  VALUES (
    p_workspace_id,
    p_artifact_type,
    p_artifact_id,
    p_title,
    COALESCE(p_metadata_json, '{}'::jsonb)
  )
  ON CONFLICT (workspace_id, artifact_type, artifact_id)
  DO UPDATE
    SET title = COALESCE(EXCLUDED.title, public.workspace_artifacts.title),
        metadata_json = COALESCE(EXCLUDED.metadata_json, public.workspace_artifacts.metadata_json),
        archived_at = NULL;
$fn$;

CREATE OR REPLACE FUNCTION public.workspace_artifacts_sync()
RETURNS trigger
LANGUAGE plpgsql
AS $fn$
BEGIN
  IF TG_TABLE_NAME = 'maps' THEN
    PERFORM public.workspace_artifacts_upsert(
      NEW.workspace_id, 'map', NEW.map_id, 'Citation map', jsonb_build_object('graph_draft_id', NEW.graph_draft_id)
    );
  ELSIF TG_TABLE_NAME = 'rank_jobs' THEN
    PERFORM public.workspace_artifacts_upsert(
      NEW.workspace_id, 'rank_job', NEW.rank_job_id, 'Rank run', jsonb_build_object('status', NEW.status)
    );
  ELSIF TG_TABLE_NAME = 'graph_drafts' THEN
    PERFORM public.workspace_artifacts_upsert(
      NEW.workspace_id, 'graph_draft', NEW.graph_draft_id, 'Graph draft', '{}'::jsonb
    );
  ELSIF TG_TABLE_NAME = 'gap_analysis_jobs' THEN
    PERFORM public.workspace_artifacts_upsert(
      NEW.workspace_id, 'gap_analysis_job', NEW.id, 'Gap analysis job', jsonb_build_object('status', NEW.status)
    );
  ELSIF TG_TABLE_NAME = 'gap_analysis_results' THEN
    PERFORM public.workspace_artifacts_upsert(
      NEW.workspace_id, 'gap_analysis_result', NEW.id, 'Gap analysis result', jsonb_build_object('map_id', NEW.map_id)
    );
  END IF;

  RETURN NEW;
END;
$fn$;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_workspace_artifacts_sync_maps') THEN
    CREATE TRIGGER trg_workspace_artifacts_sync_maps
      AFTER INSERT OR UPDATE OF workspace_id, graph_draft_id ON public.maps
      FOR EACH ROW
      WHEN (NEW.workspace_id IS NOT NULL)
      EXECUTE FUNCTION public.workspace_artifacts_sync();
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_workspace_artifacts_sync_rank_jobs') THEN
    CREATE TRIGGER trg_workspace_artifacts_sync_rank_jobs
      AFTER INSERT OR UPDATE OF workspace_id, status ON public.rank_jobs
      FOR EACH ROW
      WHEN (NEW.workspace_id IS NOT NULL)
      EXECUTE FUNCTION public.workspace_artifacts_sync();
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_workspace_artifacts_sync_graph_drafts') THEN
    CREATE TRIGGER trg_workspace_artifacts_sync_graph_drafts
      AFTER INSERT OR UPDATE OF workspace_id ON public.graph_drafts
      FOR EACH ROW
      WHEN (NEW.workspace_id IS NOT NULL)
      EXECUTE FUNCTION public.workspace_artifacts_sync();
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_workspace_artifacts_sync_gap_jobs') THEN
    CREATE TRIGGER trg_workspace_artifacts_sync_gap_jobs
      AFTER INSERT OR UPDATE OF workspace_id, status ON public.gap_analysis_jobs
      FOR EACH ROW
      WHEN (NEW.workspace_id IS NOT NULL)
      EXECUTE FUNCTION public.workspace_artifacts_sync();
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_workspace_artifacts_sync_gap_results') THEN
    CREATE TRIGGER trg_workspace_artifacts_sync_gap_results
      AFTER INSERT OR UPDATE OF workspace_id, map_id ON public.gap_analysis_results
      FOR EACH ROW
      WHEN (NEW.workspace_id IS NOT NULL)
      EXECUTE FUNCTION public.workspace_artifacts_sync();
  END IF;
END $$;

-- Initial backfill into workspace_artifacts for existing rows.
INSERT INTO public.workspace_artifacts (workspace_id, artifact_type, artifact_id, title, metadata_json)
SELECT m.workspace_id, 'map', m.map_id, 'Citation map', jsonb_build_object('graph_draft_id', m.graph_draft_id)
  FROM public.maps m
 WHERE m.workspace_id IS NOT NULL
ON CONFLICT (workspace_id, artifact_type, artifact_id) DO NOTHING;

INSERT INTO public.workspace_artifacts (workspace_id, artifact_type, artifact_id, title, metadata_json)
SELECT rj.workspace_id, 'rank_job', rj.rank_job_id, 'Rank run', jsonb_build_object('status', rj.status)
  FROM public.rank_jobs rj
 WHERE rj.workspace_id IS NOT NULL
ON CONFLICT (workspace_id, artifact_type, artifact_id) DO NOTHING;

INSERT INTO public.workspace_artifacts (workspace_id, artifact_type, artifact_id, title, metadata_json)
SELECT gd.workspace_id, 'graph_draft', gd.graph_draft_id, 'Graph draft', '{}'::jsonb
  FROM public.graph_drafts gd
 WHERE gd.workspace_id IS NOT NULL
ON CONFLICT (workspace_id, artifact_type, artifact_id) DO NOTHING;

INSERT INTO public.workspace_artifacts (workspace_id, artifact_type, artifact_id, title, metadata_json)
SELECT gj.workspace_id, 'gap_analysis_job', gj.id, 'Gap analysis job', jsonb_build_object('status', gj.status)
  FROM public.gap_analysis_jobs gj
 WHERE gj.workspace_id IS NOT NULL
ON CONFLICT (workspace_id, artifact_type, artifact_id) DO NOTHING;

INSERT INTO public.workspace_artifacts (workspace_id, artifact_type, artifact_id, title, metadata_json)
SELECT gr.workspace_id, 'gap_analysis_result', gr.id, 'Gap analysis result', jsonb_build_object('map_id', gr.map_id)
  FROM public.gap_analysis_results gr
 WHERE gr.workspace_id IS NOT NULL
ON CONFLICT (workspace_id, artifact_type, artifact_id) DO NOTHING;

-- ---------------------------------------------------------------------
-- Row Level Security for workspace-scoped artifacts
-- Requires user-bootstrap.sql tables (public.users/public.workspaces).
-- ---------------------------------------------------------------------
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'users')
     AND EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'workspaces') THEN

    ALTER TABLE public.candidate_sets ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.candidate_set_items ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.graph_drafts ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.graph_draft_nodes ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.graph_draft_edges ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.maps ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.map_nodes ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.map_edges ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.rank_jobs ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.rank_results ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.gap_analysis_results ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.gap_analysis_jobs ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.gap_candidates ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.workspace_artifacts ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.workspace_collections ENABLE ROW LEVEL SECURITY;
    ALTER TABLE public.workspace_collection_items ENABLE ROW LEVEL SECURITY;

    -- candidate_sets
    DROP POLICY IF EXISTS candidate_sets_select_workspace ON public.candidate_sets;
    CREATE POLICY candidate_sets_select_workspace
    ON public.candidate_sets
    FOR SELECT
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS candidate_sets_insert_workspace ON public.candidate_sets;
    CREATE POLICY candidate_sets_insert_workspace
    ON public.candidate_sets
    FOR INSERT
    TO authenticated
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS candidate_sets_update_workspace ON public.candidate_sets;
    CREATE POLICY candidate_sets_update_workspace
    ON public.candidate_sets
    FOR UPDATE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    )
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS candidate_sets_delete_workspace ON public.candidate_sets;
    CREATE POLICY candidate_sets_delete_workspace
    ON public.candidate_sets
    FOR DELETE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );

    -- candidate_set_items (via parent candidate_set)
    DROP POLICY IF EXISTS candidate_set_items_select_workspace ON public.candidate_set_items;
    CREATE POLICY candidate_set_items_select_workspace
    ON public.candidate_set_items
    FOR SELECT
    TO authenticated
    USING (
      EXISTS (
        SELECT 1
        FROM public.candidate_sets cs
        JOIN public.users u ON u.workspace_id = cs.workspace_id
        WHERE cs.candidate_set_id = candidate_set_items.candidate_set_id
          AND u.user_id = auth.uid()
      )
    );
    DROP POLICY IF EXISTS candidate_set_items_insert_workspace ON public.candidate_set_items;
    CREATE POLICY candidate_set_items_insert_workspace
    ON public.candidate_set_items
    FOR INSERT
    TO authenticated
    WITH CHECK (
      EXISTS (
        SELECT 1
        FROM public.candidate_sets cs
        JOIN public.users u ON u.workspace_id = cs.workspace_id
        WHERE cs.candidate_set_id = candidate_set_items.candidate_set_id
          AND u.user_id = auth.uid()
      )
    );
    DROP POLICY IF EXISTS candidate_set_items_update_workspace ON public.candidate_set_items;
    CREATE POLICY candidate_set_items_update_workspace
    ON public.candidate_set_items
    FOR UPDATE
    TO authenticated
    USING (
      EXISTS (
        SELECT 1
        FROM public.candidate_sets cs
        JOIN public.users u ON u.workspace_id = cs.workspace_id
        WHERE cs.candidate_set_id = candidate_set_items.candidate_set_id
          AND u.user_id = auth.uid()
      )
    )
    WITH CHECK (
      EXISTS (
        SELECT 1
        FROM public.candidate_sets cs
        JOIN public.users u ON u.workspace_id = cs.workspace_id
        WHERE cs.candidate_set_id = candidate_set_items.candidate_set_id
          AND u.user_id = auth.uid()
      )
    );
    DROP POLICY IF EXISTS candidate_set_items_delete_workspace ON public.candidate_set_items;
    CREATE POLICY candidate_set_items_delete_workspace
    ON public.candidate_set_items
    FOR DELETE
    TO authenticated
    USING (
      EXISTS (
        SELECT 1
        FROM public.candidate_sets cs
        JOIN public.users u ON u.workspace_id = cs.workspace_id
        WHERE cs.candidate_set_id = candidate_set_items.candidate_set_id
          AND u.user_id = auth.uid()
      )
    );

    -- graph_drafts
    DROP POLICY IF EXISTS graph_drafts_select_workspace ON public.graph_drafts;
    CREATE POLICY graph_drafts_select_workspace
    ON public.graph_drafts
    FOR SELECT
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS graph_drafts_insert_workspace ON public.graph_drafts;
    CREATE POLICY graph_drafts_insert_workspace
    ON public.graph_drafts
    FOR INSERT
    TO authenticated
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS graph_drafts_update_workspace ON public.graph_drafts;
    CREATE POLICY graph_drafts_update_workspace
    ON public.graph_drafts
    FOR UPDATE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    )
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS graph_drafts_delete_workspace ON public.graph_drafts;
    CREATE POLICY graph_drafts_delete_workspace
    ON public.graph_drafts
    FOR DELETE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );

    -- graph draft children (via parent graph_draft)
    DROP POLICY IF EXISTS graph_draft_nodes_select_workspace ON public.graph_draft_nodes;
    CREATE POLICY graph_draft_nodes_select_workspace
    ON public.graph_draft_nodes
    FOR SELECT
    TO authenticated
    USING (
      EXISTS (
        SELECT 1
        FROM public.graph_drafts gd
        JOIN public.users u ON u.workspace_id = gd.workspace_id
        WHERE gd.graph_draft_id = graph_draft_nodes.graph_draft_id
          AND u.user_id = auth.uid()
      )
    );
    DROP POLICY IF EXISTS graph_draft_edges_select_workspace ON public.graph_draft_edges;
    CREATE POLICY graph_draft_edges_select_workspace
    ON public.graph_draft_edges
    FOR SELECT
    TO authenticated
    USING (
      EXISTS (
        SELECT 1
        FROM public.graph_drafts gd
        JOIN public.users u ON u.workspace_id = gd.workspace_id
        WHERE gd.graph_draft_id = graph_draft_edges.graph_draft_id
          AND u.user_id = auth.uid()
      )
    );

    -- maps
    DROP POLICY IF EXISTS maps_select_workspace ON public.maps;
    CREATE POLICY maps_select_workspace
    ON public.maps
    FOR SELECT
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS maps_insert_workspace ON public.maps;
    CREATE POLICY maps_insert_workspace
    ON public.maps
    FOR INSERT
    TO authenticated
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS maps_update_workspace ON public.maps;
    CREATE POLICY maps_update_workspace
    ON public.maps
    FOR UPDATE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    )
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS maps_delete_workspace ON public.maps;
    CREATE POLICY maps_delete_workspace
    ON public.maps
    FOR DELETE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );

    -- map children (via parent map)
    DROP POLICY IF EXISTS map_nodes_select_workspace ON public.map_nodes;
    CREATE POLICY map_nodes_select_workspace
    ON public.map_nodes
    FOR SELECT
    TO authenticated
    USING (
      EXISTS (
        SELECT 1
        FROM public.maps m
        JOIN public.users u ON u.workspace_id = m.workspace_id
        WHERE m.map_id = map_nodes.map_id
          AND u.user_id = auth.uid()
      )
    );
    DROP POLICY IF EXISTS map_edges_select_workspace ON public.map_edges;
    CREATE POLICY map_edges_select_workspace
    ON public.map_edges
    FOR SELECT
    TO authenticated
    USING (
      EXISTS (
        SELECT 1
        FROM public.maps m
        JOIN public.users u ON u.workspace_id = m.workspace_id
        WHERE m.map_id = map_edges.map_id
          AND u.user_id = auth.uid()
      )
    );

    -- rank_jobs
    DROP POLICY IF EXISTS rank_jobs_select_workspace ON public.rank_jobs;
    CREATE POLICY rank_jobs_select_workspace
    ON public.rank_jobs
    FOR SELECT
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS rank_jobs_insert_workspace ON public.rank_jobs;
    CREATE POLICY rank_jobs_insert_workspace
    ON public.rank_jobs
    FOR INSERT
    TO authenticated
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS rank_jobs_update_workspace ON public.rank_jobs;
    CREATE POLICY rank_jobs_update_workspace
    ON public.rank_jobs
    FOR UPDATE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    )
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS rank_jobs_delete_workspace ON public.rank_jobs;
    CREATE POLICY rank_jobs_delete_workspace
    ON public.rank_jobs
    FOR DELETE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );

    -- rank_results (via rank_job)
    DROP POLICY IF EXISTS rank_results_select_workspace ON public.rank_results;
    CREATE POLICY rank_results_select_workspace
    ON public.rank_results
    FOR SELECT
    TO authenticated
    USING (
      EXISTS (
        SELECT 1
        FROM public.rank_jobs rj
        JOIN public.users u ON u.workspace_id = rj.workspace_id
        WHERE rj.rank_job_id = rank_results.rank_job_id
          AND u.user_id = auth.uid()
      )
    );

    -- gap analysis
    DROP POLICY IF EXISTS gap_analysis_results_select_workspace ON public.gap_analysis_results;
    CREATE POLICY gap_analysis_results_select_workspace
    ON public.gap_analysis_results
    FOR SELECT
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS gap_analysis_results_insert_workspace ON public.gap_analysis_results;
    CREATE POLICY gap_analysis_results_insert_workspace
    ON public.gap_analysis_results
    FOR INSERT
    TO authenticated
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS gap_analysis_results_update_workspace ON public.gap_analysis_results;
    CREATE POLICY gap_analysis_results_update_workspace
    ON public.gap_analysis_results
    FOR UPDATE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    )
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS gap_analysis_results_delete_workspace ON public.gap_analysis_results;
    CREATE POLICY gap_analysis_results_delete_workspace
    ON public.gap_analysis_results
    FOR DELETE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );

    DROP POLICY IF EXISTS gap_analysis_jobs_select_workspace ON public.gap_analysis_jobs;
    CREATE POLICY gap_analysis_jobs_select_workspace
    ON public.gap_analysis_jobs
    FOR SELECT
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS gap_analysis_jobs_insert_workspace ON public.gap_analysis_jobs;
    CREATE POLICY gap_analysis_jobs_insert_workspace
    ON public.gap_analysis_jobs
    FOR INSERT
    TO authenticated
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS gap_analysis_jobs_update_workspace ON public.gap_analysis_jobs;
    CREATE POLICY gap_analysis_jobs_update_workspace
    ON public.gap_analysis_jobs
    FOR UPDATE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    )
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS gap_analysis_jobs_delete_workspace ON public.gap_analysis_jobs;
    CREATE POLICY gap_analysis_jobs_delete_workspace
    ON public.gap_analysis_jobs
    FOR DELETE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );

    DROP POLICY IF EXISTS gap_candidates_select_workspace ON public.gap_candidates;
    CREATE POLICY gap_candidates_select_workspace
    ON public.gap_candidates
    FOR SELECT
    TO authenticated
    USING (
      EXISTS (
        SELECT 1
        FROM public.gap_analysis_jobs gj
        JOIN public.users u ON u.workspace_id = gj.workspace_id
        WHERE gj.id = gap_candidates.job_id
          AND u.user_id = auth.uid()
      )
    );

    -- workspace artifacts
    DROP POLICY IF EXISTS workspace_artifacts_select_workspace ON public.workspace_artifacts;
    CREATE POLICY workspace_artifacts_select_workspace
    ON public.workspace_artifacts
    FOR SELECT
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS workspace_artifacts_insert_workspace ON public.workspace_artifacts;
    CREATE POLICY workspace_artifacts_insert_workspace
    ON public.workspace_artifacts
    FOR INSERT
    TO authenticated
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS workspace_artifacts_update_workspace ON public.workspace_artifacts;
    CREATE POLICY workspace_artifacts_update_workspace
    ON public.workspace_artifacts
    FOR UPDATE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    )
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS workspace_artifacts_delete_workspace ON public.workspace_artifacts;
    CREATE POLICY workspace_artifacts_delete_workspace
    ON public.workspace_artifacts
    FOR DELETE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );

    -- workspace collections
    DROP POLICY IF EXISTS workspace_collections_select_workspace ON public.workspace_collections;
    CREATE POLICY workspace_collections_select_workspace
    ON public.workspace_collections
    FOR SELECT
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS workspace_collections_insert_workspace ON public.workspace_collections;
    CREATE POLICY workspace_collections_insert_workspace
    ON public.workspace_collections
    FOR INSERT
    TO authenticated
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS workspace_collections_update_workspace ON public.workspace_collections;
    CREATE POLICY workspace_collections_update_workspace
    ON public.workspace_collections
    FOR UPDATE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    )
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS workspace_collections_delete_workspace ON public.workspace_collections;
    CREATE POLICY workspace_collections_delete_workspace
    ON public.workspace_collections
    FOR DELETE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );

    -- workspace collection items
    DROP POLICY IF EXISTS workspace_collection_items_select_workspace ON public.workspace_collection_items;
    CREATE POLICY workspace_collection_items_select_workspace
    ON public.workspace_collection_items
    FOR SELECT
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS workspace_collection_items_insert_workspace ON public.workspace_collection_items;
    CREATE POLICY workspace_collection_items_insert_workspace
    ON public.workspace_collection_items
    FOR INSERT
    TO authenticated
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS workspace_collection_items_update_workspace ON public.workspace_collection_items;
    CREATE POLICY workspace_collection_items_update_workspace
    ON public.workspace_collection_items
    FOR UPDATE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    )
    WITH CHECK (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );
    DROP POLICY IF EXISTS workspace_collection_items_delete_workspace ON public.workspace_collection_items;
    CREATE POLICY workspace_collection_items_delete_workspace
    ON public.workspace_collection_items
    FOR DELETE
    TO authenticated
    USING (
      workspace_id IN (SELECT workspace_id FROM public.users WHERE user_id = auth.uid())
    );

    GRANT SELECT, INSERT, UPDATE, DELETE ON public.candidate_sets TO authenticated;
    GRANT SELECT, INSERT, UPDATE, DELETE ON public.candidate_set_items TO authenticated;
    GRANT SELECT, INSERT, UPDATE, DELETE ON public.graph_drafts TO authenticated;
    GRANT SELECT ON public.graph_draft_nodes TO authenticated;
    GRANT SELECT ON public.graph_draft_edges TO authenticated;
    GRANT SELECT, INSERT, UPDATE, DELETE ON public.maps TO authenticated;
    GRANT SELECT ON public.map_nodes TO authenticated;
    GRANT SELECT ON public.map_edges TO authenticated;
    GRANT SELECT, INSERT, UPDATE, DELETE ON public.rank_jobs TO authenticated;
    GRANT SELECT ON public.rank_results TO authenticated;
    GRANT SELECT, INSERT, UPDATE, DELETE ON public.gap_analysis_results TO authenticated;
    GRANT SELECT, INSERT, UPDATE, DELETE ON public.gap_analysis_jobs TO authenticated;
    GRANT SELECT ON public.gap_candidates TO authenticated;
    GRANT SELECT, INSERT, UPDATE, DELETE ON public.workspace_artifacts TO authenticated;
    GRANT SELECT, INSERT, UPDATE, DELETE ON public.workspace_collections TO authenticated;
    GRANT SELECT, INSERT, UPDATE, DELETE ON public.workspace_collection_items TO authenticated;
  END IF;
END $$;

COMMIT;
