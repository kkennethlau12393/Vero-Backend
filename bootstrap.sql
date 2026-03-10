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
    created_at timestamptz NOT NULL DEFAULT now(),
    seed_type text,
    seed_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    params_hash text NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_candidate_sets_tenant_params_hash
    ON public.candidate_sets (tenant_id, params_hash);

CREATE INDEX IF NOT EXISTS idx_candidate_sets_tenant
    ON public.candidate_sets (tenant_id);

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
    candidate_set_id uuid,
    created_at timestamptz NOT NULL DEFAULT now()
);

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
-- Query decomposition cache (structured query analysis)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.query_decomposition_cache (
    query_hash text PRIMARY KEY,
    query_text text NOT NULL,
    decomposition jsonb NOT NULL,
    model_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_decomposition_cache_created
    ON public.query_decomposition_cache (created_at DESC);

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
-- LLM evaluation paragraph cache (per paper per query)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.llm_evaluation_cache (
    paper_id text NOT NULL,
    query_hash text NOT NULL,
    evaluation_text text NOT NULL,
    model_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (paper_id, query_hash)
);

CREATE INDEX IF NOT EXISTS idx_llm_evaluation_cache_query
    ON public.llm_evaluation_cache (query_hash);

CREATE INDEX IF NOT EXISTS idx_llm_evaluation_cache_created
    ON public.llm_evaluation_cache (created_at DESC);

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

-- ---------------------------------------------------------------------
-- Permanent novelty assessment storage (Feature 3)
-- Separate from version-gated cache; survives ASSESSMENT_VERSION bumps.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.novelty_assessments (
    work_id text PRIMARY KEY,
    novelty_level text NOT NULL,
    confidence text NOT NULL,
    whats_new text,
    compared_to_prior_work text,
    novelty_explanation text NOT NULL,
    grounding_papers jsonb NOT NULL DEFAULT '[]',
    context_depth text NOT NULL DEFAULT 'abstract_only',
    assessment_unavailable_reason text,
    model_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- Saved papers (workspace-level paper library)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.saved_papers (
    workspace_id uuid NOT NULL,
    paper_work_id text NOT NULL,
    user_id uuid NOT NULL,
    source text NOT NULL DEFAULT 'manual',
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, paper_work_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_saved_papers_workspace
    ON public.saved_papers (workspace_id);
CREATE INDEX IF NOT EXISTS idx_saved_papers_user
    ON public.saved_papers (user_id);

-- Migration: add source column for existing tables
ALTER TABLE public.saved_papers
    ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT 'manual';

-- ---------------------------------------------------------------------
-- Paper tags (user-defined tags on papers, workspace-scoped)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.paper_tags (
    tag_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id uuid NOT NULL,
    user_id uuid NOT NULL,
    work_id text NOT NULL,
    tag text NOT NULL,
    color text DEFAULT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(workspace_id, user_id, work_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_paper_tags_workspace
    ON public.paper_tags (workspace_id);
CREATE INDEX IF NOT EXISTS idx_paper_tags_user
    ON public.paper_tags (user_id, workspace_id);

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
    tenant_id text NOT NULL,
    work_ids text[] NOT NULL,
    result_json jsonb NOT NULL,
    model_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_methodology_comparison_created
    ON public.methodology_comparison_cache (created_at DESC);

-- Persistent methodology comparisons (survives version bumps)
CREATE TABLE IF NOT EXISTS public.methodology_comparisons (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    rank_job_id text NOT NULL,
    work_ids text[] NOT NULL,
    result jsonb NOT NULL,
    created_at timestamptz DEFAULT now(),
    UNIQUE(rank_job_id, work_ids)
);

CREATE INDEX IF NOT EXISTS idx_methodology_comparisons_rank_job
    ON public.methodology_comparisons (rank_job_id);

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

-- Timeline narrative cache (rich per-node timeline narratives)
CREATE TABLE IF NOT EXISTS public.timeline_narrative_cache (
    work_id text PRIMARY KEY,
    narrative_json jsonb NOT NULL,
    narrative_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_timeline_narrative_cache_created
    ON public.timeline_narrative_cache (created_at DESC);

-- Persisted node timelines (rank-job scoped, survives version bumps)
CREATE TABLE IF NOT EXISTS public.node_timelines (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    rank_job_id text NOT NULL,
    work_id text NOT NULL,
    result jsonb NOT NULL,
    created_at timestamptz DEFAULT now(),
    UNIQUE(rank_job_id, work_id)
);

CREATE INDEX IF NOT EXISTS idx_node_timelines_rank_job
    ON public.node_timelines (rank_job_id);

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
    map_id uuid REFERENCES public.maps(map_id) ON DELETE CASCADE,
    rank_job_id uuid REFERENCES public.rank_jobs(rank_job_id) ON DELETE CASCADE,
    tenant_id uuid NOT NULL,
    gaps jsonb NOT NULL DEFAULT '[]'::jsonb,
    data_sources_used text[] NOT NULL DEFAULT '{}',
    coverage_pct double precision NOT NULL,
    total_candidates_detected integer NOT NULL DEFAULT 0,
    candidates_validated integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_gap_analysis_results_map
    ON public.gap_analysis_results (map_id);

CREATE INDEX IF NOT EXISTS idx_gap_analysis_results_rank_job
    ON public.gap_analysis_results (rank_job_id);

CREATE INDEX IF NOT EXISTS idx_gap_analysis_results_tenant
    ON public.gap_analysis_results (tenant_id);

CREATE INDEX IF NOT EXISTS idx_gap_analysis_results_created
    ON public.gap_analysis_results (created_at DESC);

-- Gap analysis jobs (async processing)
CREATE TABLE IF NOT EXISTS public.gap_analysis_jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    map_id uuid NOT NULL REFERENCES public.maps(map_id) ON DELETE CASCADE,
    tenant_id uuid NOT NULL,
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

-- ============================================================================
-- Research Activity Log (Feature 5 - Activity Tracking)
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.research_activity_log (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid NOT NULL,
    map_id uuid REFERENCES public.maps(map_id) ON DELETE CASCADE,
    rank_job_id uuid REFERENCES public.rank_jobs(rank_job_id) ON DELETE CASCADE,
    activity_type text NOT NULL,
    work_id text,
    node_count integer DEFAULT 0,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_activity_log_tenant
    ON public.research_activity_log (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_log_map
    ON public.research_activity_log (map_id, activity_type);
CREATE INDEX IF NOT EXISTS idx_activity_log_rank
    ON public.research_activity_log (rank_job_id, activity_type);

-- =========================================================================
-- Feature 1: Citation Map Persistence
-- =========================================================================
CREATE TABLE IF NOT EXISTS public.citation_maps (
    citation_map_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid NOT NULL,
    graph_draft_id uuid REFERENCES public.graph_drafts(graph_draft_id) ON DELETE SET NULL,
    seed_work_id text NOT NULL,
    query_text text,
    seed_doi text,
    seed_title text,
    response_json jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS citation_maps_tenant_idx
    ON public.citation_maps (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS citation_maps_seed_idx
    ON public.citation_maps (seed_work_id);

COMMIT;
