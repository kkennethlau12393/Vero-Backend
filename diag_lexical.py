"""Diagnostic: show lexical scores for ranked papers to validate lexical filter idea."""
import json
import os
from pathlib import Path
from uuid import UUID
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from app.db import make_engine
from app.feature2.rank_service import direct_rank_prod

TENANT_ID = UUID(os.environ["DEV_TENANT_ID"])

QUERIES = [
    "attention mechanisms transformers NLP",
    "reinforcement learning robotics manipulation",
]

engine = make_engine()

for query in QUERIES:
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")

    result = direct_rank_prod(
        engine,
        tenant_id=TENANT_ID,
        query_text=query,
        context_json={},
        filters_json={},
        rank_params_json={},
    )

    for cat_name in ["foundational", "methodology", "reviews", "applications"]:
        items = result.get(cat_name, [])
        if not items:
            continue
        print(f"\n  --- {cat_name.upper()} ---")
        for item in items:
            preview = item.get("preview", {})
            title = (preview.get("title", "?") or "?")[:70]
            year = preview.get("year", "?")
            bd = item.get("breakdown", {})
            norm = bd.get("norm", {})
            lex = norm.get("lexical", 0.0)
            llm = norm.get("llm_relevance", 0.0)
            prov = item.get("provenance", [])
            sources = [p.get("source", "?") for p in prov]
            print(f"    [{year}] {title}")
            print(f"           llm={llm:.2f}  lex={lex:.3f}  sources={sources}")
