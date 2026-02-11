"""
Ranking Benchmark: Run 5 queries, collect results, output for grading.
"""
import json
import os
import sys
import time
from pathlib import Path
from uuid import UUID

# Ensure .env is loaded
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from app.db import make_engine
from app.feature2.rank_service import direct_rank_prod

TENANT_ID = UUID(os.environ["DEV_TENANT_ID"])

QUERIES = [
    "attention mechanisms transformers NLP",
    "CRISPR gene therapy",
    "reinforcement learning robotics manipulation",
    "deep learning computer vision",
    "climate change impacts",
]

def run_benchmark(label: str):
    engine = make_engine()
    results = {}

    for query in QUERIES:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print(f"{'='*70}")

        start = time.time()
        try:
            result = direct_rank_prod(
                engine,
                tenant_id=TENANT_ID,
                query_text=query,
                context_json={},
                filters_json={},
                rank_params_json={},
            )
            elapsed = time.time() - start

            # Extract categories
            foundational = result.get("foundational", [])
            methodology = result.get("methodology", [])
            reviews = result.get("reviews", [])
            applications = result.get("applications", [])

            print(f"\nTime: {elapsed:.1f}s")
            print(f"Categories: foundational={len(foundational)}, methodology={len(methodology)}, reviews={len(reviews)}, applications={len(applications)}")
            print(f"Total papers: {len(foundational) + len(methodology) + len(reviews) + len(applications)}")

            # Print each category
            for cat_name, cat_items in [
                ("FOUNDATIONAL", foundational),
                ("METHODOLOGY", methodology),
                ("REVIEWS", reviews),
                ("APPLICATIONS", applications),
            ]:
                if cat_items:
                    print(f"\n  --- {cat_name} ({len(cat_items)} papers) ---")
                    for item in cat_items:
                        preview = item.get("preview", {})
                        title = preview.get("title", "?")[:80]
                        year = preview.get("year", "?")
                        cites = preview.get("cited_by_count", 0)
                        bd = item.get("breakdown", item.get("score_breakdown", {}))
                        norm = bd.get("norm", {})
                        llm = norm.get("llm_relevance", "?")
                        score = item.get("score", 0)
                        print(f"    [{year}] {title}")
                        print(f"           cites={cites}, llm={llm}, score={score:.3f}")

            results[query] = {
                "elapsed": elapsed,
                "foundational": [
                    {
                        "title": i.get("preview", {}).get("title", ""),
                        "year": i.get("preview", {}).get("year"),
                        "cites": i.get("preview", {}).get("cited_by_count", 0),
                        "llm": i.get("breakdown", i.get("score_breakdown", {})).get("norm", {}).get("llm_relevance"),
                        "score": i.get("score", 0),
                    }
                    for i in foundational
                ],
                "methodology": [
                    {
                        "title": i.get("preview", {}).get("title", ""),
                        "year": i.get("preview", {}).get("year"),
                        "cites": i.get("preview", {}).get("cited_by_count", 0),
                        "llm": i.get("breakdown", i.get("score_breakdown", {})).get("norm", {}).get("llm_relevance"),
                        "score": i.get("score", 0),
                    }
                    for i in methodology
                ],
                "reviews": [
                    {
                        "title": i.get("preview", {}).get("title", ""),
                        "year": i.get("preview", {}).get("year"),
                        "cites": i.get("preview", {}).get("cited_by_count", 0),
                        "llm": i.get("breakdown", i.get("score_breakdown", {})).get("norm", {}).get("llm_relevance"),
                        "score": i.get("score", 0),
                    }
                    for i in reviews
                ],
                "applications": [
                    {
                        "title": i.get("preview", {}).get("title", ""),
                        "year": i.get("preview", {}).get("year"),
                        "cites": i.get("preview", {}).get("cited_by_count", 0),
                        "llm": i.get("breakdown", i.get("score_breakdown", {})).get("norm", {}).get("llm_relevance"),
                        "score": i.get("score", 0),
                    }
                    for i in applications
                ],
            }

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[query] = {"error": str(e)}

    # Save to file
    outfile = f"benchmark_{label}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to {outfile}")
    return results


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "pre_fix"
    run_benchmark(label)
