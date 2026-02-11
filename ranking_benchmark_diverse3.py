"""
Ranking Benchmark: 10 diverse queries (round 3) - fresh validation set.
"""
import json
import os
import sys
import time
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from app.db import make_engine
from app.feature2.rank_service import direct_rank_prod

TENANT_ID = UUID(os.environ["DEV_TENANT_ID"])

QUERIES = [
    "CRISPR gene editing",
    "quantum computing error correction",
    "microplastics environmental impact",
    "large language models",
    "gut microbiome human health",
    "gravitational wave detection",
    "lithium ion battery degradation",
    "reinforcement learning robotics",
    "antibiotic resistance mechanisms",
    "federated learning privacy",
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

            foundational = result.get("foundational", [])
            methodology = result.get("methodology", [])
            reviews = result.get("reviews", [])
            applications = result.get("applications", [])
            textbooks = result.get("textbooks", [])

            print(f"\nTime: {elapsed:.1f}s")
            print(f"Categories: foundational={len(foundational)}, methodology={len(methodology)}, reviews={len(reviews)}, applications={len(applications)}, textbooks={len(textbooks)}")
            print(f"Total papers: {len(foundational) + len(methodology) + len(reviews) + len(applications) + len(textbooks)}")

            for cat_name, cat_items in [
                ("FOUNDATIONAL", foundational),
                ("METHODOLOGY", methodology),
                ("REVIEWS", reviews),
                ("APPLICATIONS", applications),
                ("TEXTBOOKS", textbooks),
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

            def _extract(items):
                return [
                    {
                        "title": i.get("preview", {}).get("title", ""),
                        "year": i.get("preview", {}).get("year"),
                        "cites": i.get("preview", {}).get("cited_by_count", 0),
                        "llm": i.get("breakdown", i.get("score_breakdown", {})).get("norm", {}).get("llm_relevance"),
                        "score": i.get("score", 0),
                    }
                    for i in items
                ]

            results[query] = {
                "elapsed": elapsed,
                "foundational": _extract(foundational),
                "methodology": _extract(methodology),
                "reviews": _extract(reviews),
                "applications": _extract(applications),
                "textbooks": _extract(textbooks),
            }

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[query] = {"error": str(e)}

    outfile = f"benchmark_diverse3_{label}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to {outfile}")
    return results


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "v1"
    run_benchmark(label)
