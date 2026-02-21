#!/usr/bin/env python3
"""
End-to-end tests: DOI/work_id/PDF → ranking output.

Tests the full pipeline:
1. Input: DOI, work_id, or PDF
2. Extract title from input
3. Generate query from title
4. Run ranking pipeline
5. Verify ranking results

Run after server is started:
    uvicorn app.main:app --host 127.0.0.1 --port 8000 &
    pytest tests/live/test_seed_resolver_e2e_ranking.py -v -s
"""

import json
import time
from pathlib import Path

import pytest
import requests


BASE_URL = "http://127.0.0.1:8000"
RESULTS_DIR = Path("tests/live/results")
RESULTS_DIR.mkdir(exist_ok=True)


# Test cases: 6 DOI + 4 PDF = 10 total
DOI_CASES = [
    {
        "seed_doi": "10.48550/arXiv.1706.03762",
        "expected_query_substring": "attention",
        "note": "Attention Is All You Need (Transformer paper)",
    },
    {
        "seed_doi": "10.48550/arXiv.1512.03385",
        "expected_query_substring": "residual",
        "note": "ResNet paper",
    },
    {
        "seed_doi": "10.48550/arXiv.2005.14165",
        "expected_query_substring": "language",
        "note": "GPT-3 paper",
    },
    {
        "seed_doi": "10.48550/arXiv.1409.1556",
        "expected_query_substring": "convolutional",
        "note": "VGG paper",
    },
    {
        "seed_doi": "10.48550/arXiv.1406.2661",
        "expected_query_substring": "generative",
        "note": "GAN paper",
    },
    {
        "seed_doi": "10.48550/arXiv.1312.6114",
        "expected_query_substring": "variational",
        "note": "VAE paper",
    },
]

PDF_CASES = [
    {
        "pdf_path": "tests/live/seed_resolver_test_data/pdfs/1706.03762_transformer.pdf",
        "expected_query_substring": "attention",
        "note": "Attention Is All You Need (PDF upload)",
    },
    {
        "pdf_path": "tests/live/seed_resolver_test_data/pdfs/1810.04805_bert.pdf",
        "expected_query_substring": "language",
        "note": "BERT (PDF upload)",
    },
    {
        "pdf_path": "tests/live/seed_resolver_test_data/pdfs/1512.03385_resnet.pdf",
        "expected_query_substring": "residual",
        "note": "ResNet (PDF upload)",
    },
    {
        "pdf_path": "tests/live/seed_resolver_test_data/pdfs/2005.14165_gpt3.pdf",
        "expected_query_substring": "language",
        "note": "GPT-3 (PDF upload)",
    },
]


def wait_for_job_completion(rank_job_id: str, timeout: int = 120) -> dict:
    """Poll /v1/rank/{job_id} until complete or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{BASE_URL}/v1/rank/{rank_job_id}")
        if resp.status_code != 200:
            raise RuntimeError(f"Job status check failed: {resp.status_code} {resp.text}")

        data = resp.json()
        status = data.get("job", {}).get("status")

        if status == "completed":
            return data
        elif status == "failed":
            raise RuntimeError(f"Ranking job failed: {data}")

        time.sleep(2)

    raise TimeoutError(f"Job {rank_job_id} did not complete in {timeout}s")


@pytest.mark.live
def test_doi_to_ranking_e2e():
    """DOI → title → query → ranking pipeline (4 cases)."""
    results = []

    for case in DOI_CASES:
        print(f"\n{'='*60}")
        print(f"Testing DOI: {case['seed_doi']}")
        print(f"Note: {case['note']}")

        # Submit ranking request
        resp = requests.post(
            f"{BASE_URL}/v1/rank",
            json={"seed_doi": case["seed_doi"]},
            timeout=30,
        )

        if resp.status_code not in (200, 202):
            results.append({
                "input": case["seed_doi"],
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text}",
                "note": case["note"],
            })
            continue

        data = resp.json()

        # If async job, wait for completion
        if resp.status_code == 202:
            rank_job_id = data.get("rank_job_id")
            print(f"  Waiting for job {rank_job_id}...")
            data = wait_for_job_completion(rank_job_id)

        # Verify results - ranking returns category fields, not a single "papers" field
        all_papers = []
        for category in ["foundational", "methodology", "reviews", "applications", "items", "textbooks"]:
            all_papers.extend(data.get(category, []))

        success = len(all_papers) > 0

        results.append({
            "input": case["seed_doi"],
            "success": success,
            "paper_count": len(all_papers),
            "top_3_titles": [p.get("preview", {}).get("title", "")[:60] for p in all_papers[:3]],
            "categories": {k: len(data.get(k, [])) for k in ["foundational", "methodology", "reviews", "applications", "items"]},
            "note": case["note"],
        })

        print(f"  Success: {success}")
        print(f"  Papers returned: {len(all_papers)}")
        print(f"  By category: foundational={len(data.get('foundational', []))}, methodology={len(data.get('methodology', []))}, reviews={len(data.get('reviews', []))}")
        if all_papers:
            print(f"  Top paper: {all_papers[0].get('preview', {}).get('title', '')[:80]}")

    # Save results
    output_file = RESULTS_DIR / "doi_to_ranking_e2e.json"
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")

    # Assert all succeeded
    success_count = sum(r["success"] for r in results)
    assert success_count == len(DOI_CASES), \
        f"DOI→ranking: {success_count}/{len(DOI_CASES)} succeeded"




@pytest.mark.live
def test_pdf_to_ranking_e2e():
    """PDF → title → query → ranking pipeline (2 cases)."""
    results = []

    for case in PDF_CASES:
        pdf_path = Path(case["pdf_path"])

        print(f"\n{'='*60}")
        print(f"Testing PDF: {pdf_path.name}")
        print(f"Note: {case['note']}")

        if not pdf_path.exists():
            results.append({
                "input": str(pdf_path),
                "success": False,
                "error": f"PDF not found: {pdf_path}",
                "note": case["note"],
            })
            continue

        # Submit PDF upload
        with open(pdf_path, "rb") as f:
            files = {"pdf_file": (pdf_path.name, f, "application/pdf")}
            resp = requests.post(
                f"{BASE_URL}/v1/rank/pdf",
                files=files,
                timeout=30,
            )

        if resp.status_code not in (200, 202):
            results.append({
                "input": str(pdf_path),
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text}",
                "note": case["note"],
            })
            continue

        data = resp.json()

        # If async job, wait for completion
        if resp.status_code == 202:
            rank_job_id = data.get("rank_job_id")
            print(f"  Waiting for job {rank_job_id}...")
            data = wait_for_job_completion(rank_job_id)

        # Verify results - ranking returns category fields, not a single "papers" field
        all_papers = []
        for category in ["foundational", "methodology", "reviews", "applications", "items", "textbooks"]:
            all_papers.extend(data.get(category, []))

        success = len(all_papers) > 0

        results.append({
            "input": str(pdf_path),
            "success": success,
            "paper_count": len(all_papers),
            "top_3_titles": [p.get("preview", {}).get("title", "")[:60] for p in all_papers[:3]],
            "categories": {k: len(data.get(k, [])) for k in ["foundational", "methodology", "reviews", "applications", "items"]},
            "extracted_title": data.get("seed_extraction", {}).get("extracted_title", ""),
            "note": case["note"],
        })

        print(f"  Success: {success}")
        print(f"  Papers returned: {len(all_papers)}")
        print(f"  By category: foundational={len(data.get('foundational', []))}, methodology={len(data.get('methodology', []))}, reviews={len(data.get('reviews', []))}")
        if all_papers:
            print(f"  Top paper: {all_papers[0].get('preview', {}).get('title', '')[:80]}")

    # Save results
    output_file = RESULTS_DIR / "pdf_to_ranking_e2e.json"
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")

    # Assert all succeeded
    success_count = sum(r["success"] for r in results)
    assert success_count == len(PDF_CASES), \
        f"PDF→ranking: {success_count}/{len(PDF_CASES)} succeeded"


if __name__ == "__main__":
    # Can run directly: python tests/live/test_seed_resolver_e2e_ranking.py
    print("Running 10 end-to-end test cases...")
    print("DOI cases: 6, PDF cases: 4")

    test_doi_to_ranking_e2e()
    test_pdf_to_ranking_e2e()

    print("\n" + "="*60)
    print("All 10 test cases completed!")
