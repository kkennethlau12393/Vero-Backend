"""
Live end-to-end tests for seed_resolver.py - DOI/work_id/PDF → title extraction.

BURNS API CREDITS - hits real OpenAlex, Semantic Scholar, ArXiv APIs.
Run explicitly: pytest tests/live/test_seed_resolver_e2e.py -v -s --timeout=600

Requirements:
- DOI parser: ≥95/100 success rate (95%)
- work_id parser: ≥95/100 success rate (95%)
- PDF parser: ≥95/100 success rate (95%)

Test cases loaded from JSON files in tests/live/seed_resolver_test_data/:
- doi_test_cases.json (100 DOIs from literature)
- work_id_test_cases.json (100 work_ids from dev DB)
- pdf_test_cases.json (100 PDF metadata specs)

Results saved to tests/live/results/ for manual review and regression detection.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from typing import List, Dict, Any


TEST_DATA_DIR = Path(__file__).parent / "seed_resolver_test_data"
RESULTS_DIR = Path(__file__).parent / "results"


# ===== DOI Parser Live Tests =====


@pytest.mark.live
def test_doi_parser_100_cases():
    """
    DOI → title extraction must achieve ≥95% success rate on 100 diverse DOIs.

    Test cases sourced from:
    - ArXiv DOIs (30): 10.48550/arXiv.XXXX format
    - Journal DOIs (20): Nature, Science, IEEE, ACM
    - Conference DOIs (15): NeurIPS, ICLR, CVPR, ACL
    - Preprint DOIs (10): bioRxiv, medRxiv
    - Edge cases (10): special chars, redirects, case variations
    - Error cases (10): invalid DOI, 404, malformed
    - S2-only papers (5): DOI not in OpenAlex
    """
    from app.feature2.seed_resolver import resolve_doi_to_title

    # Load test cases
    test_cases_path = TEST_DATA_DIR / "doi_test_cases.json"

    if not test_cases_path.exists():
        pytest.skip(f"DOI test cases not found: {test_cases_path}")

    with open(test_cases_path) as f:
        test_cases = json.load(f)

    # Allow running with <100 cases during development, but warn
    num_cases = len(test_cases)
    if num_cases < 100:
        print(f"\n⚠️  WARNING: Running with {num_cases}/100 test cases. Need 100 for final validation.\n")

    results = []
    passed = 0
    failed = 0

    for i, case in enumerate(test_cases, 1):
        doi = case["doi"]
        expected_substring = case.get("expected_title_substring", "")
        expect_error = case.get("expect_error", False)

        try:
            title, metadata = resolve_doi_to_title(doi)

            if expect_error:
                # Expected an error but got success
                results.append({
                    "test_num": i,
                    "doi": doi,
                    "success": False,
                    "error": "Expected error but resolution succeeded",
                    "title": title,
                })
                failed += 1
            elif expected_substring and expected_substring.lower() not in title.lower():
                # Title doesn't match expected
                results.append({
                    "test_num": i,
                    "doi": doi,
                    "success": False,
                    "error": f"Title mismatch. Expected substring: '{expected_substring}', got: '{title}'",
                })
                failed += 1
            else:
                # Success
                results.append({
                    "test_num": i,
                    "doi": doi,
                    "success": True,
                    "title": title,
                    "work_id": metadata.get("work_id"),
                    "source": metadata.get("source"),
                })
                passed += 1

        except Exception as e:
            if expect_error:
                # Expected an error and got one
                results.append({
                    "test_num": i,
                    "doi": doi,
                    "success": True,
                    "note": "Expected error occurred",
                    "error_message": str(e),
                })
                passed += 1
            else:
                # Unexpected error
                results.append({
                    "test_num": i,
                    "doi": doi,
                    "success": False,
                    "error": str(e),
                })
                failed += 1

    success_rate = passed / len(results) if results else 0

    # Save detailed results
    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = RESULTS_DIR / "doi_parser_100_cases.json"
    with open(results_file, "w") as f:
        json.dump({
            "total_cases": len(results),
            "passed": passed,
            "failed": failed,
            "success_rate": success_rate,
            "results": results,
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"DOI Parser Results: {passed}/{len(results)} passed ({success_rate:.1%})")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")

    # Require ≥95% success rate only if we have 100 cases
    if len(results) >= 100:
        assert success_rate >= 0.95, \
            f"DOI parser success rate {success_rate:.1%} < 95% threshold. See {results_file} for details."
    else:
        print(f"⚠️  Development mode: Need 100 test cases for final validation (currently {len(results)})")


# ===== work_id Parser Live Tests =====


@pytest.mark.live
def test_work_id_parser_100_cases():
    """
    work_id → title extraction must achieve ≥95% success rate on 100 diverse work_ids.

    Test cases sourced from:
    - OpenAlex IDs (40): W... format, diverse domains (CV, NLP, Bio, Physics)
    - Semantic Scholar IDs (30): S2:... format
    - ArXiv IDs (20): AX:... format
    - Edge cases (5): mixed-case, legacy formats
    - Error cases (5): malformed ID, 404
    """
    from app.feature2.seed_resolver import resolve_work_id_to_title

    # Load test cases
    test_cases_path = TEST_DATA_DIR / "work_id_test_cases.json"

    if not test_cases_path.exists():
        pytest.skip(f"work_id test cases not found: {test_cases_path}")

    with open(test_cases_path) as f:
        test_cases = json.load(f)

    # Allow running with <100 cases during development, but warn
    num_cases = len(test_cases)
    if num_cases < 100:
        print(f"\n⚠️  WARNING: Running with {num_cases}/100 test cases. Need 100 for final validation.\n")

    results = []
    passed = 0
    failed = 0

    for i, case in enumerate(test_cases, 1):
        work_id = case["work_id"]
        expected_substring = case.get("expected_title_substring", "")
        expect_error = case.get("expect_error", False)

        try:
            title, metadata = resolve_work_id_to_title(work_id)

            if expect_error:
                results.append({
                    "test_num": i,
                    "work_id": work_id,
                    "success": False,
                    "error": "Expected error but resolution succeeded",
                    "title": title,
                })
                failed += 1
            elif expected_substring and expected_substring.lower() not in title.lower():
                results.append({
                    "test_num": i,
                    "work_id": work_id,
                    "success": False,
                    "error": f"Title mismatch. Expected substring: '{expected_substring}', got: '{title}'",
                })
                failed += 1
            else:
                results.append({
                    "test_num": i,
                    "work_id": work_id,
                    "success": True,
                    "title": title,
                    "source": metadata.get("source"),
                })
                passed += 1

        except Exception as e:
            if expect_error:
                results.append({
                    "test_num": i,
                    "work_id": work_id,
                    "success": True,
                    "note": "Expected error occurred",
                    "error_message": str(e),
                })
                passed += 1
            else:
                results.append({
                    "test_num": i,
                    "work_id": work_id,
                    "success": False,
                    "error": str(e),
                })
                failed += 1

    success_rate = passed / len(results) if results else 0

    # Save detailed results
    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = RESULTS_DIR / "work_id_parser_100_cases.json"
    with open(results_file, "w") as f:
        json.dump({
            "total_cases": len(results),
            "passed": passed,
            "failed": failed,
            "success_rate": success_rate,
            "results": results,
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"work_id Parser Results: {passed}/{len(results)} passed ({success_rate:.1%})")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")

    # Require ≥95% success rate only if we have 100 cases
    if len(results) >= 100:
        assert success_rate >= 0.95, \
            f"work_id parser success rate {success_rate:.1%} < 95% threshold. See {results_file} for details."
    else:
        print(f"⚠️  Development mode: Need 100 test cases for final validation (currently {len(results)})")


# ===== PDF Parser Live Tests =====


@pytest.mark.live
def test_pdf_parser_100_cases():
    """
    PDF → title extraction must achieve ≥95% success rate on 100 diverse PDFs.

    Test cases sourced from:
    - ArXiv PDFs (30): varying header formats, ligatures (ﬁ→fi)
    - Conference PDFs (20): IEEE, ACM, Springer templates
    - Journal PDFs (15): Nature, Science, PLOS formats
    - Preprint PDFs (10): bioRxiv, medRxiv
    - Multi-line titles (10): ALL CAPS, split across lines
    - Metadata-only PDFs (5): title in PDF metadata, not text
    - Scanned PDFs (5): OCR-extracted text (known limitation)
    - Error cases (5): corrupt PDF, no text, no title
    """
    from app.feature2.seed_resolver import resolve_pdf_to_title

    # Load test cases
    test_cases_path = TEST_DATA_DIR / "pdf_test_cases.json"

    if not test_cases_path.exists():
        pytest.skip(f"PDF test cases not found: {test_cases_path}")

    with open(test_cases_path) as f:
        test_cases = json.load(f)

    if len(test_cases) < 100:
        pytest.skip(f"Need 100 PDF test cases, found {len(test_cases)}")

    results = []
    passed = 0
    failed = 0

    for i, case in enumerate(test_cases[:100], 1):
        pdf_path = case["pdf_path"]
        expected_substring = case.get("expected_title_substring", "")
        expect_error = case.get("expect_error", False)

        pdf_full_path = TEST_DATA_DIR / "pdfs" / pdf_path

        if not pdf_full_path.exists():
            results.append({
                "test_num": i,
                "pdf_path": pdf_path,
                "success": False,
                "error": f"PDF file not found: {pdf_full_path}",
            })
            failed += 1
            continue

        try:
            with open(pdf_full_path, "rb") as f:
                pdf_bytes = f.read()

            title, metadata = resolve_pdf_to_title(pdf_bytes)

            if expect_error:
                results.append({
                    "test_num": i,
                    "pdf_path": pdf_path,
                    "success": False,
                    "error": "Expected error but extraction succeeded",
                    "title": title,
                })
                failed += 1
            elif expected_substring and expected_substring.lower() not in title.lower():
                results.append({
                    "test_num": i,
                    "pdf_path": pdf_path,
                    "success": False,
                    "error": f"Title mismatch. Expected substring: '{expected_substring}', got: '{title}'",
                })
                failed += 1
            else:
                results.append({
                    "test_num": i,
                    "pdf_path": pdf_path,
                    "success": True,
                    "title": title,
                    "extraction_method": metadata.get("extraction_method"),
                    "work_id": metadata.get("work_id"),
                })
                passed += 1

        except Exception as e:
            if expect_error:
                results.append({
                    "test_num": i,
                    "pdf_path": pdf_path,
                    "success": True,
                    "note": "Expected error occurred",
                    "error_message": str(e),
                })
                passed += 1
            else:
                results.append({
                    "test_num": i,
                    "pdf_path": pdf_path,
                    "success": False,
                    "error": str(e),
                })
                failed += 1

    success_rate = passed / len(results) if results else 0

    # Save detailed results
    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = RESULTS_DIR / "pdf_parser_100_cases.json"
    with open(results_file, "w") as f:
        json.dump({
            "total_cases": len(results),
            "passed": passed,
            "failed": failed,
            "success_rate": success_rate,
            "results": results,
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"PDF Parser Results: {passed}/{len(results)} passed ({success_rate:.1%})")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")

    # Require ≥95% success rate
    assert success_rate >= 0.95, \
        f"PDF parser success rate {success_rate:.1%} < 95% threshold. See {results_file} for details."
