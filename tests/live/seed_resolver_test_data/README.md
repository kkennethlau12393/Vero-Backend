# Seed Resolver Test Data

This directory contains test cases for live end-to-end testing of the seed resolver parsers (DOI, work_id, PDF).

## Current Status

| Parser | Test Cases | Requirement | Status |
|--------|-----------|-------------|--------|
| DOI | 10/100 | ≥95% success | ⚠️ Need 90 more |
| work_id | 10/100 | ≥95% success | ⚠️ Need 90 more |
| PDF | 4/100 | ≥95% success | ⚠️ Need 96 more (+ PDFs) |

## Running Live Tests

```bash
# DOI parser (burns API credits)
pytest tests/live/test_seed_resolver_e2e.py::test_doi_parser_100_cases -v -s --timeout=600

# work_id parser (burns API credits)
pytest tests/live/test_seed_resolver_e2e.py::test_work_id_parser_100_cases -v -s --timeout=600

# PDF parser (burns API credits)
pytest tests/live/test_seed_resolver_e2e.py::test_pdf_parser_100_cases -v -s --timeout=600
```

## Test Case Format

### DOI Test Cases (`doi_test_cases.json`)

```json
{
  "doi": "10.48550/arXiv.1706.03762",
  "expected_title_substring": "Attention Is All You Need",
  "expect_error": false,
  "category": "arxiv",
  "note": "Transformer paper"
}
```

**Sources for DOI test cases:**
- ArXiv papers (30): `10.48550/arXiv.XXXX` format
- Nature papers (10): `10.1038/nature...`
- Science papers (5): `10.1126/science...`
- ACL papers (10): `10.18653/v1/...`
- NeurIPS papers (10): `10.5555/...`
- ICLR papers (10): via OpenReview or ACM
- IEEE papers (10): `10.1109/...`
- bioRxiv/medRxiv (10): `10.1101/...`
- Edge cases (10): special chars, redirects
- Error cases (10): invalid/404 DOIs

### work_id Test Cases (`work_id_test_cases.json`)

```json
{
  "work_id": "W2963663278",
  "expected_title_substring": "Attention Is All You Need",
  "expect_error": false,
  "category": "openalex",
  "note": "Transformer paper"
}
```

**Sources for work_id test cases:**
- Query dev DB for W... IDs (40 cases)
- Query S2 API for S2:... IDs (30 cases)
- Extract AX:... IDs from ArXiv (20 cases)
- Edge cases (5): mixed-case, legacy formats
- Error cases (5): malformed/404 IDs

**Script to generate work_id test cases from dev DB:**
```python
# Connect to dev DB, query:
# SELECT DISTINCT work_id FROM works LIMIT 50;
# Filter to ensure diverse domains (CV, NLP, Bio, Physics, etc.)
```

### PDF Test Cases (`pdf_test_cases.json`)

```json
{
  "pdf_path": "attention_paper.pdf",
  "expected_title_substring": "Attention",
  "expect_error": false,
  "category": "arxiv",
  "note": "Attention Is All You Need - ArXiv PDF"
}
```

**Sources for PDF test cases:**
- Download ArXiv PDFs (30): `wget https://arxiv.org/pdf/XXXX.pdf`
- Download conference PDFs (20): IEEE, ACM, Springer templates
- Download journal PDFs (15): Nature, Science, PLOS
- Download preprints (10): bioRxiv, medRxiv
- Edge cases (20): multi-line titles, ALL CAPS, ligatures (ﬁ→fi)
- Error cases (5): corrupt PDF, no text, no title

**Script to download ArXiv PDFs:**
```bash
#!/bin/bash
# Download top ArXiv papers
arxiv_ids=(
  "1706.03762"  # Attention
  "1810.04805"  # BERT
  "1406.2661"   # GAN
  # ... add 27 more
)

for id in "${arxiv_ids[@]}"; do
  wget "https://arxiv.org/pdf/${id}.pdf" -O "pdfs/arxiv_${id}.pdf"
  sleep 2  # Rate limiting
done
```

## Expanding Test Cases

To reach 100 test cases per parser:

1. **DOI test cases:** Add 90 more diverse DOIs to `doi_test_cases.json`
   - Mine from existing test results (`tests/live/results/*.json`)
   - Extract from TESTED_QUERIES.md benchmark papers
   - Add DOIs from various publishers (Nature, Science, IEEE, ACM, etc.)

2. **work_id test cases:** Add 90 more work_ids to `work_id_test_cases.json`
   - Query dev DB: `SELECT work_id, title FROM works WHERE year >= 2015 LIMIT 100;`
   - Ensure diverse formats: W..., S2:..., AX:...
   - Mix high-cited (>1000 citations) and low-cited (<100 citations) papers

3. **PDF test cases:** Collect 96 more PDFs + add to `pdf_test_cases.json`
   - Download from ArXiv API
   - Download from conference proceedings (open access)
   - Create synthetic PDFs for edge cases (using PyMuPDF)

## Acceptance Criteria

**Only proceed with integration into `rank_api.py` when:**
- DOI parser: ≥95/100 test cases pass
- work_id parser: ≥95/100 test cases pass
- PDF parser: ≥95/100 test cases pass

Results saved to `tests/live/results/`:
- `doi_parser_100_cases.json`
- `work_id_parser_100_cases.json`
- `pdf_parser_100_cases.json`
