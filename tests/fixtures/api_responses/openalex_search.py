"""
Saved OpenAlex API responses for integration test mocking.
"""

# OpenAlex search response (truncated but realistic)
SEARCH_RESPONSE = {
    "meta": {"count": 3, "db_response_time_ms": 42, "page": 1, "per_page": 200},
    "results": [
        {
            "id": "https://openalex.org/W2000000000",
            "title": "A Few Useful Things to Know about Machine Learning",
            "publication_year": 2012,
            "cited_by_count": 5000,
            "primary_topic": {
                "id": "https://openalex.org/T100",
                "display_name": "Machine Learning",
                "score": 0.95,
            },
            "authorships": [
                {"author": {"display_name": "Pedro Domingos"}},
            ],
            "primary_location": {
                "source": {"display_name": "Communications of the ACM"},
            },
            "is_retracted": False,
            "open_access": {"is_oa": True, "oa_url": None},
        },
        {
            "id": "https://openalex.org/W2000000001",
            "title": "Random Forests",
            "publication_year": 2001,
            "cited_by_count": 80000,
            "primary_topic": {
                "id": "https://openalex.org/T100",
                "display_name": "Machine Learning",
                "score": 0.92,
            },
            "authorships": [
                {"author": {"display_name": "Leo Breiman"}},
            ],
            "primary_location": {
                "source": {"display_name": "Machine Learning"},
            },
            "is_retracted": False,
            "open_access": {"is_oa": False, "oa_url": None},
        },
        {
            "id": "https://openalex.org/W2000000002",
            "title": "Deep Learning Applications in Healthcare",
            "publication_year": 2020,
            "cited_by_count": 300,
            "primary_topic": {
                "id": "https://openalex.org/T200",
                "display_name": "Healthcare AI",
                "score": 0.88,
            },
            "authorships": [
                {"author": {"display_name": "Jane Smith"}},
            ],
            "primary_location": {
                "source": {"display_name": "Nature Medicine"},
            },
            "is_retracted": False,
            "open_access": {"is_oa": True, "oa_url": "https://example.com/paper.pdf"},
        },
    ],
}

# Empty search result
EMPTY_SEARCH_RESPONSE = {
    "meta": {"count": 0, "db_response_time_ms": 10, "page": 1, "per_page": 200},
    "results": [],
}
