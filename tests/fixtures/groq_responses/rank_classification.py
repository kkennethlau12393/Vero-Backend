"""
Saved Groq LLM responses for integration test mocking.
"""

# Query classification response
CLASSIFICATION_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": '{"query_type": "topical", "confidence": 0.85, "methodological_indicators": [], "domain_keywords": ["machine learning"], "is_emerging_field": false, "field_emergence_year": null, "query_specificity": "broad"}'
            }
        }
    ],
    "usage": {"prompt_tokens": 200, "completion_tokens": 50},
}

# Query expansion response
EXPANSION_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": '{"concepts": [{"term": "machine learning", "importance": 0.95, "synonyms": ["ML", "statistical learning"], "related": ["artificial intelligence"], "foundational_works": ["A Few Useful Things to Know about Machine Learning"]}]}'
            }
        }
    ],
    "usage": {"prompt_tokens": 150, "completion_tokens": 80},
}

# LLM relevance scoring response (batch)
RELEVANCE_SCORING_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": '{"W2000000000": {"relevance": "HIGH", "type": "foundational"}, "W2000000001": {"relevance": "MEDIUM", "type": "methodology"}, "W2000000002": {"relevance": "LOW", "type": "application"}}'
            }
        }
    ],
    "usage": {"prompt_tokens": 500, "completion_tokens": 100},
}

# Subtopic generation response
SUBTOPIC_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": '{"subtopics": [{"label": "Gradient Boosting & XGBoost", "description": "Tree-based ensemble methods", "paper_ids": [0, 1, 2]}, {"label": "Neural Architecture Search", "description": "Automated model design", "paper_ids": [3, 4, 5]}, {"label": "Federated Learning Protocols", "description": "Distributed privacy-preserving training", "paper_ids": [6, 7]}, {"label": "Bayesian Optimization", "description": "Sample-efficient hyperparameter tuning", "paper_ids": [8, 9]}]}'
            }
        }
    ],
    "usage": {"prompt_tokens": 300, "completion_tokens": 150},
}

# Garbage / unparseable response
GARBAGE_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": "I'm sorry, I can't help with that. Here's a poem instead:\nRoses are red..."
            }
        }
    ],
    "usage": {"prompt_tokens": 50, "completion_tokens": 30},
}
