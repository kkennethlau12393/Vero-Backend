"""
Title-to-query inference for Feature 2.

This module generates search queries from paper titles for the "Research this topic"
feature when a node doesn't have an OpenAlex topic_id.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# Groq Llama 4 Maverick - fast and high quality
MODEL_VERSION = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


TITLE_TO_QUERY_PROMPT = """Given this academic paper title, extract the core research topic as a search query.

Paper title: "{title}"

Return ONLY a concise search query (2-5 words) that captures the main research topic or methodology.

Examples:
- "Attention Is All You Need" → "transformer attention mechanism"
- "Deep Residual Learning for Image Recognition" → "residual neural networks"
- "BERT: Pre-training of Deep Bidirectional Transformers" → "BERT language models"
- "ImageNet Classification with Deep Convolutional Neural Networks" → "convolutional neural networks"
- "A Neural Probabilistic Language Model" → "neural language models"
- "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" → "dropout regularization"

Guidelines:
- Focus on the methodology or technique, not the application
- Use common terms researchers would search for
- Be specific enough to be meaningful but not overly narrow
- Output ONLY the search query, no quotes or explanations"""


def generate_topic_query_from_title(title: str) -> Optional[str]:
    """
    Generate a research topic search query from a paper title.

    This is used for "Research this topic" when the paper doesn't have
    an OpenAlex topic_id. The LLM extracts the core methodology/topic
    from the title to create an effective search query.

    Args:
        title: The paper title

    Returns:
        A concise search query (2-5 words) or None if inference fails
    """
    if not title or len(title) < 5:
        return None

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, falling back to title as query")
        return _fallback_clean_title(title)

    try:
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

        resp = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[
                {
                    "role": "system",
                    "content": "You extract research topics from paper titles. Return only the topic query, nothing else.",
                },
                {
                    "role": "user",
                    "content": TITLE_TO_QUERY_PROMPT.format(title=title),
                },
            ],
            max_tokens=50,
            temperature=0.1,
            timeout=15.0,
        )

        query = (resp.choices[0].message.content or "").strip()

        # Remove quotes if LLM added them
        query = query.strip('"\'')

        # Validate: should be 2-8 words
        word_count = len(query.split())
        if word_count < 1 or word_count > 10:
            logger.warning(f"LLM query too short/long ({word_count} words): '{query}'")
            return _fallback_clean_title(title)

        logger.info(f"Generated query from title: '{title[:50]}...' → '{query}'")
        return query

    except Exception as e:
        logger.warning(f"LLM title-to-query error: {e}")
        return _fallback_clean_title(title)


def _fallback_clean_title(title: str) -> str:
    """
    Fallback: clean up the title to use as a query.

    Removes common filler words and truncates to reasonable length.
    """
    # Remove common academic title patterns
    stop_words = {
        "a", "an", "the", "of", "for", "in", "on", "to", "with", "and", "or",
        "using", "via", "towards", "toward", "based", "learning", "approach",
        "novel", "new", "improved", "efficient", "simple", "way", "method",
    }

    words = title.lower().split()
    filtered = [w.strip(",:;.!?()[]{}") for w in words if w.lower() not in stop_words]

    # Take first 5 meaningful words
    query = " ".join(filtered[:5])

    logger.info(f"Fallback query from title: '{title[:50]}...' → '{query}'")
    return query if query else title[:50]
