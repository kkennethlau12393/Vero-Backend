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


QUERY_TO_TITLE_PROMPT = """Extract the core research topic from this user query. Return a concise title (2-6 words) suitable as a header.

User query: "{query}"

Examples:
- "What are the best papers on CRISPR gene editing?" → "CRISPR Gene Editing"
- "Can you find me research on transformer architectures?" → "Transformer Architectures"
- "How does reinforcement learning work in robotics?" → "Reinforcement Learning in Robotics"
- "I want to learn about graph neural networks" → "Graph Neural Networks"
- "attention mechanisms in transformers" → "Attention Mechanisms in Transformers"
- "show me papers about protein folding" → "Protein Folding"

Guidelines:
- Strip conversational fluff (find me, show me, best papers on, etc.)
- Keep the core research topic
- Use Title Case
- Output ONLY the title, no quotes or explanations"""


def extract_display_title(query_text: str) -> str:
    """Extract a clean display title from a user query for workspace headers.

    Completely separate from the ranking pipeline — this is only used
    for UI display purposes. Falls back to regex cleaning if LLM fails.
    """
    if not query_text or len(query_text.strip()) < 3:
        return query_text or ""

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _fallback_display_title(query_text)

    try:
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

        resp = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[
                {
                    "role": "system",
                    "content": "You extract research topics from user queries. Return only a concise title in Title Case.",
                },
                {
                    "role": "user",
                    "content": QUERY_TO_TITLE_PROMPT.format(query=query_text),
                },
            ],
            max_tokens=30,
            temperature=0.0,
            timeout=10.0,
        )

        title = (resp.choices[0].message.content or "").strip().strip('"\'')

        word_count = len(title.split())
        if word_count < 1 or word_count > 8:
            logger.warning(f"Display title too short/long ({word_count} words): '{title}'")
            return _fallback_display_title(query_text)

        logger.info(f"Display title: '{query_text[:50]}' → '{title}'")
        return title

    except Exception as e:
        logger.warning(f"Display title LLM error: {e}")
        return _fallback_display_title(query_text)


def _fallback_display_title(query_text: str) -> str:
    """Regex fallback for display title extraction."""
    import re
    cleaned = query_text.strip()
    # Strip trailing ?
    cleaned = re.sub(r'\?$', '', cleaned).strip()
    # Strip conversational prefixes (run before question-word strip)
    cleaned = re.sub(
        r'^(?:can\s+you\s+|could\s+you\s+|please\s+)?'
        r'(?:find\s+me|show\s+me|tell\s+me\s+about|give\s+me|get\s+me'
        r'|help\s+me\s+(?:find|understand|learn\s+about))\s+',
        '', cleaned, flags=re.IGNORECASE,
    )
    # Strip question prefixes (what are the, how does, etc.)
    cleaned = re.sub(
        r'^(?:what|how|why|when|where|which|who|can|could|should|would|does|do|is|are)\s+'
        r'(?:is|are|does|do|was|were|will|would|could|should|can)?\s*'
        r'(?:the|a|an|some)?\s*',
        '', cleaned, flags=re.IGNORECASE,
    )
    # Strip "I want to learn about", "I need to understand"
    cleaned = re.sub(
        r'^i\s+(?:want|need|\'d\s+like)\s+to\s+(?:learn|know|read|find\s+out|understand|study)\s+'
        r'(?:about|more\s+about)?\s*',
        '', cleaned, flags=re.IGNORECASE,
    )
    # Strip "best/top/latest papers on", "the top papers on", "research on/about"
    cleaned = re.sub(
        r'^(?:the\s+)?(?:best|top|latest|recent|good|important|key|seminal)?\s*'
        r'(?:papers?|articles?|research|studies|work|literature)\s+'
        r'(?:on|about|in|for|regarding)\s+',
        '', cleaned, flags=re.IGNORECASE,
    )
    # Strip leftover prepositions at start
    cleaned = re.sub(
        r'^(?:about|on|in|for|regarding)\s+',
        '', cleaned, flags=re.IGNORECASE,
    )
    # Strip "work in" from "reinforcement learning work in robotics"
    cleaned = re.sub(r'\s+work\s+in\s+', ' in ', cleaned, flags=re.IGNORECASE)
    # Title case
    return cleaned.strip().title() if cleaned.strip() else query_text.strip()


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
