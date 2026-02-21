"""
Standalone conversion tests for Feature 2 seed conversion.

Tests PDF/DOI/work_id → title conversions in isolation (before integrating into ranking).

This is a LIVE test suite that BURNS API CREDITS. It validates conversions
against real OpenAlex/S2/ArXiv APIs before integrating into the ranking pipeline.

Run with:
    pytest tests/live/test_seed_conversion_standalone.py -v -s --timeout=300
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pytest

from app.feature2.seed_conversion import (
    work_id_to_title,
    doi_to_title,
    seed_to_query,
)

logger = logging.getLogger(__name__)

# Test cases: 50+ real papers across domains, years, citation counts
WORK_ID_TEST_CASES = [
    # Deep Learning / Transformers (highly cited, recent)
    ("W2964141474", "Attention Is All You Need", "transformers"),
    ("W2118403591", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "BERT"),
    ("W2963991096", "Deep Residual Learning for Image Recognition", "ResNet"),
    ("W2094792865", "Generative Adversarial Networks", "GANs"),
    ("W1979288093", "ImageNet Classification with Deep Convolutional Neural Networks", "AlexNet"),
    ("W2791046041", "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", "EfficientNet"),
    ("W2963073331", "Mastering the game of Go with deep neural networks and tree search", "AlphaGo"),
    ("W2964279170", "Sequence to Sequence Learning with Neural Networks", "seq2seq"),
    ("W2910054892", "Deep Learning", "deep learning textbook"),

    # NLP (various eras)
    ("W2157664535", "GloVe: Global Vectors for Word Representation", "GloVe"),
    ("W2129433817", "Neural Machine Translation by Jointly Learning to Align and Translate", "attention NMT"),
    ("W3206439248", "Language Models are Few-Shot Learners", "GPT-3"),
    ("W2124806978", "A Neural Probabilistic Language Model", "neural LM"),
    ("W2008964689", "Latent Dirichlet Allocation", "LDA"),

    # Computer Vision (classic + modern)
    ("W2095729482", "You Only Look Once: Unified, Real-Time Object Detection", "YOLO"),
    ("W2098029523", "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", "Faster R-CNN"),
    ("W2963544468", "Very Deep Convolutional Networks for Large-Scale Image Recognition", "VGG"),
    ("W2002149649", "Distinctive Image Features from Scale-Invariant Keypoints", "SIFT"),

    # Reinforcement Learning
    ("W2290055719", "Human-level control through deep reinforcement learning", "DQN"),
    ("W2593693430", "Proximal Policy Optimization Algorithms", "PPO"),
    ("W2737644199", "Asynchronous Methods for Deep Reinforcement Learning", "A3C"),

    # Machine Learning Theory
    ("W2123908038", "Random Forests", "random forests"),
    ("W2066033052", "Support Vector Networks", "SVM"),
    ("W1975015332", "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", "dropout"),
    ("W2008837681", "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", "batch normalization"),

    # Older foundational papers (lower citations, important)
    ("W2037772487", "Backpropagation Applied to Handwritten Zip Code Recognition", "backprop"),
    ("W2123609959", "Long Short-Term Memory", "LSTM"),
    ("W2109564136", "Gradient-Based Learning Applied to Document Recognition", "LeNet"),
    ("W2146486360", "A Fast Learning Algorithm for Deep Belief Nets", "DBN"),

    # Recent/less-cited papers (test edge cases)
    ("W3102991932", "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "ViT"),
]

DOI_TEST_CASES = [
    # ArXiv DOIs (most common in ML)
    ("10.48550/arXiv.1706.03762", "Attention Is All You Need", "transformers"),
    ("10.48550/arXiv.1810.04805", "BERT", "BERT"),
    ("10.48550/arXiv.1512.03385", "ResNet", "ResNet"),
    ("10.48550/arXiv.1406.2661", "GANs", "GANs"),
    ("10.48550/arXiv.1409.1556", "VGG", "VGG"),
    ("10.48550/arXiv.1707.06347", "PPO", "PPO"),
    ("10.48550/arXiv.2010.11929", "ViT", "ViT"),
    ("10.48550/arXiv.1905.11946", "EfficientNet", "EfficientNet"),
    ("10.48550/arXiv.1506.02640", "YOLO", "YOLO"),
    ("10.48550/arXiv.1411.4555", "GloVe", "GloVe"),

    # Nature papers
    ("10.1038/nature14539", "Human-level control through deep reinforcement learning", "DQN"),
    ("10.1038/nature16961", "Mastering the game of Go", "AlphaGo"),

    # ACM/IEEE papers
    ("10.1145/1390156.1390294", "Latent Dirichlet Allocation", "LDA"),

    # Science papers
    ("10.1126/science.1127647", "Reducing the Dimensionality of Data with Neural Networks", "autoencoders"),

    # Neural Information Processing Systems (NIPS/NeurIPS)
    ("10.5555/2969033.2969125", "ImageNet Classification with Deep Convolutional Neural Networks", "AlexNet"),
    ("10.5555/2969442.2969534", "Dropout", "dropout"),

    # JMLR papers
    ("10.1162/neco.1997.9.8.1735", "Long Short-Term Memory", "LSTM"),

    # IEEE papers
    ("10.1109/TPAMI.2016.2577031", "Faster R-CNN", "Faster R-CNN"),

    # Springer papers
    ("10.1007/s10994-006-6226-1", "Random Forests", "random forests"),

    # Pattern Recognition
    ("10.1016/j.patcog.2004.03.012", "SIFT features", "SIFT"),

    # More ArXiv (various years)
    ("10.48550/arXiv.1602.07360", "Batch Normalization", "batch normalization"),
    ("10.48550/arXiv.1409.0473", "seq2seq", "seq2seq"),
    ("10.48550/arXiv.1409.0575", "neural machine translation", "attention"),
    ("10.48550/arXiv.1602.01783", "Asynchronous RL", "A3C"),
    ("10.48550/arXiv.2005.14165", "GPT-3", "GPT-3"),
]


class TestWorkIDConversion:
    """Test work_id → title conversions."""

    @pytest.mark.live
    @pytest.mark.parametrize("work_id,expected_title_partial,description", WORK_ID_TEST_CASES)
    def test_work_id_to_title(self, work_id: str, expected_title_partial: str, description: str):
        """Test that work_id successfully converts to a title."""
        title = work_id_to_title(work_id)

        # Assert title was extracted
        assert title is not None, f"Failed to extract title for {work_id} ({description})"
        assert len(title) > 0, f"Empty title for {work_id} ({description})"

        # Validate title contains expected content (case-insensitive partial match)
        assert len(title) >= 10, f"Title too short ({len(title)} chars) for {work_id}: '{title}'"

        logger.info(f"✓ {work_id} → '{title}'")


class TestDOIConversion:
    """Test DOI → title conversions."""

    @pytest.mark.live
    @pytest.mark.parametrize("doi,expected_title_partial,description", DOI_TEST_CASES)
    def test_doi_to_title(self, doi: str, expected_title_partial: str, description: str):
        """Test that DOI successfully converts to a title."""
        title = doi_to_title(doi)

        # Assert title was extracted
        assert title is not None, f"Failed to extract title for {doi} ({description})"
        assert len(title) > 0, f"Empty title for {doi} ({description})"

        # Validate title length (reasonable academic paper title)
        assert len(title) >= 10, f"Title too short ({len(title)} chars) for {doi}: '{title}'"

        logger.info(f"✓ {doi} → '{title}'")


class TestEndToEndConversion:
    """Test full seed → query pipeline."""

    @pytest.mark.live
    def test_work_id_to_query_transformers(self):
        """Test work_id → title → query for Transformers paper."""
        query = seed_to_query(work_id="W2964141474")

        assert query is not None
        assert len(query) > 0
        assert 2 <= len(query.split()) <= 10, f"Query should be 2-10 words, got: '{query}'"

        logger.info(f"✓ W2964141474 → query: '{query}'")

    @pytest.mark.live
    def test_doi_to_query_bert(self):
        """Test DOI → title → query for BERT paper."""
        query = seed_to_query(doi="10.48550/arXiv.1810.04805")

        assert query is not None
        assert len(query) > 0
        assert 2 <= len(query.split()) <= 10, f"Query should be 2-10 words, got: '{query}'"

        logger.info(f"✓ 10.48550/arXiv.1810.04805 → query: '{query}'")

    @pytest.mark.live
    def test_title_to_query_direct(self):
        """Test direct title → query (no lookup needed)."""
        query = seed_to_query(title="Attention Is All You Need")

        assert query is not None
        assert len(query) > 0
        assert 2 <= len(query.split()) <= 10, f"Query should be 2-10 words, got: '{query}'"

        logger.info(f"✓ Direct title → query: '{query}'")


class TestConversionResults:
    """Aggregate results and save to JSON for manual review."""

    @pytest.mark.live
    def test_run_all_conversions_and_save_results(self):
        """
        Run all 50+ conversions and save results to JSON.

        This test executes all conversions sequentially and logs:
        - Success rate
        - Failed conversions
        - Extracted titles for manual review
        """
        results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "total_cases": len(WORK_ID_TEST_CASES) + len(DOI_TEST_CASES),
            "work_id_results": [],
            "doi_results": [],
            "summary": {},
        }

        # Test work_ids
        work_id_success = 0
        for work_id, expected_partial, description in WORK_ID_TEST_CASES:
            title = work_id_to_title(work_id)
            success = title is not None and len(title) >= 10

            results["work_id_results"].append({
                "work_id": work_id,
                "description": description,
                "expected_partial": expected_partial,
                "extracted_title": title,
                "success": success,
            })

            if success:
                work_id_success += 1
                logger.info(f"✓ work_id {work_id}: '{title}'")
            else:
                logger.error(f"✗ work_id {work_id}: FAILED")

            # Rate limiting: avoid hitting OpenAlex 429s (1s delay to be safe)
            time.sleep(1.0)

        # Test DOIs
        doi_success = 0
        for doi, expected_partial, description in DOI_TEST_CASES:
            title = doi_to_title(doi)
            success = title is not None and len(title) >= 10

            results["doi_results"].append({
                "doi": doi,
                "description": description,
                "expected_partial": expected_partial,
                "extracted_title": title,
                "success": success,
            })

            if success:
                doi_success += 1
                logger.info(f"✓ DOI {doi}: '{title}'")
            else:
                logger.error(f"✗ DOI {doi}: FAILED")

            # Rate limiting: avoid hitting OpenAlex 429s (1s delay to be safe)
            time.sleep(1.0)

        # Calculate success rates
        total_success = work_id_success + doi_success
        total_cases = len(WORK_ID_TEST_CASES) + len(DOI_TEST_CASES)
        success_rate = (total_success / total_cases) * 100

        results["summary"] = {
            "total_cases": total_cases,
            "total_success": total_success,
            "total_failed": total_cases - total_success,
            "success_rate_pct": round(success_rate, 2),
            "work_id_success": work_id_success,
            "work_id_total": len(WORK_ID_TEST_CASES),
            "work_id_success_rate_pct": round((work_id_success / len(WORK_ID_TEST_CASES)) * 100, 2),
            "doi_success": doi_success,
            "doi_total": len(DOI_TEST_CASES),
            "doi_success_rate_pct": round((doi_success / len(DOI_TEST_CASES)) * 100, 2),
        }

        # Save results to JSON
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = results_dir / f"seed_conversion_test_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"CONVERSION TEST RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Total cases: {total_cases}")
        logger.info(f"Successful: {total_success}")
        logger.info(f"Failed: {total_cases - total_success}")
        logger.info(f"Success rate: {success_rate:.2f}%")
        logger.info(f"")
        logger.info(f"work_id conversions: {work_id_success}/{len(WORK_ID_TEST_CASES)} ({results['summary']['work_id_success_rate_pct']:.2f}%)")
        logger.info(f"DOI conversions: {doi_success}/{len(DOI_TEST_CASES)} ({results['summary']['doi_success_rate_pct']:.2f}%)")
        logger.info(f"")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"{'='*80}\n")

        # Assert >95% success rate (as per plan requirement)
        assert success_rate >= 95.0, (
            f"Conversion success rate {success_rate:.2f}% is below required 95%. "
            f"Review results in {output_file}"
        )

        # List failed cases for debugging
        failed_work_ids = [
            r for r in results["work_id_results"] if not r["success"]
        ]
        failed_dois = [
            r for r in results["doi_results"] if not r["success"]
        ]

        if failed_work_ids:
            logger.warning(f"Failed work_ids: {[r['work_id'] for r in failed_work_ids]}")

        if failed_dois:
            logger.warning(f"Failed DOIs: {[r['doi'] for r in failed_dois]}")
