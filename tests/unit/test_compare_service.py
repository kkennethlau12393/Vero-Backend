"""
Unit tests for app/feature4/compare_service.py

Pure logic tests: no DB, no network.
Tests the self-reference scrubber, depth validator, shallow pattern detection,
and _call_llm retry/fallback logic.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.feature4.compare_service import (
    _scrub_self_references,
    _validate_depth,
    _validate_synthesis,
    _SHALLOW_PATTERN,
    _extract_search_terms_from_gaps,
    _extract_validation_metrics,
    _detect_paper_type,
    _compute_paper_overlap,
    _call_llm,
)


@pytest.mark.unit
class TestScrubSelfReferences:
    """Test that self-referencing work_ids are replaced with 'this paper'."""

    def test_replaces_own_work_id_in_handles_well(self):
        synthesis = {
            "strengths_weaknesses_matrix": [
                {
                    "work_id": "W12345",
                    "title": "Test Paper",
                    "handles_well": [
                        {
                            "capability": "Image generation",
                            "mechanism": "W12345 uses a U-Net architecture",
                            "evidence": "W12345 achieves state-of-the-art FID",
                        }
                    ],
                    "struggles_with": [],
                    "assumptions": [],
                    "complemented_by": [],
                }
            ]
        }
        result = _scrub_self_references(synthesis)
        hw = result["strengths_weaknesses_matrix"][0]["handles_well"][0]
        assert "W12345" not in hw["mechanism"]
        assert "this paper" in hw["mechanism"]
        assert "W12345" not in hw["evidence"]
        assert "this paper" in hw["evidence"]

    def test_replaces_bracketed_work_id(self):
        synthesis = {
            "strengths_weaknesses_matrix": [
                {
                    "work_id": "W99999",
                    "title": "Test",
                    "handles_well": [
                        {
                            "capability": "X",
                            "mechanism": "[W99999] introduces a novel loss",
                            "evidence": "As shown by (W99999)",
                        }
                    ],
                    "struggles_with": [],
                    "assumptions": [],
                    "complemented_by": [],
                }
            ]
        }
        result = _scrub_self_references(synthesis)
        hw = result["strengths_weaknesses_matrix"][0]["handles_well"][0]
        assert "W99999" not in hw["mechanism"]
        assert "W99999" not in hw["evidence"]

    def test_preserves_other_work_ids(self):
        synthesis = {
            "strengths_weaknesses_matrix": [
                {
                    "work_id": "W11111",
                    "title": "Paper A",
                    "handles_well": [
                        {
                            "capability": "X",
                            "mechanism": "Unlike W22222, this uses attention",
                            "evidence": "Outperforms W22222 by 5%",
                        }
                    ],
                    "struggles_with": [],
                    "assumptions": [],
                    "complemented_by": [],
                }
            ]
        }
        result = _scrub_self_references(synthesis)
        hw = result["strengths_weaknesses_matrix"][0]["handles_well"][0]
        assert "W22222" in hw["mechanism"]
        assert "W22222" in hw["evidence"]

    def test_does_not_touch_complemented_by(self):
        synthesis = {
            "strengths_weaknesses_matrix": [
                {
                    "work_id": "W11111",
                    "title": "Paper A",
                    "handles_well": [],
                    "struggles_with": [],
                    "assumptions": [],
                    "complemented_by": [
                        {
                            "other_work_id": "W22222",
                            "coverage": "W22222 addresses the gap by using diffusion",
                        }
                    ],
                }
            ]
        }
        result = _scrub_self_references(synthesis)
        comp = result["strengths_weaknesses_matrix"][0]["complemented_by"][0]
        assert "W22222" in comp["coverage"]

    def test_scrubs_struggles_with(self):
        synthesis = {
            "strengths_weaknesses_matrix": [
                {
                    "work_id": "W55555",
                    "title": "Paper X",
                    "handles_well": [],
                    "struggles_with": [
                        {
                            "limitation": "W55555 has limited scalability",
                            "cause": "W55555 uses O(n^2) attention",
                            "consequence": "W55555 cannot process long sequences",
                        }
                    ],
                    "assumptions": [],
                    "complemented_by": [],
                }
            ]
        }
        result = _scrub_self_references(synthesis)
        sw = result["strengths_weaknesses_matrix"][0]["struggles_with"][0]
        assert "W55555" not in sw["limitation"]
        assert "W55555" not in sw["cause"]
        assert "W55555" not in sw["consequence"]
        assert "this paper" in sw["limitation"]

    def test_scrubs_assumptions(self):
        synthesis = {
            "strengths_weaknesses_matrix": [
                {
                    "work_id": "W77777",
                    "title": "Paper Y",
                    "handles_well": [],
                    "struggles_with": [],
                    "assumptions": [
                        {
                            "assumption": "W77777 assumes data is normally distributed",
                            "if_violated": "W77777 performance degrades significantly",
                        }
                    ],
                    "complemented_by": [],
                }
            ]
        }
        result = _scrub_self_references(synthesis)
        assum = result["strengths_weaknesses_matrix"][0]["assumptions"][0]
        assert "W77777" not in assum["assumption"]
        assert "W77777" not in assum["if_violated"]

    def test_scrubs_single_paper_paradigm(self):
        synthesis = {
            "convergence_divergence": {
                "paradigms": [
                    {
                        "name": "Attention-based",
                        "papers": ["W33333"],
                        "mechanism": "W33333 uses multi-head self-attention",
                        "philosophy": "W33333 believes attention is all you need",
                    }
                ]
            },
            "strengths_weaknesses_matrix": [],
        }
        result = _scrub_self_references(synthesis)
        para = result["convergence_divergence"]["paradigms"][0]
        assert "W33333" not in para["mechanism"]
        assert "W33333" not in para["philosophy"]
        assert "this paper" in para["mechanism"]

    def test_multi_paper_paradigm_not_scrubbed(self):
        synthesis = {
            "convergence_divergence": {
                "paradigms": [
                    {
                        "name": "GAN-based",
                        "papers": ["W11111", "W22222"],
                        "mechanism": "W11111 and W22222 use adversarial training",
                        "philosophy": "Learning through competition",
                    }
                ]
            },
            "strengths_weaknesses_matrix": [],
        }
        result = _scrub_self_references(synthesis)
        para = result["convergence_divergence"]["paradigms"][0]
        # Multi-paper paradigms should NOT be scrubbed
        assert "W11111" in para["mechanism"]
        assert "W22222" in para["mechanism"]


@pytest.mark.unit
class TestValidateDepth:
    """Test that shallow content is caught by depth validation."""

    def test_catches_shallow_mechanism(self):
        result = {
            "strengths_weaknesses_matrix": [
                {
                    "work_id": "W1",
                    "handles_well": [
                        {
                            "capability": "Image generation",
                            "mechanism": "Uses deep learning",
                            "evidence": "Good results on benchmarks",
                        }
                    ],
                    "struggles_with": [],
                    "complemented_by": [],
                }
            ],
            "recommendation": {"decision_matrix": []},
        }
        violations = _validate_depth(result)
        assert any("mechanism" in v and "shallow" in v.lower() for v in violations)

    def test_catches_shallow_coverage(self):
        result = {
            "strengths_weaknesses_matrix": [
                {
                    "work_id": "W1",
                    "handles_well": [],
                    "struggles_with": [],
                    "complemented_by": [
                        {
                            "other_work_id": "W2",
                            "coverage": "Provides a method for editing",
                        }
                    ],
                }
            ],
            "recommendation": {"decision_matrix": []},
        }
        violations = _validate_depth(result)
        assert any("coverage" in v and "shallow" in v.lower() for v in violations)

    def test_catches_shallow_recommendation_why(self):
        result = {
            "strengths_weaknesses_matrix": [],
            "recommendation": {
                "decision_matrix": [
                    {
                        "scenario": "High-quality image generation",
                        "use": "W1",
                        "why": "Better quality",
                    }
                ]
            },
        }
        violations = _validate_depth(result)
        assert any("why" in v and ("shallow" in v.lower() or "too short" in v.lower()) for v in violations)

    def test_passes_deep_content(self):
        result = {
            "strengths_weaknesses_matrix": [
                {
                    "work_id": "W1",
                    "handles_well": [
                        {
                            "capability": "Training very deep networks",
                            "mechanism": (
                                "Identity shortcuts provide a gradient highway that "
                                "bypasses non-linear transformations, allowing gradients "
                                "to flow directly through skip connections"
                            ),
                            "evidence": (
                                "Successfully trained 152-layer network on ImageNet "
                                "achieving 3.57% top-5 error rate on the test set"
                            ),
                        }
                    ],
                    "struggles_with": [
                        {
                            "limitation": "Feature reuse across distant layers",
                            "cause": (
                                "Each residual block only accesses the output of the "
                                "immediately previous block plus the skip connection"
                            ),
                            "consequence": (
                                "Features from early convolutional layers (edges, textures) "
                                "are not directly available to later classification layers"
                            ),
                        }
                    ],
                    "complemented_by": [
                        {
                            "other_work_id": "W2",
                            "coverage": (
                                "DenseNet concatenates all preceding feature maps as input "
                                "to each layer, providing O(L) direct connections vs ResNet's "
                                "O(1) skip connections, enabling full feature reuse across depths"
                            ),
                        }
                    ],
                }
            ],
            "recommendation": {
                "decision_matrix": [
                    {
                        "scenario": "Training 100+ layer networks on limited GPU memory",
                        "use": "W1",
                        "why": (
                            "Additive residual shortcuts have O(1) memory overhead per block "
                            "compared to DenseNet's O(L) concatenation, enabling a 152-layer "
                            "ResNet to fit within 12GB GPU memory. By using identity mappings "
                            "that add rather than concatenate, each block reuses the same "
                            "activation memory, achieving 3.57% top-5 error on ImageNet "
                            "without exceeding standard GPU memory constraints."
                        ),
                    }
                ]
            },
        }
        violations = _validate_depth(result)
        assert violations == []

    def test_catches_vague_provides_method(self):
        result = {
            "strengths_weaknesses_matrix": [
                {
                    "work_id": "W1",
                    "handles_well": [
                        {
                            "capability": "Editing",
                            "mechanism": "Provides a method for editing images using diffusion",
                            "evidence": "Works on various datasets",
                        }
                    ],
                    "struggles_with": [],
                    "complemented_by": [],
                }
            ],
            "recommendation": {"decision_matrix": []},
        }
        violations = _validate_depth(result)
        assert any("vague" in v.lower() for v in violations)

    def test_catches_vague_increased_computational(self):
        result = {
            "strengths_weaknesses_matrix": [
                {
                    "work_id": "W1",
                    "handles_well": [],
                    "struggles_with": [
                        {
                            "limitation": "Scalability",
                            "cause": "Architecture leads to increased computational requirements for inference",
                            "consequence": "Processing large images requires significant GPU resources and processing time",
                        }
                    ],
                    "complemented_by": [],
                }
            ],
            "recommendation": {"decision_matrix": []},
        }
        violations = _validate_depth(result)
        assert any("vague" in v.lower() for v in violations)


@pytest.mark.unit
class TestShallowPatterns:
    """Test the shallow pattern regex directly."""

    def test_matches_provides_method(self):
        assert _SHALLOW_PATTERN.search("provides a method for editing")
        assert _SHALLOW_PATTERN.search("provide a technique for generation")
        assert _SHALLOW_PATTERN.search("provides the approach for training")

    def test_matches_vague_quality(self):
        assert _SHALLOW_PATTERN.search("high-quality images")
        assert _SHALLOW_PATTERN.search("better quality results")
        assert _SHALLOW_PATTERN.search("low quality output")

    def test_matches_vague_computational(self):
        assert _SHALLOW_PATTERN.search("increased computational requirements")
        assert _SHALLOW_PATTERN.search("improved performance on benchmarks")
        assert _SHALLOW_PATTERN.search("reduced efficiency in practice")

    def test_does_not_match_specific(self):
        assert not _SHALLOW_PATTERN.search(
            "uses DDPM inversion to find the noise map"
        )
        assert not _SHALLOW_PATTERN.search(
            "cascading through 3 upsampling stages (64→256→1024)"
        )
        assert not _SHALLOW_PATTERN.search(
            "identity shortcuts provide gradient highway"
        )


@pytest.mark.unit
class TestExtractSearchTermsFromGaps:
    """Test search term extraction from fingerprint limitations."""

    def test_extracts_consequence_after_arrow(self):
        fp = {
            "limitations": [
                "Cascaded architecture → requires 3× inference time",
                "Pixel-aligned data → fails on unpaired images",
            ],
            "domain": "Computer Vision",
        }
        terms = _extract_search_terms_from_gaps(fp)
        assert "Computer Vision" in terms
        assert "inference time" in terms
        assert "unpaired images" in terms

    def test_handles_no_arrow(self):
        fp = {
            "limitations": ["Limited to supervised learning settings"],
            "domain": "NLP",
        }
        terms = _extract_search_terms_from_gaps(fp)
        assert "NLP" in terms
        assert "supervised learning" in terms

    def test_handles_empty_limitations(self):
        fp = {"limitations": [], "domain": "ML"}
        terms = _extract_search_terms_from_gaps(fp)
        assert "ML" in terms

    def test_handles_no_domain(self):
        fp = {"limitations": ["Slow convergence → 10x training time"]}
        terms = _extract_search_terms_from_gaps(fp)
        assert "training time" in terms


@pytest.mark.unit
class TestDetectPaperType:
    """Test survey vs methodology detection from fingerprints."""

    def test_detects_survey_from_approach(self):
        fp = {
            "approach": "This paper surveys existing methods for image generation",
            "limitations": [],
        }
        assert _detect_paper_type(fp) == "survey"

    def test_detects_review_from_approach(self):
        fp = {
            "approach": "A comprehensive review of transformer architectures",
            "limitations": [],
        }
        assert _detect_paper_type(fp) == "survey"

    def test_detects_taxonomy_from_approach(self):
        fp = {
            "approach": "Provides a taxonomy of reinforcement learning algorithms",
            "limitations": [],
        }
        assert _detect_paper_type(fp) == "survey"

    def test_detects_survey_from_limitations(self):
        fp = {
            "approach": "Analyzes deep learning methods",
            "limitations": ["Limited scope of review → misses recent methods"],
        }
        assert _detect_paper_type(fp) == "survey"

    def test_methodology_paper(self):
        fp = {
            "approach": "Proposes a residual learning framework with identity shortcuts",
            "limitations": ["Fixed topology → cannot adapt depth at inference time"],
        }
        assert _detect_paper_type(fp) == "methodology"

    def test_empty_fingerprint(self):
        assert _detect_paper_type({}) == "methodology"

    def test_overview_keyword(self):
        fp = {
            "approach": "Presents an overview of deep learning techniques for NLP",
            "limitations": [],
        }
        assert _detect_paper_type(fp) == "survey"


@pytest.mark.unit
class TestComputePaperOverlap:
    """Test keyword overlap computation between fingerprints."""

    def test_identical_fingerprints(self):
        fp = {
            "approach": "deep residual learning framework",
            "domain": "computer vision",
            "key_components": ["residual block", "skip connection"],
        }
        assert _compute_paper_overlap(fp, fp) == 1.0

    def test_disjoint_fingerprints(self):
        fp1 = {
            "approach": "transformer attention mechanism for language modeling",
            "domain": "natural language processing",
            "key_components": ["multi-head attention"],
        }
        fp2 = {
            "approach": "lithium battery degradation electrochemical",
            "domain": "materials science",
            "key_components": ["cathode cycling"],
        }
        overlap = _compute_paper_overlap(fp1, fp2)
        assert overlap < 0.10

    def test_related_fingerprints(self):
        fp1 = {
            "approach": "deep convolutional neural network for image classification",
            "domain": "computer vision",
            "key_components": ["convolutional layer", "batch normalization"],
        }
        fp2 = {
            "approach": "dense convolutional network for image recognition",
            "domain": "computer vision",
            "key_components": ["dense block", "batch normalization"],
        }
        overlap = _compute_paper_overlap(fp1, fp2)
        assert overlap > 0.10

    def test_empty_fingerprints(self):
        assert _compute_paper_overlap({}, {}) == 0.0

    def test_one_empty(self):
        fp = {
            "approach": "deep learning model",
            "domain": "ML",
            "key_components": ["attention"],
        }
        assert _compute_paper_overlap(fp, {}) == 0.0


@pytest.mark.unit
class TestCircularComplementValidation:
    """Test that circular complements (A→B, B→A) are caught."""

    def test_catches_circular_complement(self):
        result = {
            "convergence_divergence": {
                "common_problem": {"domain": "CV", "challenge": "X", "why_hard": "Y"},
                "paradigms": [{"name": "P1", "papers": ["W1"], "mechanism": "M", "philosophy": "Ph"}],
                "divergence_summary": "Summary",
            },
            "strengths_weaknesses_matrix": [
                {
                    "work_id": "W1",
                    "handles_well": [{"capability": "X", "mechanism": "Uses attention mechanism with Q/K/V projections", "evidence": "Achieves SOTA on benchmark"}],
                    "struggles_with": [{"limitation": "L", "cause": "C is caused by design choice of fixed topology", "consequence": "Cannot adapt at runtime inference"}],
                    "assumptions": [{"assumption": "Data is IID distributed", "if_violated": "Performance degrades on non-IID data"}],
                    "complemented_by": [{"other_work_id": "W2", "coverage": "W2 addresses this gap by introducing dynamic topology through neural architecture search with evolutionary algorithms"}],
                },
                {
                    "work_id": "W2",
                    "handles_well": [{"capability": "Y", "mechanism": "Employs dense concatenation of feature maps from all preceding layers", "evidence": "Reduces parameters by 3x on CIFAR-10"}],
                    "struggles_with": [{"limitation": "M", "cause": "Dense connections cause quadratic memory growth O(L^2)", "consequence": "Cannot fit networks deeper than 100 layers in 12GB GPU memory"}],
                    "assumptions": [{"assumption": "Feature reuse across layers is beneficial", "if_violated": "Redundant features waste capacity in the model"}],
                    "complemented_by": [{"other_work_id": "W1", "coverage": "W1 addresses the memory issue by using additive identity shortcuts with constant O(1) memory per block"}],
                },
            ],
            "recommendation": {
                "summary": "Use W1 for memory-constrained settings, W2 for parameter-efficient models",
                "decision_matrix": [
                    {"scenario": "S1", "use": "W1", "why": "Constant memory overhead per block allows scaling to 152+ layers on a single GPU"},
                    {"scenario": "S2", "use": "W2", "why": "Dense connections achieve equivalent accuracy with 3x fewer parameters via feature reuse"},
                    {"scenario": "S3", "use": "W1", "why": "Identity shortcuts enable training 1000-layer networks on CIFAR-10 without degradation"},
                ],
                "can_combine": False,
            },
        }
        violations = _validate_synthesis(result, ["W1", "W2"])
        assert any("Circular complement" in v for v in violations)

    def test_no_circular_when_different_complements(self):
        result = {
            "convergence_divergence": {
                "common_problem": {"domain": "CV", "challenge": "X", "why_hard": "Y"},
                "paradigms": [{"name": "P1", "papers": ["W1"], "mechanism": "M", "philosophy": "Ph"}],
                "divergence_summary": "Summary",
            },
            "strengths_weaknesses_matrix": [
                {
                    "work_id": "W1",
                    "handles_well": [{"capability": "X", "mechanism": "Uses attention mechanism with Q/K/V projections", "evidence": "Achieves SOTA on benchmark"}],
                    "struggles_with": [{"limitation": "L", "cause": "Fixed depth topology prevents runtime adaptation", "consequence": "Cannot handle variable-length sequences efficiently"}],
                    "assumptions": [{"assumption": "Input dimension is fixed", "if_violated": "Requires padding or truncation"}],
                    "complemented_by": [{"other_work_id": "W3", "coverage": "W3 introduces dynamic depth via early-exit mechanisms, allowing runtime adaptation of compute"}],
                },
                {
                    "work_id": "W2",
                    "handles_well": [{"capability": "Y", "mechanism": "Dense feature concatenation from all prior layers", "evidence": "3x parameter reduction on CIFAR"}],
                    "struggles_with": [{"limitation": "M", "cause": "O(L^2) memory from dense connections", "consequence": "Cannot scale beyond 100 layers in 12GB GPU memory"}],
                    "assumptions": [{"assumption": "Feature reuse is beneficial", "if_violated": "Redundant features waste model capacity"}],
                    "complemented_by": [{"other_work_id": "W4", "coverage": "W4 uses memory-efficient dense connections via checkpoint gradient computation"}],
                },
            ],
            "recommendation": {
                "summary": "Use W1 for deep training, W2 for parameter efficiency",
                "decision_matrix": [
                    {"scenario": "S1", "use": "W1", "why": "Constant O(1) memory per block enables 152-layer networks on single GPU"},
                    {"scenario": "S2", "use": "W2", "why": "Dense connections achieve equivalent accuracy with 3x fewer parameters"},
                    {"scenario": "S3", "use": "W1", "why": "Identity shortcuts allow 1000-layer networks without gradient degradation"},
                ],
                "can_combine": False,
            },
        }
        violations = _validate_synthesis(result, ["W1", "W2"], valid_complement_wids={"W1", "W2", "W3", "W4"})
        assert not any("Circular complement" in v for v in violations)


@pytest.mark.unit
class TestExpandedShallowPatterns:
    """Test new shallow patterns added in v3-robust."""

    def test_matches_significantly_more(self):
        assert _SHALLOW_PATTERN.search("significantly more efficient")
        assert _SHALLOW_PATTERN.search("significantly better results")
        assert _SHALLOW_PATTERN.search("significantly lower error")

    def test_matches_leads_to_better(self):
        assert _SHALLOW_PATTERN.search("leads to better performance")
        assert _SHALLOW_PATTERN.search("results in improved accuracy")
        assert _SHALLOW_PATTERN.search("leads to degraded quality")

    def test_matches_more_efficient(self):
        assert _SHALLOW_PATTERN.search("more efficient than baseline")
        assert _SHALLOW_PATTERN.search("less effective in practice")
        assert _SHALLOW_PATTERN.search("more robust to noise")

    def test_still_matches_original_patterns(self):
        assert _SHALLOW_PATTERN.search("provides a method for editing")
        assert _SHALLOW_PATTERN.search("increased computational requirements")
        assert _SHALLOW_PATTERN.search("high-quality images")

    def test_does_not_match_specific_claims(self):
        assert not _SHALLOW_PATTERN.search(
            "uses O(n log n) attention via locality-sensitive hashing"
        )
        assert not _SHALLOW_PATTERN.search(
            "3-stage cascade requires 3 sequential U-Net forward passes"
        )
        assert not _SHALLOW_PATTERN.search(
            "PatchGAN discriminator classifies 70x70 pixel patches"
        )

    def test_does_not_match_significantly_in_isolation(self):
        # "significantly" alone should not match — only with vague comparators
        assert not _SHALLOW_PATTERN.search(
            "The architecture differs significantly in its use of bottleneck blocks"
        )


def _make_llm_response(content: str):
    """Build a mock Groq response object."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.unit
class TestCallLlmBestAttemptFallback:
    """Tests for _call_llm best-attempt fallback and progressive relaxation."""

    @patch("app.feature4.compare_service.time.sleep")
    @patch("app.feature4.compare_service._get_client")
    @patch("app.feature4.compare_service.extract_json_from_llm_response")
    def test_returns_best_attempt_on_json_parse_failure_last_retry(
        self, mock_extract, mock_client, mock_sleep
    ):
        """If attempt 1 parses OK (with violations) but attempts 2-4 fail JSON parse,
        return the best-attempt result instead of None."""
        client = MagicMock()
        mock_client.return_value = client
        client.chat.completions.create.return_value = _make_llm_response('{"ok": true}')

        good_result = {"paradigms": [{"name": "A"}], "sw_matrix": []}
        # Attempt 0: parses, has violations
        # Attempt 1-3: JSON parse fails
        mock_extract.side_effect = [
            (good_result, None),   # attempt 0: parses OK
            (None, "bad json"),    # attempt 1: parse fail
            (None, "bad json"),    # attempt 2: parse fail
            (None, "bad json"),    # attempt 3: parse fail
        ]

        def validator(result):
            return ["some violation"]

        result = _call_llm("sys", "usr", validator=validator, validator_args=())
        assert result is good_result

    @patch("app.feature4.compare_service.time.sleep")
    @patch("app.feature4.compare_service._get_client")
    @patch("app.feature4.compare_service.extract_json_from_llm_response")
    def test_returns_result_with_fewest_violations(
        self, mock_extract, mock_client, mock_sleep
    ):
        """When multiple attempts have violations, return the one with fewest."""
        client = MagicMock()
        mock_client.return_value = client
        client.chat.completions.create.return_value = _make_llm_response('{}')

        result_3v = {"attempt": 0, "data": "three_violations"}
        result_1v = {"attempt": 1, "data": "one_violation"}
        result_2v = {"attempt": 2, "data": "two_violations"}
        result_4v = {"attempt": 3, "data": "four_violations"}

        mock_extract.side_effect = [
            (result_3v, None),
            (result_1v, None),
            (result_2v, None),
            (result_4v, None),
        ]

        call_count = [0]

        def validator(result):
            c = call_count[0]
            call_count[0] += 1
            violations_per_attempt = [
                ["v1", "v2", "v3"],           # attempt 0: 3 violations
                ["v1"],                        # attempt 1: 1 violation (BEST)
                ["v1", "v2"],                  # attempt 2: 2 violations
                ["v1", "v2", "v3", "v4"],      # attempt 3: 4 violations
            ]
            return violations_per_attempt[c]

        result = _call_llm("sys", "usr", validator=validator, validator_args=())
        # Should return result_1v (fewest violations from attempt 1)
        assert result is result_1v

    @patch("app.feature4.compare_service.time.sleep")
    @patch("app.feature4.compare_service._get_client")
    @patch("app.feature4.compare_service.extract_json_from_llm_response")
    def test_returns_clean_result_immediately(
        self, mock_extract, mock_client, mock_sleep
    ):
        """Clean result (no violations) should be returned on first attempt."""
        client = MagicMock()
        mock_client.return_value = client
        client.chat.completions.create.return_value = _make_llm_response('{}')

        clean = {"clean": True}
        mock_extract.return_value = (clean, None)

        def validator(result):
            return []  # no violations

        result = _call_llm("sys", "usr", validator=validator, validator_args=())
        assert result is clean
        # Should only call LLM once
        assert client.chat.completions.create.call_count == 1

    @patch("app.feature4.compare_service.time.sleep")
    @patch("app.feature4.compare_service._get_client")
    @patch("app.feature4.compare_service.extract_json_from_llm_response")
    def test_progressive_relaxation_uses_relaxed_args_on_attempt_2(
        self, mock_extract, mock_client, mock_sleep
    ):
        """On attempt 2+, should use relaxed_validator_args instead of validator_args."""
        client = MagicMock()
        mock_client.return_value = client
        client.chat.completions.create.return_value = _make_llm_response('{}')

        parsed = {"data": "ok"}
        mock_extract.return_value = (parsed, None)

        received_args = []

        def validator(result, wids, complements, relaxed):
            received_args.append(relaxed)
            if len(received_args) <= 2:
                return ["violation"]  # fail first 2 attempts
            return []  # pass on attempt 2 (3rd call)

        result = _call_llm(
            "sys", "usr",
            validator=validator,
            validator_args=(["W1"], set(), False),
            relaxed_validator_args=(["W1"], set(), True),
        )
        assert result is parsed
        # Attempt 0: relaxed=False, attempt 1: relaxed=False, attempt 2: relaxed=True
        assert received_args == [False, False, True]

    @patch("app.feature4.compare_service.time.sleep")
    @patch("app.feature4.compare_service._get_client")
    @patch("app.feature4.compare_service.extract_json_from_llm_response")
    def test_returns_best_attempt_on_exception(
        self, mock_extract, mock_client, mock_sleep
    ):
        """If attempt 0 succeeds with violations but attempt 1 raises a fatal
        exception, return the best-attempt result."""
        client = MagicMock()
        mock_client.return_value = client

        parsed = {"data": "best_attempt"}
        # Attempt 0: succeeds, Attempt 1: exception
        client.chat.completions.create.side_effect = [
            _make_llm_response('{}'),
            ValueError("fatal error"),
        ]
        mock_extract.return_value = (parsed, None)

        def validator(result):
            return ["one violation"]

        result = _call_llm("sys", "usr", validator=validator, validator_args=())
        assert result is parsed

    @patch("app.feature4.compare_service.time.sleep")
    @patch("app.feature4.compare_service._get_client")
    @patch("app.feature4.compare_service.extract_json_from_llm_response")
    def test_returns_none_when_all_attempts_fail_parse_no_prior_success(
        self, mock_extract, mock_client, mock_sleep
    ):
        """If all attempts fail JSON parse and no prior success, return None."""
        client = MagicMock()
        mock_client.return_value = client
        client.chat.completions.create.return_value = _make_llm_response('bad')
        mock_extract.return_value = (None, "parse error")

        result = _call_llm("sys", "usr")
        assert result is None


@pytest.mark.unit
class TestValidateDepthFingerprintGrounding:
    """Test that _validate_depth checks synthesis references key_components."""

    def _make_sw_matrix(self, mechanism="", evidence="", cause="", consequence=""):
        """Helper: build minimal SW matrix with configurable text fields."""
        return {
            "strengths_weaknesses_matrix": [
                {
                    "work_id": "W123",
                    "handles_well": [
                        {
                            "capability": "Something",
                            "mechanism": mechanism or ("x" * 50),
                            "evidence": evidence or ("x" * 50),
                        }
                    ],
                    "struggles_with": [
                        {
                            "limitation": "Something",
                            "cause": cause or ("x" * 40),
                            "consequence": consequence or ("x" * 40),
                        }
                    ],
                    "complemented_by": [],
                }
            ],
            "recommendation": {
                "summary": "x" * 60,
                "decision_matrix": [
                    {"scenario": "s1", "use": "W123", "why": "x" * 80},
                    {"scenario": "s2", "use": "W123", "why": "x" * 80},
                    {"scenario": "s3", "use": "W123", "why": "x" * 80},
                ],
                "can_combine": True,
                "combination_notes": "n/a",
            },
        }

    def test_catches_missing_key_components_in_synthesis(self):
        """Violation when synthesis text references none of the fingerprint's key_components."""
        result = self._make_sw_matrix(
            mechanism="Uses a generic neural network for classification",
            evidence="Achieves good accuracy on benchmarks",
        )
        fingerprints = {
            "W123": {
                "key_components": [
                    "bottleneck residual block (1×1→3×3→1×1)",
                    "global average pooling (GAP) classifier",
                ],
            }
        }
        violations = _validate_depth(result, fingerprints=fingerprints)
        grounding_violations = [v for v in violations if "key_components" in v]
        assert len(grounding_violations) == 1
        assert "0/2" in grounding_violations[0] or "Missing" in grounding_violations[0]

    def test_catches_insufficient_key_components(self):
        """Violation when only 1 of 3 key_components is referenced (need 2)."""
        result = self._make_sw_matrix(
            mechanism="Uses bottleneck blocks for feature extraction",
        )
        fingerprints = {
            "W123": {
                "key_components": [
                    "bottleneck residual block (1×1→3×3→1×1)",
                    "global average pooling (GAP) classifier",
                    "batch normalization (after each convolutional layer)",
                ],
            }
        }
        violations = _validate_depth(result, fingerprints=fingerprints)
        grounding_violations = [v for v in violations if "key_components" in v]
        assert len(grounding_violations) == 1
        assert "1/3" in grounding_violations[0]

    def test_passes_when_enough_key_components_referenced(self):
        """No violation when at least half the key_components appear in synthesis."""
        result = self._make_sw_matrix(
            mechanism="Uses bottleneck residual blocks with batch normalization after each layer",
        )
        fingerprints = {
            "W123": {
                "key_components": [
                    "bottleneck residual block (1×1→3×3→1×1)",
                    "global average pooling (GAP) classifier",
                    "batch normalization (after each convolutional layer)",
                ],
            }
        }
        violations = _validate_depth(result, fingerprints=fingerprints)
        grounding_violations = [v for v in violations if "key_components" in v]
        assert len(grounding_violations) == 0

    def test_no_violation_without_fingerprints(self):
        """No grounding check when fingerprints not provided."""
        result = self._make_sw_matrix(
            mechanism="Uses a generic neural network approach",
        )
        violations = _validate_depth(result, fingerprints=None)
        grounding_violations = [v for v in violations if "key_components" in v]
        assert len(grounding_violations) == 0

    def test_no_violation_with_few_key_components(self):
        """Skip grounding check when fingerprint has <2 key_components."""
        result = self._make_sw_matrix(
            mechanism="Uses a generic neural network approach",
        )
        fingerprints = {
            "W123": {
                "key_components": ["single component only"],
            }
        }
        violations = _validate_depth(result, fingerprints=fingerprints)
        grounding_violations = [v for v in violations if "key_components" in v]
        assert len(grounding_violations) == 0

    def test_catches_missing_numbers_from_fingerprint(self):
        """Violation when fingerprint has numbers but synthesis has none."""
        result = self._make_sw_matrix(
            mechanism="Uses residual blocks with identity shortcuts for gradient flow",
            cause="Deep architecture increases training time significantly",
            consequence="Slower convergence compared to shallower networks",
        )
        fingerprints = {
            "W123": {
                "key_components": [
                    "residual block (two 3x3 layers with identity shortcut)",
                    "SGD with momentum (0.9) and weight decay (0.0001)",
                ],
                "approach": "Trained with SGD, batch size 256, lr 0.1, on ImageNet with 1000 classes",
            }
        }
        violations = _validate_depth(result, fingerprints=fingerprints)
        number_violations = [v for v in violations if "numbers" in v]
        assert len(number_violations) == 1

    def test_passes_when_numbers_present(self):
        """No numeric violation when synthesis references specific values."""
        result = self._make_sw_matrix(
            mechanism="Uses residual blocks trained with SGD (batch size 256, lr 0.1)",
        )
        fingerprints = {
            "W123": {
                "key_components": [
                    "residual block (two 3x3 layers with identity shortcut)",
                    "SGD with momentum (0.9) and weight decay (0.0001)",
                ],
                "approach": "Trained with SGD, batch size 256, lr 0.1, on ImageNet with 1000 classes",
            }
        }
        violations = _validate_depth(result, fingerprints=fingerprints)
        number_violations = [v for v in violations if "numbers" in v]
        assert len(number_violations) == 0


@pytest.mark.unit
class TestExtractValidationMetrics:
    """Test regex extraction of numeric metrics from validation prose."""

    def test_r_squared(self):
        metrics = _extract_validation_metrics("R² = 0.92 on test set")
        assert len(metrics) == 1
        assert metrics[0]["name"] == "R²"
        assert metrics[0]["value"] == "0.92"

    def test_rmse_with_unit(self):
        metrics = _extract_validation_metrics("RMSE = 3.4 mm on 200 test specimens")
        names = {m["name"] for m in metrics}
        assert "RMSE" in names
        rmse = next(m for m in metrics if m["name"] == "RMSE")
        assert rmse["value"] == "3.4"
        assert rmse["unit"] == "mm"

    def test_accuracy_percent(self):
        metrics = _extract_validation_metrics("accuracy = 95.3%")
        assert len(metrics) == 1
        assert metrics[0]["name"] == "Accuracy"
        assert metrics[0]["value"] == "95.3"
        assert metrics[0]["unit"] == "%"

    def test_multiple_metrics(self):
        text = "R² = 0.89, RMSE = 2.1 MPa, MAE = 1.5 MPa on 150 test specimens"
        metrics = _extract_validation_metrics(text)
        names = {m["name"] for m in metrics}
        assert "R²" in names
        assert "RMSE" in names
        assert "MAE" in names
        assert "Test samples" in names

    def test_f1_score(self):
        metrics = _extract_validation_metrics("F1-score = 0.87 on validation set")
        assert len(metrics) == 1
        assert metrics[0]["name"] == "F1"
        assert metrics[0]["value"] == "0.87"

    def test_auc(self):
        metrics = _extract_validation_metrics("AUC of 0.95")
        assert len(metrics) == 1
        assert metrics[0]["name"] == "AUC"
        assert metrics[0]["value"] == "0.95"

    def test_bleu(self):
        metrics = _extract_validation_metrics("BLEU = 32.5")
        assert len(metrics) == 1
        assert metrics[0]["name"] == "BLEU"
        assert metrics[0]["value"] == "32.5"

    def test_test_samples(self):
        metrics = _extract_validation_metrics("Evaluated on 500 test images")
        assert len(metrics) == 1
        assert metrics[0]["name"] == "Test samples"
        assert metrics[0]["value"] == "500"

    def test_empty_string(self):
        assert _extract_validation_metrics("") == []

    def test_no_metrics(self):
        assert _extract_validation_metrics("Validated using cross-validation approach") == []

    def test_deduplication(self):
        text = "R² = 0.85 and R2 = 0.85 on different splits"
        metrics = _extract_validation_metrics(text)
        r2_count = sum(1 for m in metrics if m["name"] == "R²")
        assert r2_count == 1

    def test_map_metric(self):
        metrics = _extract_validation_metrics("mAP = 0.412 on COCO")
        assert len(metrics) == 1
        assert metrics[0]["name"] == "mAP"
        assert metrics[0]["value"] == "0.412"

    def test_iou(self):
        metrics = _extract_validation_metrics("IoU of 0.76")
        assert len(metrics) == 1
        assert metrics[0]["name"] == "IoU"
        assert metrics[0]["value"] == "0.76"
