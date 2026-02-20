"""
Unit tests for app/feature3/methodology.py

Pure logic tests: no DB, no network.
"""
import pytest

from app.feature3.methodology import get_methodology, is_methodology_mismatch


@pytest.mark.unit
class TestGetMethodology:
    def test_neural_network_detected(self):
        assert get_methodology("deep learning transformer model") == "neural_network"

    def test_ensemble_tree_detected(self):
        assert get_methodology("random forest gradient boosting model") == "ensemble_tree"

    def test_kernel_svm_detected(self):
        assert get_methodology("support vector machine classification") == "kernel_svm"

    def test_graph_network_detected(self):
        assert get_methodology("graph neural network for node classification") == "graph_network"

    def test_optimization_detected(self):
        assert get_methodology("stochastic gradient descent convergence") == "optimization"

    def test_bayesian_detected(self):
        assert get_methodology("bayesian posterior inference with mcmc") == "bayesian"

    def test_no_match_returns_none(self):
        assert get_methodology("ocean temperature study") is None

    def test_empty_string(self):
        assert get_methodology("") is None

    def test_none_input(self):
        assert get_methodology(None) is None

    def test_longest_match_wins(self):
        # "graph neural" (12 chars) should beat "neural" (6 chars)
        result = get_methodology("graph neural network architecture")
        assert result == "graph_network"

    def test_case_insensitive(self):
        assert get_methodology("LSTM Recurrent Network") == "neural_network"


@pytest.mark.unit
class TestIsMethodologyMismatch:
    def test_same_methodology_no_mismatch(self):
        assert is_methodology_mismatch(
            "deep learning CNN", "neural networks for vision",
            "convolutional neural network", "image classification with CNN"
        ) is False

    def test_different_methodology_mismatch(self):
        assert is_methodology_mismatch(
            "deep learning transformer", "attention mechanism model",
            "random forest ensemble", "gradient boosting for tabular data"
        ) is True

    def test_target_no_methodology(self):
        # Can't determine target methodology -> no mismatch
        assert is_methodology_mismatch(
            "ocean currents", "temperature study",
            "neural network", "deep learning model"
        ) is False

    def test_paper_no_methodology(self):
        # Can't determine paper methodology -> no mismatch
        assert is_methodology_mismatch(
            "deep learning model", "neural network architecture",
            "ocean currents", "temperature patterns"
        ) is False

    def test_both_none_no_mismatch(self):
        assert is_methodology_mismatch(None, None, None, None) is False
