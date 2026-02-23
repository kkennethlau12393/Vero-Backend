"""
Centralized methodology categorization for cross-domain filtering.

This module provides a single source of truth for determining what
methodology a paper uses, enabling proper filtering of irrelevant
grounding papers (e.g., neural network papers shouldn't appear as
landmarks for ensemble method papers).
"""

from typing import Optional

# METHODOLOGY CATEGORIES - papers in different categories are irrelevant to each other
METHODOLOGY_CATEGORIES = {
    "neural_network": {
        "lstm", "rnn", "recurrent", "neural", "backpropagation", "backprop",
        "perceptron", "deep learning", "cnn", "convolutional", "transformer",
        "attention", "encoder", "decoder", "activation", "feedforward",
        "short-term memory", "long short", "hidden layer", "dropout",
        "batch norm", "relu", "sigmoid", "softmax", "multilayer",
        "rprop", "adaptive learning", "oscillator",
    },
    "ensemble_tree": {
        "random forest", "decision tree", "bagging", "boosting",
        "gradient boosting", "xgboost", "adaboost", "cart", "gini",
        "tree-based", "leaf node", "split criterion", "feature importance",
        "classification tree", "regression tree",
    },
    "kernel_svm": {
        "svm", "support vector", "kernel", "margin", "hyperplane",
        "rbf", "polynomial kernel", "slack variable", "hinge loss",
        "linear kernel",
    },
    "graph_network": {
        "graph neural", "gnn", "node embedding", "graph embedding", "network embedding",
        "node2vec", "deepwalk", "graph convolution", "message passing", "adjacency",
    },
    "optimization": {
        "gradient descent", "sgd", "adam", "momentum", "learning rate",
        "convergence", "stochastic optimization", "adagrad", "rmsprop",
    },
    "bayesian": {
        "bayesian", "posterior", "prior", "likelihood", "mcmc", "variational",
        "probabilistic", "inference", "gibbs", "metropolis",
    },
}


def get_methodology(text: str) -> Optional[str]:
    """Determine which methodology category a paper belongs to.

    Prioritizes longer/more specific matches to avoid false positives.
    E.g., "graph neural network" should match graph_network, not neural_network.

    Args:
        text: Title and/or abstract text to analyze

    Returns:
        Methodology category name, or None if no match
    """
    if not text:
        return None
    text_lower = text.lower()

    # Find all matching categories with their best (longest) matching keyword
    matches = []
    for category, keywords in METHODOLOGY_CATEGORIES.items():
        for kw in keywords:
            if kw in text_lower:
                matches.append((category, len(kw)))
                break  # Found a match for this category

    if not matches:
        return None

    # Return the category with the longest matching keyword
    # This ensures "graph neural" (12 chars) beats "neural" (6 chars)
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0][0]


def is_methodology_mismatch(
    target_title: Optional[str],
    target_abstract: Optional[str],
    paper_title: Optional[str],
    paper_abstract: Optional[str],
) -> bool:
    """Check if a paper uses a different methodology than the target.

    Args:
        target_title: Title of the target paper
        target_abstract: Abstract of the target paper
        paper_title: Title of the paper to check
        paper_abstract: Abstract of the paper to check

    Returns:
        True if methodologies are different, False if same or can't determine
    """
    target_text = f"{target_title or ''} {target_abstract or ''}"
    target_methodology = get_methodology(target_text)

    if not target_methodology:
        return False  # Can't determine target methodology

    paper_text = f"{paper_title or ''} {paper_abstract or ''}"
    paper_methodology = get_methodology(paper_text)

    if paper_methodology and paper_methodology != target_methodology:
        return True

    return False
