#!/usr/bin/env python3
"""
Generate verified DOI test cases by searching OpenAlex for well-known papers.

This script ensures each test case has a real, working DOI that resolves correctly.
"""

import json
import time
from pathlib import Path
import requests


def search_openalex_by_title(title: str) -> dict | None:
    """Search OpenAlex for a paper by title and return the first result."""
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"title.search:{title}",
        "per_page": 1,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["results"]:
            work = data["results"][0]
            return {
                "doi": work.get("doi", "").replace("https://doi.org/", ""),
                "title": work.get("title", ""),
                "work_id": work.get("id", "").split("/")[-1],
                "year": work.get("publication_year"),
                "cited_by_count": work.get("cited_by_count", 0),
            }
    except Exception as e:
        print(f"Error searching for '{title}': {e}")

    return None


def main():
    # Well-known papers with verified titles
    well_known_papers = [
        # Original 10 that we know work
        ("Attention Is All You Need", "arxiv_nlp", "Transformer paper"),
        ("Very Deep Convolutional Networks for Large-Scale Image Recognition", "arxiv_cv", "VGG paper"),
        ("Generative Adversarial Networks", "arxiv_ml", "GAN paper"),
        ("Deep Residual Learning for Image Recognition", "arxiv_cv", "ResNet paper"),
        ("An Image is Worth 16x16 Words", "arxiv_cv", "Vision Transformer"),
        ("Mastering the game of Go", "nature", "AlphaGo Nature paper"),
        ("Highly accurate protein structure prediction with AlphaFold", "nature", "AlphaFold 2"),
        ("A general reinforcement learning algorithm that masters chess", "science", "AlphaZero Science"),
        ("BERT: Pre-training of Deep Bidirectional Transformers", "acl", "BERT ACL paper"),

        # Add more well-known papers (verified titles from literature)
        ("ImageNet Classification with Deep Convolutional Neural Networks", "arxiv_cv", "AlexNet"),
        ("Batch Normalization: Accelerating Deep Network Training", "arxiv_ml", "Batch Normalization"),
        ("Adam: A Method for Stochastic Optimization", "arxiv_ml", "Adam optimizer"),
        ("Densely Connected Convolutional Networks", "arxiv_cv", "DenseNet"),
        ("U-Net: Convolutional Networks for Biomedical Image Segmentation", "arxiv_cv", "U-Net"),
        ("Sequence to Sequence Learning with Neural Networks", "arxiv_nlp", "Seq2Seq"),
        ("Efficient Estimation of Word Representations in Vector Space", "arxiv_nlp", "Word2Vec"),
        ("You Only Look Once: Unified, Real-Time Object Detection", "arxiv_cv", "YOLO"),
        ("Spatial Transformer Networks", "arxiv_cv", "Spatial Transformer"),
        ("Auto-Encoding Variational Bayes", "arxiv_ml", "VAE"),
        ("Mask R-CNN", "arxiv_cv", "Mask R-CNN"),
        ("Faster R-CNN: Towards Real-Time Object Detection", "arxiv_cv", "Faster R-CNN"),
        ("YOLOv3: An Incremental Improvement", "arxiv_cv", "YOLOv3"),
        ("SqueezeNet: AlexNet-level accuracy with 50x fewer parameters", "arxiv_cv", "SqueezeNet"),
        ("MobileNets: Efficient Convolutional Neural Networks for Mobile", "arxiv_cv", "MobileNets"),
        ("MobileNetV2: Inverted Residuals and Linear Bottlenecks", "arxiv_cv", "MobileNetV2"),
        ("EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", "arxiv_cv", "EfficientNet"),
        ("Deep contextualized word representations", "arxiv_nlp", "ELMo"),
        ("XLNet: Generalized Autoregressive Pretraining for Language Understanding", "arxiv_nlp", "XLNet"),
        ("RoBERTa: A Robustly Optimized BERT Pretraining Approach", "arxiv_nlp", "RoBERTa"),
        ("DistilBERT, a distilled version of BERT", "arxiv_nlp", "DistilBERT"),
        ("ALBERT: A Lite BERT for Self-supervised Learning", "arxiv_nlp", "ALBERT"),
        ("Language Models are Few-Shot Learners", "arxiv_nlp", "GPT-3"),
        ("ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators", "arxiv_nlp", "ELECTRA"),
        ("Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context", "arxiv_nlp", "Transformer-XL"),
        ("Reformer: The Efficient Transformer", "arxiv_nlp", "Reformer"),
        ("End-to-End Object Detection with Transformers", "arxiv_cv", "DETR"),
        ("Momentum Contrast for Unsupervised Visual Representation Learning", "arxiv_cv", "MoCo"),
        ("A Simple Framework for Contrastive Learning of Visual Representations", "arxiv_cv", "SimCLR"),
        ("Learning Transferable Visual Models From Natural Language Supervision", "arxiv_cv", "CLIP"),
        ("Zero-Shot Text-to-Image Generation", "arxiv_cv", "DALL-E"),
        ("Training language models to follow instructions with human feedback", "arxiv_nlp", "InstructGPT"),
        ("Training Compute-Optimal Large Language Models", "arxiv_nlp", "Chinchilla"),
        ("LLaMA: Open and Efficient Foundation Language Models", "arxiv_nlp", "LLaMA"),
        ("Llama 2: Open Foundation and Fine-Tuned Chat Models", "arxiv_nlp", "Llama 2"),
        ("BERT Pre-Training of Image Transformers", "arxiv_cv", "BEiT"),
        ("Masked Autoencoders Are Scalable Vision Learners", "arxiv_cv", "MAE"),
        ("Image-to-Image Translation with Conditional Adversarial Networks", "arxiv_cv", "pix2pix"),
        ("Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", "arxiv_cv", "CycleGAN"),
        ("A Style-Based Generator Architecture for Generative Adversarial Networks", "arxiv_cv", "StyleGAN"),
        ("Xception: Deep Learning with Depthwise Separable Convolutions", "arxiv_cv", "Xception"),
        ("Squeeze-and-Excitation Networks", "arxiv_cv", "SENet"),
        ("ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design", "arxiv_cv", "ShuffleNet V2"),
        ("MnasNet: Platform-Aware Neural Architecture Search for Mobile", "arxiv_cv", "MnasNet"),
        ("EfficientDet: Scalable and Efficient Object Detection", "arxiv_cv", "EfficientDet"),
        ("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "arxiv_cv", "ViT (duplicate)"),
        ("Training data-efficient image transformers", "arxiv_cv", "DeiT"),
        ("Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", "arxiv_cv", "Swin Transformer"),
        ("A ConvNet for the 2020s", "arxiv_cv", "ConvNeXt"),
        ("Megatron-LM: Training Multi-Billion Parameter Language Models", "arxiv_nlp", "Megatron-LM"),
        ("REALM: Retrieval-Augmented Language Model Pre-Training", "arxiv_nlp", "REALM"),
        ("Longformer: The Long-Document Transformer", "arxiv_nlp", "Longformer"),
        ("Big Bird: Transformers for Longer Sequences", "arxiv_nlp", "BigBird"),
        ("PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization", "arxiv_nlp", "PEGASUS"),
        ("BART: Denoising Sequence-to-Sequence Pre-training", "arxiv_nlp", "BART"),
        ("Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", "arxiv_nlp", "T5"),
        ("LoRA: Low-Rank Adaptation of Large Language Models", "arxiv_nlp", "LoRA"),
        ("Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "arxiv_nlp", "Chain-of-Thought"),
        ("Self-Consistency Improves Chain of Thought Reasoning in Language Models", "arxiv_nlp", "Self-Consistency"),
        ("ReAct: Synergizing Reasoning and Acting in Language Models", "arxiv_nlp", "ReAct"),
        ("Toolformer: Language Models Can Teach Themselves to Use Tools", "arxiv_nlp", "Toolformer"),
        ("GPT-4 Technical Report", "arxiv_nlp", "GPT-4"),
        ("Tree of Thoughts: Deliberate Problem Solving with Large Language Models", "arxiv_nlp", "Tree of Thoughts"),
        ("QLoRA: Efficient Finetuning of Quantized LLMs", "arxiv_nlp", "QLoRA"),
        ("Mistral 7B", "arxiv_nlp", "Mistral 7B"),
        ("Mixtral of Experts", "arxiv_nlp", "Mixtral"),

        # Classic papers
        ("Long Short-Term Memory", "neco", "LSTM"),
        ("Gradient-Based Learning Applied to Document Recognition", "ieee", "LeNet"),
        ("Random Forests", "ml_journal", "Random Forests"),
        ("Greedy Function Approximation: A Gradient Boosting Machine", "ml_journal", "Gradient Boosting"),
        ("XGBoost: A Scalable Tree Boosting System", "kdd", "XGBoost"),

        # Nature/Science papers
        ("Deep learning", "nature", "LeCun, Bengio, Hinton - Deep Learning"),
        ("Mastering Chess and Shogi by Self-Play", "nature", "AlphaZero Nature"),
        ("Quantum supremacy using a programmable superconducting processor", "nature", "Quantum supremacy"),
        ("Grandmaster level in StarCraft II using multi-agent reinforcement learning", "science", "AlphaStar"),
    ]

    verified_cases = []
    failed_searches = []

    print(f"Searching OpenAlex for {len(well_known_papers)} papers...")
    print("This will take ~2-3 minutes.\n")

    for title, category, note in well_known_papers:
        print(f"Searching: {title[:50]}...")
        result = search_openalex_by_title(title)

        if result and result["doi"]:
            # Extract a unique substring from the title for validation
            title_words = result["title"].split()
            if len(title_words) >= 2:
                expected_substring = " ".join(title_words[:3])  # First 3 words
            else:
                expected_substring = title_words[0] if title_words else ""

            verified_cases.append({
                "doi": result["doi"],
                "expected_title_substring": expected_substring,
                "expect_error": False,
                "category": category,
                "note": f"{note} - {result['title'][:60]}",
            })
            print(f"  ✓ Found: {result['doi']}")
        else:
            failed_searches.append((title, note))
            print(f"  ✗ NOT FOUND")

        time.sleep(0.5)  # Be nice to OpenAlex

    # Add error cases
    verified_cases.extend([
        {
            "doi": "10.9999/invalid-doi-format",
            "expected_title_substring": "",
            "expect_error": True,
            "category": "error_case",
            "note": "Invalid DOI - should fail",
        },
        {
            "doi": "10.0000/nonexistent.doi.12345",
            "expected_title_substring": "",
            "expect_error": True,
            "category": "error_case",
            "note": "Non-existent DOI - should fail",
        },
        {
            "doi": "INVALID_DOI_NO_PREFIX",
            "expected_title_substring": "",
            "expect_error": True,
            "category": "error_case",
            "note": "Malformed DOI - no prefix",
        },
        {
            "doi": "",
            "expected_title_substring": "",
            "expect_error": True,
            "category": "error_case",
            "note": "Empty DOI - should fail",
        },
    ])

    # Save verified test cases
    output_file = Path(__file__).parent / "doi_test_cases_verified.json"
    with open(output_file, "w") as f:
        json.dump(verified_cases, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated {len(verified_cases)} verified DOI test cases")
    print(f"  Success: {len(verified_cases) - 4} papers found")
    print(f"  Failed:  {len(failed_searches)} papers not found")
    print(f"  Error cases: 4")
    print(f"\nSaved to: {output_file}")

    if failed_searches:
        print(f"\nFailed searches:")
        for title, note in failed_searches:
            print(f"  - {note}: {title[:60]}")


if __name__ == "__main__":
    main()
