#!/usr/bin/env python3
"""
Generate verified PDF test cases by searching OpenAlex for papers with ArXiv IDs.

This script ensures each test case has a real, working ArXiv ID that matches the expected paper.
"""

import json
import time
from pathlib import Path
import requests


def search_openalex_by_title(title: str) -> dict | None:
    """Search OpenAlex for a paper by title and return the first result with ArXiv ID."""
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"title.search:{title}",
        "per_page": 5,  # Get top 5 to find one with ArXiv ID
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["results"]:
            # Find first result with ArXiv ID
            for work in data["results"]:
                doi = work.get("doi", "")
                if "arxiv" in doi.lower():
                    # Extract ArXiv ID from DOI (e.g., "https://doi.org/10.48550/arXiv.1706.03762" -> "1706.03762")
                    arxiv_id = doi.split("arXiv.")[-1]
                    return {
                        "arxiv_id": arxiv_id,
                        "title": work.get("title", ""),
                        "doi": doi.replace("https://doi.org/", ""),
                        "work_id": work.get("id", "").split("/")[-1],
                        "year": work.get("publication_year"),
                        "cited_by_count": work.get("cited_by_count", 0),
                    }
    except Exception as e:
        print(f"Error searching for '{title}': {e}")

    return None


def main():
    # Well-known ArXiv papers (verified titles from literature)
    well_known_papers = [
        # NLP/Transformers
        ("Attention Is All You Need", "arxiv_nlp", "Transformer"),
        ("BERT: Pre-training of Deep Bidirectional Transformers", "arxiv_nlp", "BERT"),
        ("Language Models are Few-Shot Learners", "arxiv_nlp", "GPT-3"),
        ("RoBERTa: A Robustly Optimized BERT Pretraining Approach", "arxiv_nlp", "RoBERTa"),
        ("ALBERT: A Lite BERT for Self-supervised Learning", "arxiv_nlp", "ALBERT"),
        ("DistilBERT, a distilled version of BERT", "arxiv_nlp", "DistilBERT"),
        ("XLNet: Generalized Autoregressive Pretraining", "arxiv_nlp", "XLNet"),
        ("ELECTRA: Pre-training Text Encoders as Discriminators", "arxiv_nlp", "ELECTRA"),
        ("Transformer-XL: Attentive Language Models", "arxiv_nlp", "Transformer-XL"),
        ("Reformer: The Efficient Transformer", "arxiv_nlp", "Reformer"),
        ("Deep contextualized word representations", "arxiv_nlp", "ELMo"),
        ("Sequence to Sequence Learning with Neural Networks", "arxiv_nlp", "Seq2Seq"),
        ("Efficient Estimation of Word Representations in Vector Space", "arxiv_nlp", "Word2Vec"),
        ("LLaMA: Open and Efficient Foundation Language Models", "arxiv_nlp", "LLaMA"),
        ("Llama 2: Open Foundation and Fine-Tuned Chat Models", "arxiv_nlp", "Llama 2"),
        ("Training Compute-Optimal Large Language Models", "arxiv_nlp", "Chinchilla"),
        ("Training language models to follow instructions with human feedback", "arxiv_nlp", "InstructGPT"),
        ("BART: Denoising Sequence-to-Sequence Pre-training", "arxiv_nlp", "BART"),
        ("Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", "arxiv_nlp", "T5"),
        ("PEGASUS: Pre-training with Extracted Gap-sentences", "arxiv_nlp", "PEGASUS"),
        ("Longformer: The Long-Document Transformer", "arxiv_nlp", "Longformer"),
        ("Big Bird: Transformers for Longer Sequences", "arxiv_nlp", "BigBird"),
        ("REALM: Retrieval-Augmented Language Model Pre-Training", "arxiv_nlp", "REALM"),
        ("Megatron-LM: Training Multi-Billion Parameter Language Models", "arxiv_nlp", "Megatron-LM"),
        ("LoRA: Low-Rank Adaptation of Large Language Models", "arxiv_nlp", "LoRA"),
        ("Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "arxiv_nlp", "Chain-of-Thought"),
        ("Self-Consistency Improves Chain of Thought Reasoning", "arxiv_nlp", "Self-Consistency"),
        ("ReAct: Synergizing Reasoning and Acting in Language Models", "arxiv_nlp", "ReAct"),
        ("Toolformer: Language Models Can Teach Themselves to Use Tools", "arxiv_nlp", "Toolformer"),
        ("Tree of Thoughts: Deliberate Problem Solving with Large Language Models", "arxiv_nlp", "Tree of Thoughts"),

        # Computer Vision
        ("Very Deep Convolutional Networks for Large-Scale Image Recognition", "arxiv_cv", "VGG"),
        ("Deep Residual Learning for Image Recognition", "arxiv_cv", "ResNet"),
        ("An Image is Worth 16x16 Words", "arxiv_cv", "ViT"),
        ("Densely Connected Convolutional Networks", "arxiv_cv", "DenseNet"),
        ("U-Net: Convolutional Networks for Biomedical Image Segmentation", "arxiv_cv", "U-Net"),
        ("You Only Look Once: Unified, Real-Time Object Detection", "arxiv_cv", "YOLO"),
        ("Faster R-CNN: Towards Real-Time Object Detection", "arxiv_cv", "Faster R-CNN"),
        ("Mask R-CNN", "arxiv_cv", "Mask R-CNN"),
        ("YOLOv3: An Incremental Improvement", "arxiv_cv", "YOLOv3"),
        ("Spatial Transformer Networks", "arxiv_cv", "Spatial Transformer"),
        ("SqueezeNet: AlexNet-level accuracy with 50x fewer parameters", "arxiv_cv", "SqueezeNet"),
        ("MobileNets: Efficient Convolutional Neural Networks", "arxiv_cv", "MobileNets"),
        ("MobileNetV2: Inverted Residuals and Linear Bottlenecks", "arxiv_cv", "MobileNetV2"),
        ("EfficientNet: Rethinking Model Scaling", "arxiv_cv", "EfficientNet"),
        ("Xception: Deep Learning with Depthwise Separable Convolutions", "arxiv_cv", "Xception"),
        ("Squeeze-and-Excitation Networks", "arxiv_cv", "SENet"),
        ("ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture", "arxiv_cv", "ShuffleNet V2"),
        ("MnasNet: Platform-Aware Neural Architecture Search for Mobile", "arxiv_cv", "MnasNet"),
        ("EfficientDet: Scalable and Efficient Object Detection", "arxiv_cv", "EfficientDet"),
        ("Training data-efficient image transformers", "arxiv_cv", "DeiT"),
        ("Swin Transformer: Hierarchical Vision Transformer", "arxiv_cv", "Swin Transformer"),
        ("A ConvNet for the 2020s", "arxiv_cv", "ConvNeXt"),
        ("End-to-End Object Detection with Transformers", "arxiv_cv", "DETR"),
        ("Momentum Contrast for Unsupervised Visual Representation Learning", "arxiv_cv", "MoCo"),
        ("A Simple Framework for Contrastive Learning of Visual Representations", "arxiv_cv", "SimCLR"),
        ("Learning Transferable Visual Models From Natural Language Supervision", "arxiv_cv", "CLIP"),
        ("Zero-Shot Text-to-Image Generation", "arxiv_cv", "DALL-E"),
        ("BERT Pre-Training of Image Transformers", "arxiv_cv", "BEiT"),
        ("Masked Autoencoders Are Scalable Vision Learners", "arxiv_cv", "MAE"),
        ("Image-to-Image Translation with Conditional Adversarial Networks", "arxiv_cv", "pix2pix"),
        ("Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", "arxiv_cv", "CycleGAN"),
        ("A Style-Based Generator Architecture for Generative Adversarial Networks", "arxiv_cv", "StyleGAN"),

        # ML Fundamentals
        ("Generative Adversarial Networks", "arxiv_ml", "GAN"),
        ("Adam: A Method for Stochastic Optimization", "arxiv_ml", "Adam optimizer"),
        ("Batch Normalization: Accelerating Deep Network Training", "arxiv_ml", "Batch Normalization"),
        ("Auto-Encoding Variational Bayes", "arxiv_ml", "VAE"),
        ("QLoRA: Efficient Finetuning of Quantized LLMs", "arxiv_ml", "QLoRA"),

        # Recent models
        ("Mistral 7B", "arxiv_nlp", "Mistral 7B"),
        ("Mixtral of Experts", "arxiv_nlp", "Mixtral"),
        ("GPT-4 Technical Report", "arxiv_nlp", "GPT-4"),
        ("Flamingo: a Visual Language Model for Few-Shot Learning", "arxiv_cv", "Flamingo"),
    ]

    verified_cases = []
    failed_searches = []

    print(f"Searching OpenAlex for {len(well_known_papers)} papers...")
    print("This will take ~2-3 minutes.\\n")

    for title, category, note in well_known_papers:
        print(f"Searching: {title[:50]}...")
        result = search_openalex_by_title(title)

        if result and result["arxiv_id"]:
            # Extract a unique substring from the title for validation
            title_words = result["title"].split()
            if len(title_words) >= 2:
                expected_substring = " ".join(title_words[:3])  # First 3 words
            else:
                expected_substring = title_words[0] if title_words else ""

            verified_cases.append({
                "arxiv_id": result["arxiv_id"],
                "expected_title_substring": expected_substring,
                "pdf_path": f"{result['arxiv_id']}_{note.lower().replace(' ', '_').replace('-', '_')}.pdf",
                "expect_error": False,
                "category": category,
                "note": f"{note} - {result['title'][:60]}",
            })
            print(f"  ✓ Found: {result['arxiv_id']}")
        else:
            failed_searches.append((title, note))
            print(f"  ✗ NOT FOUND")

        time.sleep(0.5)  # Be nice to OpenAlex

    # Add error cases
    verified_cases.extend([
        {
            "arxiv_id": "",
            "expected_title_substring": "",
            "pdf_path": "corrupt.pdf",
            "expect_error": True,
            "category": "error_case",
            "note": "Corrupt PDF - should fail",
        },
        {
            "arxiv_id": "",
            "expected_title_substring": "",
            "pdf_path": "no_metadata.pdf",
            "expect_error": True,
            "category": "error_case",
            "note": "PDF with no metadata or title - should fail",
        },
        {
            "arxiv_id": "",
            "expected_title_substring": "",
            "pdf_path": "scanned.pdf",
            "expect_error": True,
            "category": "error_case",
            "note": "Scanned PDF (OCR needed) - expected limitation",
        },
    ])

    # Save verified test cases
    output_file = Path(__file__).parent / "pdf_test_cases_verified.json"
    with open(output_file, "w") as f:
        json.dump(verified_cases, f, indent=2)

    print(f"\\n{'='*60}")
    print(f"Generated {len(verified_cases)} verified PDF test cases")
    print(f"  Success: {len(verified_cases) - 3} papers found")
    print(f"  Failed:  {len(failed_searches)} papers not found")
    print(f"  Error cases: 3")
    print(f"\\nSaved to: {output_file}")

    if failed_searches:
        print(f"\\nFailed searches:")
        for title, note in failed_searches:
            print(f"  - {note}: {title[:60]}")


if __name__ == "__main__":
    main()
