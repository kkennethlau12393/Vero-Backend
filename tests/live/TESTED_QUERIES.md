# Tested Queries Tracker

> Auto-updated by live tests. Also update manually when adding new queries.

## Benchmark Queries (13)

| ID | Query | Domain | Type | Added |
|---|---|---|---|---|
| transformers | transformer architecture attention mechanisms in deep learning | CS/ML | broad | 2026-02-11 |
| graph_neural_networks | graph neural networks node classification | CS/ML | specific | 2026-02-11 |
| reinforcement_learning | reinforcement learning policy gradient methods | CS/ML | specific | 2026-02-11 |
| crispr_gene_editing | CRISPR-Cas9 gene editing therapeutic applications | Biology | specific | 2026-02-11 |
| quantum_computing | quantum computing error correction fault tolerance | Physics | specific | 2026-02-11 |
| climate_modeling | climate change modeling global circulation models projections | Earth Science | broad | 2026-02-11 |
| federated_learning | federated learning privacy-preserving machine learning | CS/ML | specific | 2026-02-11 |
| diffusion_models | diffusion models denoising score matching image generation | CS/ML | specific | 2026-02-11 |
| protein_folding | protein structure prediction deep learning AlphaFold | Biology/ML | intersection | 2026-02-11 |
| llm_alignment | large language model alignment RLHF safety | CS/ML | specific | 2026-02-11 |
| attention_medical_imaging | attention mechanisms in medical image segmentation | CS/Medical | intersection | 2026-02-11 |
| gnn_drug_discovery | graph neural networks for drug discovery molecular property prediction | CS/Chemistry | intersection | 2026-02-11 |
| rl_robotics | reinforcement learning sim-to-real transfer robotic manipulation | CS/Robotics | intersection | 2026-02-11 |

## Coverage Summary

**Domains covered:** CS/ML, Biology, Physics, Earth Science, Medical, Chemistry, Robotics
**Query types:** 1 broad, 8 specific, 4 intersection
**Total unique queries:** 13

### Gaps — domains NOT yet tested:
- Mathematics / pure math
- Social sciences / economics
- Materials science / engineering
- Neuroscience
- Linguistics / NLP-specific
- Energy / sustainability
- Astronomy / astrophysics

### Gaps — query types NOT yet tested:
- Methodological queries ("what methods exist for X")
- Historical/timeline queries ("evolution of X over time")
- Comparison queries ("X vs Y")
- Very broad queries (single word like "photosynthesis")
- Very niche queries (narrow subfield with <100 papers)
- Ambiguous queries (terms with multiple meanings)
- Non-English terms / transliterated queries

---

## Feature 1: Citation Map Queries

### E2E Benchmark Queries — Query Mode (12)

| ID | Query | Domain | Difficulty | Added |
|---|---|---|---|---|
| transformers | transformer attention mechanism | CS/ML | easy | 2026-02-12 |
| graph_neural_networks | graph neural networks | CS/ML | medium | 2026-02-12 |
| rl_robotics | reinforcement learning robotics | CS/Robotics | medium | 2026-02-12 |
| diffusion_models | denoising diffusion probabilistic models image generation | CS/ML | medium | 2026-02-12 |
| crispr | CRISPR gene editing | Biology | medium | 2026-02-12 |
| protein_folding | protein structure prediction AlphaFold | Biology/ML | medium | 2026-02-12 |
| quantum_error_correction | quantum error correction surface codes | Physics | hard | 2026-02-12 |
| behavioral_economics | prospect theory behavioral economics | Economics | medium | 2026-02-12 |
| gnn_drug_discovery | graph neural networks drug discovery molecular property prediction | CS/Chemistry | hard | 2026-02-12 |
| attention_medical_imaging | attention mechanisms medical image segmentation | CS/Medical | hard | 2026-02-12 |
| synaptic_plasticity | synaptic plasticity long-term potentiation | Neuroscience | medium | 2026-02-12 |
| graphene | graphene electronic properties synthesis | Materials Science | medium | 2026-02-12 |

### E2E Benchmark Queries — Seed Mode (1)

| ID | Seed Work ID | Notes | Added |
|---|---|---|---|
| seed_attention_is_all_you_need | W2963403868 | Attention Is All You Need | 2026-02-12 |

### Seed Selection Quality Queries (12)

| ID | Query | Difficulty | Expected Seed | Added |
|---|---|---|---|---|
| transformers_seminal | attention is all you need transformer | easy | Vaswani et al. 2017 | 2026-02-12 |
| resnet | deep residual learning image recognition | easy | He et al. 2015 | 2026-02-12 |
| adam_optimizer | adam optimizer adaptive learning rate | easy | Kingma & Ba 2014 | 2026-02-12 |
| gan | generative adversarial networks | easy | Goodfellow et al. 2014 | 2026-02-12 |
| bert | BERT pre-training language representations | easy | Devlin et al. 2018 | 2026-02-12 |
| crispr_broad | CRISPR gene editing technology | medium | Jinek et al. 2012 | 2026-02-12 |
| alphafold | protein structure prediction deep learning | medium | AlphaFold paper | 2026-02-12 |
| word2vec | word embeddings distributed representations | medium | Mikolov et al. 2013 | 2026-02-12 |
| broad_machine_learning | machine learning | hard | any highly cited ML paper | 2026-02-12 |
| niche_neural_ode | neural ordinary differential equations continuous depth | hard | Chen et al. 2018 | 2026-02-12 |
| ambiguous_attention | attention mechanism | hard | Bahdanau 2014 or Vaswani 2017 | 2026-02-12 |
| niche_capsule_networks | capsule networks dynamic routing | hard | Sabour/Hinton 2017 | 2026-02-12 |

### F1 Coverage Summary

**Domains covered:** CS/ML, Biology, Physics, Economics, Chemistry, Medical, Neuroscience, Materials Science, Robotics
**Query difficulties:** 5 easy, 9 medium, 5 hard
**Total unique F1 queries:** 25 (12 e2e + 1 seed mode + 12 seed quality)

### F1 Gaps — not yet tested:
- Astronomy / astrophysics queries
- Humanities / social science beyond economics
- Very new fields (<2 years old)
- Non-English paper seeds
- Queries with no clear seed paper (emerging fields)

---

## How to use this file

When generating new test queries, read this file first to:
1. Avoid duplicating existing queries
2. Fill domain gaps
3. Fill query type gaps
4. Increase difficulty progressively
