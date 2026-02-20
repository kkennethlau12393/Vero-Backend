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

## Feature 3: Novelty Assessment Benchmark Papers

### E2E Benchmark Papers (36)

| ID | Work ID | Paper | Domain | Novelty Level | Type | Added |
|---|---|---|---|---|---|---|
| resnet | W2194775991 | Deep Residual Learning for Image Recognition | CV | pioneering | foundational | 2026-02-20 |
| imagenet | W2108598243 | ImageNet: A large-scale hierarchical image database | CV | pioneering | foundational | 2026-02-19 |
| faster_rcnn | W639708223 | Faster R-CNN | CV | pioneering | foundational | 2026-02-20 |
| densenet | W2963446712 | Densely Connected Convolutional Networks | CV | high | foundational | 2026-02-20 |
| rcnn | W2102605133 | Rich Feature Hierarchies for Object Detection | CV | high | foundational | 2026-02-20 |
| mask_rcnn | W2963150697 | Mask R-CNN | CV | high | foundational | 2026-02-20 |
| senet | W2752782242 | Squeeze-and-Excitation Networks | CV | high | foundational | 2026-02-20 |
| cyclegan | W2962793481 | CycleGAN | CV | high | foundational | 2026-02-20 |
| selective_search | W2088049833 | Selective Search for Object Recognition | CV | high | foundational | 2026-02-20 |
| playing_atari_drl | W1757796397 | Playing Atari with Deep Reinforcement Learning | RL | pioneering | foundational | 2026-02-19 |
| reinforce | W2119717200 | REINFORCE gradient-following algorithms | RL | pioneering | foundational | 2026-02-20 |
| continuous_control_ddpg | W2963864421 | DDPG: Continuous control with deep RL | RL | high | foundational | 2026-02-19 |
| dpg | W2165150801 | Deterministic policy gradient algorithms | RL | high | foundational | 2026-02-20 |
| soft_actor_critic | W2781726626 | Soft Actor-Critic | RL | high | foundational | 2026-02-19 |
| sac_applications | W2904246096 | Soft Actor-Critic Algorithms and Applications | RL | high | applied | 2026-02-20 |
| evolution_strategies | W2596367596 | Evolution Strategies as Alternative to RL | RL | high | foundational | 2026-02-20 |
| visuomotor_policies | W2964161785 | End-to-end training of deep visuomotor policies | robotics | high | foundational | 2026-02-20 |
| domain_randomization | W2605102758 | Domain randomization for sim-to-real transfer | robotics | high | foundational | 2026-02-20 |
| target_driven_nav | W2962887844 | Target-driven visual navigation using deep RL | robotics | high | foundational | 2026-02-20 |
| robotic_grasps | W1999156278 | Deep learning for detecting robotic grasps | robotics | high | foundational | 2026-02-20 |
| drl_robotic_manipulation | W2575705757 | DRL for robotic manipulation | robotics | high | applied | 2026-02-20 |
| tactile_glove | W2947434510 | Learning the signatures of the human grasp | robotics | high | foundational | 2026-02-20 |
| nn_robot_manipulators | W1517236425 | NN Control of Robot Manipulators | robotics | medium | handbook | 2026-02-20 |
| deep_learning_overview | W2076063813 | Deep learning in neural networks: An overview | CS/ML | medium | review | 2026-02-19 |
| drl_brief_survey | W3100789280 | Deep Reinforcement Learning: A Brief Survey | RL | medium | review | 2026-02-20 |
| dnn_tutorial_survey | W2604319603 | Efficient Processing of DNNs: Tutorial and Survey | CS/ML | medium | review | 2026-02-20 |
| cnn_classification_review | W2622826443 | Deep CNNs for Image Classification: Comprehensive Review | CV | medium | review | 2026-02-20 |
| rl_robotics_survey | W1977655452 | Reinforcement learning in robotics: A survey | robotics | medium | review | 2026-02-20 |
| dl_theory_survey | W2919358988 | A State-of-the-Art Survey on DL Theory | CS/ML | medium | review | 2026-02-20 |
| dl_medical_imaging | W2777186991 | Deep Learning in Medical Image Analysis | medical | medium | review | 2026-02-20 |
| drl_that_matters | W2754517384 | Deep Reinforcement Learning That Matters | RL | high | benchmark | 2026-02-20 |
| drl_multiagent_review | W2908261578 | DRL for Multiagent Systems: Review | RL | medium | review | 2026-02-20 |
| offline_rl_tutorial | W3022566517 | Offline RL: Tutorial, Review, and Perspectives | RL | medium | review | 2026-02-20 |
| nn_control_survey_1992 | W1969705022 | Neural networks for control systems: A survey | control | medium | review | 2026-02-20 |
| robotic_surgery | W2077544344 | Robotic Surgery | medical | medium | review | 2026-02-20 |
| visuomotor_policies_v2 | W2155007355 | End-to-End Training of Deep Visuomotor Policies | robotics | high | foundational | 2026-02-20 |

### Grounding Quality Benchmark Papers (12)

| ID | Work ID | Paper | Focus | Added |
|---|---|---|---|---|
| resnet_grounding | W2194775991 | ResNet | CV grounding quality | 2026-02-20 |
| imagenet_grounding | W2108598243 | ImageNet | CV grounding quality | 2026-02-19 |
| faster_rcnn_grounding | W639708223 | Faster R-CNN | CV grounding quality | 2026-02-20 |
| playing_atari_grounding | W1757796397 | Playing Atari | RL grounding quality | 2026-02-19 |
| cyclegan_grounding | W2962793481 | CycleGAN | CV grounding quality | 2026-02-20 |
| ddpg_grounding | W2963864421 | DDPG | RL grounding quality | 2026-02-20 |
| sac_grounding | W2781726626 | SAC | RL grounding quality | 2026-02-20 |
| visuomotor_grounding | W2964161785 | Visuomotor Policies | Robotics grounding | 2026-02-20 |
| robotic_grasps_grounding | W1999156278 | Robotic Grasps | Robotics grounding | 2026-02-20 |
| dl_overview_grounding | W2076063813 | DL Overview | Review grounding | 2026-02-20 |
| rl_robotics_survey_grounding | W1977655452 | RL in Robotics Survey | Survey grounding | 2026-02-20 |
| dl_medical_grounding | W2777186991 | DL Medical Imaging | Medical grounding | 2026-02-20 |

### F3 Coverage Summary

**Domains covered:** CV (9), RL (9), robotics (8), CS/ML (4), medical (2), control (1) — 6 domains
**Novelty levels tested:** pioneering (5), high (18), medium (13) — 3 levels
**Paper types:** foundational (22), review (11), applied (2), benchmark (1), handbook (1) — 5 types
**Total unique F3 papers:** 36 (e2e) + 12 (grounding) = 48 test cases

### F3 Gaps — not yet tested:
- Biology / life sciences papers (CRISPR, AlphaFold)
- Economics / social science papers
- Statistics / methodology papers
- Software tool papers (medium novelty via Q1)
- Low novelty papers (incremental improvements)
- Pioneering papers from non-CS domains
- Papers with missing abstracts (fallback behavior)
- Papers with no grounding data (assessment unavailable path)

---

## How to use this file

When generating new test queries, read this file first to:
1. Avoid duplicating existing queries
2. Fill domain gaps
3. Fill query type gaps
4. Increase difficulty progressively
