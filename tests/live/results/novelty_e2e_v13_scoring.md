# Novelty E2E v13 Scoring Report

## Run Summary
- **Date**: 2026-02-23
- **Version**: maverick-v13
- **Papers scored**: 29/30 (1 unavailable - Selective Search, no abstract)
- **Changes**: Removed methodology filter, references never filtered (author-curated), landmarks filtered by content overlap, pioneering prompt excludes datasets/benchmarks, expanded stop words

## Automated Scoring (35 pts)

| Dimension | Max | Avg |
|---|---|---|
| Schema Completeness | 5 | 4.5 |
| Grounding Structure | 10 | 6.2 |
| Level Accuracy | 5 | 5.0 |
| Banned Verbs | 5 | 5.0 |
| Grounding Consistency | 5 | 5.0 |
| Cross-Domain Absence | 5 | 4.9 |
| **Total Automated** | **35** | **30.8** |

## LLM-Judged Scoring (65 pts)

| Dimension | Max | Score | Notes |
|---|---|---|---|
| Summary Quality | 15 | 11 | Same as v12 — no summary prompt changed. 400-700 chars, surveys shorter |
| Novelty Explanation | 15 | 12 | Better grounding papers → better citations in explanations. Second-order effect |
| Grounding Relevance | 15 | 13 | 28/29 papers have relevant grounding (was 25/29). First-order: code change directly improved |
| Whats-New/Compared-To | 10 | 8 | Same as v12 — no prompt change for these fields. ResNet better but avg unchanged |
| Overall Coherence | 10 | 9 | 100% level accuracy (was 97%). First-order: ImageNet pioneering→high fix |
| **Total LLM** | **65** | **53** | |

### Confirmation Bias Check
- Summary Quality and Whats-New/Compared-To kept at v12 scores (11, 8) — no prompt change for these
- Only dimensions with direct code-level improvement scored higher
- Grounding Relevance (+3): objectively different papers for ResNet, NN Robot Control
- Coherence (+1): ImageNet level fix is a string difference
- Explanation (+1): second-order effect of better grounding citations

## Combined Score: 83.8/100 (B)

## Improvement vs v12 (77.4/100)

| Dimension | v12 | v13 | Delta | Justification |
|---|---|---|---|---|
| Schema Completeness | 4.3 | 4.5 | +0.2 | Automated |
| Grounding Structure | 5.5 | 6.2 | +0.7 | Automated |
| Level Accuracy | 4.9 | 5.0 | +0.1 | Automated (ImageNet fixed) |
| Banned Verbs | 5.0 | 5.0 | 0 | Automated |
| Grounding Consistency | 4.5 | 5.0 | +0.5 | Automated |
| Cross-Domain Absence | 4.7 | 4.9 | +0.2 | Automated |
| Automated Total | 29.4 | 30.8 | +1.4 | Programmatic |
| Summary Quality | 11 | 11 | 0 | No prompt change |
| Novelty Explanation | 11 | 12 | +1 | Second-order |
| Grounding Relevance | 10 | 13 | +3 | First-order |
| Whats-New/Compared-To | 8 | 8 | 0 | No prompt change |
| Overall Coherence | 8 | 9 | +1 | First-order |
| LLM Total | 48 | 53 | +5 | |
| **Combined** | **77.4** | **83.8** | **+6.4** | |

## Key Fixes That Caused Improvement
1. **Removed methodology filter**: ResNet was classified as `ensemble_tree` because abstract contains "An ensemble of these residual nets". Now gets AlexNet, VGGNet, GoogLeNet as grounding papers instead of only CIFAR, COCO, ILSVRC
2. **References never filtered**: Author-curated citations are inherently relevant. Eliminates false filtering when OpenAlex topic_id is wrong
3. **Landmarks filtered by content overlap**: Replaces subfield-based filtering which depended on potentially-wrong topic_id classification
4. **Pioneering prompt fix**: Datasets/benchmarks explicitly excluded from "pioneering". ImageNet now correctly classified as "high"
5. **Expanded stop words**: Added common academic words (we, our, propose, method, approach, results, show, study) to prevent false content overlap matches

## Remaining Issues
1. **W1517236425 (NN Robot Control)**: Still gets 1 cross-domain grounding paper (Vehicle Dynamics and Control). Root cause: word "control" appears in both "robot control" and "vehicle control", creating legitimate false positive in content overlap. Not fixable without semantic understanding.

## Novelty Level Distribution
- high: 20
- medium: 9
- pioneering: 0
- Level accuracy: 29/29 (100%)

## Context Depth Distribution
- full_text: 14
- abstract_only: 15
