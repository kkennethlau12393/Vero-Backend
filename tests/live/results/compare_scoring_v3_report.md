# Feature 4: Methodology Comparison — v3-robust Scoring Report

**Date**: 2026-02-24
**Pipeline Version**: `v3-robust` (COMPARISON_VERSION)
**Baseline**: `v2-depth` (81.3/100 avg)
**Test Count**: 30 benchmark cases

---

## TL;DR — v3-robust REGRESSED vs v2-depth

| Metric | v2-depth | v3-robust | Delta |
|--------|----------|-----------|-------|
| Automated avg (35 pts) | 34.4 | 33.9 | **-0.5** |
| LLM-judged avg (65 pts) | 46.9 | 41.7 | **-5.2** |
| **Combined avg (100 pts)** | **81.3** | **75.6** | **-5.7** |
| Catastrophic failures (<55) | 2 | 5 | **+3** |
| Non-catastrophic avg | 83.8 | 80.7 | **-3.1** |

**Root cause**: The stricter validation rules (expanded shallow pattern regex, circular complement check) cause the LLM to fail all 4 retry attempts on 3 cases that previously passed with lighter validation. The visuomotor fix worked (+29 pts) but was offset by 3 new catastrophic failures.

---

## What IMPROVED in v3-robust

| Case | v2 Score | v3 Score | Delta | Why |
|------|----------|----------|-------|-----|
| visuomotor_vs_async_manip | 50 | 79 | **+29** | Low-overlap detection + relaxed validation → output instead of empty |
| sac_vs_ddpg | 82 | 84 | **+2** | Shallow language fixed (34→35 auto) |
| deep_rl_survey_vs_deep_learning_overview | 80 | 72 | -8 | Survey detection active but LLM quality was lower this run |

**visuomotor fix is the clear win** — the low-overlap detection correctly identified these papers as coming from different sub-domains and applied relaxed validation, allowing the LLM to produce output.

## What REGRESSED in v3-robust

| Case | v2 Score | v3 Score | Delta | Root Cause |
|------|----------|----------|-------|------------|
| resnet_densenet_senet_3way | 84 | 40 | **-44** | NEW catastrophic: empty SW matrix, paradigms, recommendations |
| robot_grasp_lenz_vs_supersizing | 83 | 44 | **-39** | NEW catastrophic: empty output |
| multiagent_rl_vs_offline_rl | 83 | 44 | **-39** | NEW catastrophic: empty output |
| time_freq_transformer_vs_transformers_survey | 52 | 45 | **-7** | Still catastrophic despite survey detection |
| faster_rcnn_vs_mask_rcnn | 82 | 77 | **-5** | Slightly lower LLM quality |
| rcnn_vs_selective_search | 82 | 77 | **-5** | Lower LLM quality |

**3 new catastrophic failures are the primary damage.** These cases previously worked under v2-depth's lighter validation. The expanded shallow pattern regex and/or circular complement check likely cause validation to reject outputs that v2-depth would have accepted, exhausting all 4 retries.

---

## Per-Case Scoring (All 30 Cases)

### CV Architecture Comparisons (Map a000)

| # | Case | Auto /35 | LLM /65 | Total /100 | v2 Total | Delta |
|---|------|----------|---------|------------|----------|-------|
| 1 | resnet_vs_densenet | 35 | 59 | **94** | 94 | 0 |
| 2 | senet_vs_resnet | 35 | 49 | **84** | 88 | -4 |
| 3 | densenet_vs_cyclegan | 35 | 49 | **84** | 83 | +1 |
| 4 | faster_rcnn_vs_mask_rcnn | 32 | 45 | **77** | 82 | -5 |
| 5 | cyclegan_vs_mask_rcnn | 33 | 45 | **78** | 76 | +2 |
| 6 | rcnn_vs_selective_search | 32 | 45 | **77** | 82 | -5 |

**Subtotal avg**: 33.7 auto / 48.7 LLM = **82.3/100** (v2: 84.2, delta: -1.9)

### Deep RL & Robotics Comparisons (Map 577e)

| # | Case | Auto /35 | LLM /65 | Total /100 | v2 Total | Delta |
|---|------|----------|---------|------------|----------|-------|
| 7 | dqn_vs_ddpg | 35 | 51 | **86** | 87 | -1 |
| 8 | sac_vs_ddpg | 35 | 49 | **84** | 82 | +2 |
| 9 | policy_gradient_vs_human_level | 35 | 45 | **80** | 80 | 0 |
| 10 | evolution_strategies_vs_reinforce | 35 | 47 | **82** | 85 | -3 |
| 11 | multiagent_rl_vs_offline_rl | 30 | 14 | **44** | 83 | **-39** |
| 12 | deep_rl_survey_vs_deep_learning_overview | 35 | 37 | **72** | 80 | -8 |
| 13 | domain_rand_vs_visual_nav | 35 | 48 | **83** | 85 | -2 |
| 14 | sac_paper_vs_sac_app | 34 | 51 | **85** | 87 | -2 |
| 15 | visuomotor_vs_async_manip | 35 | 44 | **79** | 50 | **+29** |
| 16 | time_freq_transformer_vs_transformers_survey | 30 | 15 | **45** | 52 | -7 |

**Subtotal avg**: 33.9 auto / 40.1 LLM = **74.0/100** (v2: 77.1, delta: -3.1)

### Robotic Manipulation (Rank Job)

| # | Case | Auto /35 | LLM /65 | Total /100 | v2 Total | Delta |
|---|------|----------|---------|------------|----------|-------|
| 17 | robot_grasp_lenz_vs_supersizing | 30 | 14 | **44** | 83 | **-39** |
| 18 | dexterous_vs_deformable | 35 | 44 | **79** | 82 | -3 |
| 19 | imitation_vs_contact_rich | 34.5 | 40 | **74.5** | 79.5 | -5 |

**Subtotal avg**: 33.2 auto / 32.7 LLM = **65.8/100** (v2: 81.5, delta: -15.7)

### Battery Degradation (Rank Job)

| # | Case | Auto /35 | LLM /65 | Total /100 | v2 Total | Delta |
|---|------|----------|---------|------------|----------|-------|
| 20 | battery_storage_vs_degradation | 35 | 42 | **77** | 80 | -3 |
| 21 | electrode_deg_vs_modeling | 35 | 44 | **79** | 75 | +4 |
| 22 | lio2_lis_vs_cathode_zinc | 34.5 | 35 | **69.5** | 67.5 | +2 |

**Subtotal avg**: 34.8 auto / 40.3 LLM = **75.2/100** (v2: 74.2, delta: +1.0)

### Federated Learning (Rank Job)

| # | Case | Auto /35 | LLM /65 | Total /100 | v2 Total | Delta |
|---|------|----------|---------|------------|----------|-------|
| 23 | fedavg_vs_dp_federated | 35 | 54 | **89** | 89 | 0 |
| 24 | fedml_concept_vs_challenges | 35 | 47 | **82** | 82 | 0 |
| 25 | noniid_rl_vs_patient_clustering | 34 | 47 | **81** | 77 | +4 |

**Subtotal avg**: 34.7 auto / 49.3 LLM = **84.0/100** (v2: 82.7, delta: +1.3)

### Drug Discovery GNNs (Rank Job)

| # | Case | Auto /35 | LLM /65 | Total /100 | v2 Total | Delta |
|---|------|----------|---------|------------|----------|-------|
| 26 | potentialnet_vs_gcn_drug | 35 | 40 | **75** | 79 | -4 |
| 27 | smiles_bert_vs_graph_transformer | 34 | 51 | **85** | 82 | +3 |
| 28 | struct_drug_design_vs_attention_gnn | 35 | 47 | **82** | 81 | +1 |

**Subtotal avg**: 34.7 auto / 46.0 LLM = **80.7/100** (v2: 80.7, delta: 0.0)

### Attention Mechanisms (Rank Job)

| # | Case | Auto /35 | LLM /65 | Total /100 | v2 Total | Delta |
|---|------|----------|---------|------------|----------|-------|
| 29 | dilated_attention_vs_twins | 33 | 43 | **76** | 79 | -3 |
| 30 | resnet_densenet_senet_3way | 30 | 10 | **40** | 84 | **-44** |

**Subtotal avg**: 31.5 auto / 26.5 LLM = **58.0/100** (v2: 81.5, delta: -23.5)

---

## Failure Analysis

### NEW Catastrophic Failures (3 cases — did NOT fail in v2-depth)

These are the critical regressions:

**1. resnet_densenet_senet_3way** (40/100, was 84)
- Empty SW matrix, paradigms, and recommendations
- This is a 3-way comparison (unusual — most are 2-way)
- Hypothesis: the stricter validation rejects the LLM's output on all 4 retries because 3-way comparisons generate more potential violations (more shallow patterns, more circular complement opportunities)

**2. robot_grasp_lenz_vs_supersizing** (44/100, was 83)
- Empty output
- Two closely related robotics papers
- Not a survey, not low-overlap → relaxed mode NOT triggered
- Hypothesis: the expanded shallow regex or circular complement check catches something in every retry

**3. multiagent_rl_vs_offline_rl** (44/100, was 83)
- Empty output
- Two RL sub-domains with reasonable overlap
- Same hypothesis as above — stricter validation exhausts retries

### Persistent Catastrophic Failures (2 cases — also failed in v2)

**4. time_freq_transformer_vs_transformers_survey** (45/100, was 52)
- Survey detection should have triggered relaxed mode, but output is still empty
- Root cause needs investigation — is `_detect_paper_type()` correctly classifying the survey paper?

**5. visuomotor_vs_async_manip** — **FIXED** (79/100, was 50)
- This is now a success. Low-overlap detection correctly triggered relaxed validation.

---

## Diagnosis: Why v3-robust Regressed

The core issue is that **the expanded validation catches more violations but doesn't give the LLM enough guidance to fix them**. The retry loop:

1. LLM generates output
2. `_validate_synthesis()` catches violations (shallow patterns, circular complements, etc.)
3. Violations fed back to LLM as corrections
4. LLM regenerates — but the regenerated output often triggers DIFFERENT violations
5. After 4 retries, fallback to empty output

In v2-depth, the simpler validation accepted outputs that v3-robust now rejects. The rejected outputs weren't perfect, but they were far better than empty output (scoring 82-84 vs 40-44).

### The Fundamental Trade-off

```
Stricter validation → catches more quality issues → but more retries fail → more empty output → lower overall scores
Lighter validation  → accepts lower quality → but fewer empty outputs → higher average scores
```

v3-robust moved too far toward "strict" without improving the LLM's ability to satisfy the stricter rules.

---

## Recommendations for v4

1. **Relax validation for non-edge cases too**: The `relaxed` flag currently only triggers for surveys and low-overlap papers. Extend it: if retry 3 or 4 is reached for ANY case, switch to relaxed validation to avoid empty fallback.

2. **Never return empty output**: If all 4 retries fail validation, return the BEST-scoring retry (fewest violations) instead of empty. A 75/100 output with some shallow language is far better than a 40/100 empty output.

3. **Reduce expanded shallow pattern scope**: The 3 new patterns (`significantly more/less`, `leads to better`, `more/less efficient`) may be too aggressive. Consider removing them or only flagging (not rejecting) when they appear.

4. **Investigate the 3 new failures specifically**: Read the actual retry logs for `resnet_densenet_senet_3way`, `robot_grasp`, and `multiagent_rl` to see which validation rule is triggering the rejections.

5. **Keep the wins**: Survey detection, anti-hallucination rule, and circular complement detection are good additions — the issue is validation strictness, not the rules themselves.

---

## Score Distribution Comparison

```
              v2-depth          v3-robust
90-100:  ██████  2 (7%)    ██████  2 (7%)
80-89:   ████████████████████ 16 (53%)  ████████████████ 13 (43%)
70-79:   ██████████  6 (20%)  ████████████  8 (27%)
60-69:   ██  1 (3%)    ██  1 (3%)
50-59:   ████  2 (7%)    0 (0%)
40-49:   0 (0%)      ██████  3 (10%)
<40:     0 (0%)      ██  1 (3%)
N/A:     ██████  3 (10%)   ██  2 (7%)
```

The distribution shifted: v3 has fewer 80-89 cases and more 40-49 cases (the new catastrophic failures).

---

## Honest Assessment

**v3-robust is a net regression.** The fixes addressed real problems (visuomotor now works, survey detection is sound, anti-hallucination rule is correct) but the expanded validation strictness introduced worse problems than it solved.

**Score: 75.6/100 vs 81.3/100 baseline — FAIL.**

The path forward is clear: keep the detection logic (survey, overlap, circular complements) but make the validation more forgiving — return best-attempt instead of empty, and soften the retry threshold.
