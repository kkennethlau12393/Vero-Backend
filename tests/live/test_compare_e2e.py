"""
Live e2e tests for Feature 4 (Methodology Comparison).

Burns API credits: Groq LLM (2 calls per comparison), S2 content fetching,
and external complement retrieval (OpenAlex/S2/ArXiv).

Usage:
    pytest tests/live/test_compare_e2e.py -v -s --timeout=600
    pytest tests/live/test_compare_e2e.py -k "resnet_densenet" -v -s
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List
from uuid import UUID

import pytest
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAP_ROBOTICS = "577e9a86-6fd5-47ee-9589-5f953370ae03"
MAP_CV = "a0000000-0000-0000-0000-000000000001"

RJ_ROBOT_MANIP = "3bd17bd7-4727-40b8-bf9f-c5a8d4b47b2e"
RJ_BATTERY = "c66fbf09-d0d8-4e99-8341-dbb1c3e8234b"
RJ_FEDERATED = "d8b75d65-4c4d-4fc0-a779-2dd07896e951"
RJ_DRUG_GNN = "f2440685-6b3e-4af1-a6c0-a7845a49a73a"
RJ_ATTENTION = "e7246a57-400e-4898-808f-4cf8d09d94bc"

# ---------------------------------------------------------------------------
# 30 Benchmark test cases — diverse domains, mix of abstract+full text
# ---------------------------------------------------------------------------

BENCHMARK_CASES = [
    # === CV Architectures (Map: a000) ===
    {
        "id": "resnet_vs_densenet",
        "desc": "Residual vs Dense connectivity for deep networks",
        "source": "map", "source_id": MAP_CV,
        "work_ids": ["W2194775991", "W2963446712"],
        "domain": "Computer Vision",
    },
    {
        "id": "faster_rcnn_vs_mask_rcnn",
        "desc": "Two-stage object detection evolution",
        "source": "map", "source_id": MAP_CV,
        "work_ids": ["W639708223", "W2963150697"],
        "domain": "Object Detection",
    },
    {
        "id": "senet_vs_resnet",
        "desc": "Channel attention vs residual learning",
        "source": "map", "source_id": MAP_CV,
        "work_ids": ["W2752782242", "W2194775991"],
        "domain": "Computer Vision",
    },
    {
        "id": "cyclegan_vs_mask_rcnn",
        "desc": "Image translation vs instance segmentation (cross-task)",
        "source": "map", "source_id": MAP_CV,
        "work_ids": ["W2962793481", "W2963150697"],
        "domain": "Computer Vision",
    },
    {
        "id": "resnet_densenet_senet_3way",
        "desc": "Three CNN architecture paradigms compared",
        "source": "map", "source_id": MAP_CV,
        "work_ids": ["W2194775991", "W2963446712", "W2752782242"],
        "domain": "Computer Vision",
    },
    {
        "id": "densenet_vs_cyclegan",
        "desc": "Dense features vs cycle consistency (cross-task)",
        "source": "map", "source_id": MAP_CV,
        "work_ids": ["W2963446712", "W2962793481"],
        "domain": "Computer Vision",
    },
    # === Deep RL / Robotics (Map: 577e) ===
    {
        "id": "dqn_vs_ddpg",
        "desc": "Discrete DQN vs continuous DDPG for RL",
        "source": "map", "source_id": MAP_ROBOTICS,
        "work_ids": ["W1757796397", "W2963864421"],
        "domain": "Reinforcement Learning",
    },
    {
        "id": "sac_vs_ddpg",
        "desc": "Maximum entropy RL vs deterministic policy gradient",
        "source": "map", "source_id": MAP_ROBOTICS,
        "work_ids": ["W2781726626", "W2963864421"],
        "domain": "Reinforcement Learning",
    },
    {
        "id": "rcnn_vs_selective_search",
        "desc": "Deep detection vs handcrafted proposals",
        "source": "map", "source_id": MAP_ROBOTICS,
        "work_ids": ["W2102605133", "W2088049833"],
        "domain": "Object Detection",
    },
    {
        "id": "domain_rand_vs_visual_nav",
        "desc": "Sim-to-real transfer vs target-driven navigation",
        "source": "map", "source_id": MAP_ROBOTICS,
        "work_ids": ["W2605102758", "W2962887844"],
        "domain": "Robotics",
    },
    {
        "id": "robot_grasp_lenz_vs_supersizing",
        "desc": "Two approaches to robotic grasping",
        "source": "map", "source_id": MAP_ROBOTICS,
        "work_ids": ["W1999156278", "W2201912979"],
        "domain": "Robotic Grasping",
    },
    {
        "id": "evolution_strategies_vs_reinforce",
        "desc": "Evolution strategies vs policy gradient (abstract-only mix)",
        "source": "map", "source_id": MAP_ROBOTICS,
        "work_ids": ["W2596367596", "W2119717200"],
        "domain": "Policy Optimization",
    },
    {
        "id": "sac_paper_vs_sac_app",
        "desc": "SAC original vs SAC algorithms+applications",
        "source": "map", "source_id": MAP_ROBOTICS,
        "work_ids": ["W2781726626", "W2904246096"],
        "domain": "Reinforcement Learning",
    },
    {
        "id": "deep_rl_survey_vs_deep_learning_overview",
        "desc": "RL survey vs deep learning overview (survey comparison)",
        "source": "map", "source_id": MAP_ROBOTICS,
        "work_ids": ["W3100789280", "W2076063813"],
        "domain": "Deep Learning",
    },
    {
        "id": "visuomotor_vs_async_manip",
        "desc": "End-to-end visuomotor vs async deep RL manipulation",
        "source": "map", "source_id": MAP_ROBOTICS,
        "work_ids": ["W2964161785", "W2575705757"],
        "domain": "Robotic Manipulation",
    },
    {
        "id": "multiagent_rl_vs_offline_rl",
        "desc": "Multi-agent vs offline RL paradigms",
        "source": "map", "source_id": MAP_ROBOTICS,
        "work_ids": ["W2908261578", "W3022566517"],
        "domain": "Reinforcement Learning",
    },
    # === Robotic Manipulation (Rank job) ===
    {
        "id": "dexterous_vs_deformable",
        "desc": "Dexterous in-hand vs deformable object manipulation",
        "source": "rank", "source_id": RJ_ROBOT_MANIP,
        "work_ids": ["W2990747716", "W3040477237"],
        "domain": "Robotic Manipulation",
    },
    {
        "id": "policy_gradient_vs_human_level",
        "desc": "Classical policy gradient vs DQN",
        "source": "rank", "source_id": RJ_ROBOT_MANIP,
        "work_ids": ["W2155027007", "W2145339207"],
        "domain": "Reinforcement Learning",
    },
    {
        "id": "imitation_vs_contact_rich",
        "desc": "Imitation learning vs contact-rich RL review",
        "source": "rank", "source_id": RJ_ROBOT_MANIP,
        "work_ids": ["W2963713397", "W4313442619"],
        "domain": "Robotic Manipulation",
    },
    # === Battery Degradation (Rank job) ===
    {
        "id": "battery_storage_vs_degradation",
        "desc": "Grid storage challenges vs LiB degradation mechanisms",
        "source": "rank", "source_id": RJ_BATTERY,
        "work_ids": ["W2089525884", "W3139510216"],
        "domain": "Battery Science",
    },
    {
        "id": "electrode_deg_vs_modeling",
        "desc": "Electrode degradation vs cell lifetime modeling",
        "source": "rank", "source_id": RJ_BATTERY,
        "work_ids": ["W2996908450", "W2417460170"],
        "domain": "Battery Degradation",
    },
    {
        "id": "lio2_lis_vs_cathode_zinc",
        "desc": "Li-O2/Li-S vs aqueous zinc cathode materials",
        "source": "rank", "source_id": RJ_BATTERY,
        "work_ids": ["W2151207643", "W4366278627"],
        "domain": "Battery Chemistry",
    },
    # === Federated Learning (Rank job) ===
    {
        "id": "fedavg_vs_dp_federated",
        "desc": "FedAvg communication efficiency vs differential privacy",
        "source": "rank", "source_id": RJ_FEDERATED,
        "work_ids": ["W2541884796", "W3016632787"],
        "domain": "Federated Learning",
    },
    {
        "id": "fedml_concept_vs_challenges",
        "desc": "FedML concepts vs challenges survey",
        "source": "rank", "source_id": RJ_FEDERATED,
        "work_ids": ["W2949522309", "W3021654819"],
        "domain": "Federated Learning",
    },
    {
        "id": "noniid_rl_vs_patient_clustering",
        "desc": "RL for non-IID vs patient clustering in federated",
        "source": "rank", "source_id": RJ_FEDERATED,
        "work_ids": ["W3047304572", "W2977072935"],
        "domain": "Federated Learning",
    },
    # === Drug Discovery / GNNs (Rank job) ===
    {
        "id": "potentialnet_vs_gcn_drug",
        "desc": "PotentialNet vs GCN for drug discovery",
        "source": "rank", "source_id": RJ_DRUG_GNN,
        "work_ids": ["W2895884529", "W2948035163"],
        "domain": "Drug Discovery",
    },
    {
        "id": "smiles_bert_vs_graph_transformer",
        "desc": "SMILES-BERT vs algebraic graph transformers",
        "source": "rank", "source_id": RJ_DRUG_GNN,
        "work_ids": ["W2973114758", "W3166272013"],
        "domain": "Molecular Representation",
    },
    {
        "id": "struct_drug_design_vs_attention_gnn",
        "desc": "Structure-based drug design vs attention GNN",
        "source": "rank", "source_id": RJ_DRUG_GNN,
        "work_ids": ["W4321769975", "W4315708854"],
        "domain": "Drug Discovery",
    },
    # === Attention Mechanisms (Rank job) ===
    {
        "id": "dilated_attention_vs_twins",
        "desc": "Dilated neighborhood attention vs Twins spatial attention",
        "source": "rank", "source_id": RJ_ATTENTION,
        "work_ids": ["AX:2209.15001", "W3157528469"],
        "domain": "Vision Transformers",
    },
    {
        "id": "time_freq_transformer_vs_transformers_survey",
        "desc": "Time-frequency transformer vs comprehensive transformer survey",
        "source": "rank", "source_id": RJ_ATTENTION,
        "work_ids": ["AX:2104.09079", "W4380992741"],
        "domain": "Transformers",
    },
]

assert len(BENCHMARK_CASES) == 30, f"Expected 30 cases, got {len(BENCHMARK_CASES)}"


# ---------------------------------------------------------------------------
# Automated Scoring (35 pts)
# ---------------------------------------------------------------------------

def compute_automated_score(result: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    """Compute 35-point automated score for a methodology comparison result."""
    scores = {}
    violations = []

    # 1. Schema Completeness (5 pts)
    schema_score = 5
    required_top = ["work_ids", "papers", "lineage", "convergence_divergence",
                    "strengths_weaknesses_matrix", "recommendation", "confidence"]
    for field in required_top:
        if field not in result or result[field] is None:
            schema_score -= 1
            violations.append(f"Missing top-level field: {field}")
    scores["schema_completeness"] = max(0, schema_score)

    # 2. Fingerprint Depth (5 pts)
    fp_score = 5
    papers = result.get("papers", [])
    if not papers:
        fp_score = 0
        violations.append("No paper profiles returned")
    else:
        for p in papers:
            fp = p.get("methodology_fingerprint", {})
            approach = fp.get("approach", "") or ""
            if len(approach) < 200:
                fp_score -= 1
                violations.append(f"{p.get('work_id', '?')}: approach too short ({len(approach)} chars)")
            if len(fp.get("key_components", [])) < 2:
                fp_score -= 0.5
                violations.append(f"{p.get('work_id', '?')}: fewer than 2 key_components")
            if len(fp.get("limitations", [])) < 1:
                fp_score -= 0.5
                violations.append(f"{p.get('work_id', '?')}: no limitations extracted")
    scores["fingerprint_depth"] = max(0, min(5, fp_score))

    # 3. Convergence/Divergence Quality (5 pts)
    cd_score = 5
    cd = result.get("convergence_divergence", {})
    cp = cd.get("common_problem", {})
    if not cp.get("domain"):
        cd_score -= 1; violations.append("Missing common_problem.domain")
    if not cp.get("challenge"):
        cd_score -= 1; violations.append("Missing common_problem.challenge")
    if not cp.get("why_hard"):
        cd_score -= 1; violations.append("Missing common_problem.why_hard")
    paradigms = cd.get("paradigms", [])
    if len(paradigms) < 1:
        cd_score -= 1; violations.append("No paradigms identified")
    if not cd.get("divergence_summary"):
        cd_score -= 1; violations.append("No divergence_summary")
    scores["convergence_divergence"] = max(0, cd_score)

    # 4. Strengths/Weaknesses Matrix (5 pts)
    sw_score = 5
    sw_matrix = result.get("strengths_weaknesses_matrix", [])
    input_wids = set(result.get("work_ids", []))
    covered_wids = {sw.get("work_id") for sw in sw_matrix}
    missing = input_wids - covered_wids
    if missing:
        sw_score -= 2; violations.append(f"SW matrix missing: {missing}")
    for sw in sw_matrix:
        wid = sw.get("work_id", "?")
        if not sw.get("handles_well"):
            sw_score -= 0.5; violations.append(f"{wid}: no handles_well")
        if not sw.get("struggles_with"):
            sw_score -= 0.5; violations.append(f"{wid}: no struggles_with")
        if not sw.get("assumptions"):
            sw_score -= 0.5; violations.append(f"{wid}: no assumptions")
    scores["sw_matrix"] = max(0, min(5, sw_score))

    # 5. Self-Reference Absence (5 pts)
    sr_score = 5
    for sw in sw_matrix:
        wid = sw.get("work_id", "")
        if not wid:
            continue
        # Check handles_well, struggles_with, assumptions for self-refs
        for h in sw.get("handles_well", []):
            for field in ["mechanism", "evidence", "capability"]:
                text = h.get(field, "") or ""
                if wid in text:
                    sr_score -= 0.5
                    violations.append(f"{wid} self-ref in handles_well.{field}")
        for s in sw.get("struggles_with", []):
            for field in ["limitation", "cause", "consequence"]:
                text = s.get(field, "") or ""
                if wid in text:
                    sr_score -= 0.5
                    violations.append(f"{wid} self-ref in struggles_with.{field}")
        for a in sw.get("assumptions", []):
            for field in ["assumption", "if_violated"]:
                text = a.get(field, "") or ""
                if wid in text:
                    sr_score -= 0.5
                    violations.append(f"{wid} self-ref in assumptions.{field}")
    scores["self_ref_absence"] = max(0, min(5, sr_score))

    # 6. Recommendation Quality (5 pts)
    rec_score = 5
    rec = result.get("recommendation", {})
    if not rec.get("summary"):
        rec_score -= 1; violations.append("No recommendation summary")
    dm = rec.get("decision_matrix", [])
    if len(dm) < 3:
        rec_score -= 2; violations.append(f"Only {len(dm)} decision scenarios (need 3+)")
    for i, d in enumerate(dm):
        if not d.get("why") or len(d.get("why", "")) < 40:
            rec_score -= 0.5
            violations.append(f"Decision {i+1} 'why' too short")
    if "can_combine" not in rec:
        rec_score -= 1; violations.append("Missing can_combine field")
    scores["recommendation"] = max(0, min(5, rec_score))

    # 7. Technical Depth (5 pts) — no shallow patterns
    depth_score = 5
    import re
    shallow_re = re.compile(
        r'(provides? (?:a |the )?(?:method|way|approach|technique|means)\b|'
        r'(?:increased?|reduced?|improved?|better|worse) (?:computational|performance|quality|efficiency)\b|'
        r'(?:high|low|good|bad|better|worse)[\s-]quality\b)',
        re.IGNORECASE,
    )
    full_text = json.dumps(result)
    shallow_matches = shallow_re.findall(full_text)
    if shallow_matches:
        depth_score -= min(5, len(shallow_matches))
        violations.append(f"Shallow language found ({len(shallow_matches)}x): {shallow_matches[:3]}")
    # Check coverage depth
    for sw in sw_matrix:
        for c in sw.get("complemented_by", []):
            cov = c.get("coverage", "")
            if len(cov) < 80:
                depth_score -= 0.5
                violations.append(f"Shallow complemented_by coverage ({len(cov)} chars)")
    scores["technical_depth"] = max(0, min(5, depth_score))

    total = sum(scores.values())
    return {
        "total_automated": round(total, 1),
        "max_automated": 35,
        "breakdown": {k: round(v, 1) for k, v in scores.items()},
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def comparison_engine():
    from app.db import make_engine
    engine = make_engine()
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def tenant_id() -> UUID:
    return UUID(os.getenv("DEV_TENANT_ID", "00000000-0000-0000-0000-000000000001"))


@pytest.mark.live
@pytest.mark.parametrize("case", BENCHMARK_CASES, ids=[c["id"] for c in BENCHMARK_CASES])
def test_methodology_comparison(case, comparison_engine, tenant_id, results_dir):
    """Run a methodology comparison and compute automated score."""
    from app.feature4.compare_service import (
        compare_methodologies,
        compare_methodologies_for_rank_job,
    )

    case_id = case["id"]
    work_ids = case["work_ids"]
    source = case["source"]
    source_id = case["source_id"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Test: {case_id} — {case['desc']}")
    logger.info(f"Papers: {work_ids}")
    logger.info(f"Source: {source} {source_id[:12]}...")
    logger.info(f"{'='*60}")

    start = time.time()

    try:
        if source == "map":
            result = compare_methodologies(
                engine=comparison_engine,
                tenant_id=tenant_id,
                map_id=source_id,
                work_ids=work_ids,
            )
        else:
            result = compare_methodologies_for_rank_job(
                engine=comparison_engine,
                tenant_id=tenant_id,
                rank_job_id=source_id,
                work_ids=work_ids,
            )
    except Exception as e:
        # Save error for manual review
        error_data = {
            "case_id": case_id,
            "desc": case["desc"],
            "domain": case["domain"],
            "work_ids": work_ids,
            "error": str(e),
            "elapsed_s": round(time.time() - start, 1),
        }
        save_path = results_dir / f"compare_{case_id}_latest.json"
        save_path.write_text(json.dumps(error_data, indent=2, default=str))
        pytest.fail(f"Comparison failed: {e}")

    elapsed = round(time.time() - start, 1)
    result_dict = result.model_dump()

    # Compute automated score
    auto_score = compute_automated_score(result_dict, case)

    # Build summary for review
    summary = {
        "case_id": case_id,
        "desc": case["desc"],
        "domain": case["domain"],
        "work_ids": work_ids,
        "elapsed_s": elapsed,
        "confidence": result_dict.get("confidence"),
        "source_qualities": [
            {"work_id": p["work_id"], "quality": p["source_quality"]}
            for p in result_dict.get("papers", [])
        ],
        "automated_score": auto_score,
        "full_result": result_dict,
    }

    # Save results
    save_path = results_dir / f"compare_{case_id}_latest.json"
    save_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"Saved: {save_path}")

    # Print summary
    print(f"\n--- {case_id} ({elapsed}s) ---")
    print(f"  Confidence: {result_dict.get('confidence')}")
    print(f"  Automated: {auto_score['total_automated']}/35")
    for dim, score in auto_score["breakdown"].items():
        max_pts = 5
        print(f"    {dim}: {score}/{max_pts}")
    if auto_score["violations"]:
        print(f"  Violations ({len(auto_score['violations'])}):")
        for v in auto_score["violations"][:5]:
            print(f"    - {v}")

    # Assertions
    assert auto_score["total_automated"] >= 20, (
        f"Automated score too low: {auto_score['total_automated']}/35. "
        f"Violations: {auto_score['violations'][:5]}"
    )


# ---------------------------------------------------------------------------
# Aggregate scoring (run after all individual tests)
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_aggregate_scores(results_dir):
    """Aggregate all comparison scores and check overall quality."""
    import glob

    result_files = sorted(results_dir.glob("compare_*_latest.json"))
    if not result_files:
        pytest.skip("No comparison results found")

    total_auto = 0
    count = 0
    all_scores = []

    for f in result_files:
        data = json.loads(f.read_text())
        if "error" in data:
            continue
        auto = data.get("automated_score", {})
        if "total_automated" in auto:
            total_auto += auto["total_automated"]
            count += 1
            all_scores.append({
                "case_id": data.get("case_id"),
                "auto_score": auto["total_automated"],
                "confidence": data.get("confidence"),
                "violations": len(auto.get("violations", [])),
            })

    if count == 0:
        pytest.skip("No valid results to aggregate")

    avg_auto = round(total_auto / count, 1)

    print(f"\n{'='*60}")
    print(f"AGGREGATE METHODOLOGY COMPARISON SCORES")
    print(f"{'='*60}")
    print(f"Tests completed: {count}")
    print(f"Average automated score: {avg_auto}/35")
    print(f"\nPer-case scores:")
    for s in sorted(all_scores, key=lambda x: x["auto_score"]):
        print(f"  {s['case_id']:40s} {s['auto_score']:5.1f}/35  conf={s['confidence']}  violations={s['violations']}")

    # Save aggregate
    agg_path = results_dir / "compare_aggregate_latest.json"
    agg_path.write_text(json.dumps({
        "count": count,
        "avg_automated": avg_auto,
        "per_case": all_scores,
    }, indent=2))

    assert avg_auto >= 25, f"Average automated score {avg_auto}/35 is below threshold (25)"


def save_result(results_dir, name, data):
    """Local save helper."""
    path = results_dir / f"{name}_latest.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    return path
