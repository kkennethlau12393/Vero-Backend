"""
End-to-end novelty assessment quality tests.

Hits real /v1/maps/{map_id}/nodes/{work_id}/details?include_novelty=true
Uses existing maps and nodes from the dev DB (created by F1/F2 pipelines).
Results saved to tests/live/results/ for manual LLM-judged scoring.

Mark: @pytest.mark.live

WARNING: Burns API credits (Groq, OpenAlex, S2, ArXiv)
"""
import json
import os
from pathlib import Path
from uuid import UUID

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=True)

from app.feature3.node_details_service import _has_banned_verbs


# ============================================================================
# Benchmark Papers — use existing nodes from citation maps / rank jobs
# ============================================================================

BENCHMARK_PAPERS = [
    # ── Pioneering / Foundational (>50k cites) ────────────────────────────
    {
        "id": "resnet",
        "work_id": "W2194775991",
        "expected_novelty": "pioneering",
        "expected_novelty_alt": ["high"],
        "domain": "CV",
        "paper_type": "foundational",
        "difficulty": "easy",
    },
    {
        "id": "imagenet",
        "work_id": "W2108598243",
        "expected_novelty": "pioneering",
        "expected_novelty_alt": ["high"],
        "domain": "CV",
        "paper_type": "foundational",
        "difficulty": "easy",
    },
    {
        "id": "faster_rcnn",
        "work_id": "W639708223",
        "expected_novelty": "pioneering",
        "expected_novelty_alt": ["high"],
        "domain": "CV",
        "paper_type": "foundational",
        "difficulty": "easy",
    },
    # ── High Novelty — Computer Vision ────────────────────────────────────
    {
        "id": "densenet",
        "work_id": "W2963446712",
        "expected_novelty": "high",
        "expected_novelty_alt": ["pioneering"],
        "domain": "CV",
        "paper_type": "foundational",
        "difficulty": "easy",
    },
    {
        "id": "rcnn",
        "work_id": "W2102605133",
        "expected_novelty": "high",
        "expected_novelty_alt": ["pioneering"],
        "domain": "CV",
        "paper_type": "foundational",
        "difficulty": "easy",
    },
    {
        "id": "mask_rcnn",
        "work_id": "W2963150697",
        "expected_novelty": "high",
        "expected_novelty_alt": ["pioneering"],
        "domain": "CV",
        "paper_type": "foundational",
        "difficulty": "easy",
    },
    {
        "id": "senet",
        "work_id": "W2752782242",
        "expected_novelty": "high",
        "expected_novelty_alt": [],
        "domain": "CV",
        "paper_type": "foundational",
        "difficulty": "medium",
    },
    {
        "id": "cyclegan",
        "work_id": "W2962793481",
        "expected_novelty": "high",
        "expected_novelty_alt": ["pioneering"],
        "domain": "CV",
        "paper_type": "foundational",
        "difficulty": "easy",
    },
    {
        "id": "selective_search",
        "work_id": "W2088049833",
        "expected_novelty": "high",
        "expected_novelty_alt": [],
        "domain": "CV",
        "paper_type": "foundational",
        "difficulty": "medium",
    },
    # ── High Novelty — Reinforcement Learning ─────────────────────────────
    {
        "id": "playing_atari_drl",
        "work_id": "W1757796397",
        "expected_novelty": "pioneering",
        "expected_novelty_alt": ["high"],
        "domain": "RL",
        "paper_type": "foundational",
        "difficulty": "easy",
    },
    {
        "id": "reinforce",
        "work_id": "W2119717200",
        "expected_novelty": "pioneering",
        "expected_novelty_alt": ["high"],
        "domain": "RL",
        "paper_type": "foundational",
        "difficulty": "medium",
    },
    {
        "id": "continuous_control_ddpg",
        "work_id": "W2963864421",
        "expected_novelty": "high",
        "expected_novelty_alt": ["pioneering"],
        "domain": "RL",
        "paper_type": "foundational",
        "difficulty": "easy",
    },
    {
        "id": "dpg",
        "work_id": "W2165150801",
        "expected_novelty": "high",
        "expected_novelty_alt": [],
        "domain": "RL",
        "paper_type": "foundational",
        "difficulty": "medium",
    },
    {
        "id": "soft_actor_critic",
        "work_id": "W2781726626",
        "expected_novelty": "high",
        "expected_novelty_alt": [],
        "domain": "RL",
        "paper_type": "foundational",
        "difficulty": "medium",
    },
    {
        "id": "sac_applications",
        "work_id": "W2904246096",
        "expected_novelty": "high",
        "expected_novelty_alt": ["medium"],
        "domain": "RL",
        "paper_type": "applied",
        "difficulty": "medium",
    },
    {
        "id": "evolution_strategies",
        "work_id": "W2596367596",
        "expected_novelty": "high",
        "expected_novelty_alt": [],
        "domain": "RL",
        "paper_type": "foundational",
        "difficulty": "medium",
    },
    # ── High Novelty — Robotics / Applied ─────────────────────────────────
    {
        "id": "visuomotor_policies",
        "work_id": "W2964161785",
        "expected_novelty": "high",
        "expected_novelty_alt": [],
        "domain": "robotics",
        "paper_type": "foundational",
        "difficulty": "medium",
    },
    {
        "id": "domain_randomization",
        "work_id": "W2605102758",
        "expected_novelty": "high",
        "expected_novelty_alt": [],
        "domain": "robotics",
        "paper_type": "foundational",
        "difficulty": "medium",
    },
    {
        "id": "target_driven_nav",
        "work_id": "W2962887844",
        "expected_novelty": "high",
        "expected_novelty_alt": [],
        "domain": "robotics",
        "paper_type": "foundational",
        "difficulty": "medium",
    },
    {
        "id": "robotic_grasps",
        "work_id": "W1999156278",
        "expected_novelty": "high",
        "expected_novelty_alt": [],
        "domain": "robotics",
        "paper_type": "foundational",
        "difficulty": "medium",
    },
    {
        "id": "drl_robotic_manipulation",
        "work_id": "W2575705757",
        "expected_novelty": "high",
        "expected_novelty_alt": [],
        "domain": "robotics",
        "paper_type": "applied",
        "difficulty": "medium",
    },
    {
        "id": "tactile_glove",
        "work_id": "W2947434510",
        "expected_novelty": "high",
        "expected_novelty_alt": [],
        "domain": "robotics",
        "paper_type": "foundational",
        "difficulty": "hard",
    },
    {
        "id": "nn_robot_manipulators",
        "work_id": "W1517236425",
        "expected_novelty": "medium",
        "expected_novelty_alt": ["high"],
        "domain": "robotics",
        "paper_type": "handbook",
        "difficulty": "hard",
    },
    # ── Medium Novelty — Reviews & Surveys ────────────────────────────────
    {
        "id": "deep_learning_overview",
        "work_id": "W2076063813",
        "expected_novelty": "medium",
        "expected_novelty_alt": ["low"],
        "domain": "CS/ML",
        "paper_type": "review",
        "difficulty": "medium",
    },
    {
        "id": "drl_brief_survey",
        "work_id": "W3100789280",
        "expected_novelty": "medium",
        "expected_novelty_alt": ["low"],
        "domain": "RL",
        "paper_type": "review",
        "difficulty": "easy",
    },
    {
        "id": "dnn_tutorial_survey",
        "work_id": "W2604319603",
        "expected_novelty": "medium",
        "expected_novelty_alt": ["low"],
        "domain": "CS/ML",
        "paper_type": "review",
        "difficulty": "easy",
    },
    {
        "id": "cnn_classification_review",
        "work_id": "W2622826443",
        "expected_novelty": "medium",
        "expected_novelty_alt": ["low"],
        "domain": "CV",
        "paper_type": "review",
        "difficulty": "easy",
    },
    {
        "id": "rl_robotics_survey",
        "work_id": "W1977655452",
        "expected_novelty": "medium",
        "expected_novelty_alt": ["low"],
        "domain": "robotics",
        "paper_type": "review",
        "difficulty": "easy",
    },
    {
        "id": "dl_theory_survey",
        "work_id": "W2919358988",
        "expected_novelty": "medium",
        "expected_novelty_alt": ["low"],
        "domain": "CS/ML",
        "paper_type": "review",
        "difficulty": "easy",
    },
    {
        "id": "dl_medical_imaging",
        "work_id": "W2777186991",
        "expected_novelty": "medium",
        "expected_novelty_alt": ["low", "high"],
        "domain": "medical",
        "paper_type": "review",
        "difficulty": "medium",
    },
    {
        "id": "drl_that_matters",
        "work_id": "W2754517384",
        "expected_novelty": "high",
        "expected_novelty_alt": ["medium"],
        "domain": "RL",
        "paper_type": "benchmark",
        "difficulty": "medium",
    },
    {
        "id": "drl_multiagent_review",
        "work_id": "W2908261578",
        "expected_novelty": "medium",
        "expected_novelty_alt": ["low"],
        "domain": "RL",
        "paper_type": "review",
        "difficulty": "easy",
    },
    {
        "id": "offline_rl_tutorial",
        "work_id": "W3022566517",
        "expected_novelty": "medium",
        "expected_novelty_alt": ["low"],
        "domain": "RL",
        "paper_type": "review",
        "difficulty": "easy",
    },
    {
        "id": "nn_control_survey_1992",
        "work_id": "W1969705022",
        "expected_novelty": "medium",
        "expected_novelty_alt": ["high"],
        "domain": "control",
        "paper_type": "review",
        "difficulty": "hard",
    },
    {
        "id": "robotic_surgery",
        "work_id": "W2077544344",
        "expected_novelty": "medium",
        "expected_novelty_alt": ["low", "high"],
        "domain": "medical",
        "paper_type": "review",
        "difficulty": "hard",
    },
    {
        "id": "visuomotor_policies_v2",
        "work_id": "W2155007355",
        "expected_novelty": "high",
        "expected_novelty_alt": [],
        "domain": "robotics",
        "paper_type": "foundational",
        "difficulty": "medium",
    },
]


# ============================================================================
# Helpers
# ============================================================================

def _find_map_for_work(engine, work_id: str):
    """Find an existing map_id + tenant_id that contains this work_id."""
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT mn.map_id, m.tenant_id
                FROM map_nodes mn
                JOIN maps m ON m.map_id = mn.map_id
                WHERE mn.work_id = :work_id
                LIMIT 1
            """),
            {"work_id": work_id},
        ).mappings().first()
        if row:
            return str(row["map_id"]), str(row["tenant_id"])
        return None, None


def _compute_automated_score(response_data, expected):
    """
    Compute 35-point automated score per the rubric.

    Returns (score, breakdown_dict)
    """
    score = 0
    breakdown = {}

    novelty = response_data.get("novelty_assessment") or {}
    gps = novelty.get("grounding_papers", [])

    # 1. Schema Completeness (5 pts)
    schema_score = 5
    if not novelty.get("novelty_level"):
        schema_score -= 2
    if not novelty.get("confidence"):
        schema_score -= 1
    if not response_data.get("summary") or len(response_data.get("summary", "")) < 50:
        schema_score -= 1
    if not response_data.get("keywords") or len(response_data.get("keywords", [])) < 3:
        schema_score -= 1
    schema_score = max(0, schema_score)
    breakdown["schema"] = schema_score
    score += schema_score

    # 2. Grounding Paper Structure (10 pts)
    gp_score = 10
    gp_count = len(gps)
    if gp_count < 3:
        gp_score = 0
    elif gp_count < 5:
        gp_score = 4
    elif gp_count > 7:
        gp_score -= 2

    landmark_count = sum(1 for gp in gps if gp.get("relationship") == "field_landmark")
    if landmark_count < 3:
        gp_score = min(gp_score, 7)

    # Check self-citation
    target_wid = response_data.get("work_id")
    if any(gp.get("work_id") == target_wid for gp in gps):
        gp_score = 0

    # Check relevance length
    short_relevance = sum(1 for gp in gps if len(gp.get("relevance", "")) < 20)
    if short_relevance > 0:
        gp_score = min(gp_score, 7)

    gp_score = max(0, gp_score)
    breakdown["grounding_structure"] = gp_score
    score += gp_score

    # 3. Novelty Level Accuracy (5 pts)
    actual_level = novelty.get("novelty_level", "")
    expected_level = expected["expected_novelty"]
    alt_levels = expected.get("expected_novelty_alt", [])

    levels_ordered = ["low", "medium", "high", "pioneering"]
    if actual_level == expected_level:
        level_score = 5
    elif actual_level in alt_levels:
        level_score = 5  # Acceptable alternative
    elif actual_level in levels_ordered and expected_level in levels_ordered:
        diff = abs(levels_ordered.index(actual_level) - levels_ordered.index(expected_level))
        level_score = {0: 5, 1: 3, 2: 1}.get(diff, 0)
    else:
        level_score = 0
    breakdown["novelty_accuracy"] = level_score
    score += level_score

    # 4. Banned Verb Absence (5 pts)
    banned_count = 0
    for field in [response_data.get("summary", ""),
                  novelty.get("whats_new", "") or "",
                  novelty.get("compared_to_prior_work", "") or "",
                  novelty.get("novelty_explanation", "") or ""]:
        if _has_banned_verbs(field):
            banned_count += 1
    for gp in gps:
        if _has_banned_verbs(gp.get("relevance", "")):
            banned_count += 1
    verb_score = 5 if banned_count == 0 else (3 if banned_count == 1 else 0)
    breakdown["banned_verbs"] = verb_score
    score += verb_score

    # 5. Grounding Consistency (5 pts)
    import re
    narrative = f"{novelty.get('whats_new', '') or ''} {novelty.get('compared_to_prior_work', '') or ''} {novelty.get('novelty_explanation', '') or ''}"
    cited_ids = set(re.findall(r'W\d{8,}', narrative))
    gp_ids = {gp["work_id"] for gp in gps}
    orphaned = cited_ids - gp_ids
    consistency_score = 5 if len(orphaned) == 0 else (3 if len(orphaned) == 1 else 0)
    breakdown["consistency"] = consistency_score
    score += consistency_score

    # 6. Cross-Domain Absence (5 pts) — placeholder, needs manual verification
    breakdown["cross_domain"] = 5  # Assume pass for automated; manual review adjusts
    score += 5

    breakdown["total_automated"] = score
    return score, breakdown


# ============================================================================
# Tests
# ============================================================================

@pytest.fixture(scope="session")
def db_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not set")
    return create_engine(db_url)


@pytest.fixture(scope="session")
def results_dir():
    d = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(d, exist_ok=True)
    return d


@pytest.mark.live
@pytest.mark.parametrize("paper", BENCHMARK_PAPERS, ids=lambda p: p["id"])
def test_novelty_assessment_e2e(db_engine, results_dir, paper):
    """End-to-end test: fetch node details with novelty assessment."""
    from app.feature3.node_details_service import get_node_details

    work_id = paper["work_id"]

    # Find a map containing this work_id
    map_id, tenant_id = _find_map_for_work(db_engine, work_id)
    if not map_id:
        pytest.skip(f"No map found containing {work_id} — run F1 citation map first")

    # Run the actual assessment
    result = get_node_details(
        engine=db_engine,
        tenant_id=UUID(tenant_id),
        map_id=UUID(map_id),
        work_id=work_id,
        include_novelty=True,
        include_timeline=False,
    )

    # Convert to dict for assertions
    result_dict = result.model_dump() if hasattr(result, "model_dump") else result

    # Basic structural assertions
    assert result_dict.get("work_id") == work_id
    assert result_dict.get("summary")
    assert len(result_dict.get("summary", "")) >= 30

    novelty = result_dict.get("novelty_assessment")
    if novelty:
        assert novelty["novelty_level"] in ("low", "medium", "high", "pioneering")
        assert novelty["confidence"] in ("low", "medium", "high")
        assert len(novelty.get("grounding_papers", [])) >= 3

    # Compute automated score
    auto_score, breakdown = _compute_automated_score(result_dict, paper)
    assert auto_score >= 20, f"Automated score too low: {auto_score}/35 — {breakdown}"

    # Save results for manual review
    from datetime import datetime
    save_data = {
        "paper_id": paper["id"],
        "work_id": work_id,
        "expected_novelty": paper["expected_novelty"],
        "actual_novelty": novelty["novelty_level"] if novelty else None,
        "automated_score": auto_score,
        "automated_breakdown": breakdown,
        "full_response": result_dict,
        "timestamp": datetime.now().isoformat(),
    }

    # Save timestamped + latest
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"novelty_{paper['id']}"
    with open(os.path.join(results_dir, f"{name}_{ts}.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    with open(os.path.join(results_dir, f"{name}_latest.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  {paper['id']}: novelty={novelty['novelty_level'] if novelty else 'N/A'}, "
          f"auto_score={auto_score}/35, expected={paper['expected_novelty']}")
