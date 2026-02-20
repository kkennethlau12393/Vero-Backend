"""
Tests focused on grounding paper quality for Feature 3.

Verifies that the grounding supplement pipeline produces relevant,
domain-appropriate grounding papers for various paper types.

Mark: @pytest.mark.live

WARNING: Burns API credits (Groq, OpenAlex, S2, ArXiv)
"""
import json
import os
import re
from pathlib import Path
from uuid import UUID

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=True)

from app.feature3.node_details_service import _has_banned_verbs


GROUNDING_BENCHMARK_PAPERS = [
    # ── Pioneering papers — expect strong grounding ───────────────────────
    {
        "id": "resnet_grounding",
        "work_id": "W2194775991",
        "expected_landmark_keywords": ["image", "recognition", "deep", "network"],
        "expected_ref_count_min": 2,
        "expected_landmark_count_min": 2,
        "domain": "CV",
    },
    {
        "id": "imagenet_grounding",
        "work_id": "W2108598243",
        "expected_landmark_keywords": ["image", "recognition", "classification", "dataset"],
        "expected_ref_count_min": 2,
        "expected_landmark_count_min": 2,
        "domain": "CV",
    },
    {
        "id": "faster_rcnn_grounding",
        "work_id": "W639708223",
        "expected_landmark_keywords": ["object", "detection", "region", "proposal"],
        "expected_ref_count_min": 2,
        "expected_landmark_count_min": 2,
        "domain": "CV",
    },
    {
        "id": "playing_atari_grounding",
        "work_id": "W1757796397",
        "expected_landmark_keywords": ["reinforcement", "learning", "deep", "agent"],
        "expected_ref_count_min": 2,
        "expected_landmark_count_min": 2,
        "domain": "RL",
    },
    # ── High novelty papers — different domains ───────────────────────────
    {
        "id": "cyclegan_grounding",
        "work_id": "W2962793481",
        "expected_landmark_keywords": ["image", "generative", "adversarial", "translation"],
        "expected_ref_count_min": 2,
        "expected_landmark_count_min": 2,
        "domain": "CV",
    },
    {
        "id": "ddpg_grounding",
        "work_id": "W2963864421",
        "expected_landmark_keywords": ["reinforcement", "learning", "continuous", "control"],
        "expected_ref_count_min": 2,
        "expected_landmark_count_min": 2,
        "domain": "RL",
    },
    {
        "id": "sac_grounding",
        "work_id": "W2781726626",
        "expected_landmark_keywords": ["reinforcement", "learning", "entropy", "actor"],
        "expected_ref_count_min": 2,
        "expected_landmark_count_min": 2,
        "domain": "RL",
    },
    {
        "id": "visuomotor_grounding",
        "work_id": "W2964161785",
        "expected_landmark_keywords": ["robot", "policy", "visual", "learning"],
        "expected_ref_count_min": 2,
        "expected_landmark_count_min": 2,
        "domain": "robotics",
    },
    {
        "id": "robotic_grasps_grounding",
        "work_id": "W1999156278",
        "expected_landmark_keywords": ["grasp", "robot", "deep", "learning"],
        "expected_ref_count_min": 2,
        "expected_landmark_count_min": 2,
        "domain": "robotics",
    },
    # ── Review papers — should ground to original methods ─────────────────
    {
        "id": "dl_overview_grounding",
        "work_id": "W2076063813",
        "expected_landmark_keywords": ["neural", "network", "deep", "learning"],
        "expected_ref_count_min": 2,
        "expected_landmark_count_min": 2,
        "domain": "CS/ML",
    },
    {
        "id": "rl_robotics_survey_grounding",
        "work_id": "W1977655452",
        "expected_landmark_keywords": ["reinforcement", "learning", "robot"],
        "expected_ref_count_min": 2,
        "expected_landmark_count_min": 2,
        "domain": "robotics",
    },
    {
        "id": "dl_medical_grounding",
        "work_id": "W2777186991",
        "expected_landmark_keywords": ["medical", "image", "deep", "learning"],
        "expected_ref_count_min": 2,
        "expected_landmark_count_min": 2,
        "domain": "medical",
    },
]


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
@pytest.mark.parametrize("paper", GROUNDING_BENCHMARK_PAPERS, ids=lambda p: p["id"])
def test_grounding_quality(db_engine, results_dir, paper):
    """Test that grounding papers are relevant and well-distributed."""
    from app.feature3.node_details_service import get_node_details

    work_id = paper["work_id"]
    map_id, tenant_id = _find_map_for_work(db_engine, work_id)
    if not map_id:
        pytest.skip(f"No map found containing {work_id}")

    result = get_node_details(
        engine=db_engine,
        tenant_id=UUID(tenant_id),
        map_id=UUID(map_id),
        work_id=work_id,
        include_novelty=True,
        include_timeline=False,
    )

    result_dict = result.model_dump() if hasattr(result, "model_dump") else result
    novelty = result_dict.get("novelty_assessment")

    if not novelty:
        pytest.skip(f"Assessment unavailable for {work_id}")

    gps = novelty.get("grounding_papers", [])

    # Count by relationship type
    ref_count = sum(1 for gp in gps if gp.get("relationship") == "cited_reference")
    landmark_count = sum(1 for gp in gps if gp.get("relationship") == "field_landmark")

    # Assert minimum counts
    assert ref_count >= paper["expected_ref_count_min"], \
        f"Expected >= {paper['expected_ref_count_min']} refs, got {ref_count}"
    assert landmark_count >= paper["expected_landmark_count_min"], \
        f"Expected >= {paper['expected_landmark_count_min']} landmarks, got {landmark_count}"

    # Assert total grounding count
    assert 5 <= len(gps) <= 7, f"Expected 5-7 grounding papers, got {len(gps)}"

    # Assert no self-citation
    assert all(gp["work_id"] != work_id for gp in gps), "Self-citation detected"

    # Assert relevance strings are not boilerplate
    for gp in gps:
        assert len(gp.get("relevance", "")) > 20, f"Short relevance: {gp}"
        assert not _has_banned_verbs(gp.get("relevance", ""))

    # Save results
    from datetime import datetime
    save_data = {
        "paper_id": paper["id"],
        "work_id": work_id,
        "grounding_papers": gps,
        "ref_count": ref_count,
        "landmark_count": landmark_count,
        "total_count": len(gps),
        "timestamp": datetime.now().isoformat(),
    }
    name = f"grounding_{paper['id']}"
    with open(os.path.join(results_dir, f"{name}_latest.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  {paper['id']}: refs={ref_count}, landmarks={landmark_count}, "
          f"total={len(gps)}")
