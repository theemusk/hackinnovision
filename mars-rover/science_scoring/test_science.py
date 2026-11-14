import os
import tempfile

import pandas as pd

from science import *


def test_scoring_and_logic():
    df = load_targets()
    assert len(df) == 10
    s = calculate_scores(df)
    t6 = s[s.target_id == "T6"].mission_score.iloc[0]
    assert abs(t6 - 95.20) < 1e-9, t6
    assert classify_water_potential(0.70) == "High"
    assert classify_water_potential(0.40) == "Medium"
    assert classify_water_potential(0.39) == "Low"
    assert calculate_confidence(0.08) == 92
    # more risk -> lower score; more ice/science -> higher score
    base = df.iloc[[0]].copy()
    b = calculate_scores(base).mission_score.iloc[0]
    for col, sign in [("terrain_risk", -1), ("ice_probability", 1), ("scientific_value", 1)]:
        m = base.copy()
        m[col] = min(m[col].iloc[0] + 0.1, 1.0)
        assert sign * (calculate_scores(m).mission_score.iloc[0] - b) > 0, col
    ranked, text = run_pipeline()
    assert ranked["mission_score"].is_monotonic_decreasing
    assert ranked.iloc[0].target_id in text
    best = find_best_resource(ranked)
    assert best.target_id == df.loc[df.ice_probability.idxmax(), "target_id"]
    print("Scoring and logic tests passed")


def test_outputs_and_validation():
    run_pipeline()
    for f in ["ranked_targets.csv", "recommendation.txt", "target_scores.png",
              "science_opportunity_map.png", "spatial_confidence_map.png"]:
        assert (OUT_DIR / f).exists(), f
    tmp = tempfile.gettempdir()
    # value out of range
    bad = pd.read_csv(DATA_DIR / "targets.csv")
    bad.loc[0, "ice_probability"] = 1.5
    p1 = os.path.join(tmp, "bad_range.csv")
    bad.to_csv(p1, index=False)
    # missing column
    bad2 = pd.read_csv(DATA_DIR / "targets.csv").drop(columns=["uncertainty"])
    p2 = os.path.join(tmp, "bad_cols.csv")
    bad2.to_csv(p2, index=False)
    for p in (p1, p2):
        try:
            load_targets(p)
            assert False, "validation should have raised"
        except ValueError:
            pass
    print("Output and validation tests passed")


def test_integration():
    from integration import get_best_target, get_ranked_targets
    assert len(get_ranked_targets()) == 10
    g = get_best_target()
    assert g["target_id"] == "T6" and g["x"] == 40 and g["y"] == 90
    g2 = get_best_target(grid_width=200, grid_height=200)
    assert g2["grid_x"] == round(40 / 100 * 199) and g2["grid_y"] == round(90 / 100 * 199)
    print("Integration tests passed")


if __name__ == "__main__":
    test_scoring_and_logic()
    test_outputs_and_validation()
    test_integration()
    print("All tests passed")