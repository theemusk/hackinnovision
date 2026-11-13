from science import *

def test_all():
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
    assert find_best_resource(ranked).target_id == df.loc[df.ice_probability.idxmax(), "target_id"]
    print("All tests passed")

if __name__ == "__main__":
    test_all()