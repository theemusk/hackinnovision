# Scientific Discovery & Uncertainty Reasoning (Odyssey)

Ranks candidate Mars exploration targets and recommends the best one.

## Run (from this folder)
    py science.py        # scoring, ranking, recommendation, 3 charts
    py robustness.py     # Monte Carlo ranking-stability check
    py test_science.py   # all tests

## Formula
Mission Score = 100 x [0.4 x IceProb + 0.4 x SciValue + 0.2 x (1 - TerrainRisk)]
Confidence (%) = (1 - Uncertainty) x 100  (heuristic, not statistically calibrated)
Ice class: High >= 0.70, Medium 0.40-0.69, Low < 0.40

## Files
- data/targets.csv: 10 simulated candidate targets (x, y from 0 to 100)
- outputs/ranked_targets.csv: ranked table with scores, ice class, confidence
- outputs/recommendation.txt: automatic recommendation and explanation
- outputs/target_scores.png, science_opportunity_map.png, spatial_confidence_map.png
- outputs/rank_stability.csv / .png: Monte Carlo ranking stability

## Integration (for planner and dashboard)
From the repo root:

    from science_scoring.integration import get_best_target, get_ranked_targets
    goal = get_best_target(grid_width=200, grid_height=200)
    # goal -> target_id, x, y, grid_x, grid_y, mission_score, confidence ...
    ranked = get_ranked_targets()   # full pandas DataFrame

Dashboard (Streamlit example):

    import streamlit as st, pandas as pd
    st.dataframe(pd.read_csv("science_scoring/outputs/ranked_targets.csv"))
    st.text(open("science_scoring/outputs/recommendation.txt").read())
    st.image("science_scoring/outputs/science_opportunity_map.png")

## Limitations
Simulated data. No imagery or DEM processing. Confidence is a heuristic, not a
calibrated probability. Water-ice probability is not derived from real sensors.

## Future work
Feed perception hazard output into terrain_risk, use real imagery/DEM features
for scientific value, calibrate uncertainty, and update scores as new
observations arrive.