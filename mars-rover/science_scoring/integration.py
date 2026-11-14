"""Simple entry points for the planner and dashboard.

From the repo root:   from science_scoring.integration import get_best_target
From this folder:     from integration import get_best_target
"""
try:
    from .science import run_pipeline
except ImportError:
    from science import run_pipeline


def get_ranked_targets():
    """Return the full ranked DataFrame (rank 1 = highest mission score)."""
    ranked, _ = run_pipeline()
    return ranked


def get_recommendation_text():
    _, text = run_pipeline()
    return text


def get_best_target(grid_width=None, grid_height=None, map_size=100.0):
    """Return the top-ranked target as a dict.

    Target coordinates run from 0 to map_size. If the planner's grid size is
    given, grid_x and grid_y are also returned, scaled to that grid.
    """
    ranked, _ = run_pipeline()
    t = ranked.iloc[0]
    result = {
        "target_id": t.target_id,
        "x": float(t.x),
        "y": float(t.y),
        "mission_score": float(t.mission_score),
        "ice_probability": float(t.ice_probability),
        "confidence": float(t.confidence),
        "terrain_risk": float(t.terrain_risk),
    }
    if grid_width and grid_height:
        result["grid_x"] = int(round(t.x / map_size * (grid_width - 1)))
        result["grid_y"] = int(round(t.y / map_size * (grid_height - 1)))
    return result


if __name__ == "__main__":
    print(get_best_target(grid_width=200, grid_height=200))