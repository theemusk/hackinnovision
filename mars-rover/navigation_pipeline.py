import json
import os
import numpy as np
from cost_mapper import generate_cost_map
from energy_estimator import estimate_energy_wh
from path_planner import a_star, compute_metrics, find_safe_cell
from config import START, GOAL


def load_terrain(terrain_path: str = "outputs/terrain_classes.npy") -> np.ndarray:
    if not os.path.exists(terrain_path):
        raise FileNotFoundError(f"{terrain_path} not found. Run perception.py first.")
    return np.load(terrain_path)


def plan_multi_waypoint(
    cost_map: np.ndarray, start: tuple[int, int], waypoints: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    full_path = []
    current = find_safe_cell(cost_map, start)

    for wp in waypoints:
        target = find_safe_cell(cost_map, wp)
        segment = a_star(cost_map, current, target)
        if not segment:
            continue
        full_path.extend(segment if not full_path else segment[1:])
        current = target

    return full_path


def simulate_hazard_replan(
    cost_map: np.ndarray,
    primary_path: list[tuple[int, int]],
    start: tuple[int, int],
    waypoints: list[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[int, int]], list[int] | None]:
    if len(primary_path) < 10:
        return cost_map, [], None

    blocked_idx = len(primary_path) // 2
    br, bc = primary_path[blocked_idx]

    updated_map = cost_map.copy()
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            nr, nc = br + dr, bc + dc
            if 0 <= nr < updated_map.shape[0] and 0 <= nc < updated_map.shape[1]:
                updated_map[nr, nc] = 100.0

    replanned = plan_multi_waypoint(updated_map, start, waypoints)
    return updated_map, replanned, [br, bc]


def run_navigation(
    terrain_path: str = "outputs/terrain_classes.npy",
    targets_path: str = "science_targets.json",
) -> dict:
    terrain = load_terrain(terrain_path)
    cost_map = generate_cost_map(terrain)
    np.save("cost_map.npy", cost_map)

    start = START
    if os.path.exists(targets_path):
        with open(targets_path, "r", encoding="utf-8") as f:
            raw_targets = json.load(f)
        waypoints = [tuple(t["coords"]) for t in raw_targets.get("targets", [])[:3]]
    else:
        waypoints = [GOAL]

    primary_path = plan_multi_waypoint(cost_map, start, waypoints)
    _, replanned_path, hazard_center = simulate_hazard_replan(
        cost_map, primary_path, start, waypoints
    )

    payload = {
        "status": "SUCCESS" if primary_path else "NO_PATH",
        "start": list(find_safe_cell(cost_map, start)),
        "waypoints": [list(find_safe_cell(cost_map, w)) for w in waypoints],
        "path": [list(p) for p in primary_path],
        "alternate_path": [list(p) for p in replanned_path],
        "simulated_hazard_at": hazard_center,
        "metrics": compute_metrics(cost_map, primary_path),
        "alternate_metrics": compute_metrics(cost_map, replanned_path),
        "energy": estimate_energy_wh(cost_map, primary_path),
        "alternate_energy": estimate_energy_wh(cost_map, replanned_path),
    }

    with open("planned_path.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


if __name__ == "__main__":
    result = run_navigation()
    print(
        "Pipeline status:",
        result["status"],
        "| Metrics:",
        result["metrics"],
        "| Energy:",
        result["energy"],
    )
