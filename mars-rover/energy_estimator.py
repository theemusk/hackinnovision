import numpy as np


def estimate_energy_wh(
    cost_map: np.ndarray,
    path: list[tuple[int, int]],
    base_wh_per_step: float = 1.2,
    terrain_energy_factor: float = 0.15,
) -> dict:
    if len(path) < 2:
        return {
            "total_energy_wh": 0.0,
            "remaining_battery_pct": 100.0,
            "is_feasible": True,
        }

    total_energy = 0.0
    for i in range(1, len(path)):
        r1, c1 = path[i - 1]
        r2, c2 = path[i]
        step_dist = float(np.hypot(r2 - r1, c2 - c1))
        cell_cost = float(cost_map[r2, c2])
        total_energy += step_dist * (base_wh_per_step + terrain_energy_factor * cell_cost)

    battery_capacity_wh = 500.0
    remaining_pct = max(0.0, round(100.0 * (1.0 - total_energy / battery_capacity_wh), 2))

    return {
        "total_energy_wh": round(total_energy, 2),
        "remaining_battery_pct": remaining_pct,
        "is_feasible": total_energy <= (battery_capacity_wh * 0.85),
    }
