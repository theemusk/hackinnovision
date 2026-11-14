import numpy as np


def inflate_hazard_zones(
    cost_map: np.ndarray, buffer_radius: int = 2, buffer_penalty: float = 25.0
) -> np.ndarray:
    inflated = cost_map.copy()
    rows, cols = cost_map.shape
    hazard_coords = np.argwhere(cost_map >= 100.0)

    for hr, hc in hazard_coords:
        r_min = max(0, hr - buffer_radius)
        r_max = min(rows, hr + buffer_radius + 1)
        c_min = max(0, hc - buffer_radius)
        c_max = min(cols, hc + buffer_radius + 1)

        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                if inflated[r, c] < 100.0:
                    dist = max(abs(r - hr), abs(c - hc))
                    decay_penalty = buffer_penalty / dist
                    inflated[r, c] = max(inflated[r, c], cost_map[r, c] + decay_penalty)

    return inflated


def generate_cost_map(
    terrain_grid: np.ndarray, apply_safety_buffer: bool = True
) -> np.ndarray:
    cost_lookup = {
        0: 1.0,
        1: 5.0,
        2: 10.0,
        3: 100.0,
    }

    cost_map = np.vectorize(cost_lookup.get)(terrain_grid).astype(float)

    if apply_safety_buffer:
        cost_map = inflate_hazard_zones(cost_map, buffer_radius=2, buffer_penalty=25.0)

    np.save("cost_map.npy", cost_map)
    return cost_map
