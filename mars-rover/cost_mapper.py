import numpy as np


def generate_cost_map(terrain_grid: np.ndarray) -> np.ndarray:
    
   
    cost_lookup = {
        0: 1.0,    # Safe
        1: 5.0,    # Sand
        2: 10.0,   # Rock
        3: 100.0,  # Crater / Hazard
    }

    cost_map = np.vectorize(cost_lookup.get)(terrain_grid).astype(float)
    np.save("cost_map.npy", cost_map)
    return cost_map


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dummy_grid = rng.choice([0, 1, 2, 3], size=(100, 100), p=[0.6, 0.2, 0.15, 0.05])
    cm = generate_cost_map(dummy_grid)
    print("Saved cost_map.npy | shape:", cm.shape)
