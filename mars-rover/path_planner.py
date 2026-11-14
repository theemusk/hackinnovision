import heapq
import numpy as np


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def find_safe_cell(cost_map: np.ndarray, preferred: tuple[int, int]) -> tuple[int, int]:
    rows, cols = cost_map.shape
    r0, c0 = preferred
    if 0 <= r0 < rows and 0 <= c0 < cols and cost_map[r0, c0] < 100.0:
        return preferred

    for dist in range(1, max(rows, cols)):
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < rows and 0 <= nc < cols and cost_map[nr, nc] < 100.0:
                    return (nr, nc)
    return preferred


def a_star(cost_map: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
    rows, cols = cost_map.shape
    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}

    neighbors = [
        (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
        (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2)),
    ]

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        if current_g > g_score.get(current, float("inf")):
            continue

        for dr, dc, step_scale in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            cell_cost = float(cost_map[nr, nc])
            if cell_cost >= 100.0:
                continue

            tentative_g = current_g + (cell_cost * step_scale)
            neighbor = (nr, nc)

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    return []


def compute_metrics(cost_map: np.ndarray, path: list[tuple[int, int]]) -> dict:
    if not path:
        return {"path_length": 0, "risk_score": 0.0}
    risk_score = sum(float(cost_map[r, c]) for r, c in path)
    return {
        "path_length": len(path),
        "risk_score": round(risk_score, 2),
    }
