import numpy as np
import plotly.graph_objects as go
from config import START, GOAL

EXTENT_KM = 31.0
HAZ = ["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"]
HAZ_SCALE = [[0, HAZ[0]], [0.167, HAZ[0]], [0.167, HAZ[1]], [0.5, HAZ[1]],
             [0.5, HAZ[2]], [0.833, HAZ[2]], [0.833, HAZ[3]], [1, HAZ[3]]]


def build_figure(mode="elevation", exaggeration=3.0, path_rc=None):
    elev = np.load("outputs/elevation_map.npy")
    classes = np.load("outputs/terrain_classes.npy")
    n = elev.shape[0]

    x = np.linspace(0, EXTENT_KM, n)
    y = np.linspace(EXTENT_KM, 0, n)

    if mode == "hazard":
        surf = go.Surface(x=x, y=y, z=elev, surfacecolor=classes, cmin=0, cmax=3,
                          colorscale=HAZ_SCALE, showscale=False)
    else:
        surf = go.Surface(x=x, y=y, z=elev, surfacecolor=elev, colorscale="Earth",
                          colorbar=dict(title="Elevation (m)"))
    surf.update(lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.9))
    fig = go.Figure(surf)

    def to_xyz(r, c):
        return (c / (n - 1) * EXTENT_KM, EXTENT_KM - r / (n - 1) * EXTENT_KM, elev[r, c] + 15)

    if path_rc:
        pts = np.array([to_xyz(int(r), int(c)) for r, c in path_rc])
        fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
                                   line=dict(color="white", width=6), name="Route"))
    for label, (r, c) in [("Start", START), ("Goal", GOAL)]:
        px, py, pz = to_xyz(r, c)
        fig.add_trace(go.Scatter3d(x=[px], y=[py], z=[pz + 30], mode="markers+text",
                                   text=[label], marker=dict(size=5, color="black"),
                                   name=label, showlegend=False))

    z_ratio = exaggeration * (elev.max() - elev.min()) / (EXTENT_KM * 1000)
    fig.update_layout(
        title="Jezero Crater: 3D elevation model",
        scene=dict(aspectratio=dict(x=1, y=1, z=max(z_ratio, 0.05)),
                   xaxis_title="East (km)", yaxis_title="North (km)",
                   zaxis_title="Elevation (m)"),
        margin=dict(l=0, r=0, t=40, b=0))
    return fig


if __name__ == "__main__":
    build_figure("elevation").write_html("outputs/terrain_3d_elevation.html")
    build_figure("hazard").write_html("outputs/terrain_3d_hazard.html")
    print("Saved outputs/terrain_3d_elevation.html and terrain_3d_hazard.html")