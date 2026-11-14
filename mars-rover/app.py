"""MARS-09 mission dashboard (Streamlit).
Run from inside mars-rover/:   python -m streamlit run app.py
Reads the files written by perception.py, navigation_pipeline.py and science_scoring/.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from config import GOAL, GRID_SIZE, START

st.set_page_config(page_title="MARS-09 Rover Mission", page_icon="🚀", layout="wide")

CLASS_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"]
CLASS_NAMES = ["Safe", "Sand", "Rock", "Hazard"]
CMAP = ListedColormap(CLASS_COLORS)
CELL_KM = 0.125          # roughly 125 m per grid cell (about 32 km area / 256 cells)
SCI = "science_scoring/outputs"


# ---------------------------------------------------------------- loading
@st.cache_data
def load_npy(path):
    return np.load(path) if os.path.exists(path) else None


@st.cache_data
def load_route():
    for p in ("planned_path.json", "outputs/planned_path.json"):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return None


@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def read_text(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return None


# ---------------------------------------------------------------- helpers
def route_km(path):
    if not path or len(path) < 2:
        return 0.0
    steps = np.hypot(*np.diff(np.array(path), axis=0).T)
    return float(steps.sum() * CELL_KM)


def targets_ok(df):
    if df is None or not {"x", "y"} <= set(df.columns):
        return False
    return bool(df["x"].between(0, GRID_SIZE - 1).all() and df["y"].between(0, GRID_SIZE - 1).all())


def terrain_patches():
    return [Patch(color=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)]


def class_map(classes, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(classes, cmap=CMAP, vmin=0, vmax=3)
    ax.set_title(title)
    ax.axis("off")
    ax.legend(handles=terrain_patches(), loc="upper right", fontsize=8)
    return fig


def mission_figure(terrain, path=None, alt=None, hazard_at=None, targets=None, title="Mission map"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(terrain, cmap=CMAP, vmin=0, vmax=3)
    ax.set_title(title)
    ax.axis("off")
    if path:
        p = np.array(path)
        ax.plot(p[:, 1], p[:, 0], color="white", lw=2, label="Planned route")
    if alt:
        a = np.array(alt)
        ax.plot(a[:, 1], a[:, 0], color="cyan", lw=2, ls="--", label="Alternate route")
    if hazard_at:
        ax.scatter(hazard_at[1], hazard_at[0], marker="X", s=150, color="black",
                   edgecolor="white", zorder=5, label="Simulated new hazard")
    if targets is not None:
        ax.scatter(targets["x"], targets["y"], marker="*", s=230, color="#8e44ad",
                   edgecolor="white", zorder=6, label="Science targets")
        for i, (_, r) in enumerate(targets.iterrows()):
            ax.annotate(str(int(r["rank"])) if "rank" in targets.columns else str(i + 1),
                        (r["x"], r["y"]), color="white", fontsize=7, ha="center", va="bottom",
                        xytext=(0, 7), textcoords="offset points")
    ax.scatter([START[1]], [START[0]], marker="o", s=110, color="white", edgecolor="black",
               zorder=7, label="Start")
    ax.scatter([GOAL[1]], [GOAL[0]], marker="s", s=110, color="white", edgecolor="black",
               zorder=7, label="Goal")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=terrain_patches() + handles, loc="upper right", fontsize=7, framealpha=0.85)
    return fig


def show_image(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption)
    else:
        st.caption(f"{caption}: file not found ({path})")


# ---------------------------------------------------------------- data
terrain = load_npy("outputs/terrain_classes.npy")
if terrain is None:
    st.error("outputs/terrain_classes.npy not found. Run `python run_all.py` first.")
    st.stop()

image_only = load_npy("outputs/classes_image_only.npy")
slope = load_npy("outputs/slope_map.npy")
route = load_route() or {}
path = [tuple(p) for p in route.get("path", [])]
alt_path = [tuple(p) for p in route.get("alternate_path", [])]
hazard_at = route.get("simulated_hazard_at")
metrics = route.get("metrics", {})
alt_metrics = route.get("alternate_metrics", {})
energy = route.get("energy", {})
targets = load_csv(f"{SCI}/ranked_targets.csv")
shares = np.bincount(terrain.ravel(), minlength=4) / terrain.size

# ---------------------------------------------------------------- header
st.title("🚀 MARS-09: Rover Mission Planner")
st.caption("Jezero Crater · real NASA/USGS CTX imagery and elevation · Track A: Rover Exploration")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Safe terrain", f"{shares[0]:.0%}")
k2.metric("Hazard terrain", f"{shares[3]:.0%}")
k3.metric("Route length", f"≈ {route_km(path):.0f} km" if path else "n/a",
          f"{len(path)} cells" if path else None, delta_color="off")
k4.metric("Route risk score", f"{metrics['risk_score']:.0f}" if "risk_score" in metrics else "n/a")
k5.metric("Targets ranked", f"{len(targets)}" if targets is not None else "n/a")

tab_over, tab_ter, tab_nav, tab_sci = st.tabs(
    ["🛰️ Mission Overview", "🪨 Terrain Analysis", "🛣️ Navigation", "🔬 Science Targets"])

# ---------------------------------------------------------------- overview
with tab_over:
    left, right = st.columns([3, 2])
    with left:
        show_t = targets if targets_ok(targets) else None
        st.pyplot(mission_figure(terrain, path, targets=show_t, title="Terrain, route and top science targets"))
    with right:
        st.subheader("Mission summary")
        st.markdown(
            "This dashboard gives a mission operator the full picture of Jezero Crater: "
            "where the terrain is hazardous, the safest route from start to goal, and the "
            "locations most worth exploring."
        )
        st.markdown(
            f"- **Terrain:** {shares[0]:.0%} safe, {shares[1]:.0%} sand, "
            f"{shares[2]:.0%} rock, {shares[3]:.0%} hazard\n"
            f"- **Route:** {len(path)} cells, ≈ {route_km(path):.0f} km, risk score "
            f"{metrics.get('risk_score', 'n/a')}\n"
            f"- **Science:** {len(targets) if targets is not None else 0} ranked targets with "
            "uncertainty and rank-stability analysis"
        )
        if energy:
            st.info(f"Energy estimate for the primary route: {energy.get('total_energy_wh', 'n/a')} Wh "
                    f"(battery remaining {energy.get('remaining_battery_pct', 'n/a')}%).")
        if targets is not None and not targets_ok(targets):
            st.warning("Target coordinates are missing or outside the 256 x 256 grid, so targets are not plotted.")

# ---------------------------------------------------------------- terrain
with tab_ter:
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(class_map(terrain, "Terrain classification"))
    with c2:
        if image_only is not None and slope is not None:
            limit = st.slider("What-if: slope limit for hazards (degrees)", 5, 30, 15)
            what_if = np.where(slope > limit, 3, image_only)
            st.pyplot(class_map(what_if, f"Hazard map with slope limit {limit}°"))
            over = sum(1 for r, c in path if slope[r, c] > limit) if path else 0
            st.caption(f"Hazard share at {limit}°: {(what_if == 3).mean():.1%}. "
                       f"Route cells steeper than the limit: {over} of {len(path)}. "
                       "The planned route itself is not recomputed here.")
        else:
            show_image("outputs/hazard_map.png", "Hazard map")
    st.subheader("3D elevation model")
    try:
        from terrain_3d import build_figure
        mode = st.radio("Color by", ["elevation", "hazard"], horizontal=True)
        exaggeration = st.slider("Vertical exaggeration", 1.0, 10.0, 3.0)
        st.plotly_chart(build_figure(mode, exaggeration, path_rc=path or None))
        st.caption("Vertical exaggeration makes relief visible; real relief is about 1.2 km over 31 km.")
    except Exception as exc:
        st.warning(f"3D view unavailable: {exc}")

# ---------------------------------------------------------------- navigation
with tab_nav:
    if not path:
        st.warning("planned_path.json not found. Run `python navigation_pipeline.py`.")
    else:
        n1, n2 = st.columns([3, 2])
        with n1:
            show_alt = st.checkbox("Show alternate route (re-plan after a simulated new hazard)", value=False)
            st.pyplot(mission_figure(terrain, path, alt=alt_path if show_alt else None,
                                     hazard_at=hazard_at if show_alt else None,
                                     title="Planned route over terrain"))
        with n2:
            st.subheader("Primary route")
            a, b = st.columns(2)
            a.metric("Length", f"{len(path)} cells")
            b.metric("Risk score", f"{metrics.get('risk_score', 'n/a')}")
            st.caption(f"≈ {route_km(path):.0f} km at about 125 m per cell")
            st.subheader("Energy")
            e1, e2 = st.columns(2)
            e1.metric("Total energy", f"{energy.get('total_energy_wh', 'n/a')} Wh")
            e2.metric("Battery left", f"{energy.get('remaining_battery_pct', 'n/a')}%")
            if energy and not energy.get("is_feasible", True):
                st.warning("The energy model reports the battery is not enough for the full traverse.")
            if alt_path:
                st.subheader("Alternate route")
                st.write(f"{len(alt_path)} cells, risk score {alt_metrics.get('risk_score', 'n/a')}")

# ---------------------------------------------------------------- science
with tab_sci:
    if targets is None:
        st.warning(f"{SCI}/ranked_targets.csv not found.")
    else:
        s1, s2 = st.columns([3, 2])
        with s1:
            st.subheader("Top ranked exploration targets")
            st.dataframe(targets)
        with s2:
            if targets_ok(targets):
                st.pyplot(mission_figure(terrain, targets=targets, title="Target locations"))
        text = read_text(f"{SCI}/recommendation.txt")
        if text:
            st.info(text)
        st.subheader("Scores, uncertainty and ranking stability")
        i1, i2 = st.columns(2)
        with i1:
            show_image(f"{SCI}/target_scores.png", "Target scores")
            show_image(f"{SCI}/spatial_confidence_map.png", "Spatial uncertainty")
        with i2:
            show_image(f"{SCI}/science_opportunity_map.png", "Science opportunity map")
            show_image(f"{SCI}/rank_stability.png", "Rank stability under changed weights")

st.divider()
st.caption("Data: NASA/JPL MRO CTX mosaic and CTX DEM of Jezero Crater (USGS Astrogeology, via NASA Mars Trek). "
           "Terrain classes are image-texture and slope proxies; see the README for assumptions.")