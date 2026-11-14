"""Scientific Discovery & Uncertainty Reasoning module - Odyssey (HackInnovision 2026).

Other modules can either:
    from science_scoring.science import run_pipeline; ranked, text = run_pipeline()
or just read science_scoring/outputs/ranked_targets.csv.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "outputs"
W_ICE, W_SCI, W_SAFE = 0.4, 0.4, 0.2
REQUIRED = ["target_id", "x", "y", "ice_probability",
            "scientific_value", "terrain_risk", "uncertainty"]
UNIT_COLS = ["ice_probability", "scientific_value", "terrain_risk", "uncertainty"]


def load_targets(path=DATA_DIR / "targets.csv"):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[REQUIRED].isnull().any().any():
        raise ValueError("Dataset contains missing values")
    if df["target_id"].duplicated().any():
        raise ValueError("Duplicate target_id found")
    for col in UNIT_COLS:
        if not df[col].between(0, 1).all():
            raise ValueError(f"{col} must be between 0 and 1")
    return df


def calculate_scores(df):
    df = df.copy()
    df["mission_score"] = 100 * (W_ICE * df["ice_probability"]
                                 + W_SCI * df["scientific_value"]
                                 + W_SAFE * (1 - df["terrain_risk"]))
    df["mission_score"] = df["mission_score"].round(2)
    return df


def classify_water_potential(p):
    if p >= 0.70:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"


def calculate_confidence(uncertainty):
    return round((1 - uncertainty) * 100, 1)


def rank_targets(df):
    df = df.sort_values("mission_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def find_best_resource(df):
    return df.loc[df["ice_probability"].idxmax()]


def generate_recommendation(ranked, best_resource):
    t = ranked.iloc[0]
    reasons = []
    if t.ice_probability >= 0.70:
        reasons.append(f"high estimated water-ice potential ({t.ice_probability:.2f})")
    elif t.ice_probability >= 0.40:
        reasons.append(f"moderate estimated water-ice potential ({t.ice_probability:.2f})")
    if t.scientific_value >= 0.70:
        reasons.append(f"high scientific value ({t.scientific_value:.2f})")
    if t.terrain_risk <= 0.30:
        reasons.append(f"low terrain risk ({t.terrain_risk:.2f})")
    elif t.terrain_risk >= 0.60:
        reasons.append(f"note: terrain risk is high ({t.terrain_risk:.2f})")
    if t.confidence >= 70:
        reasons.append(f"high estimated confidence ({t.confidence:.0f}%)")
    elif t.confidence < 50:
        reasons.append(f"note: confidence is low ({t.confidence:.0f}%)")

    lines = [
        "SCIENTIFIC EXPLORATION RECOMMENDATION",
        "=" * 40,
        f"Recommended target : {t.target_id}  (position {t.x:g}, {t.y:g})",
        f"Mission score      : {t.mission_score:.2f} / 100",
        f"Water-ice prob.    : {t.ice_probability:.2f}  ({t.water_potential})",
        f"Scientific value   : {t.scientific_value:.2f}",
        f"Terrain risk       : {t.terrain_risk:.2f}",
        f"Confidence         : {t.confidence:.0f}%  (heuristic, not calibrated)",
        "",
        f"Why {t.target_id}: it ranks first because of " + "; ".join(reasons) + ".",
        "",
    ]
    b = best_resource
    if b.target_id == t.target_id:
        lines.append(f"Best water-ice resource: {b.target_id} (also the top overall target).")
    else:
        lines.append(
            f"Best water-ice resource: {b.target_id} (ice probability {b.ice_probability:.2f}, "
            f"confidence {b.confidence:.0f}%, terrain risk {b.terrain_risk:.2f}). "
            f"It is NOT the top overall target because its mission score is "
            f"{b.mission_score:.2f} (rank {int(b['rank'])}).")
    return "\n".join(lines)


def plot_target_scores(df, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = df.sort_values("mission_score", ascending=False)
    bars = ax.bar(ordered["target_id"], ordered["mission_score"], color="#c1440e")
    ax.bar_label(bars, fmt="%.1f", padding=2)
    ax.set_xlabel("Target ID"); ax.set_ylabel("Mission Score")
    ax.set_title("Mission Priority Score by Target"); ax.set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_science_opportunities(df, out):
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(df["terrain_risk"], df["scientific_value"],
                    s=df["ice_probability"] * 1200 + 40, c=df["confidence"],
                    cmap="RdYlGn", vmin=0, vmax=100, edgecolors="black", alpha=0.85)
    for _, r in df.iterrows():
        ax.annotate(r.target_id, (r.terrain_risk, r.scientific_value),
                    ha="center", va="center", fontsize=8, fontweight="bold")
    fig.colorbar(sc, label="Confidence (%)")
    ax.set_xlabel("Terrain Risk"); ax.set_ylabel("Scientific Value")
    ax.set_title("Scientific Opportunity Map (bubble size = water-ice probability)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_uncertainty_map(df, out):
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(df["x"], df["y"], c=df["confidence"], cmap="RdYlGn",
                    vmin=0, vmax=100, s=350, edgecolors="black")
    for _, r in df.iterrows():
        ax.annotate(r.target_id, (r.x, r.y), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=9)
    fig.colorbar(sc, label="Confidence (%)  red = low, green = high")
    ax.set_xlabel("X Coordinate"); ax.set_ylabel("Y Coordinate")
    ax.set_title("Spatial Confidence Map (discrete targets)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def save_ranked_data(df, out):
    cols = ["rank", "target_id", "x", "y", "ice_probability", "scientific_value",
            "terrain_risk", "uncertainty", "mission_score", "water_potential", "confidence"]
    df[cols].to_csv(out, index=False)


def run_pipeline(input_csv=DATA_DIR / "targets.csv", out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    df = load_targets(input_csv)
    df = calculate_scores(df)
    df["water_potential"] = df["ice_probability"].apply(classify_water_potential)
    df["confidence"] = df["uncertainty"].apply(calculate_confidence)
    ranked = rank_targets(df)
    best = find_best_resource(ranked)
    text = generate_recommendation(ranked, best)
    save_ranked_data(ranked, out_dir / "ranked_targets.csv")
    (out_dir / "recommendation.txt").write_text(text, encoding="utf-8")
    plot_target_scores(ranked, out_dir / "target_scores.png")
    plot_science_opportunities(ranked, out_dir / "science_opportunity_map.png")
    plot_uncertainty_map(ranked, out_dir / "spatial_confidence_map.png")
    return ranked, text


def main():
    ranked, text = run_pipeline()
    print(ranked[["rank", "target_id", "mission_score", "water_potential",
                  "confidence", "ice_probability", "scientific_value", "terrain_risk"]]
          .to_string(index=False))
    print()
    print(text)


if __name__ == "__main__":
    main()