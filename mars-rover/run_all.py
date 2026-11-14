import os, subprocess, sys

STEPS = [
    ("Terrain and hazards", "perception.py", ["data/mars_mosaic.tif", "data/mars_dem.tif"]),
    ("Navigation", "navigation_pipeline.py", []),
    ("Integration check", "check_all.py", []),
]

for name, script, needs in STEPS:
    print(f"\n=== {name} ===")
    if any(not os.path.exists(n) for n in needs):
        print(f"Skipping {script}: input data not found (using the files already in outputs/).")
        continue
    if subprocess.run([sys.executable, script]).returncode != 0:
        print(f"STOPPED: {script} failed.")
        sys.exit(1)

print("\nPipeline finished. Start the dashboard with: streamlit run app.py")