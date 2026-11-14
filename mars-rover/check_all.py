import glob, json, os, py_compile, subprocess, sys
import numpy as np
from config import GRID_SIZE, START, GOAL

passed, failed, warned = 0, 0, 0

def check(name, ok, detail=""):
    global passed, failed
    passed, failed = passed + bool(ok), failed + (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return bool(ok)

def warn(name, detail=""):
    global warned
    warned += 1
    print(f"[WARN] {name} {detail}")

def find(*paths):
    return next((p for p in paths if os.path.exists(p)), None)

def load_npy(*paths):
    p = find(*paths)
    if not check(f"{paths[0]} found", p is not None, f"(also looked in {list(paths[1:])})" if p is None and len(paths) > 1 else (f"at {p}" if p and p != paths[0] else "")):
        return None
    return np.load(p)

def extract_path(obj, depth=0):
    if isinstance(obj, list) and obj:
        if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in obj):
            return [(int(p[0]), int(p[1])) for p in obj]
        if all(isinstance(p, dict) for p in obj):
            low = [{k.lower(): v for k, v in p.items()} for p in obj]
            if all("row" in d and "col" in d for d in low):
                return [(int(d["row"]), int(d["col"])) for d in low]
    if isinstance(obj, dict) and depth < 3:
        for k, v in obj.items():
            r = extract_path(v, depth + 1)
            if r:
                return r
    return None

if "--run" in sys.argv:
    print("\n== Running each module ==")
    for s in ["perception.py", "navigation_pipeline.py"]:
        if os.path.exists(s):
            r = subprocess.run([sys.executable, s], capture_output=True, text=True)
            err = r.stderr.strip().splitlines()[-1] if r.returncode and r.stderr.strip() else ""
            check(f"{s} runs without error", r.returncode == 0, err)
        else:
            check(f"{s} exists", False)

print("\n== Member 1: terrain ==")
terrain = load_npy("outputs/terrain_classes.npy")
if terrain is not None:
    check("terrain shape is 256x256", terrain.shape == (GRID_SIZE, GRID_SIZE), str(terrain.shape))
    check("terrain dtype is uint8", terrain.dtype == np.uint8, str(terrain.dtype))
    check("terrain only has IDs 0-3", set(np.unique(terrain)) <= {0, 1, 2, 3}, str(np.unique(terrain)))
    check("start cell is safe", terrain[START] == 0)
    check("goal cell is safe", terrain[GOAL] == 0)
    share = (terrain == 0).mean()
    check("at least 20% of map is safe", share > 0.2, f"({share:.0%} safe)")
for f in ["confidence_map", "slope_map", "elevation_map"]:
    a = load_npy(f"outputs/{f}.npy")
    if a is not None:
        check(f"{f} shape matches grid", a.shape == (GRID_SIZE, GRID_SIZE), str(a.shape))
for f in ["terrain_map", "hazard_map"]:
    check(f"outputs/{f}.png exists", os.path.exists(f"outputs/{f}.png"))

print("\n== Member 2: navigation ==")
cost = load_npy("outputs/cost_map.npy", "cost_map.npy")
if cost is not None and terrain is not None:
    check("cost map shape matches grid", cost.shape == terrain.shape, str(cost.shape))
    if (terrain == 3).any() and (terrain == 0).any():
        check("hazard cells cost more than safe cells", cost[terrain == 3].mean() > cost[terrain == 0].mean())
path = None
pp = find("outputs/planned_path.json", "planned_path.json")
if check("planned_path.json found", pp is not None, f"at {pp}" if pp else "(looked in outputs/ and mars-rover/)"):
    data = json.load(open(pp))
    if isinstance(data, dict):
        print("   keys in file:", list(data.keys()))
    path = extract_path(data["path"] if isinstance(data, dict) and "path" in data else data)
    if path is None:
        warn("could not read the route as (row, col) cells", "- check the file manually")
if path and terrain is not None:
    print(f"   path has {len(path)} cells")
    if path[0] != START and path[0] == START[::-1]:
        warn("path looks like (x, y), not (row, col)", "- classic coordinate bug")
    check("path starts at START", path[0] == START, str(path[0]))
    check("path ends at GOAL", path[-1] == GOAL, str(path[-1]))
    steps_ok = all(max(abs(a[0]-b[0]), abs(a[1]-b[1])) == 1 for a, b in zip(path, path[1:]))
    check("path moves one cell at a time", steps_ok)
    in_grid = all(0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE for r, c in path)
    check("path stays inside the grid", in_grid)
    if in_grid:
        bad = sum(terrain[r, c] == 3 for r, c in path)
        check("path avoids hazard cells", bad == 0, f"({bad} hazard cells on route)")

print("\n== Member 3: science ==")
csvs = (glob.glob("science_scoring/outputs/ranked_targets.csv") + glob.glob("outputs/*target*.csv")
        + glob.glob("science_scoring/outputs/*rank*.csv"))
if check("ranked targets CSV found", bool(csvs), str(csvs[:1])):
    import pandas as pd
    df = pd.read_csv(csvs[0])
    print("   columns:", list(df.columns))
    check("at least 10 targets", len(df) >= 10, f"({len(df)} rows)")
    cols = {c.lower(): c for c in df.columns}
    if "row" in cols and "col" in cols and terrain is not None:
        on_haz = sum(terrain[int(r), int(c)] == 3 for r, c in zip(df[cols["row"]], df[cols["col"]]))
        check("no target sits on a hazard cell", on_haz == 0, f"({on_haz} on hazard)")
    else:
        warn("no row/col columns in targets CSV", "- check target locations manually")
unc = (glob.glob("outputs/*uncertainty*") + glob.glob("outputs/*confidence*.png")
       + glob.glob("science_scoring/outputs/*confidence*") + glob.glob("science_scoring/outputs/*uncertainty*")
       + glob.glob("science_scoring/outputs/*stability*"))
check("uncertainty/confidence output exists", bool(unc), str([os.path.basename(u) for u in unc]))

print("\n== Member 4: dashboard and repo ==")
for f in ["app.py", "README.md", "requirements.txt", "config.py"]:
    check(f"{f} exists", os.path.exists(f) or os.path.exists(f"../{f}"))
if os.path.exists("app.py"):
    try:
        py_compile.compile("app.py", doraise=True); check("app.py has no syntax errors", True)
    except py_compile.PyCompileError as e:
        check("app.py has no syntax errors", False, str(e)[:80])

print(f"\n== SUMMARY: {passed} passed, {failed} failed, {warned} warnings ==")
sys.exit(1 if failed else 0)