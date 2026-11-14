import os
import cv2
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from config import GRID_SIZE, START, GOAL

MOSAIC_PATH = "data/mars_mosaic.tif"
DEM_PATH = "data/mars_dem.tif"
SLOPE_HAZARD_DEG = 15
WORK = GRID_SIZE * 4

def norm(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-9)

# ---------- 1. LOAD MOSAIC (GEOTIFF) ----------
with rasterio.open(MOSAIC_PATH) as src:
    mosaic = src.read(1).astype(np.float32)
    m_nodata = src.nodata
valid = np.ones(mosaic.shape, bool) if m_nodata is None else (mosaic != m_nodata)
mosaic[~valid] = np.median(mosaic[valid])
print("Mosaic shape:", mosaic.shape)
work = cv2.resize(mosaic, (WORK, WORK), interpolation=cv2.INTER_AREA)
gray = cv2.resize(mosaic, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA).astype(np.uint8)

# ---------- 2. FEATURES & CLEAN THE IMAGE ----------
# Sobel filter : computes an approximation of the image gradient to identify regions of high spatial frequency, which correspond to edges where pixel intensity changes abruptly
blur = cv2.GaussianBlur(work, (5, 5), 0)
flat = blur - cv2.GaussianBlur(blur, (0, 0), 100) + blur.mean()   # brightness with noise clear
gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
grad = cv2.GaussianBlur(np.sqrt(gx**2 + gy**2), (0, 0), 2)       # steepness
mean = cv2.blur(blur, (9, 9))
var = cv2.GaussianBlur(cv2.blur(blur**2, (9, 9)) - mean**2, (0, 0), 2) # roughness

def shrink(a):
    return cv2.resize(a, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA)

X = np.stack([norm(shrink(flat)), norm(shrink(grad)), norm(shrink(var))], axis=-1).reshape(-1, 3)

# ---------- 4. GROUP PIXELS INTO 4 TYPES (K-MEANS) ----------
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
labels = km.labels_.reshape(GRID_SIZE, GRID_SIZE)
c = km.cluster_centers_                     # columns: brightness, steepness, roughness

# ---------- 5. NAME THE GROUPS ----------
order = np.argsort(c[:, 1] + c[:, 2])            # smoothest -> roughest
mapping = {order[0]: 0, order[3]: 3}
mid = [order[1], order[2]]
bright_mid = mid[int(np.argmax(c[mid, 0]))]
mapping[bright_mid] = 1                      # brighter middle = sand
mapping[[m for m in mid if m != bright_mid][0]] = 2   # other middle = rock
classes = np.vectorize(mapping.get)(labels).astype(np.uint8)

# ---------- 6. REMOVE SPECKLE ----------
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
classes = cv2.morphologyEx(classes, cv2.MORPH_OPEN, k)
classes = cv2.morphologyEx(classes, cv2.MORPH_CLOSE, k).astype(np.uint8)

np.save("outputs/classes_image_only.npy", classes)

# ---------- 7. CONFIDENCE ----------
d = np.linalg.norm(X[:, None, :] - c[None, :, :], axis=2)
conf = 1 - norm(d[np.arange(len(X)), km.labels_]).reshape(GRID_SIZE, GRID_SIZE)

# ---------- 7.5 ELEVATION -> SLOPE HAZARD ----------
dem_small = None
if os.path.exists(DEM_PATH):
    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(np.float32)
        px, py = abs(src.transform.a), abs(src.transform.e)
        d_nodata = src.nodata
    bad = ~np.isfinite(dem) | (dem < -1e6)
    if d_nodata is not None:
        bad |= (dem == d_nodata)
    dem[bad] = np.median(dem[~bad])
    print("DEM shape:", dem.shape, "| pixel size (m):", px, py,
          "| elevation min/max:", float(dem.min()), float(dem.max()))

    dz_dy, dz_dx = np.gradient(dem, py, px)
    slope_full = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)
    slope = cv2.resize(slope_full, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA)
    dem_small = cv2.resize(dem, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA)

    steep = slope > SLOPE_HAZARD_DEG
    classes[steep] = 3
    p = np.percentile(slope, [50, 90, 99]).round(1)
    print(f"Slope 50/90/99th pct: {p} deg | flagged as hazard: {steep.mean():.1%}")
    np.save("outputs/elevation_map.npy", dem_small.astype(np.float32))
    np.save("outputs/slope_map.npy", slope.astype(np.float32))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(gray, cmap="gray"); ax[0].set_title("Image"); ax[0].axis("off")
    ax[1].imshow(dem_small, cmap="terrain")
    ax[1].contour(dem_small, levels=15, colors="k", linewidths=0.4)
    ax[1].set_title("Elevation + contours"); ax[1].axis("off")
    fig.savefig("outputs/elevation_check.png", dpi=150, bbox_inches="tight"); plt.close(fig)
else:
    print("No DEM found, using image-only hazards")

# ---------- 8. MAKE START AND GOAL SAFE (must stay AFTER step 7) ----------
for r, cc in [START, GOAL]:
    classes[r-4:r+5, cc-4:cc+5] = 0

# ---------- 9. SAVE FOR THE TEAM ----------
np.save("outputs/terrain_classes.npy", classes)
np.save("outputs/confidence_map.npy", conf.astype(np.float32))

# ---------- 10. MAKE THE PICTURES ----------
colors = ["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"]
names = ["Safe", "Sand", "Rock", "Crater/Hazard"]
legend = [Patch(color=colors[i], label=names[i]) for i in range(4)]

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(classes, cmap=ListedColormap(colors), vmin=0, vmax=3)
ax.legend(handles=legend, loc="upper right", fontsize=8)
ax.set_title("Terrain Classification Map"); ax.axis("off")
fig.savefig("outputs/terrain_map.png", dpi=150, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gray, cmap="gray")
hazard = np.ma.masked_where(classes < 2, classes)
ax.imshow(hazard, cmap=ListedColormap(["#e67e22", "#c0392b"]), vmin=2, vmax=3, alpha=0.55)
ax.legend(handles=legend[2:], loc="upper right", fontsize=8)
ax.set_title("Hazard Map"); ax.axis("off")
fig.savefig("outputs/hazard_map.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print("Done!", classes.shape, classes.dtype,
      "class share:", (np.bincount(classes.ravel(), minlength=4) / classes.size).round(3))