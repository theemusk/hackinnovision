"""Terrain perception module (Member 1)."""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from config import GRID_SIZE, START, GOAL

# ---------- LOAD AND CROP ----------
img = cv2.imread("data/mars_terrain.jpg")
print("Image shape:", img.shape) 

# used ai here to crop the images grey scale so we can reduce noise in later part of code
img = img[47:690, 123:798]                
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA)   # brightness with some noise

# ---------- CLEAN THE IMAGE ----------
# Sobel filter : computes an approximation of the image gradient to identify regions of high spatial frequency, which correspond to edges where pixel intensity changes abruptly
blur = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
flat = blur - cv2.GaussianBlur(blur, (0, 0), 25) + blur.mean()   # brightness with noise clear

# ---------- 3. MEASURE FEATURES ----------
gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
grad = cv2.GaussianBlur(np.sqrt(gx**2 + gy**2), (0, 0), 2)       # steepness
mean = cv2.blur(blur, (7, 7))
var = cv2.blur(blur**2, (7, 7)) - mean**2
var = cv2.GaussianBlur(var, (0, 0), 2)                           # roughness

def norm(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-9)

X = np.stack([norm(flat), norm(grad), norm(var)], axis=-1).reshape(-1, 3)



# ---------- 4. GROUP PIXELS INTO 4 TYPES ----------
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
labels = km.labels_.reshape(GRID_SIZE, GRID_SIZE)
c = km.cluster_centers_                  # columns: brightness, steepness, roughness

# ---------- 5. NAME THE GROUPS ----------
roughness = c[:, 1] + c[:, 2]
order = np.argsort(roughness)            # smoothest -> roughest
mapping = {}
mapping[order[0]] = 0                    # smoothest = safe
mapping[order[3]] = 3                    # roughest = hazard
mid = [order[1], order[2]]
bright_mid = mid[int(np.argmax(c[mid, 0]))]
mapping[bright_mid] = 1                  # brighter middle = sand
mapping[[m for m in mid if m != bright_mid][0]] = 2   # other middle = rock
classes = np.vectorize(mapping.get)(labels).astype(np.uint8)

# ---------- 6. REMOVE SPECKLE ----------
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
classes = cv2.morphologyEx(classes, cv2.MORPH_OPEN, k)
classes = cv2.morphologyEx(classes, cv2.MORPH_CLOSE, k).astype(np.uint8)

# ---------- 7. CONFIDENCE ----------
d = np.linalg.norm(X[:, None, :] - c[None, :, :], axis=2)
conf = 1 - norm(d[np.arange(len(X)), km.labels_]).reshape(GRID_SIZE, GRID_SIZE)

# ---------- 8. MAKE START AND GOAL SAFE ----------
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

print("Done!", classes.shape, classes.dtype)