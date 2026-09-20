import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import ndimage as ndi
from PIL import Image
from pathlib import Path

# 1. Load image
image_path = Path("Maharaj.png")

if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found: {image_path.resolve()}\n"
        "Place shivaji.png in your current project folder."
    )

img = np.array(Image.open(image_path).convert("L"))

# 2. Create mask
line = img < 150
bg = ~line

# Close the bottom edge (as in your original code)
bg[-2:, :] = False

# 3. Detect outside background
lab, _ = ndi.label(bg)

# Exclude label 0 (non-background pixels)
edge = (
    set(lab[0])
    | set(lab[:, 0])
    | set(lab[:, -1])
) - {0}

outside = np.isin(lab, list(edge))

# 4. Fill the silhouette
solid = ndi.binary_fill_holes(line | ~outside)

ys, xs = np.where(solid)

# Check if image contains valid pixels
if len(xs) == 0:
    raise ValueError(
        "No valid silhouette pixels found. "
        "Check your image and threshold value."
    )

# 5. Randomly select pixels safely
rng = np.random.default_rng()

n = min(5000, len(xs))
i = rng.choice(len(xs), n, replace=False)

P = np.c_[xs[i], -ys[i]].astype(float)

# Center and normalize
P -= P.mean(axis=0)

max_y = np.abs(P[:, 1]).max()

if max_y == 0:
    raise ValueError("Image silhouette has no vertical dimension.")

P /= max_y

# 6. Create scattered particles
dust = P * (2.4 + rng.random((len(P), 1)))

# 7. Setup plot
fig, ax = plt.subplots(facecolor="#0b0b0f")

ax.set(
    xlim=(-2.7, 2.7),
    ylim=(-2.7, 2.7),
    aspect="equal"
)

ax.set_facecolor("#0b0b0f")
ax.axis("off")

dots = ax.scatter(
    dust[:, 0],
    dust[:, 1],
    s=4,
    c="#ff8c1a"
)

# 8. Animation function
def frame(f):
    k = min(f / 70, 1) ** 0.65

    current = dust + (P - dust) * k

    dots.set_offsets(current)

    return dots,

# IMPORTANT: Store animation in a variable
ani = FuncAnimation(
    fig,
    frame,
    frames=260,
    interval=33,
    blit=True
)

# 9. Display animation
plt.show()