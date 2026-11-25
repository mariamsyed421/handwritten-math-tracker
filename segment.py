# segment.py: takes a photo of a multi-line solution and splits it into seperate lines
# usage: python segment.py data/A_sol1.jpg   (if no photo is given, uses data/solution.jpg)

import cv2
import numpy as np
import os  # for making folders + deleting files
import glob  # finding files by name pattern
import sys  # NEW: to read the photo name from the command line

photo = sys.argv[1] if len(sys.argv) > 1 else "data/solution.jpg"  # NEW
img = cv2.imread(photo, cv2.IMREAD_GRAYSCALE)  # load photo in grayscale
if img is None:  # NEW: clear error if the photo can't be opened
    sys.exit(f"could not open {photo}")

# NEW: shrink big phone photos so the size-based settings below work the same on every photo
MAX_W = 1000
if img.shape[1] > MAX_W:
    scale = MAX_W / img.shape[1]
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
h, w = img.shape  # height and width in pixels

# 1. clean the whole photo (same as preprocess.py in examples)

bg = cv2.medianBlur(img, 101)  # blur away the writings to just paper + shadows
norm = cv2.divide(img, bg, scale=255)  # divide out shadows so paper is evenly white
_, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # writing = white, paper = black

# find every separate blob of ink and keep ones that look like writing
n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
keep = np.zeros_like(binary)  # blank image to put kept blobs into
for i in range(1, n):  # search each blob (ignore 0, it's the background)
    x, y, bw, bh, area = stats[i]
    touches_edge = x == 0 or y == 0 or x + bw >= w or y + bh >= h
    if area > 40 and bw < w * 0.5 and not touches_edge:  # keep if not tiny noise (small enough to keep "=" bars and minus signs), not a paper ruled line, not edge junk
        keep[labels == i] = 255

# 2. count pen stroke pixels in each row
row_ink = (keep > 0).sum(axis=1)  # one number per row: how many ink pixels it has
has_ink = row_ink > 0  # True for rows with ink, False for blank rows

# 3. group rows with ink into lines (start row, end row)
lines = []
start = None  # None = not inside a line right now
for y in range(h):  # go through each row in the image
    if has_ink[y] and start is None:
        start = y  # the line begins
    elif not has_ink[y] and start is not None:
        if y - start > 20:  # skip noise lines that are too thin to be writing
            lines.append((start, y))
        start = None  # the line ended
if start is not None:
    lines.append((start, h))  # a line ran to the bottom

os.makedirs("output", exist_ok=True)  # make output folder if missing
os.makedirs("debug", exist_ok=True)  # NEW: keeps every photo's line images so you can look at them later
photo_name = os.path.splitext(os.path.basename(photo))[0]  # NEW: "data/1.jpeg" -> "1"
for f in glob.glob("output/line_*.jpg"):
    os.remove(f)  # delete old line images

# 4. crop each line and save it
pad = 20  # margin around each line
for i, (y0, y1) in enumerate(lines):  # i = line number, y0/y1 = its top/bottom rows
    strip = keep[y0:y1]  # just this line's rows
    xs = np.where(strip.any(axis=0))[0]  # columns that have ink in this line
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w)  # left and right edges of the writing
    crop = 255 - keep[max(y0 - pad, 0):min(y1 + pad, h), x0:x1]  # flip to black on white for TrOCR
    cv2.imwrite(f"output/line_{i}.jpg", crop)
    cv2.imwrite(f"debug/{photo_name}_line_{i}.jpg", crop)  # NEW: copy for checking later

print(f"{photo}: found {len(lines)} lines")