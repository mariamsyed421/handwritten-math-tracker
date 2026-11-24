# this program just reads an example single line of math (2x + 4 = 10) and cleans it up for TrOCR 
# this is just an example for testing and not needed for our actual project

import cv2 # import OpenCV to load, edit, and save images
import numpy as np # import numpy to process the image as a grid of numbers

#load the image as a grid of grayscale values
img = cv2.imread("line1.jpg", cv2.IMREAD_GRAYSCALE)
h, w = img.shape #store image dimensions


# ---------- PART 1: make the paper evenly white ----------

# blur the image a LOT to get a picture of just the paper
# chooses the median value of a 101x101 square of pixels around each pixel, and replaces the pixel 
# so the pencil strokes disappear (strokes are much thinner than 101 pixels)
bg = cv2.medianBlur(img, 101) # 101 is chosen because it is odd and large enough to cover pencil strokes



# divide the original image by the paper-only image
# both images are dark in shadowed areas, so dividing cancels it out and leaves the paper evenly white
# ink is much darker than the paper, so it stays dark after dividing
norm = cv2.divide(img, bg, scale=255) #scale = 255 stretches the result back to the normal 0 to 255 range


# ---------- PART 2: seperate ink vs paper ----------

# turn every pixel into pure black or pure white
# OTSU: looks at how many pixels there are of each brightness, sees two groups and picks the cutoff between them 
# THRESH_BINARY_INV: flips the colors: ink becomes white (255), paper becomes black (0)

# flip because the next OpenCV functions look for white
# threshold() returns: the cutoff it picked (not needed) and the black/white image we want (binary)
_, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 


# ---------- PART 3: throw away ink that isn't writing ----------

# finds all the separate blobs in the binary image
# n = how many blobs there are (including background)
# labels = a grid same size as the image; each pixel holds the blob it belongs to (0 = background, 1 = first blob, 2 = second)
# stats = for each blob: x, y (top-left corner), width, height, area (pixel count)
# _ = the center of each blob (not needed)
n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)


keep = np.zeros_like(binary) # new all black image 

# Go through each blob. Start at 1 (blob 0 is the background)
for i in range(1, n):
    # this blob's stats: position, width, height, and size (area)
    x, y, bw, bh, area = stats[i]

    # keep the blob if all three are true:
    #   area > 500       -> big enough to be a character, not a speck of noise
    #   bw < w * 0.5     -> narrower than half the image, so not a long ruled line
    #   not touches_edge -> not junk at the border
    touches_edge = x == 0 or y == 0 or x + bw >= w or y + bh >= h

    if area > 500 and bw < w * 0.5 and not touches_edge:
        # add those pixels as white into the keep image.
        keep[labels == i] = 255


# ---------- PART 4: crop to just the writing ----------

# Find every pixel in keep that is white (writing)
ys, xs = np.where(keep > 0)
pad = 20

# The box around all the ink
x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w) 
y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, h)

# keep[y0:y1, x0:x1] cuts out that box (rows first, then columns).
# 255 - image: flips the colors back, so the result is black ink on white paper (which TrOCR expects)
clean = 255 - keep[y0:y1, x0:x1]

# Save the cleaned, cropped line 
cv2.imwrite("line1_clean.jpg", clean)