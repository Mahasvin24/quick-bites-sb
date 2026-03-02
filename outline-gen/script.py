from PIL import Image
import numpy as np
import cv2

# Load image
FILE_NAME = "carillo"
img_path = f"./pre-{FILE_NAME}.png"
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Failed to load image: {img_path}")

# Convert to HSV to isolate tan region
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define tan/beige color range
lower_tan = np.array([10, 10, 200])
upper_tan = np.array([40, 80, 255])

mask = cv2.inRange(hsv, lower_tan, upper_tan)

# Clean mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create WHITE background with alpha channel (RGBA)
outline = np.full(
    (img.shape[0], img.shape[1], 4),
    (255, 255, 255, 255),
    dtype=np.uint8
)

# Make interior transparent
cv2.drawContours(outline, contours, -1, (255, 255, 255, 0), thickness=cv2.FILLED)

# Draw black outline on top (~10px)
cv2.drawContours(outline, contours, -1, (0, 0, 0, 255), thickness=10)

# Save result
out_path = f"./{FILE_NAME}.png"
cv2.imwrite(out_path, outline)

print(out_path)