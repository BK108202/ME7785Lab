import cv2
import numpy as np

# Load the original image
image = cv2.imread('./Lab6_ws/2025S_imgs/031.png')
if image is None:
    raise IOError("Image not found or incorrect path!")

# Convert the image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV ranges for red, green, and blue colors
lower_red   = np.array([0, 100, 100])
upper_red   = np.array([10, 255, 255])
lower_green = np.array([40, 100, 100])
upper_green = np.array([80, 255, 255])
lower_blue  = np.array([110, 100, 100])
upper_blue  = np.array([130, 255, 255])

# Create and combine the color masks
red_mask   = cv2.inRange(hsv_image, lower_red, upper_red)
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
blue_mask  = cv2.inRange(hsv_image, lower_blue, upper_blue)
combined_mask = cv2.bitwise_or(red_mask, green_mask)
combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

# Apply the combined mask to get the masked image
masked_image = cv2.bitwise_and(image, image, mask=combined_mask)

# Convert the masked image to HSV to detect the black arrow area
hsv_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

# Define the HSV range for "black"
# Adjust the upper V value if the arrow is not fully detected
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 40])
black_mask = cv2.inRange(hsv_masked, lower_black, upper_black)

# Create an output image with an alpha channel (4 channels: B, G, R, A)
# The size of the output image is the same as the input image
height, width = image.shape[:2]
arrow_rgba = np.zeros((height, width, 4), dtype=np.uint8)

# Choose a color for the arrow; for example, green (BGR format)
arrow_color = (0, 255, 0)  # (Blue, Green, Red)

# Set the arrow pixels: use the mask to set arrow_color with full opacity (alpha = 255)
arrow_rgba[black_mask == 255] = [arrow_color[0], arrow_color[1], arrow_color[2], 255]

# Save the result as a PNG to preserve transparency
cv2.imwrite('arrow_transparent.png', arrow_rgba)

# Display the resulting arrow with a transparent background
# (Note: cv2.imshow may not show transparency, but the saved PNG will have it)
cv2.imshow('Orginal image', image)
cv2.imshow('Arrow with Transparent Background', arrow_rgba)
cv2.waitKey(0)
cv2.destroyAllWindows()
