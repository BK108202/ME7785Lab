import cv2
import numpy as np

# Assume 'img' is your original BGR image

img = cv2.imread('./Lab6_ws/2025S_imgs/040.png')

# Convert to HSV
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define boundaries for red, green, and blue
lower_red   = np.array([0, 100, 100])
upper_red   = np.array([10, 255, 255])
lower_green = np.array([20, 0, 0])
upper_green = np.array([110, 120, 255])
lower_blue  = np.array([110, 100, 100])
upper_blue  = np.array([130, 255, 255])

# Create masks for each color
red_mask   = cv2.inRange(hsv_image, lower_red, upper_red)
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
blue_mask  = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Combine the masks
combined_mask = cv2.bitwise_or(red_mask, green_mask)
combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

# Find contours
contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Optional: Edge detection on the combined mask (not used for cropping here)
_ = cv2.Canny(combined_mask, threshold1=25, threshold2=75)

largest_area = 0
best_box = None

for c in contours:
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    # Filter out contours that are too small or too irregular
    if area > 1000 and perimeter < 1000:
        x, y, w, h = cv2.boundingRect(c)
        if area > largest_area:
            largest_area = area
            best_box = (x, y, w, h)

if best_box is not None:
    x, y, w, h = best_box
    cropped_arrow = img[y:y+h, x:x+w]
    resized_img = cropped_arrow
else:
    # If no arrow is detected, resize the whole image to guarantee a fixed output size
    resized_img = img
    
# Finally, if you want, show the original image with the drawn rectangle
cv2.imshow("Detected Arrow", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
