import cv2
import numpy as np

def preprocess_image(img, output_size=(50, 50)):
    # Convert to HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define boundaries for red, green, and blue
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])

    # Create masks for each color
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Combine the masks
    combined_mask = cv2.bitwise_or(red_mask, green_mask)
    combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # (Optional) Edge detection (not used in cropping here)
    edges1 = cv2.Canny(combined_mask, threshold1=25, threshold2=75)

    largest_area = 0
    best_box = None

    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        # Filter out small or weird contours
        if area > 1000 and perimeter < 1000:
            x, y, w, h = cv2.boundingRect(c)
            if area > largest_area:
                largest_area = area
                best_box = (x, y, w, h)

    if best_box is not None:
        x, y, w, h = best_box
        cropped_arrow = img[y:y + h, x:x + w]

        # Optionally perform edge detection on the cropped image
        edges = cv2.Canny(cropped_arrow, threshold1=50, threshold2=150)

        # Resize the cropped arrow to a fixed size so that all feature vectors are the same length
        resized_arrow = cv2.resize(cropped_arrow, output_size).flatten()
        return resized_arrow
    else:
        # If no arrow detected, resize the whole image to guarantee a fixed output size
        resized_img = cv2.resize(img, output_size).flatten()
        return resized_img