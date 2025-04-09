import cv2
import numpy as np

def preprocess_image(img, output_size=(50, 50)):
    """
    Processes the image to detect and crop the arrow based on HSV color filtering.
    The arrow is first isolated using red, green, and blue masks. Then,
    the cropped region is processed to detect a black arrow, painting it green on a black background.
    Finally, only the largest green area is kept and the result is resized.
    """
    # --- Step 1: Crop the Region containing the Arrow ---
    # Convert to HSV color space for filtering
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define HSV boundaries for red, green, and blue
    lower_red   = np.array([0, 100, 100])
    upper_red   = np.array([10, 255, 255])
    
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    
    lower_blue  = np.array([110, 100, 100])
    upper_blue  = np.array([130, 255, 255])
    
    # Create masks for each color
    red_mask   = cv2.inRange(hsv_image, lower_red, upper_red)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    blue_mask  = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Combine the masks
    combined_mask = cv2.bitwise_or(red_mask, green_mask)
    combined_mask = cv2.bitwise_or(combined_mask, blue_mask)
    
    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_area = 0
    best_box = None
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        # Filter out noise by area and perimeter constraints
        if area > 1000 and perimeter < 1000:
            x, y, w, h = cv2.boundingRect(c)
            if area > largest_area:
                largest_area = area
                best_box = (x, y, w, h)
    
    if best_box is not None:
        x, y, w, h = best_box
        cropped_img = img[y:y+h, x:x+w]
    else:
        # If no arrow-like object is detected, work with the full image.
        cropped_img = img.copy()
    
    # --- Step 2: Detect the Arrow and Change its Appearance ---
    # Convert the cropped region to HSV and detect areas that look "black" (the arrow).
    arrow_processed = detect_and_color_arrow(cropped_img)
    
    # --- Step 3: Resize the Final Result ---
    final_result = cv2.resize(arrow_processed, output_size).flatten()
    
    return final_result

def detect_and_color_arrow(img):
    """
    Detects the arrow in the provided image region by using a mask to
    find black regions. The detected black arrow is then painted in green
    (BGR: (0, 255, 0)) on a black background.
    """
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for red, green, and blue
    lower_red   = np.array([0, 100, 100])
    upper_red   = np.array([10, 255, 255])
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    lower_blue  = np.array([110, 100, 100])
    upper_blue  = np.array([130, 255, 255])
    
    # Create individual masks for red, green, and blue
    red_mask   = cv2.inRange(hsv_image, lower_red, upper_red)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    blue_mask  = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Combine the color masks
    combined_mask = cv2.bitwise_or(red_mask, green_mask)
    combined_mask = cv2.bitwise_or(combined_mask, blue_mask)
    
    # Apply the combined mask to the original image
    masked_image = cv2.bitwise_and(img, img, mask=combined_mask)
    
    # Convert the masked image to HSV to detect the black arrow
    hsv_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    
    # Define an HSV range for "black" (adjust the upper V value if needed)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 40])
    black_mask = cv2.inRange(hsv_masked, lower_black, upper_black)
    
    # Create a new, empty image (with the same size as the input)
    arrow_colored = np.zeros_like(img)
    
    # Paint the detected black arrow in a distinct color (here, green)
    arrow_color = (0, 255, 0)
    arrow_colored[black_mask == 255] = arrow_color

    # Keep only the largest green area (to avoid stray green pixels)
    arrow_colored1 = keep_largest_green_area(arrow_colored)
    
    return arrow_colored1

def keep_largest_green_area(img):
    """
    Keeps only the largest green area in the image and sets the rest to black.
    Assumes that the arrow is painted in green (BGR: (0,255,0)).
    A small tolerance is added to account for minor variations.
    """
    # Define a tolerance range for green (adjust if needed)
    # Here we assume the arrow color is near (0, 255, 0)
    lower_green = np.array([0, 250, 0])
    upper_green = np.array([0, 255, 0])
    
    # Create a binary mask where the green color is detected
    green_mask = cv2.inRange(img, lower_green, upper_green)
    
    # Find contours in the green mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # If no contours are found, return an all-black image.
        return np.zeros_like(img)
    
    # Identify the largest contour by area (assumed to be the arrow)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask for the largest green region only
    largest_mask = np.zeros_like(green_mask)
    cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Create the result image: copy over the green pixels
    result = np.zeros_like(img)
    result[largest_mask == 255] = img[largest_mask == 255]
    
    return result

# # Example usage:
# img = cv2.imread('./Lab6_ws/2025S_imgs/031.png')
# processed_img = preprocess_image(img)
# cv2.imshow("Processed Arrow", processed_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
