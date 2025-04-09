import cv2
import numpy as np

def preprocess_image(img):
    """
    Processes the image to extract the arrow using HSV color filtering.
    First, red, green, and blue areas are isolated and combined. Then, 
    the combined image is reprocessed to detect a black arrow, and finally 
    the arrow is painted in a distinct color (green) on a black background.
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
    
    return arrow_colored