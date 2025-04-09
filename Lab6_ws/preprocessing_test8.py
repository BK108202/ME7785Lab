import cv2
import numpy as np
from skimage.feature import hog

def preprocess_image(img, output_size=(50, 50)):
    # Convert to HSV for color segmentation
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define boundaries for red, green, and blue in HSV space
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

    # Combine the masks for red, green, and blue
    combined_mask = cv2.bitwise_or(red_mask, green_mask)
    combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

    # Find contours on the combined color mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Set initial values for detecting the best contour
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

    # Crop the arrow if detected; otherwise, work on the whole image.
    if best_box is not None:
        x, y, w, h = best_box
        cropped_arrow = img[y:y+h, x:x+w]
        # Optional: Apply Canny edge detection (if needed)
        _ = cv2.Canny(cropped_arrow, threshold1=50, threshold2=150)
        resized_img = cv2.resize(cropped_arrow, output_size)
    else:
        resized_img = cv2.resize(img, output_size)
    
    # ------------------------ New Gradient Extraction Step ------------------------
    # Convert the resized (or cropped) image to grayscale
    gray_resized = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    # Compute gradients in the x and y directions using the Sobel operator
    grad_x = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
    # Compute gradient magnitude
    gradient = cv2.magnitude(grad_x, grad_y)
    # Convert gradient magnitude to 8-bit image for further processing
    gradient = cv2.convertScaleAbs(gradient)
    
    # Optionally, compute a histogram of the gradient image (similar to color histograms)
    histSize = [8]         # 8 bins for the histogram
    hist_range = [0, 256]    # pixel value range
    gradient_hist = cv2.calcHist([gradient], [0], None, histSize, hist_range)
    gradient_hist = cv2.normalize(gradient_hist, gradient_hist).flatten()
    # --------------------- End of Gradient Extraction Step ---------------------

    # === Extract Color Histograms from the resized image ===
    color_hist = []
    channels = [0, 1, 2]
    for ch in channels:
        hist = cv2.calcHist([resized_img], [ch], None, histSize, hist_range)
        # Normalize the histogram for the channel
        hist = cv2.normalize(hist, hist).flatten()
        color_hist.append(hist)
    color_hist = np.concatenate(color_hist)

    # === HOG Feature Extraction ===
    # Convert the resized image to grayscale for HOG feature extraction
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    eq_img = cv2.equalizeHist(gray_img)
    normalized_img = eq_img.astype(np.float32) / 255.0

    hog_features = hog(
        normalized_img,
        orientations=9,
        pixels_per_cell=(8, 8),   # You can experiment with (8,8) vs. (4,4)
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )

    # Combine color histograms with the new gradient histogram and HOG features into one feature vector.
    combined_features = np.concatenate((color_hist, gradient_hist, hog_features))

    return combined_features
