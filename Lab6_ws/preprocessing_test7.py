import cv2
import numpy as np

def preprocess_image(img, output_size=(50, 50)):
    # Convert to HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define boundaries for red, green, and blue
    lower_red   = np.array([0, 100, 100])
    upper_red   = np.array([10, 255, 255])
    lower_green = np.array([20, 0, 0])
    upper_green = np.array([238, 120, 255])
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
        # Optional: Edge detection on the cropped image
        _ = cv2.Canny(cropped_arrow, threshold1=50, threshold2=150)
        resized_img = cv2.resize(cropped_arrow, output_size)
    else:
        # If no arrow is detected, resize the whole image to guarantee a fixed output size
        resized_img = cv2.resize(img, output_size)

    # === Extract Color Histograms ===
    # Compute a histogram for each color channel from the resized image.
    # Using 8 bins per channel (adjustable) over the range [0, 256].
    histSize = [8]
    hist_range = [0, 256]
    channels = [0, 1, 2]
    color_hist = []
    for ch in channels:
        hist = cv2.calcHist([resized_img], [ch], None, histSize, hist_range)
        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()
        color_hist.append(hist)
    color_hist = np.concatenate(color_hist)
    # === End of Color Histograms Extraction ===

    # Convert the resized image to grayscale for HOG computation
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Create a HOGDescriptor object with parameters suited for a 50x50 image.
    # Adjust the following parameters if needed:
    winSize    = output_size
    blockSize  = (10, 10)
    blockStride= (4, 4)
    cellSize   = (5, 5)
    nbins      = 12

    hog_descriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    # Compute the HOG features on the gray image.
    hog_features = hog_descriptor.compute(gray_img)

    # Check if hog_features is valid and non-empty.
    if hog_features is None or (isinstance(hog_features, tuple) and len(hog_features) == 0):
        hog_features = np.zeros((1,))
    else:
        if isinstance(hog_features, tuple):
            hog_features = hog_features[0]
    # Flatten to ensure a 1D feature vector.
    hog_features = hog_features.flatten()

    # Combine color histograms with HOG features into one feature vector.
    combined_features = np.concatenate((color_hist, hog_features))

    return combined_features
