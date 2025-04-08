import cv2
import numpy as np
from skimage.feature import hog

def preprocess_image(img):
    # --- Extract Color Histogram Features ---
    # Convert the image from BGR to HSV color space for a more perceptually relevant histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the number of bins for each channel
    h_bins = 32  # Hue bins; note that hue ranges between 0-180 in OpenCV
    s_bins = 32  # Saturation bins; saturation ranges between 0-256
    v_bins = 32  # Value bins; value ranges between 0-256

    # Compute the histogram for each channel
    hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])

    # Normalize the histograms (so that the feature vector is scale invariant)
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)

    # Flatten and concatenate the histograms into one feature vector
    color_hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])

    # --- Compute HOG Features ---
    # Convert the input image to grayscale (HOG typically operates on grayscale images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute HOG features.
    # The parameters provided below (orientations, pixels_per_cell, cells_per_block, etc.) 
    # can be tuned based on your application.
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )

    # --- Combine the Features ---
    # Concatenate the color histogram features and HOG features into one feature vector.
    combined_features = np.concatenate([color_hist, hog_features])

    return combined_features
