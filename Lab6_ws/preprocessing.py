import cv2
import numpy as np
from skimage.feature import hog

def preprocess_image(img, target_size=(25, 33)):
    """
    Preprocess an image by:
      1. Resizing the image to a fixed target size.
      2. Converting it to grayscale.
      3. Enhancing contrast using histogram equalization.
      4. Reducing noise using a Gaussian blur.
      5. Extracting the area of the largest contour from a thresholded and morphologically processed version of the image.
      6. Computing HOG features from the preprocessed image.
      7. Combining raw pixel values (normalized), the contour area, and HOG features into a single feature vector.

    Args:
        img (numpy.ndarray): Input image.
        target_size (tuple): Target size to resize the image (width, height).

    Returns:
        numpy.ndarray: Combined feature vector.
    """
    # 1. Resize the image
    resized_img = cv2.resize(img, target_size)
    
    # 2. Convert to grayscale
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    
    # 3. Enhance contrast with histogram equalization
    gray_eq = cv2.equalizeHist(gray_img)
    
    # 4. Reduce noise using Gaussian blur
    gray_blur = cv2.GaussianBlur(gray_eq, (3, 3), 0)
    
    # 5. Extract the largest contour's area
    # Apply a threshold to get a binary image
    _, thresh = cv2.threshold(gray_blur, 50, 255, cv2.THRESH_BINARY)
    # Apply morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Find external contours
    contours, _ = cv2.findContours(thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
    else:
        contour_area = 0.0  # Default if no contours are found
    
    # 6. Compute HOG features from the preprocessed image (using the blurred image)
    hog_features = hog(
        gray_blur,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    
    # 7. Extract normalized raw pixel values from the blurred image
    raw_features = gray_blur.flatten().astype(np.float32) / 255.0
    
    # Combine raw pixel features, the contour area, and HOG features into one feature vector
    combined_features = np.concatenate([raw_features, np.array([contour_area], dtype=np.float32), hog_features])
    
    return combined_features
