import cv2
import numpy as np
from skimage.feature import hog

def preprocess_image(img, target_size=(32, 32)):
    """
    Preprocess an image by:
      1. Converting it to grayscale.
      2. Resizing the image to a fixed target size.
      3. Applying histogram equalization to enhance contrast.
      4. Normalizing the pixel values.
      5. Extracting HOG features from the normalized image.
      
    Args:
        img (numpy.ndarray): Input image.
        target_size (tuple): Target size to resize the image (width, height).

    Returns:
        numpy.ndarray: HOG feature vector.
    """
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to target size (experiment with different sizes)
    resized_img = cv2.resize(gray_img, target_size)
    
    # Apply histogram equalization to improve contrast
    eq_img = cv2.equalizeHist(resized_img)
    
    # Normalize pixel values to [0, 1]
    normalized_img = eq_img.astype(np.float32) / 255.0
    
    # Extract HOG features with adjusted parameters
    hog_features = hog(
        normalized_img,
        orientations=9,
        pixels_per_cell=(8, 8),   # Experiment with (8,8) vs. (4,4)
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    
    return hog_features
