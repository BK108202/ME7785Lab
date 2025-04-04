import cv2
import numpy as np
from skimage.feature import hog

def preprocess_image(img, target_size=(25, 33)):
    """
    Preprocess an image by:
      1. Converting it to grayscale.
      2. Resizing the image to a fixed target size.
      3. Normalizing the pixel values.
      4. Extracting HOG features from the normalized image.
      
    Args:
        img (numpy.ndarray): Input image.
        target_size (tuple): Target size to resize the image (width, height).

    Returns:
        numpy.ndarray: HOG feature vector.
    """
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the target size
    resized_img = cv2.resize(gray_img, target_size)
    
    # Normalize the pixel values to the range [0, 1]
    normalized_img = resized_img.astype(np.float32) / 255.0
    
    # Extract HOG features from the normalized image
    hog_features = hog(
        normalized_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    
    return hog_features
