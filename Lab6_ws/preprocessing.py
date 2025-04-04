import cv2
import numpy as np
from skimage.feature import hog

def preprocess_image(img, target_size=(25, 33)):
    """
    Preprocess an image by:
      1. Resizing the image to a fixed target size.
      2. Converting it to grayscale.
      3. Extracting the area of the largest contour from a thresholded version of the grayscale image.
      4. Computing HOG features from the grayscale image.
      5. Combining raw pixel values, the contour area, and HOG features into a single feature vector.

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
    
    # 3. Extract the largest contour's area
    # Apply a threshold to obtain a binary image
    _, thresh = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
    else:
        contour_area = 0.0  # If no contours are found, default to 0
    
    # 4. Compute HOG features from the grayscale image
    hog_features = hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    
    # Optionally include raw grayscale pixel features (normalized)
    raw_features = gray_img.flatten().astype(np.float32) / 255.0
    
    # 5. Combine raw pixel features, contour area (as a single feature), and HOG features
    combined_features = np.concatenate([raw_features, np.array([contour_area], dtype=np.float32), hog_features])
    
    return combined_features

def display_image(image_list, target_size=(25, 33)):
    """
    Processes the first five images using the preprocess_image function.
    From each processed output (a feature vector), the raw pixel features (i.e. the grayscale image)
    are extracted and reshaped to their 2D form (height, width). This enables visualization of the preprocessed images.

    Args:
        image_list (list): List of images (numpy arrays).
        target_size (tuple): Target size used in preprocessing (width, height).

    Returns:
        list: A list of processed grayscale images (each of shape (target_size[1], target_size[0])).
    """
    processed_images = []
    # Determine how many images to process (up to five)
    num_images = min(5, len(image_list))
    for img in image_list[:num_images]:
        features = preprocess_image(img, target_size)
        # The first part of the feature vector corresponds to the raw grayscale pixels.
        num_raw_pixels = target_size[0] * target_size[1]
        raw_pixels = features[:num_raw_pixels]
        # Reshape back to 2D form: (height, width)
        processed_img = raw_pixels.reshape((target_size[1], target_size[0]))
        processed_images.append(processed_img)
    return processed_images
