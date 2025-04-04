# preprocessing.py
import cv2
import numpy as np

def preprocess_image(img, target_size=(25, 33)):
    """
    Preprocess an image by resizing, normalizing, extracting edge features,
    and combining all features into a single vector.

    Args:
        img (numpy.ndarray): Input image.
        target_size (tuple): Target size to resize the image.

    Returns:
        numpy.ndarray: Combined feature vector.
    """
    # Resize the image
    resized_img = cv2.resize(img, target_size)
    
    # Normalize the image to [0,1]
    normalized_img = resized_img / 255.0
    
    # Compute edge features using Canny
    edges = cv2.Canny((normalized_img * 255).astype(np.uint8), 100, 200)
    edges_normalized = edges.astype(np.float32) / 255.0
    
    # Flatten and combine features
    features_img = normalized_img.flatten()      # Image features
    features_edges = edges_normalized.flatten()    # Edge features
    
    combined_features = np.concatenate((features_img, features_edges))
    return combined_features
