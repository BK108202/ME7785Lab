import cv2
import numpy as np
from preprocessing import preprocess_image
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