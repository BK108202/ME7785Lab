import cv2
import numpy as np
from skimage.feature import hog

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Define contrast and brightness adjustments
    alpha = 1.97  # Increase contrast by 10%
    beta = -15    # Increase brightness by 1 

    # Apply the contrast and brightness adjustments
    adjusted_gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # # Simple thresholding
    # ret, thresh1 = cv2.threshold(adjusted_gray, 127, 255, cv2.THRESH_BINARY)

    # (2) Threshold Adaptive
    thresh2 = cv2.adaptiveThreshold(adjusted_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # # Otsu's thresholding
    # ret3, thresh3 = cv2.threshold(adjusted_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel, iterations=1)

    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    updated_img = closed



    ## Add Edge Detection Here
    edges = cv2.Canny(updated_img, threshold1=50, threshold2=150)


    ## (3) Find the min-area contour
    # cnts = cv2.findContours(updated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # cnts = sorted(cnts, key=cv2.contourArea)
    # for cnt in cnts:
    #     if cv2.contourArea(cnt) > 100:
    #         break

    # contours, _ = cv2.findContours(updated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # min_area = 1000  # tweak this based on your image size
    # filtered = np.zeros_like(closed)
    # for cnt in contours:
    #     if cv2.contourArea(cnt) > min_area:
    #         cv2.drawContours(filtered, [cnt], -1, 255, -1)

    # ## (4) Create mask and do bitwise-op
    # # mask = np.zeros(img.shape[:2],np.uint8)
    # # cv2.drawContours(mask, [cnt],-1, 255, -1)
    # # dst = cv2.bitwise_and(img, img, mask=mask)

    # isolated_arrow = cv2.bitwise_and(adjusted_gray, adjusted_gray, mask=filtered)
    
    # Extract HOG features with adjusted parameters
    # hog_features = hog(
    #     edges,
    #     orientations=9,
    #     pixels_per_cell=(8, 8),   # Experiment with (8,8) vs. (4,4)
    #     cells_per_block=(2, 2),
    #     block_norm='L2-Hys',
    #     transform_sqrt=True,
    #     feature_vector=True
    # )
    
    return edges



