# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 18:33:07 2025

@author: mizuc
"""

import cv2
import numpy as np
#import matplotlib.pyplot as plt

# read the input
#img = cv2.imread('./New_Images/IMG3.jfif')

img = cv2.imread('./lab6_ws/test1')
output_size = (50, 50)

# Looking at Red Colors First
# convert to LAB
LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# separate channels
L, A, B = cv2.split(LAB)

# get the maximum between the A and B pixel by pixel
ABmax = np.maximum(A, B)

# threshold
thresh = cv2.threshold(ABmax, 180, 255, cv2.THRESH_BINARY)[1]

# morphology close and open
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# get contours (fixed unpacking)
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

no_contour = False
min_area = 0.1

# initial rough check for any contour above tiny area
image_with_contour = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
if image_with_contour:
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area < 20000 and perimeter > 100:
            M = cv2.moments(c)
            # image_with_contours = cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
            ix, iy, iw, ih = cv2.boundingRect(c)
            # cv2.rectangle(image_with_contours, (ix, iy), (ix+iw, iy+ih), (0, 255, 0), 5)
            # if M['m00'] != 0:
            #     x = int(M['m10']/M['m00'])
            #     y = int(M['m01']/M['m00'])
            #     cv2.drawMarker(img, (x, y), (0, 255, 0), cv2.MARKER_DIAMOND, markerSize=10, thickness=2)

if not image_with_contour:
    no_contour = True
    print(no_contour)
else:
    no_contour = True

# If no arrow is found, try green/blue threshold fallback
if no_contour:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # Define contrast and brightness adjustments
    alpha = 1.0  # Increase contrast by 0%
    beta = 25   # Increase brightness by 10 

    # # # Apply the contrast and brightness adjustments
    bright_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    blurred_image = cv2.GaussianBlur(bright_img,(15,15),0)
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # # Define the lower and upper bounds for red, green, and blue
    # lower_red = np.array([0, 100, 100])
    # upper_red = np.array([10, 255, 255])

    # lower_green = np.array([40, 100, 100])
    # upper_green = np.array([238, 255, 255])

    # lower_blue = np.array([90, 100, 100])
    # upper_blue = np.array([130, 255, 255])

    # # Create masks for each color
    # red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    # green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    # blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # # Combine the masks
    # combined_mask = cv2.bitwise_or(red_mask, green_mask)
    # combined_mask = cv2.bitwise_or(combined_mask, blue_mask)


    threshold = 300
    # Create a binary mask: 
    # - pixels with value greater than threshold are set to 255 (white)
    # - pixels with value less than or equal to threshold are set to 0 (black)
    _, white_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Invert the mask to select the non-white regions
    white_mask_inv = cv2.bitwise_not(white_mask)

    result = cv2.bitwise_and(img, img, mask=white_mask_inv)

    lower_green = np.array([20, 0, 0])
    upper_green = np.array([235, 255, 255])

    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])

    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    blue_mask  = cv2.inRange(hsv_image, lower_blue, upper_blue)

    combined_mask = cv2.bitwise_or(green_mask, blue_mask)
    combined_mask = cv2.bitwise_or(combined_mask, white_mask)
    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(img, img, mask=combined_mask)

    ret, thresh1 = cv2.threshold(combined_mask, 180, 255, cv2.THRESH_BINARY)
    #ret, thresh1 = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
    # # Threshold Adaptive Mean
    thresh2 = cv2.adaptiveThreshold(combined_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel1 = np.ones((7,7))
    morph1 = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel1, iterations=1)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(img, img, mask=combined_mask)


    # Find contours
    contours, _ = cv2.findContours(morph1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# ——— Updated: pick contour whose centroid is closest to image center ———
img_h, img_w = img.shape[:2]
center_x, center_y = img_w / 2, img_h / 2
min_dist = float('inf')
best_box = None

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area < 25000 and perimeter > 200:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        dist = np.hypot(cx - center_x, cy - center_y)
        if dist < min_dist:
            min_dist = dist
            best_box = cv2.boundingRect(cnt)

if best_box is not None:
    x, y, w_box, h_box = best_box
    cropped_arrow = img[y:y+h_box, x:x+w_box]
    resized_img = cv2.resize(cropped_arrow, output_size)
else:
    # center‑200×200 fallback crop + resize
    h, w = img.shape[:2]
    crop_w, crop_h = 200, 200
    start_x = max((w - crop_w) // 2, 0)
    start_y = max((h - crop_h) // 2, 0)
    if h < crop_h or w < crop_w:
        center_crop = cv2.resize(img, (crop_w, crop_h))
    else:
        center_crop = img[start_y:start_y+crop_h, start_x:start_x+crop_w]
    resized_img = cv2.resize(center_crop, output_size)
  
cv2.imshow('CropOutput.jpg', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
    # for i in contours:
    #     area = cv2.contourArea(i)
    #     perimeter = cv2.arcLength(i, True)
    #     if (cv2.contourArea(i)) < 25000 and perimeter > 370:
    #         midPoint = cv2.moments(i)
    #         image_with_contours = cv2.drawContours(img, contours, -1, (255, 0, 0), 2) #Draw Contours need -1 to show
    #         ix, iy, iw, ih = cv2.boundingRect(i)  
    #         cv2.rectangle(image_with_contours,(ix,iy), (ix+iw,iy+ih), (0, 255, 0), 5)
    #       #if midPoint['m00']!=0:
    #         x = int(midPoint['m10']/midPoint['m00'])
    #         y = int(midPoint['m01']/midPoint['m00'])
    #         cv2.drawMarker(img, (x, y), (0, 255, 0), cv2.MARKER_DIAMOND, markerSize=10, thickness=2)
    
    # else

# # # Filter contours
# result = np.zeros_like(img)
# for cntr in contours:
#     area = cv2.contourArea(cntr)
#     perimeter = cv2.arcLength(cntr, True)
#     if area > 10 and perimeter < 20:
#         cv2.drawContours(result, [cntr], 0, (255,255,255), -1)