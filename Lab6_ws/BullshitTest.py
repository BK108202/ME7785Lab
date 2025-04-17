# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 18:43:04 2025

@author: mizuc
"""

import cv2
import numpy as np

# Load the image
#image = cv2.imread('./New_Images/IMG8.jfif')
image = cv2.imread('./2025S_imgs/003.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # Define contrast and brightness adjustments
alpha = 1.0  # Increase contrast by 0%
beta = 25   # Increase brightness by 10 

# # # Apply the contrast and brightness adjustments
bright_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
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

result = cv2.bitwise_and(image, image, mask=white_mask_inv)

lower_green = np.array([20, 0, 0])
upper_green = np.array([235, 255, 255])

lower_blue = np.array([90, 100, 100])
upper_blue = np.array([130, 255, 255])

green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
blue_mask  = cv2.inRange(hsv_image, lower_blue, upper_blue)

combined_mask = cv2.bitwise_or(green_mask, blue_mask)
combined_mask = cv2.bitwise_or(combined_mask, white_mask)
# Apply the mask to the original image
masked_image = cv2.bitwise_and(image, image, mask=combined_mask)

ret, thresh1 = cv2.threshold(combined_mask, 180, 255, cv2.THRESH_BINARY)
#ret, thresh1 = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
# # Threshold Adaptive Mean
thresh2 = cv2.adaptiveThreshold(combined_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
kernel1 = np.ones((7,7))
morph1 = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel1, iterations=1)

# Apply the mask to the original image
masked_image = cv2.bitwise_and(image, image, mask=combined_mask)


# Find contours
contours, _ = cv2.findContours(morph1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

largest_area = 0
best_box = None
midPoint = 0
output_size=(50, 50)
for c in contours:
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    # Filter out contours that are too small or too irregular
    if area < 25000 and perimeter > 100:
        midPoint = cv2.moments(c)
        #image_with_contours = cv2.drawContours(image, contours, 0, (255, 0, 0), 2) #Draw Contours need -1 to show
        ix, iy, iw, ih = cv2.boundingRect(c)  
        #cv2.rectangle(image_with_contours,(ix,iy), (ix+iw,iy+ih), (0, 255, 0), 5)
      #if midPoint['m00']!=0:
        x = int(midPoint['m10']/midPoint['m00'])
        y = int(midPoint['m01']/midPoint['m00'])
        #cv2.drawMarker(image, (x, y), (0, 255, 0), cv2.MARKER_DIAMOND, markerSize=10, thickness=2)
        
        if area > largest_area:
            largest_area = area
            best_box = (ix, iy, iw, ih)
            
        if best_box is not None:
            ix, iy, iw, ih = best_box
            cropped_arrow = image[iy:iy+ih, ix:ix+iw]
            resized_img = cv2.resize(cropped_arrow, output_size)
        else:
            # If no arrow is detected, resize the whole image to guarantee a fixed output size
            resized_img = cv2.resize(image, output_size)

# Display the result
cv2.imwrite('BSTestImage.jpg', resized_img)
cv2.imwrite('BSTestMask.jpg', masked_image)
cv2.imwrite('BSTestMorphThresh.jpg', thresh1)
cv2.imwrite('BSTestMorphClose.jpg', morph1)