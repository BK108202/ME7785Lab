#!/usr/bin/env python3
import cv2
import argparse
import csv
import math
import numpy as np
import random
import os
import glob
import joblib
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_image

# --- Set Random Seeds for Reproducibility ---
random.seed(42)
np.random.seed(42)

def check_split_value_range(val):
    try:
        float_val = float(val)
        if float_val < 0 or float_val > 1:
            raise argparse.ArgumentTypeError(
                "Received data split ratio of %s which is invalid. It must be in the range [0, 1]!" % float_val)
        return float_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Received '{val}' which is not a valid float!")

def check_k_value(val):
    try:
        int_val = int(val)
        if float(val) != int_val:
            raise argparse.ArgumentTypeError(
                f"Received '{val}' which is a float, not an integer. The KNN value must be an integer!")
        if int_val % 2 == 0 or int_val < 1:
            raise argparse.ArgumentTypeError(
                f"Received '{val}' which is not a positive, odd integer!")
        return int_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Received '{val}' which is not a valid integer!")

# def preprocess_image(img):
#     """
#     Preprocess an input image by:
#       1. Cropping using the largest contour (if detected)
#       2. Resizing the cropped image to a fixed size (to ensure uniform feature vector lengths)
#       3. Extracting color histograms from the HSV channels (using 32 bins per channel)
#       4. Computing HOG features from the grayscale image
#     Returns a combined feature vector.
#     """
#     # # --- Image Cropping ---
#     # gray_crop = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # _, thresh_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     # contours, _ = cv2.findContours(thresh_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # if contours:
#     #     # Use the largest contour as the region of interest.
#     #     c = max(contours, key=cv2.contourArea)
#     #     x, y, w, h = cv2.boundingRect(c)
#     #     margin = 5  # Optional margin.
#     #     x = max(x - margin, 0)
#     #     y = max(y - margin, 0)
#     #     w = w + 2 * margin
#     #     h = h + 2 * margin
#     #     img = img[y:y+h, x:x+w]

#     # # --- Resize to a Fixed Size ---
#     # fixed_size = (128, 128)  # Adjust as needed.
#     # img = cv2.resize(img, fixed_size)

#     # # --- Color Histogram Extraction (HSV) ---
#     # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     # h_bins, s_bins, v_bins = 32, 32, 32
#     # hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
#     # hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
#     # hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])
#     # cv2.normalize(hist_h, hist_h)
#     # cv2.normalize(hist_s, hist_s)
#     # cv2.normalize(hist_v, hist_v)
#     # color_hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])

#     # # --- HOG Feature Extraction ---
#     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # hog_features = hog(
#     #     gray,
#     #     orientations=9,
#     #     pixels_per_cell=(8, 8),
#     #     cells_per_block=(2, 2),
#     #     block_norm='L2-Hys',
#     #     transform_sqrt=True,
#     #     feature_vector=True
#     # )

#     # # --- Combine Features ---
#     # combined_features = np.concatenate([color_hist, hog_features])
#     # return combined_features

#     # --- Extract Color Histogram Features ---
#     # Convert the image from BGR to HSV color space for a more perceptually relevant histogram
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     # Define the number of bins for each channel
#     h_bins = 32  # Hue bins; note that hue ranges between 0-180 in OpenCV
#     s_bins = 32  # Saturation bins; saturation ranges between 0-256
#     v_bins = 32  # Value bins; value ranges between 0-256

#     # Compute the histogram for each channel
#     hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
#     hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
#     hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])

#     # Normalize the histograms (so that the feature vector is scale invariant)
#     cv2.normalize(hist_h, hist_h)
#     cv2.normalize(hist_s, hist_s)
#     cv2.normalize(hist_v, hist_v)

#     # Flatten and concatenate the histograms into one feature vector
#     color_hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])

#     # --- Compute HOG Features ---
#     # Convert the input image to grayscale (HOG typically operates on grayscale images)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Compute HOG features.
#     # The parameters provided below (orientations, pixels_per_cell, cells_per_block, etc.) 
#     # can be tuned based on your application.
#     hog_features = hog(
#         gray,
#         orientations=9,
#         pixels_per_cell=(8, 8),
#         cells_per_block=(2, 2),
#         block_norm='L2-Hys',
#         transform_sqrt=True,
#         feature_vector=True
#     )

#     # --- Combine the Features ---
#     # Concatenate the color histogram features and HOG features into one feature vector.
#     combined_features = np.concatenate([color_hist, hog_features])

#     return combined_features

def load_labels(data_path, image_ext):
    """
    Reads the labels.txt file located in data_path.
    Expected format per line: filename_without_extension,label
    Returns a dictionary mapping filenames to label values.
    """
    label_file = os.path.join(data_path, "labels.txt")
    label_dict = {}
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines.
            parts = line.split(",")
            if len(parts) < 2:
                continue  # Skip malformed lines.
            filename, label = parts[0].strip(), parts[1].strip()
            label_dict[filename] = int(label)
    return label_dict

def load_data(data_path, image_ext, label_dict):
    """
    Returns a sorted list of image file paths from data_path that have a matching entry in label_dict.
    """
    all_image_files = sorted(glob.glob(os.path.join(data_path, "*" + image_ext)))
    # Filter out files that don't have a corresponding label.
    all_image_files = [file for file in all_image_files if os.path.basename(file)[:-len(image_ext)] in label_dict]
    return all_image_files

def train_model(X_train_images, y_train, knn_value):
    """
    Extracts features from the training images, scales them,
    and trains a KNN classifier (using OpenCV's ml.KNearest) with the specified number of neighbors.
    Returns the trained KNN classifier and the scaler.
    """
    # Flatten each image to convert the 3D array into a 1D feature vector.
    X_train = [preprocess_image(img).flatten() for img in X_train_images]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Create and train the KNN model.
    knn = cv2.ml.KNearest_create()
    knn.train(np.array(X_train_scaled, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(y_train, dtype=np.int32))
    return knn, scaler

def test_model(knn, scaler, X_test_images, y_test):
    """
    Extracts features from test images, applies the scaler,
    and predicts using the trained KNN model.
    Computes and prints the accuracy.
    """
    # Flatten each test image as well.
    X_test = [preprocess_image(img).flatten() for img in X_test_images]
    X_test_scaled = scaler.transform(X_test)
    ret, results, neighbours, dist = knn.findNearest(np.array(X_test_scaled, dtype=np.float32), k=3)
    predictions = results.flatten().astype(np.int32)
    correct = sum(int(pred) == int(true) for pred, true in zip(predictions, y_test))
    accuracy = correct / len(y_test)
    print("Test Accuracy:", accuracy)
    return predictions


def main(args):
    # Load label mappings from labels.txt.
    label_dict = load_labels(args.path, args.image_type)
    
    # List all image files matching the given extension and sort them.
    all_image_files = load_data(args.path, args.image_type, label_dict)
    if not all_image_files:
        print("No images found with extension", args.image_type)
        return

    # Split the image files into training and testing sets using the provided ratio.
    train_files, test_files = train_test_split(all_image_files, train_size=args.ratio, random_state=42)
    
    # Load the images and their corresponding labels for training.
    X_train_images, y_train = [], []
    for file in train_files:
        img = cv2.imread(file)
        if img is not None:
            X_train_images.append(img)
            base_name = os.path.basename(file)[:-len(args.image_type)]
            y_train.append(label_dict[base_name])
        else:
            print("Warning: Unable to read", file)
    
    # Load the images and labels for testing.
    X_test_images, y_test = [], []
    for file in test_files:
        img = cv2.imread(file)
        if img is not None:
            X_test_images.append(img)
            base_name = os.path.basename(file)[:-len(args.image_type)]
            y_test.append(label_dict[base_name])
        else:
            print("Warning: Unable to read", file)
    
    # Train the KNN classifier.
    knn, scaler = train_model(X_train_images, y_train, args.knn)
    print("Training completed.")
    
    # Optionally save the model if the '-s' flag is set.
    if args.save:
        knn.save(args.model_filename + '.xml')
        joblib.dump({'scaler': scaler}, args.model_filename + '_scaler.pkl')
        print("Model saved to", args.model_filename + '.xml')
    
    # Test the model and display the predictions.
    predictions = test_model(knn, scaler, X_test_images, y_test)
    print("Predictions on test data:", predictions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to image folder containing images and labels.txt')
    parser.add_argument('-r', '--ratio', type=check_split_value_range, required=True,
                        help='Train-test split ratio (the training portion, e.g., 0.8)')
    parser.add_argument('-k', '--knn', type=check_k_value, required=True,
                        help='Number of neighbors to use for KNN')
    parser.add_argument('-i', '--image_type', type=str, default=".png",
                        help='Image file extension (e.g., .png)')
    parser.add_argument('-s', '--save', action='store_true',
                        help='Flag to save the trained model')
    parser.add_argument('-n', '--model_filename', type=str, default="knn_model",
                        help='Filename for the saved KNN model')
    args = parser.parse_args()
    main(args)
