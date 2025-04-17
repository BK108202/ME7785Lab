#!/usr/bin/env python3
import cv2
import argparse
import csv
import math
import pickle
import numpy as np
import random
from preprocessing_test9 import preprocess_image

def check_split_value_range(val):
    try:
        float_val = float(val)
        if float_val < 0 or float_val > 1:
            raise argparse.ArgumentTypeError("Received data split ratio of %s which is an invalid value. The input ratio must be in range [0, 1]!" % float_val)
        return float_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Received '{val}' which is not a valid float!")

def check_k_value(val):
    try:
        int_val = int(val)
        if float(val) != int_val:
            raise argparse.ArgumentTypeError(f"Received '{val}' which is a float not an integer. The KNN value input must be an integer!")
        if int_val % 2 == 0 or int_val < 1:
            raise argparse.ArgumentTypeError(f"Received '{val}' which not a positive, odd integer. The KNN value input must be a postive, odd integer!")
        return int_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Received '{val}' which is not a valid integer!")

def zoom_image(image, zoom_factor):
    h, w = image.shape[:2]
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    cropped = image[start_y:start_y+new_h, start_x:start_x+new_w]
    zoomed = cv2.resize(cropped, (w, h))
    return zoomed

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
    return rotated

def load_and_split_data(data_path, split_ratio, image_type):
    with open(data_path + 'labels.txt', 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    # Data augmentation: create zoomed images and rotated images per original image.
    augmented_lines = []
    zoom_factors = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]
    rotation_angles = [-1, 1]
    for line in lines:
        original_name = line[0]
        label = line[1]
        img_path = data_path + original_name + image_type
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Image {img_path} not found or could not be read.")
            continue
        # Create zoomed images
        for i, z in enumerate(zoom_factors):
            zoomed_img = zoom_image(img, z)
            new_name = f"{original_name}_zoom{i+1}"
            new_image_path = data_path + new_name + image_type
            cv2.imwrite(new_image_path, zoomed_img)
            augmented_lines.append([new_name, label])
        # Create rotated images
        for angle in rotation_angles:
            rotated_img = rotate_image(img, angle)
            new_name = f"{original_name}_rot{angle}"
            new_image_path = data_path + new_name + image_type
            cv2.imwrite(new_image_path, rotated_img)
            augmented_lines.append([new_name, label])

    # Combine the original and augmented entries.
    all_lines = lines + augmented_lines
    random.shuffle(all_lines)
    train_lines = all_lines[:math.floor(len(all_lines) * split_ratio)]
    test_lines = all_lines[math.floor(len(all_lines) * split_ratio):]

    return train_lines, test_lines

def train_model(data_path, train_lines, image_type, model_filename, save_model):
    train_images = []
    for line in train_lines:
        img = cv2.imread(data_path + line[0] + image_type)
        processed_img = preprocess_image(img)
        combined_features = processed_img
        train_images.append(combined_features)

    train_data = np.array(train_images, dtype=np.float32)
    train_labels = np.array([np.int32(line[1]) for line in train_lines])
    
    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    
    print("KNN model created!")
    if save_model:
        knn.save(model_filename + '.xml')
        print(f"KNN model saved to {model_filename}.xml")
    
    return knn

def test_model(data_path, test_lines, image_type, knn_model, knn_value, show_img):
    if show_img:
        cv2.namedWindow('Original Image', cv2.WINDOW_AUTOSIZE)
    
    correct = 0.0
    confusion_matrix = np.zeros((6, 6))
    k = knn_value
    
    for line in test_lines:
        img = cv2.imread(data_path + line[0] + image_type)
        if show_img:
            cv2.imshow('Original Image', img)
            key = cv2.waitKey()
            if key == 27:  # Esc key to stop
                break
        
        combined_features = preprocess_image(img)
        test_sample = combined_features.reshape(1, -1).astype(np.float32)
        test_label = np.int32(line[1])
        
        ret, results, neighbours, dist = knn_model.findNearest(test_sample, k)
        
        if test_label == ret:
            print(f"{line[0]} Correct, {ret}")
            correct += 1
            confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
        else:
            confusion_matrix[test_label][np.int32(ret)] += 1
            print(f"{line[0]} Wrong, {test_label} classified as {ret}")
            print(f"\tneighbours: {neighbours}")
            print(f"\tdistances: {dist}")
    
    print("\n\nTotal accuracy:", correct / len(test_lines))
    print(confusion_matrix)

def main():
    parser = argparse.ArgumentParser(description="Example Model Trainer and Tester with Basic KNN for 7785 Lab 6!")
    parser.add_argument("-p","--data_path", type=str, required=True, help="Path to the valid dataset directory (must contain labels.txt and images)")
    parser.add_argument("-r","--data_split_ratio", type=check_split_value_range, required=False, default=0.5, help="Ratio of the train, test split. Must be a float between 0 and 1. The number entered is the percentage of data used for training, the remaining is used for testing!")
    parser.add_argument("-k","--knn-value", type=check_k_value, required=False, default=3, help="KNN value. Must be an odd integer greater than zero.")
    parser.add_argument("-i","--image_type", type=str, required=False, default=".png", help="Extension of the image files (e.g. .png, .jpg)")
    parser.add_argument("-s","--save_model_bool", action='store_true', required=False, help="Boolean flag to save the KNN model as a XML file for later use.")
    parser.add_argument("-n","--model_filename", type=str, required=False, default="knn_model", help="Filename of the saved KNN model.")
    parser.add_argument("-t","--dont_test_model_bool", action='store_false', required=False, help="Boolean flag to not test the created KNN model on split testing set (training only).")
    parser.add_argument("-d","--show_img", action='store_true', required=False, help="Boolean flag to show the tested images as they are classified.")

    args = parser.parse_args()
    
    random.seed(42)

    dataset_path = args.data_path
    data_split_ratio = args.data_split_ratio
    image_type = args.image_type
    save_model_bool = args.save_model_bool
    model_filename = args.model_filename
    test_model_bool = args.dont_test_model_bool
    knn_value = args.knn_value
    show_img = args.show_img

    # Pass image_type to load_and_split_data so that augmentation works correctly
    train_lines, test_lines = load_and_split_data(dataset_path, data_split_ratio, image_type)
    knn_model = train_model(dataset_path, train_lines, image_type, model_filename, save_model_bool)
    if test_model_bool:
        test_model(dataset_path, test_lines, image_type, knn_model, knn_value, show_img)

if __name__ == "__main__":
    main()
