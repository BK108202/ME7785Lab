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

def load_and_split_data(data_path, split_ratio):
    """
    Uses the provided labels.txt file to split the data into training and testing sets.

    Args:
        data_path (str): Path to the dataset.
        split_ratio (float): must be a float between 0 and 1. Split ratio will be used to split the data into training and testing sets. 
                             split_ratio of the data will be used for training and (1-split_ratio) will be used for testing. 
                             For example if split ratio was 0.7, 70% of the data will be used for training and the remaining 30% will be used for testing.

    Returns:
        list of tuples for testing and training (image_path, true_label)
    """

    with open(data_path + 'labels.txt', 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    #Randomly choose train and test data (50/50 split).
    random.shuffle(lines)
    train_lines = lines[:math.floor(len(lines)*split_ratio)][:]
    test_lines = lines[math.floor(len(lines)*split_ratio):][:]

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
    parser.add_argument("-s","--save_model_bool", action='store_true', required=False, help="Boolean flag to save the KNN model as an XML file for later use.")
    parser.add_argument("-n","--model_filename", type=str, required=False, default="knn_model", help="Filename of the saved KNN model.")
    parser.add_argument("-t","--dont_test_model_bool", action='store_false', required=False, help="Boolean flag to not test the created KNN model on split testing set (training only).")
    parser.add_argument("-d","--show_img", action='store_true', required=False, help="Boolean flag to show the tested images as they are classified.")


    args = parser.parse_args()

    random.seed(42)

    #Path to dataset directory from command line argument.
    dataset_path = args.data_path

    #Ratio of datasplit from command line argument.
    data_split_ratio = args.data_split_ratio

    #Image type from command line argument.
    image_type = args.image_type

    #Boolean if true will save the KNN model as a XML file from command line argument.
    save_model_bool = args.save_model_bool

    #Filename for the saved KNN model from command line argument.
    model_filename = args.model_filename

    #Boolean if true will test the model on the split testing set based on command line argument.
    test_model_bool = args.dont_test_model_bool

    #Number of neighbors to consider for KNN.
    knn_value = args.knn_value

    #Boolean if true will show the images as they are tested.
    show_img= args.show_img

    train_lines, test_lines = load_and_split_data(dataset_path, data_split_ratio)
    knn_model = train_model(dataset_path, train_lines, image_type, model_filename, save_model_bool)
    if(test_model_bool):
        test_model(dataset_path, test_lines, image_type, knn_model, knn_value, show_img)

if __name__ == "__main__":
    main()