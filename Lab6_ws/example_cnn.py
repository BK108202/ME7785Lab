#!/usr/bin/env python3
import cv2
import argparse
import csv
import numpy as np
import os
import glob
import math
import random

from sklearn.model_selection import train_test_split  # (not used now, but kept in case you need it)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Import the arrow extraction function from preprocessing.py
from preprocessing_test9 import preprocess_image

# Fixed image dimensions for CNN input (width, height)
FIXED_SIZE = (128, 128)

# ----------------------------
# Augmentation Functions
# ----------------------------

def zoom_image(image, zoom_factor):
    """
    Zooms into the image by a given zoom factor.
    The center of the image is cropped and then resized back to the original dimensions.

    Args:
        image (ndarray): The original image.
        zoom_factor (float): Factor by which to zoom in (e.g. 1.1 zooms a little in).

    Returns:
        ndarray: The zoomed image.
    """
    h, w = image.shape[:2]
    # Calculate new dimensions for the cropped area.
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    # Calculate starting points for the center crop.
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    # Crop and then resize back to original size.
    cropped = image[start_y:start_y+new_h, start_x:start_x+new_w]
    zoomed = cv2.resize(cropped, (w, h))
    return zoomed

def load_and_split_data(data_path, split_ratio, image_type):
    """
    Uses the provided labels.txt file to split the data into training and testing sets.
    Also performs data augmentation by zooming into each image multiple times.
    
    Args:
        data_path (str): Path to the dataset.
        split_ratio (float): Ratio of data used for training (the rest is used for testing).
        image_type (str): Extension of the image files (e.g., ".png", ".jpg").
        
    Returns:
        tuple: Two lists of tuples (image_name, true_label) for training and testing.
    """
    labels_file = os.path.join(data_path, 'labels.txt')
    with open(labels_file, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    # Data augmentation: create zoomed images per original image.
    augmented_lines = []
    # Define a list of zoom factors. (You can adjust these values as needed.)
    zoom_factors = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]
    for line in lines:
        original_name = line[0]
        label = line[1]
        img_path = os.path.join(data_path, original_name + image_type)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Image {img_path} not found or could not be read.")
            continue
        # Loop over the zoom factors and create zoomed versions.
        for i, z in enumerate(zoom_factors):
            zoomed_img = zoom_image(img, z)
            new_name = f"{original_name}_zoom{i+1}"
            new_image_path = os.path.join(data_path, new_name + image_type)
            cv2.imwrite(new_image_path, zoomed_img)
            augmented_lines.append([new_name, label])

    # Combine original and augmented image entries.
    all_lines = lines + augmented_lines

    # Randomly shuffle the complete list and then perform the split.
    random.shuffle(all_lines)
    split_index = math.floor(len(all_lines) * split_ratio)
    train_lines = all_lines[:split_index]
    test_lines = all_lines[split_index:]

    return train_lines, test_lines

# ----------------------------
# Preprocessing Function for CNN
# ----------------------------
def preprocess_image_for_cnn(img):
    """
    Applies arrow extraction using HSV filtering from preprocessing.py,
    then resizes, converts the image to RGB, and normalizes the image
    to [0, 1] for CNN input.
    """
    # Extract the arrow (for example, a green arrow on a black background)
    arrow_img = preprocess_image(img)
    
    # Resize the processed image to FIXED_SIZE
    img_resized = cv2.resize(arrow_img, FIXED_SIZE)
    
    # Convert from OpenCV's default BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_rgb.astype("float32") / 255.0
    
    return img_normalized

# ----------------------------
# CNN Model Related Functions
# ----------------------------
def build_cnn_model(input_shape, num_classes):
    """
    Builds a CNN model that processes images preprocessed via arrow extraction.
    """
    image_input = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=image_input, outputs=output)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def check_split_value_range(val):
    try:
        float_val = float(val)
        if float_val < 0 or float_val > 1:
            raise argparse.ArgumentTypeError("Train-test split ratio must be between 0 and 1!")
        return float_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Received '{val}' which is not a valid float!")

# ----------------------------
# Main Function
# ----------------------------
def main(args):
    # Use the augmentation and splitting function to generate training/test splits.
    # This function will also perform augmentation and save augmented images to the dataset folder.
    train_entries, test_entries = load_and_split_data(args.path, args.ratio, args.image_type)
    
    # Helper function to load and preprocess images based on a list of (filename, label) tuples.
    def load_images_from_entries(entries):
        imgs = []
        labs = []
        for entry in entries:
            filename, label = entry
            image_file = os.path.join(args.path, filename + args.image_type)
            img = cv2.imread(image_file)
            if img is None:
                print("Warning: Unable to load", image_file)
                continue
            processed = preprocess_image_for_cnn(img)
            imgs.append(processed)
            labs.append(int(label))
        return np.array(imgs), np.array(labs)
    
    # Load training and testing images using the helper function.
    X_train, y_train = load_images_from_entries(train_entries)
    X_test, y_test = load_images_from_entries(test_entries)
    
    if X_train.size == 0 or X_test.size == 0:
        print("No valid images loaded. Check the data path, image extension, and labels.txt.")
        return
    
    # One-hot encode the labels.
    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # Define input shape: (height, width, channels)
    input_shape = (FIXED_SIZE[1], FIXED_SIZE[0], 3)
    
    # Build the CNN model.
    model = build_cnn_model(input_shape, num_classes)
    model.summary()
    
    # Train the model.
    history = model.fit(
        X_train, y_train_cat,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test_cat),
        verbose=1
    )
    
    # Evaluate on the test set.
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print("Test Accuracy: {:.2f}%".format(test_acc * 100))
    
    # Save the model if requested.
    if args.save:
        model.save(args.model_filename + ".keras")
        print("Model saved to", args.model_filename + ".keras")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to folder containing images and labels.txt')
    parser.add_argument('-r', '--ratio', type=check_split_value_range, required=True,
                        help='Train-test split ratio (e.g., 0.8 for 80%% training)')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('-i', '--image_type', type=str, default=".png",
                        help='Image file extension (e.g., .png)')
    parser.add_argument('-s', '--save', action='store_true',
                        help='Flag to save the trained model')
    parser.add_argument('-n', '--model_filename', type=str, default="cnn_model",
                        help='Filename for the saved model')
    args = parser.parse_args()
    main(args)
