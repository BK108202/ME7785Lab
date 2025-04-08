#!/usr/bin/env python3
import cv2
import argparse
import csv
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout,
                                     Flatten, Dense, BatchNormalization, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from skimage.feature import hog

# Fixed image dimensions for CNN branch (width, height)
FIXED_SIZE = (128, 128)

def check_split_value_range(val):
    try:
        float_val = float(val)
        if float_val < 0 or float_val > 1:
            raise argparse.ArgumentTypeError("Train-test split ratio must be between 0 and 1!")
        return float_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Received '{val}' which is not a valid float!")

def load_labels(data_path):
    """
    Reads the labels.txt file located in data_path.
    Expected format per line: filename_without_extension,label
    Returns a dictionary mapping filenames to integer labels.
    """
    label_file = os.path.join(data_path, "labels.txt")
    label_dict = {}
    with open(label_file, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line:
                continue
            filename = line[0].strip()
            label = line[1].strip()
            label_dict[filename] = int(label)
    return label_dict

def preprocess_image(img):
    """
    Processes an image in two ways:
      1. For the CNN branch: Resizes, converts from BGR to RGB, and normalizes pixel values.
      2. For the handcrafted branch: Extracts a color histogram (32 bins per channel) and HOG features.
    Returns a tuple: (processed_image, handcrafted_features)
    """
    # --- Preprocessing for CNN branch ---
    # Resize and convert color to RGB
    img_resized = cv2.resize(img, FIXED_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    img_normalized = img_rgb.astype('float32') / 255.0
    
    # --- Handcrafted Features ---
    # Color Histogram: 32 bins per channel
    bins = 32
    hist_r = cv2.calcHist([img_rgb], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([img_rgb], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([img_rgb], [2], None, [bins], [0, 256])
    # Normalize each histogram and flatten
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    color_hist = np.hstack([hist_r, hist_g, hist_b])  # 32*3 = 96 features

    # HOG Features: Convert to grayscale and compute HOG features.
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    # Combine color histogram and HOG features.
    handcrafted_features = np.hstack([color_hist, hog_features])
    
    return img_normalized, handcrafted_features

def load_images_and_features(data_path, image_ext, label_dict):
    """
    Loads images from data_path that have a corresponding label.
    Applies preprocessing to generate:
      - A normalized image for the CNN branch.
      - Handcrafted features (color histogram + HOG).
    Returns three numpy arrays: images, handcrafted_features, labels.
    """
    image_files = sorted(glob.glob(os.path.join(data_path, "*" + image_ext)))
    images = []
    handcrafted_feats = []
    labels = []
    for file in image_files:
        base_name = os.path.basename(file)[:-len(image_ext)]
        if base_name in label_dict:
            img = cv2.imread(file)
            if img is not None:
                proc_img, hc_feats = preprocess_image_with_features(img)
                images.append(proc_img)
                handcrafted_feats.append(hc_feats)
                labels.append(label_dict[base_name])
            else:
                print("Warning: Unable to load", file)
    return np.array(images), np.array(handcrafted_feats), np.array(labels)

def build_hybrid_model(input_shape_image, input_shape_handcrafted, num_classes):
    """
    Builds a hybrid model with two input branches:
      - A CNN branch for raw images.
      - A Dense branch for handcrafted features.
    The branches are concatenated before the final classification layers.
    """
    # CNN branch for image input
    image_input = Input(shape=input_shape_image)
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
    
    # Dense branch for handcrafted features
    handcrafted_input = Input(shape=input_shape_handcrafted)
    y = Dense(256, activation='relu')(handcrafted_input)
    
    # Concatenate both branches
    combined = Concatenate()([x, y])
    combined = Dense(128, activation='relu')(combined)
    output = Dense(num_classes, activation='softmax')(combined)
    
    model = Model(inputs=[image_input, handcrafted_input], outputs=output)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main(args):
    # Load labels from labels.txt
    label_dict = load_labels(args.path)
    
    # Load images and handcrafted features along with labels
    images, handcrafted_feats, labels = load_images_and_features(args.path, args.image_type, label_dict)
    if images.size == 0:
        print("No valid images found. Check your image file extension and labels.txt.")
        return

    # Split into training and testing sets (using stratification for balanced classes)
    X_train_img, X_test_img, X_train_hc, X_test_hc, y_train, y_test = train_test_split(
        images, handcrafted_feats, labels, train_size=args.ratio, random_state=42, stratify=labels)

    # Convert labels to one-hot encoding
    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # Define input shapes
    image_input_shape = (FIXED_SIZE[1], FIXED_SIZE[0], 3)  # (height, width, channels)
    handcrafted_input_shape = (X_train_hc.shape[1],)
    
    # Build and summarize the hybrid model
    model = build_hybrid_model(image_input_shape, handcrafted_input_shape, num_classes)
    model.summary()
    
    # Train the model
    history = model.fit(
        [X_train_img, X_train_hc], y_train_cat,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=([X_test_img, X_test_hc], y_test_cat),
        verbose=1
    )
    
    # Evaluate on the test set
    test_loss, test_acc = model.evaluate([X_test_img, X_test_hc], y_test_cat, verbose=0)
    print("Test Accuracy: {:.2f}%".format(test_acc * 100))
    
    # Save the model if requested
    if args.save:
        # Saving using the native Keras format:
        model.save(args.model_filename + '.keras')
        print("Model saved to", args.model_filename + '.keras')

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
    parser.add_argument('-n', '--model_filename', type=str, default="hybrid_model",
                        help='Filename for the saved model')
    args = parser.parse_args()
    main(args)
