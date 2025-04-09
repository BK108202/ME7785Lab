#!/usr/bin/env python3
import cv2
import argparse
import csv
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Import the arrow extraction function from preprocessing.py
from preprocessing_test7 import preprocess_image

# Fixed image dimensions for CNN input (width, height)
FIXED_SIZE = (128, 128)

def preprocess_image_for_cnn(img):
    """
    Applies arrow extraction using HSV filtering from preprocessing.py,
    then resizes, converts the image to RGB, and normalizes the image
    to [0, 1] for CNN input.
    """
    # Extract the arrow (green arrow on a black background)
    arrow_img = preprocess_image(img)
    
    # Resize the processed image to FIXED_SIZE
    img_resized = cv2.resize(arrow_img, FIXED_SIZE)
    
    # Convert from OpenCV's default BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_rgb.astype("float32") / 255.0
    
    return img_normalized

def load_labels(data_path):
    """
    Reads the labels.txt file from data_path.
    Each line is expected to have: filename_without_extension,label
    Returns a dictionary mapping each filename to its integer label.
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

def load_images_and_labels(data_path, image_ext, label_dict):
    """
    Loads images from data_path that match image_ext and have a corresponding label.
    Applies preprocessing (using preprocess_image_for_cnn) for CNN input.
    Returns two numpy arrays: images and labels.
    """
    image_files = sorted(glob.glob(os.path.join(data_path, "*" + image_ext)))
    images = []
    labels = []
    for file in image_files:
        # Remove the extension to get the base filename
        base_name = os.path.basename(file)[:-len(image_ext)]
        if base_name in label_dict:
            img = cv2.imread(file)
            if img is not None:
                proc_img = preprocess_image_for_cnn(img)
                images.append(proc_img)
                labels.append(label_dict[base_name])
            else:
                print("Warning: Unable to load", file)
    return np.array(images), np.array(labels)

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

def main(args):
    # Load labels from labels.txt
    label_dict = load_labels(args.path)
    
    # Load images (preprocessed) and labels
    images, labels = load_images_and_labels(args.path, args.image_type, label_dict)
    
    if images.size == 0:
        print("No valid images found. Check your image file extension and labels.txt.")
        return
    
    # Split into training and testing sets (with stratification to balance classes)
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, train_size=args.ratio, random_state=42, stratify=labels)
    
    # One-hot encode the labels
    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # Define input shape: (height, width, channels)
    input_shape = (FIXED_SIZE[1], FIXED_SIZE[0], 3)
    
    # Build and summarize the CNN model
    model = build_cnn_model(input_shape, num_classes)
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train, y_train_cat,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test_cat),
        verbose=1
    )
    
    # Evaluate on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print("Test Accuracy: {:.2f}%".format(test_acc * 100))
    
    # Save the model if requested
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
