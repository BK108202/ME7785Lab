#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
from skimage.feature import hog

class ComputerSignProcessor(Node):
    def __init__(self):
        super().__init__('computer_sign_processor')
        # Publisher for the recognized sign.
        self.sign_pub = self.create_publisher(Int32, '/recognized_sign', 10)
        # Subscriber for trigger signal.
        self.create_subscription(Bool, '/trigger_sign', self.trigger_callback, 10)
        # Subscriber for camera compressed images.
        self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_callback, 10)
        
        self.bridge = CvBridge()
        # Load the pre-trained KNN model from file.
        self.knn_model = cv2.ml.KNearest_load('knn_model.xml')
        self.latest_image = None
        # You might configure k (the number of neighbors); adjust as necessary.
        self.k = 5

    def image_callback(self, msg: CompressedImage):
        try:
            # Convert from ROS compressed image to OpenCV image.
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def trigger_callback(self, msg: Bool):
        if msg.data:
            if self.latest_image is None:
                self.get_logger().warn("No image available for sign recognition.")
                return

            # Convert image from BGR to HSV before processing.
            hsv_image = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)
            features = self.preprocess_image(hsv_image)
            sample = features.reshape(1, -1).astype(np.float32)
            
            # Classify with the pre-trained KNN model.
            ret, results, neighbours, dist = self.knn_model.findNearest(sample, self.k)
            prediction = int(ret)
            self.get_logger().info(f"Predicted sign: {prediction}")
            
            # Optionally ignore sign '0' (e.g., to avoid accidental forward movement)
            if prediction == 0:
                self.get_logger().info("Sign 0 recognized (go forward) - ignoring to avoid unintended actions.")
                return

            # Publish the recognized sign.
            sign_msg = Int32()
            sign_msg.data = prediction
            self.sign_pub.publish(sign_msg)

    def preprocess_image(self, img, output_size=(50, 50)):
        """
        Process the input HSV image: generate color and texture features.
        """
        # Define boundaries for red, green, and blue in HSV.
        lower_red   = np.array([0, 100, 100])
        upper_red   = np.array([10, 255, 255])
        lower_green = np.array([20, 0, 0])
        upper_green = np.array([90, 120, 255])
        lower_blue  = np.array([110, 100, 100])
        upper_blue  = np.array([130, 255, 255])

        # Create masks for each color.
        red_mask   = cv2.inRange(img, lower_red, upper_red)
        green_mask = cv2.inRange(img, lower_green, upper_green)
        blue_mask  = cv2.inRange(img, lower_blue, upper_blue)

        # Combine the masks.
        combined_mask = cv2.bitwise_or(red_mask, green_mask)
        combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

        # Find contours in the mask.
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        largest_area = 0
        best_box = None

        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            # Filter based on area and perimeter to find the sign.
            if area < 25000 and perimeter > 370:
                x, y, w, h = cv2.boundingRect(c)
                if area > largest_area:
                    largest_area = area
                    best_box = (x, y, w, h)

        if best_box is not None:
            x, y, w, h = best_box
            cropped_arrow = img[y:y+h, x:x+w]
            resized_img = cv2.resize(cropped_arrow, output_size)
        else:
            # Use entire image if no prominent sign candidate is found.
            resized_img = cv2.resize(img, output_size)

        # Extract color histogram features.
        histSize = [8]
        hist_range = [0, 256]
        channels = [0, 1, 2]
        color_hist = []
        for ch in channels:
            hist = cv2.calcHist([resized_img], [ch], None, histSize, hist_range)
            hist = cv2.normalize(hist, hist).flatten()
            color_hist.append(hist)
        color_hist = np.concatenate(color_hist)

        # Prepare grayscale image and equalize histogram.
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        eq_img = cv2.equalizeHist(gray_img)
        normalized_img = eq_img.astype(np.float32) / 255.0
        
        # Compute HOG features.
        hog_features = hog(
            normalized_img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True
        )
        
        # Combine color histogram and HOG features.
        combined_features = np.concatenate((color_hist, hog_features))
        return combined_features

def main(args=None):
    rclpy.init(args=args)
    node = ComputerSignProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
