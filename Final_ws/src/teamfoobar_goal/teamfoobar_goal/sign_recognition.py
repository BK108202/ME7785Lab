#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
from skimage.feature import hog
import os
import importlib.resources

class SignRecognition(Node):
    def __init__(self):
        super().__init__('sign_recognition')
        # Publisher for the recognized sign.
        self.sign_pub = self.create_publisher(Int32, '/recognized_sign', 10)
        # Subscriber to the trigger signal.
        self.create_subscription(Bool, '/trigger_sign', self.trigger_callback, 10)
        # Subscriber for camera images.
        self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_callback, 10)
        
        self.bridge = CvBridge()
        # Load the pre-trained KNN model.
        # model_path = os.path.join(os.path.dirname(__file__), 'knn_model.xml')
        # self.get_logger().info(f"Loading KNN model from: {model_path}")
        # self.knn_model = cv2.ml.KNearest_load(model_path)
        # if self.knn_model.empty():
        #     self.get_logger().error("Failed to load the KNN model. Please check the knn_model.xml file.")
        #     return
        with importlib.resources.path('teamfoobar_goal', 'knn_model.xml') as model_path:
            model_path = str(model_path)  # Convert Path object to string if needed.
            self.get_logger().info(f"Loading KNN model from: {model_path}")
            self.knn_model = cv2.ml.KNearest_load(model_path)
            if self.knn_model.empty():
                self.get_logger().error("Failed to load the KNN model. Please check knn_model.xml.")

        self.latest_image = None

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image

            # Display the image.
            cv2.imshow("Robot Camera", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def trigger_callback(self, msg: Bool):
        self.get_logger().info(f"Received trigger message: {msg.data}")
        if msg.data:
            if self.latest_image is None:
                self.get_logger().warn("No image available for sign recognition.")
                return

            # Convert image from BGR to HSV and preprocess.
            hsv_image = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)
            features = self.preprocess_image(hsv_image)
            if features.size == 0:
                self.get_logger().warn("No features extracted from image. Skipping classification.")
                return

            # Debug: Check features properties.
            features = np.array(features)
            self.get_logger().info(f"Features shape: {features.shape}, dtype: {features.dtype}")
            
            # Reshape and convert to float32.
            sample = features.reshape(1, -1).astype(np.float32)
            self.get_logger().info(f"Sample shape: {sample.shape}, dtype: {sample.dtype}")

            # Define the number of neighbors to use.
            # ret, results, neighbours, dist = self.knn_model.findNearest(sample, 5)
            try:
                k = 5  # or your chosen value
                ret, results, neighbours, dist = self.knn_model.findNearest(sample, 5)
                self.get_logger().info(f"KNN result: {ret}, {results}")
            except cv2.error as e:
                self.get_logger().error(f"Error in findNearest: {e}")
            
            prediction = int(ret)
            self.get_logger().info(f"Predicted sign: {prediction}")
            
            # Ignore sign 0 to avoid accidental forward command.
            if prediction == 0:
                self.get_logger().info("Sign 0 recognized (go forward) - ignored.")
                return

            # Publish the recognized sign.
            sign_msg = Int32()
            sign_msg.data = prediction
            self.sign_pub.publish(sign_msg)
            self.get_logger().info("Published recognized sign message.")

    def preprocess_image(self, img, output_size=(50, 50)):
        # Define color boundaries.
        lower_red   = np.array([0, 100, 100])
        upper_red   = np.array([10, 255, 255])
        lower_green = np.array([20, 0, 0])
        upper_green = np.array([90, 120, 255])
        lower_blue  = np.array([110, 100, 100])
        upper_blue  = np.array([130, 255, 255])

        # Create masks for the colors.
        red_mask   = cv2.inRange(img, lower_red, upper_red)
        green_mask = cv2.inRange(img, lower_green, upper_green)
        blue_mask  = cv2.inRange(img, lower_blue, upper_blue)
        combined_mask = cv2.bitwise_or(red_mask, green_mask)
        combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

        # Find contours in the mask.
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        best_box = None
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
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
            # If no arrow is detected, resize the entire image.
            resized_img = cv2.resize(img, output_size)

        # Compute a color histogram.
        histSize = [8]
        hist_range = [0, 256]
        channels = [0, 1, 2]
        color_hist = []
        for ch in channels:
            hist = cv2.calcHist([resized_img], [ch], None, histSize, hist_range)
            hist = cv2.normalize(hist, hist).flatten()
            color_hist.append(hist)
        color_hist = np.concatenate(color_hist)

        # Convert image to grayscale, equalize, and normalize.
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        eq_img = cv2.equalizeHist(gray_img)
        normalized_img = eq_img.astype(np.float32) / 255.0
        hog_features = hog(
            normalized_img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True
        )

        combined_features = np.concatenate((color_hist, hog_features))
        return combined_features

def main(args=None):
    rclpy.init(args=args)
    node = SignRecognition()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
