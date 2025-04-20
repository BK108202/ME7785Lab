#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
from skimage.feature import hog
from scipy.stats import skew
import importlib.resources
from collections import Counter

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
        with importlib.resources.path('teamfoobar_goal', 'knn_model.xml') as model_path:
            model_path = str(model_path)
            self.get_logger().info(f"Loading KNN model from: {model_path}")
            self.knn_model = cv2.ml.KNearest_load(model_path)
            if self.knn_model.empty():
                self.get_logger().error("Failed to load the KNN model. Please check knn_model.xml.")
            else:
                self.get_logger().info("Model Loaded")

        # Buffer state for collecting multiple frames
        self.latest_image   = None
        self.collecting     = False
        self.image_buffer   = []

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image

            # Display the image.
            cv2.imshow("Robot Camera", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        # Buffer frames if we’re in a collection cycle
        if self.collecting:
            self.image_buffer.append(cv_image.copy())
            if len(self.image_buffer) >= 10:
                self.collecting = False
                self.process_buffer()

    def trigger_callback(self, msg: Bool):
        self.get_logger().info(f"Received trigger message: {msg.data}")
        if not msg.data:
            return

        if self.latest_image is None:
            self.get_logger().warn("No image available for sign recognition.")
            return

        # Start a new collection cycle
        self.image_buffer.clear()
        self.collecting = True
        self.get_logger().info("Collecting 10 frames for classification...")

    def process_buffer(self):
        """Extract features from each buffered image, run KNN once per image,
           then majority-vote and publish the result."""
        votes = []
        for img in self.image_buffer:
            features = self.preprocess_image(img)
            if features.size == 0:
                continue
            sample = features.reshape(1, -1).astype(np.float32)
            try:
                ret, _, _, _ = self.knn_model.findNearest(sample, 5)
                votes.append(int(ret))
            except cv2.error as e:
                self.get_logger().error(f"KNN error: {e}")

        if not votes:
            self.get_logger().warn("No valid predictions from buffered frames.")
            return

        # Majority vote
        vote_counts = Counter(votes)
        predicted = vote_counts.most_common(1)[0][0]
        self.get_logger().info(f"Buffered votes: {votes} → chosen: {predicted}")

        # Ignore sign 0 as before
        if predicted == 0:
            self.get_logger().info("Sign 0 recognized (go forward) - ignored.")
            return

        # Publish the majority-vote result
        sign_msg = Int32()
        sign_msg.data = predicted
        self.sign_pub.publish(sign_msg)
        self.get_logger().info("Published recognized sign message.")

    def preprocess_image(self, img, output_size=(50, 50)):
        # Convert to LAB
        LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(LAB)

        # get the maximum between the A and B pixel by pixel
        ABmax = np.maximum(A, B)

        # threshold
        thresh = cv2.threshold(ABmax, 180, 255, cv2.THRESH_BINARY)[1]

        # morphology close and open
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # get contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        no_contour = False
        min_area = 0.1

        # initial rough check for any contour above tiny area
        image_with_contour = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if not image_with_contour:
            no_contour = True
        else:
            no_contour = True

        # If no arrow is found, try green/blue threshold fallback
        if no_contour:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            alpha = 1.0
            beta = 25
            bright_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            blurred_image = cv2.GaussianBlur(bright_img, (15,15), 0)
            hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

            threshold = 300
            _, white_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            white_mask_inv = cv2.bitwise_not(white_mask)
            result = cv2.bitwise_and(img, img, mask=white_mask_inv)

            lower_green = np.array([20, 0, 0])
            upper_green = np.array([235, 255, 255])
            lower_blue  = np.array([90, 100, 100])
            upper_blue  = np.array([130, 255, 255])

            green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
            blue_mask  = cv2.inRange(hsv_image, lower_blue, upper_blue)

            combined_mask = cv2.bitwise_or(green_mask, blue_mask)
            combined_mask = cv2.bitwise_or(combined_mask, white_mask)
            masked_image = cv2.bitwise_and(img, img, mask=combined_mask)

            ret, thresh1 = cv2.threshold(combined_mask, 180, 255, cv2.THRESH_BINARY)
            thresh2 = cv2.adaptiveThreshold(combined_mask, 255,
                                            cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            kernel1 = np.ones((7,7))
            morph1 = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel1, iterations=1)
            masked_image = cv2.bitwise_and(img, img, mask=combined_mask)

            contours, _ = cv2.findContours(morph1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Pick contour whose centroid is closest to image center
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
            h, w = img.shape[:2]
            crop_w, crop_h = 200, 200
            start_x = max((w - crop_w) // 2, 0)
            start_y = max((h - crop_h) // 2, 0)
            if h < crop_h or w < crop_w:
                center_crop = cv2.resize(img, (crop_w, crop_h))
            else:
                center_crop = img[start_y:start_y+crop_h, start_x:start_x+crop_w]
            resized_img = cv2.resize(center_crop, output_size)

        # Color histogram
        histSize = [8]
        hist_range = [0, 256]
        channels = [0, 1, 2]
        color_hist = []
        for ch in channels:
            hist = cv2.calcHist([resized_img], [ch], None, histSize, hist_range)
            hist = cv2.normalize(hist, hist).flatten()
            color_hist.append(hist)
        color_hist = np.concatenate(color_hist)

        # HOG features
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        eq_img = cv2.equalizeHist(gray_img)
        normalized_img = eq_img.astype(np.float32) / 255.0

        hog_features = hog(
            gray_img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True
        )

        hog_skewness = skew(hog_features)

        combined_features = np.concatenate((color_hist, hog_features, [hog_skewness]))
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
