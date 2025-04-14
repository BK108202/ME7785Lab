import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import qos_profile_sensor_data
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import time
import math
from cv_bridge import CvBridge
from skimage.feature import hog

class test1(Node):
    def __init__(self):
        super().__init__('test1')
        # Subscription for camera images using the provided image QoS profile.
        self._video_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self._image_callback,
            qos_profile_sensor_data
        )
        
        # Publisher for movement commands.
        # self._cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Load the trained KNN model from the XML file.
        # Note: Ensure that 'knn_model.xml' is accessible at runtime.
        self.knn_model = cv2.ml.KNearest_load('knn_model.xml')
        self.get_logger().info("KNN model loaded successfully.")
    
    # def turn(self, angular_speed, duration):
    #     """
    #     Publishes the Twist message with the specified angular speed for a given duration,
    #     then stops the robot.
    #     """
    #     twist = Twist()
    #     twist.linear.x = 0.0
    #     twist.angular.z = angular_speed
    #     self._cmd_pub.publish(twist)
        
    #     time.sleep(duration)
        
    #     twist.angular.z = 0.0
    #     self._cmd_pub.publish(twist)
    #     self.get_logger().info("Turn completed; robot stopped.")
    
    def _image_callback(self, msg):
        """
        Callback for processing incoming compressed images.
        It decodes the image, pre-processes it, uses the KNN model to classify the sign,
        and publishes corresponding movement commands.
        """
        self._imgBGR = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
        image = cv2.cvtColor(self._imgBGR, cv2.COLOR_BGR2HSV) # Using HSV version instead of RGB
        
        if image is None:
            self.get_logger().warn("Empty image received.")
            return
        
        combined_features = preprocess_image(image)
        sample = combined_features.reshape(1, -1).astype(np.float32)
        
        # Use the KNN model to find the nearest neighbor.
        ret, results, neighbours, dist = self.knn_model.findNearest(sample, k)
        prediction = int(ret)
        self.get_logger().info(f"Predicted sign: {prediction}")
        
        # twist = Twist()
        # if prediction == 0:  # Empty wall: move forward.
        #     twist.linear.x = 0.5
        #     twist.angular.z = 0.0
        #     self._cmd_pub.publish(twist)
        #     self.get_logger().info("Move forward")
        # elif prediction == 1:  # Left turn sign: perform a 90° left turn.
        #     angular_speed = 1.0
        #     duration = (math.pi / 2) / angular_speed
        #     self.turn(angular_speed, duration)
        #     self.get_logger().info("Left turn")
        # elif prediction == 2:  # Right turn sign: perform a 90° right turn.
        #     angular_speed = -1.0
        #     duration = (math.pi / 2) / abs(angular_speed)
        #     self.turn(angular_speed, duration)
        #     self.get_logger().info("Right turn")
        # elif prediction == 3:  # Do not enter sign: perform a 180° turn (U-turn).
        #     angular_speed = 1.0
        #     duration = math.pi / abs(angular_speed)
        #     self.turn(angular_speed, duration)
        #     self.get_logger().info("Do not enter")
        # elif prediction == 4:  # Stop: halt the robot.
        #     twist.linear.x = 0.0
        #     twist.angular.z = 0.0
        #     self._cmd_pub.publish(twist)
        #     self.get_logger().info("Stop")
        # elif prediction == 5:  # Goal reached: stop and log the event.
        #     twist.linear.x = 0.0
        #     twist.angular.z = 0.0
        #     self._cmd_pub.publish(twist)
        #     self.get_logger().info("Goal reached")
        # else:
        #     # Default case: no recognized sign; robot stops.
        #     twist.linear.x = 0.0
        #     twist.angular.z = 0.0
        #     self._cmd_pub.publish(twist)
        #     self.get_logger().info("Not recognized sign")

    def preprocess_image(img, output_size=(50, 50)):
        # Define boundaries for red, green, and blue
        lower_red   = np.array([0, 100, 100])
        upper_red   = np.array([10, 255, 255])
        lower_green = np.array([20, 0, 0])
        upper_green = np.array([90, 120, 255])
        lower_blue  = np.array([110, 100, 100])
        upper_blue  = np.array([130, 255, 255])

        # Create masks for each color
        red_mask   = cv2.inRange(img, lower_red, upper_red)
        green_mask = cv2.inRange(img, lower_green, upper_green)
        blue_mask  = cv2.inRange(img, lower_blue, upper_blue)

        # Combine the masks
        combined_mask = cv2.bitwise_or(red_mask, green_mask)
        combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        largest_area = 0
        best_box = None

        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            # Filter out contours that are too small or too irregular
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
            # If no arrow is detected, resize the whole image to guarantee a fixed output size
            resized_img = cv2.resize(img, output_size)

        # Using 8 bins per channel (adjustable) over the range [0, 256].
        histSize = [8]
        hist_range = [0, 256]
        channels = [0, 1, 2]
        color_hist = []
        for ch in channels:
            hist = cv2.calcHist([resized_img], [ch], None, histSize, hist_range)
            # Normalize the histogram
            hist = cv2.normalize(hist, hist).flatten()
            color_hist.append(hist)
        color_hist = np.concatenate(color_hist)

        # Convert the resized image to grayscale for HOG computation
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        eq_img = cv2.equalizeHist(gray_img)
        
        # Normalize pixel values to [0, 1]
        normalized_img = eq_img.astype(np.float32) / 255.0
        
        # Extract HOG features with adjusted parameters
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
    navigator = test1()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
