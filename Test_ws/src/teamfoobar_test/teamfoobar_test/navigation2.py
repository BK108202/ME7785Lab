import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import time
import math
from preprocessing1 import preprocess_image
from cv_bridge import CvBridge

class navigation2(Node):
    def __init__(self):
        super().__init__('navigation2')

        # State definitions:
        # "MOVING" - Robot is moving forward.
        # "WAITING_FOR_SIGN" - Robot has stopped and awaits sign reading.
        # "STOPPED" - Robot is stopped (either by a stop sign or goal sign).
        self.state = "MOVING"

        # Laser scan subscriber.
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        # Camera subscriber for compressed images.
        self.video_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10
        )
        
        # Publisher for movement commands.
        self._cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Timer that continuously sends forward commands if state is MOVING.
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Load the trained KNN model from XML.
        self.knn_model = cv2.ml.KNearest_load('knn_model.xml')
        self.get_logger().info("KNN model loaded successfully.")

        self.distance_threshold = 0.5

    def timer_callback(self):
        """Continuously publish forward motion if the robot is in MOVING state."""
        if self.state == "MOVING":
            twist = Twist()
            twist.linear.x = 0.5  # Adjust forward speed as necessary.
            twist.angular.z = 0.0
            self._cmd_pub.publish(twist)

    def scan_callback(self, msg: LaserScan):
        """
        When moving forward, if any reading from the laser scan is below
        the threshold, stop the robot and switch to WAITING_FOR_SIGN state.
        """
        if self.state == "MOVING":
            if msg.ranges:
                min_distance = min(msg.ranges)
                if min_distance < self.distance_threshold:
                    self.get_logger().info(f"Wall detected at {min_distance:.2f} m. Stopping for sign reading.")
                    # Stop the robot.
                    twist = Twist()
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self._cmd_pub.publish(twist)
                    # Change state to wait for the sign.
                    self.state = "WAITING_FOR_SIGN"

    def perform_turn(self, angular_speed, duration):
        """
        Commands the robot to turn at the given angular_speed for the given
        duration (using time.sleep to block during the turn), and then stops.
        """
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = angular_speed
        self._cmd_pub.publish(twist)
        self.get_logger().info(f"Performing turn with angular speed {angular_speed} rad/s for {duration:.2f} seconds.")
        time.sleep(duration)
        twist.angular.z = 0.0
        self._cmd_pub.publish(twist)
        self.get_logger().info("Turn completed; robot stopped.")

    def image_callback(self, msg: CompressedImage):
        """
        If the robot is waiting for a sign (state == WAITING_FOR_SIGN), this
        callback processes the image to classify the sign using the loaded KNN model.
        Then, it executes the corresponding action.
        """
        # Only process the image if we're in the proper state.
        if self.state != "WAITING_FOR_SIGN":
            return
        
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
        
        # Perform actions based on the prediction:
        if prediction == 1:  # Left turn sign.
            angular_speed = 1.0
            duration = (math.pi / 2) / angular_speed
            self.perform_turn(angular_speed, duration)
            self.state = "MOVING"
        elif prediction == 2:  # Right turn sign.
            angular_speed = -1.0
            duration = (math.pi / 2) / abs(angular_speed)
            self.perform_turn(angular_speed, duration)
            self.state = "MOVING"
        elif prediction == 3:  # Do not enter sign: perform a U-turn.
            angular_speed = 1.0
            duration = math.pi / abs(angular_speed)
            self.perform_turn(angular_speed, duration)
            self.state = "MOVING"
        elif prediction == 4:  # Stop sign.
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self._cmd_pub.publish(twist)
            self.get_logger().info("Stop sign detected. Stopping robot.")
            self.state = "STOPPED"
        elif prediction == 5:  # Goal reached.
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self._cmd_pub.publish(twist)
            self.get_logger().info("Goal sign detected. Stopping robot.")
            self.state = "STOPPED"
        else:
            self.get_logger().info("Unknown sign reading. Resuming forward motion.")
            self.state = "MOVING"

def main(args=None):
    rclpy.init(args=args)
    navigator = navigation2()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
