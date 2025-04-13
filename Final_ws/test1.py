import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import qos_profile_sensor_data
import cv2
import numpy as np
from geometry_msgs.msg import Twist
import time
import math
from preprocessing1 import preprocess_image
from cv_bridge import CvBridge

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
        retval, results, neigh_resp, dists = self.knn_model.findNearest(sample, k=5)
        prediction = int(results[0][0])
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

def main(args=None):
    rclpy.init(args=args)
    navigator = navigation1()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
