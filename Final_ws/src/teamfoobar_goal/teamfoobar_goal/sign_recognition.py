#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
from preprocessing1 import preprocess_image  # Your preprocessing function

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
        self.knn_model = cv2.ml.KNearest_load('knn_model.xml')
        self.latest_image = None

    def image_callback(self, msg: CompressedImage):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def trigger_callback(self, msg: Bool):
        if msg.data:
            if self.latest_image is None:
                self.get_logger().warn("No image available for sign recognition.")
                return

            # Convert image from BGR to HSV and preprocess.
            hsv_image = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)
            features = preprocess_image(hsv_image)
            sample = features.reshape(1, -1).astype(np.float32)
            ret, results, neighbours, dist = self.knn_model.findNearest(sample, k)
            prediction = int(ret)
            self.get_logger().info(f"Predicted sign: {prediction}")
            
            # Ignore the sign '0' (go forward) to prevent accidental commands.
            if prediction == 0:
                self.get_logger().info("Sign 0 recognized (go forward) - ignored to avoid accidental forward movement.")
                return

            # Publish the recognized sign (e.g., 1=right, 2=left, 3=stop/turn-around, 4=goal reached)
            sign_msg = Int32()
            sign_msg.data = prediction
            self.sign_pub.publish(sign_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SignRecognition()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
