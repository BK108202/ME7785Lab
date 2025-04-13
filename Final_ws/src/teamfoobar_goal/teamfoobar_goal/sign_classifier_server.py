#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from teamfoobar_goal.srv import SignClassification
from cv_bridge import CvBridge
import cv2
import numpy as np
from preprocessing1 import preprocess_image  # Your preprocessing function

class SignClassifierServer(Node):
    def __init__(self):
        super().__init__('sign_classifier_server')
        # Create the service that will be called by sign_recognition.
        self.srv = self.create_service(SignClassification, 'classify_sign', self.classify_callback)
        self.bridge = CvBridge()
        # Load the pre-trained KNN model.
        self.knn_model = cv2.ml.KNearest_load('knn_model.xml')
        self.get_logger().info("Sign classifier service ready.")

    def classify_callback(self, request, response):
        # Convert the incoming compressed image to a CV image.
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(request.image, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            response.result = -1  # Error code.
            return response
        
        # Convert the image from BGR to HSV.
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # Preprocess the image.
        features = preprocess_image(hsv_image)
        sample = features.reshape(1, -1).astype(np.float32)
        retval, results, neigh_resp, dists = self.knn_model.findNearest(sample, k=5)
        prediction = int(results[0][0])
        self.get_logger().info(f"Prediction: {prediction}")
        response.result = prediction
        return response

def main(args=None):
    rclpy.init(args=args)
    node = SignClassifierServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
