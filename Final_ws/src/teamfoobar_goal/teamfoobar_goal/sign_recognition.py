#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
from cv_bridge import CvBridge
from teamfoobar_goal.srv import SignClassification

class SignRecognition(Node):
    def __init__(self):
        super().__init__('sign_recognition')
        # Publisher for the recognized sign.
        self.sign_pub = self.create_publisher(Int32, '/recognized_sign', 10)
        # Subscriber to a trigger signal.
        self.create_subscription(Bool, '/trigger_sign', self.trigger_callback, 10)
        # Subscriber for camera images.
        self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_callback, 10)
        
        self.bridge = CvBridge()
        self.latest_image = None

        # Create a service client for the sign classification service.
        self.cli = self.create_client(SignClassification, 'classify_sign')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for sign classification service...")
    
    def image_callback(self, msg: CompressedImage):
        # Store the most recent image message.
        self.latest_image = msg

    def trigger_callback(self, msg: Bool):
        if msg.data:
            if self.latest_image is None:
                self.get_logger().warn("No image available for sign recognition.")
                return

            # Create a service request with the latest image.
            req = SignClassification.Request()
            req.image = self.latest_image
            self.get_logger().info("Sending image to sign classification service...")
            future = self.cli.call_async(req)
            future.add_done_callback(self.handle_response)
    
    def handle_response(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return
        result = int(response.result)
        self.get_logger().info(f"Classified sign: {result}")
        # Ignore a sign of 0 to prevent accidental forward commands.
        if result == 0:
            self.get_logger().info("Received sign 0, ignoring.")
            return
        
        # Publish the recognized sign.
        sign_msg = Int32()
        sign_msg.data = result
        self.sign_pub.publish(sign_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SignRecognition()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
