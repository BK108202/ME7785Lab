import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage

class test(Node):
    def __init__(self):
        super().__init__('test')
        # Create a publisher for the PoseStamped message on the '/goal_pose' topic
        self.publisher_ = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.subscription = self.create_subscription(
            NavigateToPose_FeedbackMessage,
            '/navigate_to_pose/_action/feedback',
            self.feedback_callback,
            10
        )
        self.tolerance = 0.05  # meters
        # Publish at a rate of 1 Hz
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.waypoints = [
            (2.6, 0.38, 0.0),
            (2.1, 0.91, 0.0),
            (2.9, 1.4, 0.0)
        ]
        self.current_goal_index = 0

    def timer_callback(self):

        if self.current_goal_index >= len(self.waypoints):
            self.get_logger().info("All waypoints have been published.")
            return

        goal_x, goal_y, goal_z= self.waypoints[self.current_goal_index]
        # Create a new PoseStamped message
        msg = PoseStamped()
        # Use the current time for the header timestamp
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        # Set the position and orientation as specified
        msg.pose.position.x = goal_x
        msg.pose.position.y = goal_y
        msg.pose.position.z = goal_z
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        # Publish the message
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published goal_pose: {msg.pose}")
    
    def feedback_callback(self, msg):
        distance_remaining = msg.feedback.distance_remaining
        self.get_logger().info(f"Distance remaining: {distance_remaining:.2f} m")

        if self.current_goal_index >= len(self.waypoints):
            return
        
        if self.current_goal_index == 0:
            self.timer_callback()

        if distance_remaining < self.tolerance:
            self.current_goal_index = self.current_goal_index + 1
            self.timer_callback()
            self.get_logger().info("Goal reached!")

def main(args=None):
    rclpy.init(args=args)
    node = test()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # Cleanup and shutdown
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
