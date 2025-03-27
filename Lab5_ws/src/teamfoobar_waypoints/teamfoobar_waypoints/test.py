import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose  # Import the action

class WaypointPublisher(Node):
    def __init__(self):
        super().__init__('waypoint_publisher')
        # Create a publisher for the PoseStamped message on the '/goal_pose' topic
        self.publisher_ = self.create_publisher(PoseStamped, '/goal_pose', 10)
        # Subscribe to the NavigateToPose feedback
        self.subscription = self.create_subscription(
            NavigateToPose.FeedbackMessage,
            '/navigate_to_pose/_action/feedback',
            self.feedback_callback,
            10
        )
        self.tolerance = 0.05  # meters
        
        # Create a timer to publish the current goal at 1 Hz
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        # Define the waypoints as (x, y, z) tuples
        self.waypoints = [
            (1.4, 2.1, 0.0),
            # (3.2, 1.2, 0.0),
            # (4.9, 2.0, 0.0)
        ]
        self.current_goal_index = 0

    def timer_callback(self):
        if self.current_goal_index < len(self.waypoints):
            goal_x, goal_y, goal_z = self.waypoints[self.current_goal_index]
            # Create and publish the PoseStamped message
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            msg.pose.position.x = goal_x
            msg.pose.position.y = goal_y
            msg.pose.position.z = goal_z
            # Fixed orientation (no rotation)
            msg.pose.orientation.x = 0.0
            msg.pose.orientation.y = 0.0
            msg.pose.orientation.z = 0.0
            msg.pose.orientation.w = 1.0
            self.publisher_.publish(msg)
            self.get_logger().info(f"Published goal_pose: {msg.pose}")
        else:
            self.get_logger().info("All waypoints have been published.")
            # Optionally, you could stop the timer if no further publication is needed:
            # self.timer.cancel()

    def feedback_callback(self, msg):
        distance_remaining = msg.feedback.distance_remaining
        self.get_logger().info(f"Distance remaining: {distance_remaining:.2f} m")

        if self.current_goal_index < len(self.waypoints) and distance_remaining < self.tolerance:
            self.get_logger().info("Goal reached!")
            self.current_goal_index += 1
            # The new waypoint will be published on the next timer callback.

def main(args=None):
    rclpy.init(args=args)
    node = WaypointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # Cleanup and shutdown
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
