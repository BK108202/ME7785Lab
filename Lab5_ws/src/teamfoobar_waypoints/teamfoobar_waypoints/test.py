import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage

class Test(Node):
    def __init__(self):
        super().__init__('test')
        # Publisher for PoseStamped messages on '/goal_pose'
        self.publisher_ = self.create_publisher(PoseStamped, '/goal_pose', 10)
        # Subscribe to navigation feedback messages
        self.subscription = self.create_subscription(
            NavigateToPose_FeedbackMessage,
            '/navigate_to_pose/_action/feedback',
            self.feedback_callback,
            10
        )
        self.tolerance = 0.05  # meters
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.waypoints = [
            (0.2, 1.7, 0.0),
            (2.6, 1.7, 0.0),
            (4.5, 0.2, 0.0)
        ]
        self.current_goal_index = 0
        self.new_goal_sent = False  # Flag to indicate a new goal was just sent
        
        # Time delay for feedback processing after a new goal
        self.last_goal_time = self.get_clock().now()
        self.min_feedback_delay = 2.0  # seconds

    def timer_callback(self):
        if self.current_goal_index < len(self.waypoints):
            goal_x, goal_y, goal_z = self.waypoints[self.current_goal_index]
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            msg.pose.position.x = goal_x
            msg.pose.position.y = goal_y
            msg.pose.position.z = goal_z
            msg.pose.orientation.x = 0.0
            msg.pose.orientation.y = 0.0
            msg.pose.orientation.z = 0.0
            msg.pose.orientation.w = 1.0

            self.publisher_.publish(msg)
            self.get_logger().info(f"Published goal_pose: {msg.pose}")
            self.new_goal_sent = True  # Mark that a new goal was sent
            self.last_goal_time = self.get_clock().now()  # Update timestamp
        else:
            self.get_logger().info("All waypoints have been published.")
            # Optionally, you could cancel the timer here:
            # self.timer.cancel()

    def feedback_callback(self, msg):
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.last_goal_time).nanoseconds * 1e-9  # seconds
        
        # Wait a minimum period before processing feedback for the new goal
        if elapsed_time < self.min_feedback_delay:
            self.get_logger().info("Waiting for the robot to start moving. Ignoring feedback.")
            return

        distance_remaining = msg.feedback.distance_remaining
        
        # Ignore an initial 0.0 feedback if a new goal was just sent
        if self.new_goal_sent and distance_remaining == 0.0:
            self.get_logger().info("Ignoring initial 0.0 feedback.")
            return
        
        self.get_logger().info(f"Distance remaining: {distance_remaining:.2f} m")
        
        # Check if the current goal has been reached
        if self.current_goal_index < len(self.waypoints) and distance_remaining < self.tolerance:
            self.get_logger().info("Goal reached!")
            self.current_goal_index += 1
            self.new_goal_sent = False  # Reset flag for the next goal

def main(args=None):
    rclpy.init(args=args)
    node = Test()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
