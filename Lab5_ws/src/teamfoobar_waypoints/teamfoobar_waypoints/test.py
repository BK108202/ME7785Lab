import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage

class Test(Node):
    def __init__(self):
        super().__init__('test')
        # Publisher for the PoseStamped goal message
        self.publisher_ = self.create_publisher(PoseStamped, '/goal_pose', 10)
        # Subscription to the NavigateToPose feedback messages
        self.subscription = self.create_subscription(
            NavigateToPose_FeedbackMessage,
            '/navigate_to_pose/_action/feedback',
            self.feedback_callback,
            10
        )
        self.tolerance = 0.05  # meters tolerance to consider the goal reached
        # Timer to trigger the publication of goals
        self.timer = self.create_timer(1.0, self.timer_callback)
        # List of waypoints (x, y, z)
        self.waypoints = [
            (0.2, 1.7, 0.0),
            (2.6, 1.7, 0.0),
            (4.5, 0.2, 0.0)
        ]
        self.current_goal_index = 0
        self.new_goal_sent = False  # Flag to ensure a goal is published only once until reached
        
        # Timestamp to check how long since the last goal was published
        self.last_goal_time = self.get_clock().now()
        self.min_feedback_delay = 2.0  # seconds to wait before processing feedback

    def timer_callback(self):
        # Publish a new goal only if none is active
        if self.current_goal_index < len(self.waypoints) and not self.new_goal_sent:
            goal_x, goal_y, goal_z = self.waypoints[self.current_goal_index]
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            msg.pose.position.x = goal_x
            msg.pose.position.y = goal_y
            msg.pose.position.z = goal_z
            # Set a fixed orientation (no rotation)
            msg.pose.orientation.x = 0.0
            msg.pose.orientation.y = 0.0
            msg.pose.orientation.z = 0.0
            msg.pose.orientation.w = 1.0

            self.publisher_.publish(msg)
            self.get_logger().info(f"Published goal_pose: {msg.pose}")
            # Mark that a new goal was sent and update the timestamp
            self.new_goal_sent = True
            self.last_goal_time = self.get_clock().now()
        elif self.current_goal_index >= len(self.waypoints):
            self.get_logger().info("All waypoints have been published.")
            # Optionally, cancel the timer if no more goals are needed:
            # self.timer.cancel()

    def feedback_callback(self, msg):
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.last_goal_time).nanoseconds * 1e-9  # Convert to seconds

        # Only process feedback after the specified delay to let the robot start moving
        if elapsed_time < self.min_feedback_delay:
            self.get_logger().info("Waiting for the robot to start moving. Ignoring feedback.")
            return

        distance_remaining = msg.feedback.distance_remaining

        # If the distance remaining is below the tolerance, consider the goal reached
        if self.current_goal_index < len(self.waypoints) and distance_remaining < self.tolerance:
            self.get_logger().info("Goal reached!")
            self.current_goal_index += 1
            self.new_goal_sent = False  # Allow the next goal to be published
        
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
