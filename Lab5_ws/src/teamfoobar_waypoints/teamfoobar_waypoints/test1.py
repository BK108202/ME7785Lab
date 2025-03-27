import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

class NavigationActionClient(Node):
    def __init__(self):
        super().__init__('navigation_action_client')
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.waypoints = [
            (0.2, 1.7, 0.0),
            (2.6, 1.7, 0.0),
            (4.5, 0.2, 0.0)
        ]
        self.current_goal_index = 0

    def send_goal(self, pose: PoseStamped):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        self.get_logger().info(f"Sending goal: {pose.pose}")
        # Wait for the action server to be available
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return
        self.get_logger().info("Goal accepted")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # Print feedback information (distance_remaining is specific to NavigateToPose)
        self.get_logger().info(f"Distance remaining: {feedback.distance_remaining:.2f} m")

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result received: {result}")
        self.current_goal_index += 1
        if self.current_goal_index < len(self.waypoints):
            self.send_next_goal()
        else:
            self.get_logger().info("All waypoints reached.")
            rclpy.shutdown()

    def send_next_goal(self):
        waypoint = self.waypoints[self.current_goal_index]
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = waypoint[0]
        pose.pose.position.y = waypoint[1]
        pose.pose.position.z = waypoint[2]
        # Use a fixed orientation (no rotation)
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.send_goal(pose)

def main(args=None):
    rclpy.init(args=args)
    action_client = NavigationActionClient()
    # Start by sending the first goal
    action_client.send_next_goal()
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
