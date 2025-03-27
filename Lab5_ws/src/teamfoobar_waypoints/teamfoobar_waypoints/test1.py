#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage
# Note: You can also import NavigateToPose_FeedbackMessage if needed:
# from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage

class test1(Node):
    def __init__(self):
        super().__init__('combined_nav_client')
        # Create an action client for the NavigateToPose action
        self._action_client = ActionClient(self, NavigateToPose_FeedbackMessage, '/navigate_to_pose')
        
        # List of waypoints (x, y, z)
        self.waypoints = [
            (0.2, 1.7, 0.0),
            (2.6, 1.7, 0.0),
            (4.5, 0.2, 0.0)
        ]
        self.current_goal_index = 0
        self.tolerance = 0.05  # meters (for reference if using feedback)
        
        # (Optional) Subscribe to the feedback topic to log feedback (similar to RViz)
        self.create_subscription(
            NavigateToPose_FeedbackMessage,
            '/navigate_to_pose/_action/feedback',
            self.feedback_callback,
            10
        )
        
        # Timer to initiate sending the first goal
        self.send_goal_timer = self.create_timer(1.0, self.send_next_goal)

    def send_next_goal(self):
        if self.current_goal_index >= len(self.waypoints):
            self.get_logger().info("All waypoints have been processed.")
            self.send_goal_timer.cancel()
            return

        # Create a goal PoseStamped from the current waypoint
        x, y, z = self.waypoints[self.current_goal_index]
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = z
        # Set orientation (here simply facing forward)
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        self.get_logger().info(f"Sending goal: ({x}, {y}, {z})")
        self.send_goal(goal_pose)
        # Cancel the timer to avoid sending multiple goals simultaneously
        self.send_goal_timer.cancel()

    def send_goal(self, pose: PoseStamped):
        # Populate the NavigateToPose goal message
        goal_msg = NavigateToPose_FeedbackMessage.Goal()
        goal_msg.pose = pose
        
        # Wait for the action server to be available before sending the goal
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by server!")
            # Optionally, move to the next waypoint even if rejected
            self.current_goal_index += 1
            self.send_goal_timer = self.create_timer(1.0, self.send_next_goal)
            return

        self.get_logger().info("Goal accepted, waiting for the final result...")
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        status = future.result().status
        self.get_logger().info(f"Final result received, status: {status}")
        # At this point, the final action result confirms that the navigation is complete.
        self.get_logger().info("Navigation succeeded for the current goal!")
        
        # Move on to the next waypoint
        self.current_goal_index += 1
        # Restart the timer to send the next goal after a brief delay
        self.send_goal_timer = self.create_timer(1.0, self.send_next_goal)

    def feedback_callback(self, feedback_msg):
        """
        Feedback callback: logs the distance remaining.
        This is the same feedback that RViz shows as "Feedback: reached".
        """
        try:
            distance_remaining = feedback_msg.feedback.distance_remaining
            self.get_logger().info(f"Feedback: Distance remaining = {distance_remaining:.2f} m")
        except Exception as e:
            self.get_logger().error(f"Error in feedback callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = test1()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
