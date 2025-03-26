import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import math

class setwaypoints(Node):
    def __init__(self):
        super().__init__('set_waypoints')
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.amcl_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            10
        )

        self.waypoints = [
            (2.6, 0.38, 0.0, 0.05),
            (2.1, 0.91, 0.0, 0.05),
            (2.9, 1.4, 0.0, 0.05)
        ]
        self.current_goal_index = 0

        self.current_pose = None

        self.at_goal = False
        self.goal_reached_time = None
        self.stop_duration = 3.0

        self.timer = self.create_timer(0.5, self.timer_callback)

        self.publish_goal()

    def amcl_callback(self, msg):
        """
        Callback to update the robot's current pose using /amcl_pose data.
        """
        self.current_pose = msg.pose.pose

    def publish_goal(self):
        """
        Publishes the current goal (as a PoseStamped message) to /goal_pose.
        """
        if self.current_goal_index >= len(self.waypoints):
            self.get_logger().info("All waypoints have been published.")
            return

        goal_x, goal_y, goal_z, tolerance= self.waypoints[self.current_goal_index]

        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        # Use the appropriate frame (commonly "map")
        goal_msg.header.frame_id = "map"

        # Set the goal position.
        goal_msg.pose.position.x = goal_x
        goal_msg.pose.position.y = goal_y
        goal_msg.pose.position.z = goal_z

        # Set the goal orientation.
        # Here we use a fixed orientation (0, 0, 0, 1). Adjust if you need a specific heading.
        goal_msg.pose.orientation.x = 0.0
        goal_msg.pose.orientation.y = 0.0
        goal_msg.pose.orientation.z = 0.0
        goal_msg.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_msg)
        self.get_logger().info(f"Published goal {self.current_goal_index + 1}: x = {goal_x}, y = {goal_y}")

    def timer_callback(self):
        """
        Checks whether the current goal has been reached by comparing the current pose
        to the goal's position. If reached, waits for the stop duration before publishing
        the next goal.
        """
        if self.current_goal_index >= len(self.waypoints):
            return

        # Ensure we have received a current pose.
        if self.current_pose is None:
            return

        goal_x, goal_y, goal_z, tolerance = self.waypoints[self.current_goal_index]
        dx = goal_x - self.current_pose.position.x
        dy = goal_y - self.current_pose.position.y
        distance = math.sqrt(dx**2 + dy**2)

        # Check if the goal has been reached within the specified tolerance.
        if distance < tolerance:
            if not self.at_goal:
                self.get_logger().info(f"Reached waypoint {self.current_goal_index + 1}")
                self.at_goal = True
                self.goal_reached_time = self.get_clock().now().nanoseconds / 1e9  # seconds
            else:
                # Check if the stop duration has passed.
                current_time = self.get_clock().now().nanoseconds / 1e9
                if (current_time - self.goal_reached_time) >= self.stop_duration:
                    self.at_goal = False
                    self.current_goal_index += 1
                    if self.current_goal_index < len(self.waypoints):
                        self.publish_goal()
                    else:
                        self.get_logger().info("All waypoints reached. Navigation complete.")
        else:
            # Optionally, log the distance every few seconds.
            self.get_logger().info_throttle(5, f"Distance to waypoint {self.current_goal_index + 1}: {distance:.2f} m")

def main(args=None):
    rclpy.init(args=args)
    node = setwaypoints()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
