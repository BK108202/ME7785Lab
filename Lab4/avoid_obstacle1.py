import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
import math

class AvoidObstacle(Node):

    def __init__(self):
        super().__init__('avoid_obstacle')
        self._vel_publisher = self.create_publisher(Twist, '/cmd_vel', 5)
        self._object_subscriber = self.create_subscription(Pose2D, '/object_range', self.object_callback, 10)
        

    def object_callback(self, obj_pose):
        twist = Twist()
        dist_x = obj_pose.x         # forward projection (d*cos(theta))
        radian_x = obj_pose.theta   # angle corresponding to that distance

        kp_w = 2.0
        des_x = 0.3               # desired lateral distance (or “offset”) from the wall/object
        tolerance_radian = 5 * math.pi / 180
        tolerance_x = 0.1

        # If no valid object is detected, use a default wall-following behavior.
        if math.isnan(dist_x) or math.isnan(radian_x):
            twist.linear.x = 0.1
            twist.angular.z = 0
            self._vel_publisher.publish(twist)
            return

        # Estimate the lateral (sideways) distance to the object.
        # The getobjectrange node provides dist_x = d*cos(theta) and radian_x = theta.
        # Therefore, the full range d ≈ dist_x/cos(radian_x) (if cos(theta) ≠ 0),
        # and the lateral distance is:
        #   lateral_distance = d * sin(theta) ≈ dist_x * tan(theta)
        lateral_distance = dist_x * math.tan(radian_x)
        e_lat = des_x - lateral_distance   # error in lateral distance (desired - measured)

        # Also compute a forward error: if the object (or door) is farther than desired,
        # we want to drive forward (i.e. “go through” it).
        e_forward = dist_x - des_x

        # Apply tolerances to ignore small errors.
        if abs(e_lat) < tolerance_x:
            e_lat = 0.0
        if abs(radian_x) < tolerance_radian:
            radian_x = 0.0

        # Compute angular velocity to reduce lateral error.
        # This steers the robot so that the object stays at the desired lateral offset.
        w = kp_w * e_lat

        # For forward velocity, use a simple proportional controller.
        # When the object is farther away (positive e_forward), speed up a bit.
        kp_v = 0.5
        v = 0.1 + kp_v * e_forward  # base forward speed plus adjustment

        # Saturate the velocities.
        max_velocity = 0.2
        min_velocity = 0.05
        v = max(min(v, max_velocity), min_velocity)

        max_angular_velocity = 1.5
        w = max(min(w, max_angular_velocity), -max_angular_velocity)

        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self._vel_publisher.publish(twist)

        self.get_logger().info(
            f"Published velocity: linear={v:.2f} m/s, angular={w:.2f} rad/s | "
            f"lateral_error={e_lat:.2f}, forward_error={e_forward:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    avoid_obstacle = AvoidObstacle()
    rclpy.spin(avoid_obstacle)
    avoid_obstacle.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
