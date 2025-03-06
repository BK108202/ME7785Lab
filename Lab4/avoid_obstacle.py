import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
import math

# Last Modified 4:20 pm Feb18
class AvoidObstacle(Node):

    def __init__(self):
        super().__init__('avoid_obstacle')
        self._vel_publisher = self.create_publisher(Twist, '/cmd_vel', 5)
        self._object_subscriber = self.create_subscription(Pose2D, '/object_range', self.object_callback, 10)
        

    def object_callback(self, obj_pose):
        twist = Twist()
        dist_x = obj_pose.x
        radian_x = obj_pose.theta
        
        kp_w = 2.0
        des_x = 0.3
        tolerance_radian = 5*math.pi/180
        tolerance_x = 0.1

        if math.isnan(dist_x) or math.isnan(radian_x):
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self._vel_publisher.publish(twist)
            return

        e_dist = des_x - dist_x
        e_angle = -radian_x

        if abs(e_dist) < tolerance_x:
            e_dist = 0.0
        if abs(radian_x) < tolerance_radian:
            e_angle = 0.0
        
        w = kp_w * e_angle

        max_velocity = 0.1
        v = max_velocity

        # Saturation
        max_angular_velocity = 1.5
        w = max(min(w, max_angular_velocity), -max_angular_velocity)

        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self._vel_publisher.publish(twist)

        self.get_logger().info(f"Published velocity: linear={v:.2f} m/s, angular={w:.2f} rad/s")


def main(args=None):
    rclpy.init(args=args)
    avoid_obstacle = AvoidObstacle()
    rclpy.spin(avoid_obstacle)
    avoid_obstacle.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
