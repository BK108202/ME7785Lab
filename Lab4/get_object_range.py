import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import math


class GetObjectRange(Node):
    def __init__(self):
        super().__init__('get_object_range')

        qos_profile_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor
        )

        self.object_range_publisher = self.create_publisher(
            Pose2D,
            '/object_range',
            10
        )

        # Keeping the parameter name, though not used directly in this calculation.
        self.CAMERA_FOV_DEGREES = 62.2
        self.last_scan = None

    def scan_callback(self, scan_msg):
        self.last_scan = scan_msg
        scan = self.last_scan
        ranges = scan.ranges

        # Use an angular window from -30 to 30 degrees.
        lower_bound = -math.radians(30)
        upper_bound = math.radians(30)

        best_x = float('inf')  # The minimum forward (x-axis) distance found.
        best_angle = None      # The angle corresponding to that minimum x value.

        for i in range(len(ranges)):
            d = ranges[i]
            # Skip invalid readings.
            if math.isinf(d) or math.isnan(d) or d < scan_msg.range_min or d > scan_msg.range_max:
                continue

            angle = scan.angle_min + i * scan.angle_increment

            # Consider only readings within the angular window.
            if lower_bound <= angle <= upper_bound:
                # Compute the forward projection: effective distance along the robot's x-axis.
                x_proj = d * math.cos(angle)
                # Choose the reading with the smallest x projection.
                if x_proj < best_x:
                    best_x = x_proj
                    best_angle = angle

        obj_pose = Pose2D()
        if best_angle is None:
            obj_pose.x = float('nan')
            obj_pose.theta = float('nan')
        else:
            obj_pose.x = best_x
            obj_pose.theta = best_angle

        self.object_range_publisher.publish(obj_pose)


def main(args=None):
    rclpy.init(args=args)
    get_object_range = GetObjectRange()
    rclpy.spin(get_object_range)
    get_object_range.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
