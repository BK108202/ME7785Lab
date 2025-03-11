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
        lower_bound = -math.radians(70)
        upper_bound = math.radians(70)

        # Threshold (in meters) to decide if consecutive points belong to the same object.
        cluster_threshold = 0.1

        # Collect valid indices within the angular window.
        valid_indices = []
        for i in range(len(ranges)):
            d = ranges[i]
            angle = scan.angle_min + i * scan.angle_increment
            if angle < lower_bound or angle > upper_bound:
                continue
            if math.isinf(d) or math.isnan(d) or d < scan_msg.range_min or d > scan_msg.range_max:
                continue
            valid_indices.append(i)

        # Group valid indices into clusters.
        clusters = []
        current_cluster = []
        for idx in valid_indices:
            if not current_cluster:
                current_cluster.append(idx)
            else:
                prev_idx = current_cluster[-1]
                if abs(ranges[idx] - ranges[prev_idx]) < cluster_threshold:
                    current_cluster.append(idx)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [idx]
        if current_cluster:
            clusters.append(current_cluster)

        obj_pose = Pose2D()

        # If no clusters were found, publish NaN.
        if not clusters:
            obj_pose.x = float('nan')
            obj_pose.theta = float('nan')
            self.object_range_publisher.publish(obj_pose)
            return

        # Select the cluster corresponding to the closest object.
        # We assume the cluster with the smallest minimum x projection (d * cos(angle)) is the closest.
        best_cluster = None
        best_min_x = float('inf')
        for cluster in clusters:
            cluster_min_x = min(ranges[i] * math.cos(scan.angle_min + i * scan.angle_increment)
                                for i in cluster)
            if cluster_min_x < best_min_x:
                best_min_x = cluster_min_x
                best_cluster = cluster

        # Within the chosen cluster, detect the endpoint defined as the point with the maximum x projection.
        endpoint_x = -float('inf')
        endpoint_angle = None
        for i in best_cluster:
            d = ranges[i]
            angle = scan.angle_min + i * scan.angle_increment
            x_proj = d * math.cos(angle)
            if x_proj > endpoint_x:
                endpoint_x = x_proj
                endpoint_angle = angle

        if endpoint_angle is None:
            obj_pose.x = float('nan')
            obj_pose.theta = float('nan')
        else:
            obj_pose.x = endpoint_x
            obj_pose.theta = endpoint_angle

        self.object_range_publisher.publish(obj_pose)


def main(args=None):
    rclpy.init(args=args)
    get_object_range = GetObjectRange()
    rclpy.spin(get_object_range)
    get_object_range.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
