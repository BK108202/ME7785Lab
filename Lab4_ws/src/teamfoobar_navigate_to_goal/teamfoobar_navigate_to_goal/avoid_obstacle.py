# Canezo Kang Lab 4
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import UInt32
import math
import numpy as np
import time

# Last Modified 4:20 pm Feb18
class AvoidObstacle(Node):

    def __init__(self):
        super().__init__('avoid_obstacle')
        self._vel_publisher = self.create_publisher(Twist, '/cmd_vel', 5)
        self._odom_subscriber = self.create_subscription(Odometry, '/odom', self.update_Odometry, 10)

        self.Init = True
        self.Init_ang = 0.0
        self.globalAng = 0.0
        self.Init_pos = Point()
        self.globalPos = Point()

        self.waypoints = []
        self.current_goal_index = 0

        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # New subscriber: endpoint from getobjectrange
        self.endpoint = None
        self.endpoint_subscriber = self.create_subscription(
            Pose2D,
            '/object_range',
            self.endpoint_callback,
            10
        )
        self.state = 2  # Default state for avoid_obstacle logic
        self.state_subscriber = self.create_subscription(UInt32, 'state', self.state_callback, 10)

    def state_callback(self, msg):
        self.state = msg.data

    def endpoint_callback(self, msg):
        self.endpoint = msg

    def update_Odometry(self, Odom):
        position = Odom.pose.pose.position
        
        # Orientation uses the quaternion parametrization.
        # To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            # The initial data is stored to be subtracted from all the other values as we want to start at (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],
                              [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])
            self.Init_pos.x = Mrot.item((0,0)) * position.x + Mrot.item((0,1)) * position.y
            self.Init_pos.y = Mrot.item((1,0)) * position.x + Mrot.item((1,1)) * position.y
            self.Init_pos.z = position.z
        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],
                          [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])
        # We subtract the initial values
        self.globalPos.x = Mrot.item((0,0)) * position.x + Mrot.item((0,1)) * position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0)) * position.x + Mrot.item((1,1)) * position.y - self.Init_pos.y
        self.globalAng = orientation - self.Init_ang

    def timer_callback(self):
        if self.state != 2:
            return

        twist = Twist()
        kp_v = 4.0
        kp_w = 2.0
        stop_duration = 3

        obstacle_threshold = 0.30  # meters
        if self.endpoint is not None and not math.isnan(self.endpoint.x) and self.endpoint.x < obstacle_threshold:
            # Reconstruct the true range d from the endpoint message.
            if abs(math.cos(self.endpoint.theta)) > 1e-6:
                d = self.endpoint.x / math.cos(self.endpoint.theta)
            else:
                d = self.endpoint.x
            # Compute the endpoint's robot-frame coordinates.
            endpoint_robot_x = d * math.cos(self.endpoint.theta)  # ideally equals self.endpoint.x
            endpoint_robot_y = d * math.sin(self.endpoint.theta)
            # Add offsets: a forward offset to move further ahead, and a lateral offset to steer away.
            forward_offset = 0.1  # move further ahead from the obstacle
            lateral_offset = -0.3 # steer to the right (adjust sign if needed)
            avoid_robot_x = endpoint_robot_x + forward_offset
            avoid_robot_y = endpoint_robot_y + lateral_offset
            # Transform the avoidance point from the robot frame to the global frame.
            avoid_global_x = self.globalPos.x + avoid_robot_x * math.cos(self.globalAng) - avoid_robot_y * math.sin(self.globalAng)
            avoid_global_y = self.globalPos.y + avoid_robot_x * math.sin(self.globalAng) + avoid_robot_y * math.cos(self.globalAng)
            goal_x = avoid_global_x
            goal_y = avoid_global_y
            tolerance = 0.05
            self.get_logger().info(f"Avoiding obstacle: new goal set to ({goal_x:.2f}, {goal_y:.2f})")
        else:
            # No obstacle detected: use the current waypoint goal.
            if self.current_goal_index >= len(self.waypoints):
                v = 0.1
                w = 0.0
                twist.linear.x = float(v)
                twist.angular.z = float(w)
                self._vel_publisher.publish(twist)
                return
            goal = self.waypoints[self.current_goal_index]
            goal_x, goal_y, tolerance = goal

        error_x = goal_x - self.globalPos.x
        error_y = goal_y - self.globalPos.y
        e_dist = math.sqrt(error_x**2 + error_y**2)
        des_theta = math.atan2(error_y, error_x)
        des_theta = math.atan2(math.sin(des_theta), math.cos(des_theta))
        e_theta = des_theta - self.globalAng
        
        if e_dist < tolerance:
            v = 0.05
            w = -0.75
            twist.linear.x = float(v)
            twist.angular.z = float(w)
            self._vel_publisher.publish(twist)
            # time.sleep(stop_duration)
            self.current_goal_index = self.current_goal_index + 1
            return
        
        v = kp_v * e_dist
        w = kp_w * e_theta
        
        # Saturation limits.
        max_velocity = 0.1
        max_angular_velocity = 1.5
        v = max(min(v, max_velocity), -max_velocity)
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
