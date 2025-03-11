import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32
from geometry_msgs.msg import Pose2D

# State 1 - Go to Goal
# State 2 - Avoid Obstacle
class StateManager(Node):
    def __init__(self, initialState):
        super().__init__('state_manager')
        self.cur_state = UInt32()
        self.cur_state.data = initialState
        self.state_publisher = self.create_publisher(UInt32, 'state', 10)
        
        self.safe_dist = 0.3
        self.epsilon = 0.1

        self.object_range_subscriber = self.create_subscription(
            Pose2D,
            '/object_range',
            self.object_range_callback,
            10
        )
        self.obstacle_range = float('inf')
        
        timer_period = 0.08
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def object_range_callback(self, msg):
        self.obstacle_range = msg.x

    def timer_callback(self):
        if self.cur_state.data == 1:
            if self.obstacle_range <= self.safe_dist:
                self.cur_state.data = 2
                self.get_logger().info("Switching to Avoid Obstacle state.")
        elif self.cur_state.data == 2:
            if self.obstacle_range > self.safe_dist + self.epsilon:
                self.cur_state.data = 1
                self.get_logger().info("Switching to Go To Goal state.")
        self.state_publisher.publish(self.cur_state)

def main(args=None):
    rclpy.init(args=args)
    initialState = 1
    state_manager = StateManager(initialState)
    rclpy.spin(state_manager)
    state_manager.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
