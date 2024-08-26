import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from clearpath_platform_msgs.msg import Drive

class TwistToDrive(Node):
    def __init__(self):
        super().__init__('twist_to_drive_node')

        # Subscriber to the Twist topic
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1,
        )
        self.subscription = self.create_subscription(
            Drive,
            '/j100_0166/platform/motors/cmd_drive',
            self.twist_callback,
            best_effort_qos)
        
        # Publisher to the Drive topic
        self.publisher = self.create_publisher(
            Drive,
            '/platform/motors/cmd_drive',
            10)

    def twist_callback(self, msg):
        self.last_twist = msg
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TwistToDrive()
    rclpy.spin(node)

if __name__ == '__main__':
    main()