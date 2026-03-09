# goal_sender_node.py

import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class GoalSenderNode(Node):
    def __init__(self, goal_x: float, goal_y: float, goal_yaw: float = 0.0) -> None:
        super().__init__('goal_sender_node')

        self.goal_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.publish_timer = self.create_timer(0.5, self.publish_goal_message)

        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_yaw = goal_yaw
        self.published_message_count = 0

    def publish_goal_message(self) -> None:
        goal_message = PoseStamped()
        goal_message.header.frame_id = 'map'
        goal_message.header.stamp = self.get_clock().now().to_msg()
        goal_message.pose.position.x = float(self.goal_x)
        goal_message.pose.position.y = float(self.goal_y)
        goal_message.pose.orientation.w = math.cos(self.goal_yaw / 2.0)
        goal_message.pose.orientation.z = math.sin(self.goal_yaw / 2.0)

        self.goal_publisher.publish(goal_message)
        self.published_message_count += 1

        if self.published_message_count >= 3:
            self.get_logger().info(
                f'Goal published at ({self.goal_x:.2f}, {self.goal_y:.2f}).'
            )
            raise SystemExit


def main(args=None) -> None:
    rclpy.init(args=args)

    if len(sys.argv) < 3:
        print('Usage: ros2 run amr_project_lvalet2s_yakter2s goal_sender_node X Y [YAW]')
        return

    goal_x = float(sys.argv[1])
    goal_y = float(sys.argv[2])
    goal_yaw = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    goal_sender_node = GoalSenderNode(goal_x, goal_y, goal_yaw)

    try:
        rclpy.spin(goal_sender_node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        goal_sender_node.destroy_node()
        rclpy.shutdown()