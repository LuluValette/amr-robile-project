# local_navigation_node.py

import math
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from .map_processing import extract_yaw_from_pose, normalize_angle


class LocalNavigationNode(Node):
    def __init__(self) -> None:
        super().__init__('local_navigation_node')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('waypoints_topic', '/waypoints')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('control_rate', 10.0)
        self.declare_parameter('k_att', 1.2)
        self.declare_parameter('k_rep', 0.35)
        self.declare_parameter('repulsive_distance', 0.8)
        self.declare_parameter('goal_tolerance', 0.18)
        self.declare_parameter('yaw_gain', 2.0)
        self.declare_parameter('max_linear_speed', 0.22)
        self.declare_parameter('max_angular_speed', 1.2)
        self.declare_parameter('front_slowdown_distance', 0.50)
        self.declare_parameter('emergency_stop_distance', 0.16)

        self.odometry_message: Optional[Odometry] = None
        self.laser_scan_message: Optional[LaserScan] = None
        self.waypoint_list: List[Tuple[float, float]] = []
        self.current_waypoint_index = 0

        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self.on_odometry_received,
            10,
        )
        self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self.on_scan_received,
            10,
        )
        self.create_subscription(
            Path,
            self.get_parameter('waypoints_topic').value,
            self.on_waypoints_received,
            10,
        )

        self.velocity_command_publisher = self.create_publisher(
            Twist,
            self.get_parameter('cmd_vel_topic').value,
            10,
        )

        control_period_seconds = 1.0 / float(self.get_parameter('control_rate').value)
        self.create_timer(control_period_seconds, self.run_control_loop)

        self.get_logger().info('Local navigation node started.')

    def on_odometry_received(self, odometry_message: Odometry) -> None:
        self.odometry_message = odometry_message

    def on_scan_received(self, laser_scan_message: LaserScan) -> None:
        self.laser_scan_message = laser_scan_message

    def on_waypoints_received(self, waypoint_path_message: Path) -> None:
        self.waypoint_list = [
            (pose_stamped.pose.position.x, pose_stamped.pose.position.y)
            for pose_stamped in waypoint_path_message.poses
        ]
        self.current_waypoint_index = 0
        self.get_logger().info(f'Received {len(self.waypoint_list)} waypoints.')

    def run_control_loop(self) -> None:
        if (
            self.odometry_message is None
            or self.laser_scan_message is None
            or not self.waypoint_list
        ):
            return

        if self.current_waypoint_index >= len(self.waypoint_list):
            self.publish_stop_command()
            return

        robot_x = float(self.odometry_message.pose.pose.position.x)
        robot_y = float(self.odometry_message.pose.pose.position.y)
        robot_yaw = extract_yaw_from_pose(self.odometry_message.pose.pose)

        target_waypoint_x, target_waypoint_y = self.waypoint_list[self.current_waypoint_index]
        goal_delta_x = target_waypoint_x - robot_x
        goal_delta_y = target_waypoint_y - robot_y
        distance_to_waypoint = math.hypot(goal_delta_x, goal_delta_y)

        waypoint_tolerance = float(self.get_parameter('goal_tolerance').value)
        if distance_to_waypoint < waypoint_tolerance:
            self.current_waypoint_index += 1

            if self.current_waypoint_index >= len(self.waypoint_list):
                self.get_logger().info('Final waypoint reached.')
                self.publish_stop_command()
                return

            target_waypoint_x, target_waypoint_y = self.waypoint_list[self.current_waypoint_index]
            goal_delta_x = target_waypoint_x - robot_x
            goal_delta_y = target_waypoint_y - robot_y
            distance_to_waypoint = math.hypot(goal_delta_x, goal_delta_y)

        attractive_gain = float(self.get_parameter('k_att').value)
        attractive_force_world_x = attractive_gain * goal_delta_x
        attractive_force_world_y = attractive_gain * goal_delta_y

        repulsive_force_robot_x = 0.0
        repulsive_force_robot_y = 0.0

        repulsive_gain = float(self.get_parameter('k_rep').value)
        repulsive_effect_distance = float(self.get_parameter('repulsive_distance').value)
        emergency_stop_distance = float(self.get_parameter('emergency_stop_distance').value)

        minimum_front_distance = float('inf')
        scan_angle = self.laser_scan_message.angle_min

        for measured_range in self.laser_scan_message.ranges:
            if math.isfinite(measured_range):
                if abs(scan_angle) < math.radians(25.0):
                    minimum_front_distance = min(minimum_front_distance, measured_range)

                if 0.02 < measured_range < repulsive_effect_distance:
                    repulsive_strength = (
                        repulsive_gain
                        * (
                            (1.0 / max(measured_range, 1e-3))
                            - (1.0 / repulsive_effect_distance)
                        )
                        / (measured_range * measured_range)
                    )
                    repulsive_force_robot_x += -repulsive_strength * math.cos(scan_angle)
                    repulsive_force_robot_y += -repulsive_strength * math.sin(scan_angle)

            scan_angle += self.laser_scan_message.angle_increment

        if minimum_front_distance < emergency_stop_distance:
            emergency_command = Twist()
            emergency_command.linear.x = 0.0
            emergency_command.angular.z = 0.6
            self.velocity_command_publisher.publish(emergency_command)
            return

        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        repulsive_force_world_x = (
            cos_yaw * repulsive_force_robot_x - sin_yaw * repulsive_force_robot_y
        )
        repulsive_force_world_y = (
            sin_yaw * repulsive_force_robot_x + cos_yaw * repulsive_force_robot_y
        )

        total_force_world_x = attractive_force_world_x + repulsive_force_world_x
        total_force_world_y = attractive_force_world_y + repulsive_force_world_y

        desired_heading = math.atan2(total_force_world_y, total_force_world_x)
        heading_error = normalize_angle(desired_heading - robot_yaw)

        yaw_gain = float(self.get_parameter('yaw_gain').value)
        max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        slowdown_distance = float(self.get_parameter('front_slowdown_distance').value)

        heading_alignment = max(0.0, math.cos(heading_error))
        total_force_magnitude = math.hypot(total_force_world_x, total_force_world_y)

        linear_speed_command = min(max_linear_speed, 0.6 * total_force_magnitude) * heading_alignment
        angular_speed_command = max(
            -max_angular_speed,
            min(max_angular_speed, yaw_gain * heading_error),
        )

        if minimum_front_distance < slowdown_distance:
            slowdown_factor = max(
                0.0,
                (minimum_front_distance - emergency_stop_distance)
                / max(slowdown_distance - emergency_stop_distance, 1e-3),
            )
            linear_speed_command *= slowdown_factor

        velocity_command = Twist()
        velocity_command.linear.x = linear_speed_command
        velocity_command.angular.z = angular_speed_command
        self.velocity_command_publisher.publish(velocity_command)

    def publish_stop_command(self) -> None:
        self.velocity_command_publisher.publish(Twist())


def main(args=None) -> None:
    rclpy.init(args=args)
    navigation_node = LocalNavigationNode()
    try:
        rclpy.spin(navigation_node)
    except KeyboardInterrupt:
        pass
    finally:
        navigation_node.publish_stop_command()
        navigation_node.destroy_node()
        rclpy.shutdown()