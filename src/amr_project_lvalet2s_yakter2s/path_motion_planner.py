import math
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from .common import (
    OccupancyGridMap,
    compute_astar_path,
    simplify_path_to_waypoints,
    laser_scan_to_cartesian_points,
    normalize_angle,
)


class PathMotionPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('path_motion_planner')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('robot_radius_m', 0.18)
        self.declare_parameter('lookahead_waypoint_distance_m', 0.35)
        self.declare_parameter('waypoint_reached_distance_m', 0.25)
        self.declare_parameter('goal_reached_distance_m', 0.18)
        self.declare_parameter('max_linear_speed_mps', 0.22)
        self.declare_parameter('max_angular_speed_rps', 1.2)
        self.declare_parameter('attractive_gain', 1.2)
        self.declare_parameter('repulsive_gain', 0.12)
        self.declare_parameter('repulsive_effect_range_m', 0.9)
        self.declare_parameter('replan_period_s', 1.5)
        self.declare_parameter('occupied_threshold', 50)

        self.map_data = OccupancyGridMap()
        self.robot_pose: Optional[Tuple[float, float, float]] = None
        self.latest_scan: Optional[LaserScan] = None
        self.goal_position_xy: Optional[Tuple[float, float]] = None

        self.global_path_world: List[Tuple[float, float]] = []
        self.waypoints_world: List[Tuple[float, float]] = []
        self.current_waypoint_index = 0
        self.last_planning_time = self.get_clock().now()

        self.create_subscription(OccupancyGrid, self.get_parameter('map_topic').value, self.handle_map, 10)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.handle_odometry, 20)
        self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self.handle_scan, 20)
        self.create_subscription(PoseStamped, self.get_parameter('goal_topic').value, self.handle_goal, 10)

        self.cmd_vel_publisher = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/planner_markers', 10)

        self.create_timer(0.1, self.control_loop)

        self.get_logger().info('PathMotionPlannerNode started.')

    def handle_map(self, msg: OccupancyGrid) -> None:
        self.map_data.map_width = msg.info.width
        self.map_data.map_height = msg.info.height
        self.map_data.map_resolution = msg.info.resolution
        self.map_data.origin_world_x = msg.info.origin.position.x
        self.map_data.origin_world_y = msg.info.origin.position.y
        self.map_data.occupancy_data = np.array(msg.data, dtype=np.int16).reshape(
            (self.map_data.map_height, self.map_data.map_width)
        )

    def handle_odometry(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        )

        self.robot_pose = (position.x, position.y, yaw)

    def handle_scan(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def handle_goal(self, msg: PoseStamped) -> None:
        self.goal_position_xy = (msg.pose.position.x, msg.pose.position.y)
        self.current_waypoint_index = 0
        self.plan_global_path(force_replan=True)
        self.get_logger().info(f'New goal received: {self.goal_position_xy}')

    def plan_global_path(self, force_replan: bool = False) -> None:
        if not self.map_data.is_valid() or self.robot_pose is None or self.goal_position_xy is None:
            return

        current_time = self.get_clock().now()
        elapsed_time_s = (current_time - self.last_planning_time).nanoseconds * 1e-9

        if (not force_replan) and elapsed_time_s < float(self.get_parameter('replan_period_s').value):
            return

        self.last_planning_time = current_time

        robot_x, robot_y, _ = self.robot_pose
        start_cell = self.map_data.world_to_cell(robot_x, robot_y)
        goal_cell = self.map_data.world_to_cell(self.goal_position_xy[0], self.goal_position_xy[1])

        occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        robot_radius_m = float(self.get_parameter('robot_radius_m').value)

        path_cells = compute_astar_path(
            self.map_data,
            start_cell,
            goal_cell,
            occupied_threshold=occupied_threshold,
            inflation_radius_m=robot_radius_m
        )

        if not path_cells:
            self.get_logger().warn('A* failed to find a path.')
            self.global_path_world = []
            self.waypoints_world = []
            return

        waypoint_cells = simplify_path_to_waypoints(path_cells)

        self.global_path_world = [self.map_data.cell_to_world(cell_x, cell_y) for cell_x, cell_y in path_cells]
        self.waypoints_world = [self.map_data.cell_to_world(cell_x, cell_y) for cell_x, cell_y in waypoint_cells]
        self.current_waypoint_index = 0

        self.publish_visualization_markers()

    def publish_visualization_markers(self) -> None:
        marker_array = MarkerArray()

        path_marker = Marker()
        path_marker.header.frame_id = 'map'
        path_marker.header.stamp = self.get_clock().now().to_msg()
        path_marker.ns = 'path'
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.04
        path_marker.color.a = 1.0
        path_marker.color.g = 1.0

        for world_x, world_y in self.global_path_world:
            point = Point()
            point.x = world_x
            point.y = world_y
            point.z = 0.0
            path_marker.points.append(point)

        marker_array.markers.append(path_marker)

        for waypoint_index, (world_x, world_y) in enumerate(self.waypoints_world):
            waypoint_marker = Marker()
            waypoint_marker.header.frame_id = 'map'
            waypoint_marker.header.stamp = self.get_clock().now().to_msg()
            waypoint_marker.ns = 'waypoints'
            waypoint_marker.id = waypoint_index + 1
            waypoint_marker.type = Marker.SPHERE
            waypoint_marker.action = Marker.ADD
            waypoint_marker.pose.position.x = world_x
            waypoint_marker.pose.position.y = world_y
            waypoint_marker.pose.position.z = 0.0
            waypoint_marker.scale.x = 0.12
            waypoint_marker.scale.y = 0.12
            waypoint_marker.scale.z = 0.12
            waypoint_marker.color.a = 1.0
            waypoint_marker.color.r = 1.0

            marker_array.markers.append(waypoint_marker)

        self.marker_publisher.publish(marker_array)

    def select_current_target_waypoint(self) -> Optional[Tuple[float, float]]:
        if self.robot_pose is None or not self.waypoints_world:
            return None

        robot_x, robot_y, _ = self.robot_pose
        waypoint_reached_distance = float(self.get_parameter('waypoint_reached_distance_m').value)

        while self.current_waypoint_index < len(self.waypoints_world):
            waypoint_x, waypoint_y = self.waypoints_world[self.current_waypoint_index]

            if (
                math.hypot(waypoint_x - robot_x, waypoint_y - robot_y) < waypoint_reached_distance
                and self.current_waypoint_index < len(self.waypoints_world) - 1
            ):
                self.current_waypoint_index += 1
            else:
                break

        if not self.waypoints_world:
            return None

        return self.waypoints_world[min(self.current_waypoint_index, len(self.waypoints_world) - 1)]

    def compute_potential_field_command(self, target_xy: Tuple[float, float]) -> Twist:
        robot_x, robot_y, robot_yaw = self.robot_pose
        target_x, target_y = target_xy

        attractive_gain = float(self.get_parameter('attractive_gain').value)
        repulsive_gain = float(self.get_parameter('repulsive_gain').value)
        repulsive_effect_range = float(self.get_parameter('repulsive_effect_range_m').value)
        max_linear_speed = float(self.get_parameter('max_linear_speed_mps').value)
        max_angular_speed = float(self.get_parameter('max_angular_speed_rps').value)

        attractive_force_x = attractive_gain * (target_x - robot_x)
        attractive_force_y = attractive_gain * (target_y - robot_y)

        repulsive_force_robot_x = 0.0
        repulsive_force_robot_y = 0.0

        if self.latest_scan is not None:
            obstacle_points = laser_scan_to_cartesian_points(
                self.latest_scan.ranges,
                self.latest_scan.angle_min,
                self.latest_scan.angle_increment,
                self.latest_scan.range_min,
                self.latest_scan.range_max
            )

            for obstacle_x, obstacle_y in obstacle_points:
                obstacle_distance = math.hypot(obstacle_x, obstacle_y)

                if 1e-3 < obstacle_distance < repulsive_effect_range:
                    repulsive_strength = (
                        repulsive_gain
                        * ((1.0 / obstacle_distance) - (1.0 / repulsive_effect_range))
                        / (obstacle_distance * obstacle_distance)
                    )
                    repulsive_force_robot_x += -repulsive_strength * (obstacle_x / obstacle_distance)
                    repulsive_force_robot_y += -repulsive_strength * (obstacle_y / obstacle_distance)

        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        repulsive_force_world_x = cos_yaw * repulsive_force_robot_x - sin_yaw * repulsive_force_robot_y
        repulsive_force_world_y = sin_yaw * repulsive_force_robot_x + cos_yaw * repulsive_force_robot_y

        total_force_x = attractive_force_x + repulsive_force_world_x
        total_force_y = attractive_force_y + repulsive_force_world_y

        desired_heading = math.atan2(total_force_y, total_force_x)
        heading_error = normalize_angle(desired_heading - robot_yaw)
        distance_to_target = math.hypot(target_x - robot_x, target_y - robot_y)

        cmd_msg = Twist()
        cmd_msg.linear.x = max(
            -max_linear_speed,
            min(max_linear_speed, 0.8 * distance_to_target * math.cos(heading_error))
        )
        cmd_msg.angular.z = max(
            -max_angular_speed,
            min(max_angular_speed, 1.8 * heading_error)
        )

        if abs(heading_error) > 1.1:
            cmd_msg.linear.x *= 0.3

        return cmd_msg

    def stop_robot(self) -> None:
        self.cmd_vel_publisher.publish(Twist())

    def control_loop(self) -> None:
        if self.robot_pose is None or self.goal_position_xy is None or not self.map_data.is_valid():
            return

        goal_x, goal_y = self.goal_position_xy
        robot_x, robot_y, _ = self.robot_pose
        distance_to_goal = math.hypot(goal_x - robot_x, goal_y - robot_y)

        if distance_to_goal < float(self.get_parameter('goal_reached_distance_m').value):
            self.stop_robot()
            return

        if not self.waypoints_world:
            self.plan_global_path(force_replan=True)
            return

        self.plan_global_path(force_replan=False)

        target_waypoint = self.select_current_target_waypoint()
        if target_waypoint is None:
            self.stop_robot()
            return

        velocity_command = self.compute_potential_field_command(target_waypoint)
        self.cmd_vel_publisher.publish(velocity_command)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathMotionPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.stop_robot()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()