# global_planner_node.py

from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from .map_processing import OccupancyGridMap, compute_polyline_length
from .planning_algorithms import build_navigation_waypoints, compute_astar_path


class GlobalPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('global_planner_node')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('path_topic', '/global_path')
        self.declare_parameter('waypoints_topic', '/waypoints')
        self.declare_parameter('waypoint_markers_topic', '/waypoint_markers')
        self.declare_parameter('robot_radius', 0.20)
        self.declare_parameter('safety_margin', 0.10)
        self.declare_parameter('occupancy_threshold', 50)
        self.declare_parameter('unknown_is_occupied', True)
        self.declare_parameter('allow_diagonal', True)
        self.declare_parameter('waypoint_stride', 12)
        self.declare_parameter('replan_period', 1.0)

        self.occupancy_grid_message: Optional[OccupancyGrid] = None
        self.occupancy_grid_map: Optional[OccupancyGridMap] = None
        self.odometry_message: Optional[Odometry] = None
        self.goal_message: Optional[PoseStamped] = None

        latched_map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(
            OccupancyGrid,
            self.get_parameter('map_topic').value,
            self.on_map_received,
            latched_map_qos,
        )
        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self.on_odometry_received,
            10,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter('goal_topic').value,
            self.on_goal_received,
            10,
        )

        self.global_path_publisher = self.create_publisher(
            Path,
            self.get_parameter('path_topic').value,
            10,
        )
        self.waypoint_path_publisher = self.create_publisher(
            Path,
            self.get_parameter('waypoints_topic').value,
            10,
        )
        self.waypoint_marker_publisher = self.create_publisher(
            MarkerArray,
            self.get_parameter('waypoint_markers_topic').value,
            10,
        )

        replanning_period_seconds = float(self.get_parameter('replan_period').value)
        self.create_timer(replanning_period_seconds, self.compute_and_publish_plan_if_ready)

        self.get_logger().info('Global planner node started.')

    def on_map_received(self, occupancy_grid_message: OccupancyGrid) -> None:
        self.occupancy_grid_message = occupancy_grid_message
        self.occupancy_grid_map = OccupancyGridMap(occupancy_grid_message)

    def on_odometry_received(self, odometry_message: Odometry) -> None:
        self.odometry_message = odometry_message

    def on_goal_received(self, goal_message: PoseStamped) -> None:
        self.goal_message = goal_message
        self.get_logger().info(
            f'Received goal: x={goal_message.pose.position.x:.2f}, '
            f'y={goal_message.pose.position.y:.2f}'
        )
        self.compute_and_publish_plan_if_ready(force_replan=True)

    def compute_and_publish_plan_if_ready(self, force_replan: bool = False) -> None:
        del force_replan  # kept for readability / future extension

        if (
            self.occupancy_grid_map is None
            or self.odometry_message is None
            or self.goal_message is None
            or self.occupancy_grid_message is None
        ):
            return

        robot_world_position = (
            float(self.odometry_message.pose.pose.position.x),
            float(self.odometry_message.pose.pose.position.y),
        )
        goal_world_position = (
            float(self.goal_message.pose.position.x),
            float(self.goal_message.pose.position.y),
        )

        start_grid_cell = self.occupancy_grid_map.world_to_grid(*robot_world_position)
        goal_grid_cell = self.occupancy_grid_map.world_to_grid(*goal_world_position)

        if start_grid_cell is None or goal_grid_cell is None:
            self.get_logger().warn('Start or goal is outside the map.')
            return

        inflation_radius_meters = (
            float(self.get_parameter('robot_radius').value)
            + float(self.get_parameter('safety_margin').value)
        )
        occupancy_threshold = int(self.get_parameter('occupancy_threshold').value)
        treat_unknown_as_occupied = bool(self.get_parameter('unknown_is_occupied').value)
        allow_diagonal_motion = bool(self.get_parameter('allow_diagonal').value)
        waypoint_sampling_stride = int(self.get_parameter('waypoint_stride').value)

        inflated_obstacle_grid = self.occupancy_grid_map.build_inflated_obstacle_grid(
            inflation_radius_meters,
            occupancy_threshold,
            treat_unknown_as_occupied,
        )

        grid_path = compute_astar_path(
            start_grid_cell,
            goal_grid_cell,
            inflated_obstacle_grid,
            allow_diagonal_motion,
        )

        if grid_path is None:
            self.get_logger().warn('A* could not find a path.')
            return

        waypoint_grid_cells = build_navigation_waypoints(
            grid_path,
            inflated_obstacle_grid,
            stride=waypoint_sampling_stride,
        )

        world_path_points = [
            self.occupancy_grid_map.grid_to_world(grid_x, grid_y)
            for grid_x, grid_y in grid_path
        ]
        world_waypoint_points = [
            self.occupancy_grid_map.grid_to_world(grid_x, grid_y)
            for grid_x, grid_y in waypoint_grid_cells
        ]

        map_frame_id = self.occupancy_grid_message.header.frame_id

        self.global_path_publisher.publish(
            self.build_path_message(world_path_points, frame_id=map_frame_id)
        )
        self.waypoint_path_publisher.publish(
            self.build_path_message(world_waypoint_points, frame_id=map_frame_id)
        )
        self.waypoint_marker_publisher.publish(
            self.build_waypoint_marker_array(world_waypoint_points, frame_id=map_frame_id)
        )

        self.get_logger().info(
            f'Published path with {len(world_path_points)} cells / '
            f'{len(world_waypoint_points)} waypoints '
            f'(length {compute_polyline_length(world_path_points):.2f} m).'
        )

    def build_path_message(
        self,
        world_points: List[Tuple[float, float]],
        frame_id: str,
    ) -> Path:
        path_message = Path()
        path_message.header.frame_id = frame_id
        path_message.header.stamp = self.get_clock().now().to_msg()

        for point_x, point_y in world_points:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_message.header
            pose_stamped.pose.position.x = float(point_x)
            pose_stamped.pose.position.y = float(point_y)
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            path_message.poses.append(pose_stamped)

        return path_message

    def build_waypoint_marker_array(
        self,
        world_points: List[Tuple[float, float]],
        frame_id: str,
    ) -> MarkerArray:
        marker_array = MarkerArray()
        timestamp_now = self.get_clock().now().to_msg()

        for waypoint_index, (point_x, point_y) in enumerate(world_points):
            sphere_marker = Marker()
            sphere_marker.header.frame_id = frame_id
            sphere_marker.header.stamp = timestamp_now
            sphere_marker.ns = 'waypoints'
            sphere_marker.id = waypoint_index
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position.x = float(point_x)
            sphere_marker.pose.position.y = float(point_y)
            sphere_marker.pose.position.z = 0.05
            sphere_marker.pose.orientation.w = 1.0
            sphere_marker.scale.x = 0.12
            sphere_marker.scale.y = 0.12
            sphere_marker.scale.z = 0.12
            sphere_marker.color.a = 1.0
            sphere_marker.color.r = 0.0
            sphere_marker.color.g = 1.0
            sphere_marker.color.b = 0.0
            marker_array.markers.append(sphere_marker)

            text_marker = Marker()
            text_marker.header.frame_id = frame_id
            text_marker.header.stamp = timestamp_now
            text_marker.ns = 'waypoint_labels'
            text_marker.id = 1000 + waypoint_index
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(point_x)
            text_marker.pose.position.y = float(point_y)
            text_marker.pose.position.z = 0.25
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.15
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.text = str(waypoint_index)
            marker_array.markers.append(text_marker)

        return marker_array


def main(args=None) -> None:
    rclpy.init(args=args)
    planner_node = GlobalPlannerNode()
    try:
        rclpy.spin(planner_node)
    except KeyboardInterrupt:
        pass
    finally:
        planner_node.destroy_node()
        rclpy.shutdown()