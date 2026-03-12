import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import Bool

from .common import OccupancyGridMap


class FrontierExplorerNode(Node):
    def __init__(self) -> None:
        super().__init__('frontier_explorer')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('robot_pose_topic', '/pf_pose_stamped')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('min_frontier_cluster_size', 6)
        self.declare_parameter('min_goal_distance_m', 0.6)
        self.declare_parameter('goal_blacklist_radius_m', 0.6)
        self.declare_parameter('goal_publish_period_s', 2.0)
        self.declare_parameter('goal_timeout_s', 30.0)

        self.map_data = OccupancyGridMap()
        self.robot_position_xy: Optional[Tuple[float, float]] = None
        self.current_goal_xy: Optional[Tuple[float, float]] = None
        self.goal_blacklist_xy: List[Tuple[float, float]] = []
        self.exploration_complete = False
        self.goal_sent_time: Optional[float] = None

        self.create_subscription(
            OccupancyGrid,
            self.get_parameter('map_topic').value,
            self.handle_map,
            10
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter('robot_pose_topic').value,
            self.handle_robot_pose,
            20
        )

        self.goal_publisher = self.create_publisher(
            PoseStamped,
            self.get_parameter('goal_topic').value,
            10
        )
        self.exploration_complete_publisher = self.create_publisher(
            Bool,
            '/exploration_complete',
            10
        )

        self.create_timer(
            float(self.get_parameter('goal_publish_period_s').value),
            self.run_exploration_cycle
        )

        self.get_logger().info('FrontierExplorerNode started.')

    def handle_map(self, msg: OccupancyGrid) -> None:
        self.map_data.map_width = msg.info.width
        self.map_data.map_height = msg.info.height
        self.map_data.map_resolution = msg.info.resolution
        self.map_data.origin_world_x = msg.info.origin.position.x
        self.map_data.origin_world_y = msg.info.origin.position.y
        self.map_data.occupancy_data = np.array(msg.data, dtype=np.int16).reshape(
            (self.map_data.map_height, self.map_data.map_width)
        )

    def handle_robot_pose(self, msg: PoseStamped) -> None:
        self.robot_position_xy = (msg.pose.position.x, msg.pose.position.y)

    def is_frontier_cell(self, cell_x: int, cell_y: int) -> bool:
        if not self.map_data.contains_cell(cell_x, cell_y):
            return False

        occupied_threshold = int(self.get_parameter('occupied_threshold').value)

        if not self.map_data.is_free_cell(cell_x, cell_y, occupied_threshold):
            return False

        for offset_x, offset_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_x = cell_x + offset_x
            neighbor_y = cell_y + offset_y

            if self.map_data.contains_cell(neighbor_x, neighbor_y) and self.map_data.is_unknown_cell(neighbor_x, neighbor_y):
                return True

        return False

    def detect_frontier_clusters(self) -> List[List[Tuple[int, int]]]:
        visited_cells = np.zeros((self.map_data.map_height, self.map_data.map_width), dtype=bool)
        frontier_clusters: List[List[Tuple[int, int]]] = []

        for cell_y in range(self.map_data.map_height):
            for cell_x in range(self.map_data.map_width):
                if visited_cells[cell_y, cell_x] or not self.is_frontier_cell(cell_x, cell_y):
                    continue

                cluster_queue = deque([(cell_x, cell_y)])
                visited_cells[cell_y, cell_x] = True
                current_cluster = []

                while cluster_queue:
                    current_x, current_y = cluster_queue.popleft()
                    current_cluster.append((current_x, current_y))

                    for offset_x in [-1, 0, 1]:
                        for offset_y in [-1, 0, 1]:
                            neighbor_x = current_x + offset_x
                            neighbor_y = current_y + offset_y

                            if not self.map_data.contains_cell(neighbor_x, neighbor_y):
                                continue
                            if visited_cells[neighbor_y, neighbor_x]:
                                continue
                            if self.is_frontier_cell(neighbor_x, neighbor_y):
                                visited_cells[neighbor_y, neighbor_x] = True
                                cluster_queue.append((neighbor_x, neighbor_y))

                if len(current_cluster) >= int(self.get_parameter('min_frontier_cluster_size').value):
                    frontier_clusters.append(current_cluster)

        return frontier_clusters

    def choose_best_frontier_goal(
        self,
        frontier_clusters: List[List[Tuple[int, int]]]
    ) -> Optional[Tuple[float, float]]:
        if self.robot_position_xy is None:
            return None

        robot_x, robot_y = self.robot_position_xy
        best_score = -float('inf')
        best_goal_xy = None

        min_goal_distance = float(self.get_parameter('min_goal_distance_m').value)
        blacklist_radius = float(self.get_parameter('goal_blacklist_radius_m').value)

        for cluster in frontier_clusters:
            mean_cell_x = sum(cell[0] for cell in cluster) / len(cluster)
            mean_cell_y = sum(cell[1] for cell in cluster) / len(cluster)

            goal_x, goal_y = self.map_data.cell_to_world(int(mean_cell_x), int(mean_cell_y))
            distance_to_robot = math.hypot(goal_x - robot_x, goal_y - robot_y)

            if distance_to_robot < min_goal_distance:
                continue

            if any(math.hypot(goal_x - blocked_x, goal_y - blocked_y) < blacklist_radius for blocked_x, blocked_y in self.goal_blacklist_xy):
                continue

            cluster_size_bonus = 0.2 * len(cluster)
            distance_penalty = distance_to_robot
            goal_score = cluster_size_bonus - distance_penalty

            if goal_score > best_score:
                best_score = goal_score
                best_goal_xy = (goal_x, goal_y)

        return best_goal_xy

    def publish_exploration_goal(self, goal_xy: Tuple[float, float]) -> None:
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = goal_xy[0]
        goal_msg.pose.position.y = goal_xy[1]
        goal_msg.pose.orientation.w = 1.0

        self.goal_publisher.publish(goal_msg)
        self.current_goal_xy = goal_xy
        self.goal_sent_time = self.get_clock().now().nanoseconds * 1e-9

        self.get_logger().info(f'Published exploration goal: {goal_xy}')

    def run_exploration_cycle(self) -> None:
        if not self.map_data.is_valid() or self.robot_position_xy is None:
            return

        if self.exploration_complete:
            return

        # Check whether the current goal has timed out — if so, blacklist it
        if self.current_goal_xy is not None and self.goal_sent_time is not None:
            elapsed = self.get_clock().now().nanoseconds * 1e-9 - self.goal_sent_time
            goal_timeout = float(self.get_parameter('goal_timeout_s').value)
            robot_x, robot_y = self.robot_position_xy
            distance_to_goal = math.hypot(
                self.current_goal_xy[0] - robot_x,
                self.current_goal_xy[1] - robot_y
            )
            if elapsed > goal_timeout and distance_to_goal > float(self.get_parameter('min_goal_distance_m').value):
                self.get_logger().warn(
                    f'Goal {self.current_goal_xy} timed out after {elapsed:.1f}s — blacklisting.'
                )
                self.goal_blacklist_xy.append(self.current_goal_xy)
                self.current_goal_xy = None
                self.goal_sent_time = None

        frontier_clusters = self.detect_frontier_clusters()

        if not frontier_clusters:
            self.get_logger().info('No frontiers found. Exploration complete.')
            self.exploration_complete = True
            msg = Bool()
            msg.data = True
            self.exploration_complete_publisher.publish(msg)
            return

        next_goal_xy = self.choose_best_frontier_goal(frontier_clusters)

        if next_goal_xy is None:
            self.get_logger().info('No valid frontier goal found after filtering.')
            return

        if self.current_goal_xy is not None:
            if math.hypot(next_goal_xy[0] - self.current_goal_xy[0], next_goal_xy[1] - self.current_goal_xy[1]) < 0.3:
                # Goal unchanged — republish so the planner keeps tracking it
                self.publish_exploration_goal(self.current_goal_xy)
                return

        self.publish_exploration_goal(next_goal_xy)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrontierExplorerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()