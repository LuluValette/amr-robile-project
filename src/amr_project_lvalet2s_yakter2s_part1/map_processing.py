# map_processing.py

import math
from typing import List, Optional, Tuple

import numpy as np
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid

GridCell = Tuple[int, int]
WorldPoint = Tuple[float, float]


class OccupancyGridMap:
    def __init__(self, occupancy_grid_message: OccupancyGrid):
        self.occupancy_grid_message = occupancy_grid_message
        self.resolution = float(occupancy_grid_message.info.resolution)
        self.width = int(occupancy_grid_message.info.width)
        self.height = int(occupancy_grid_message.info.height)
        self.origin_x = float(occupancy_grid_message.info.origin.position.x)
        self.origin_y = float(occupancy_grid_message.info.origin.position.y)
        self.occupancy_matrix = np.array(
            occupancy_grid_message.data,
            dtype=np.int16,
        ).reshape((self.height, self.width))

    def world_to_grid(self, world_x: float, world_y: float) -> Optional[GridCell]:
        grid_x = int((world_x - self.origin_x) / self.resolution)
        grid_y = int((world_y - self.origin_y) / self.resolution)

        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return grid_x, grid_y

        return None

    def grid_to_world(self, grid_x: int, grid_y: int) -> WorldPoint:
        world_x = self.origin_x + (grid_x + 0.5) * self.resolution
        world_y = self.origin_y + (grid_y + 0.5) * self.resolution
        return world_x, world_y

    def is_occupied(self, grid_x: int, grid_y: int, occupancy_threshold: int = 50) -> bool:
        occupancy_value = int(self.occupancy_matrix[grid_y, grid_x])
        return occupancy_value >= occupancy_threshold

    def is_unknown(self, grid_x: int, grid_y: int) -> bool:
        return int(self.occupancy_matrix[grid_y, grid_x]) < 0

    def is_free(
        self,
        grid_x: int,
        grid_y: int,
        occupancy_threshold: int = 50,
        treat_unknown_as_occupied: bool = True,
    ) -> bool:
        occupancy_value = int(self.occupancy_matrix[grid_y, grid_x])

        if occupancy_value < 0:
            return not treat_unknown_as_occupied

        return occupancy_value < occupancy_threshold

    def build_inflated_obstacle_grid(
        self,
        inflation_radius_meters: float,
        occupancy_threshold: int = 50,
        treat_unknown_as_occupied: bool = True,
    ) -> np.ndarray:
        inflated_obstacle_grid = np.zeros((self.height, self.width), dtype=np.uint8)

        inflation_radius_cells = max(
            0,
            int(math.ceil(inflation_radius_meters / max(self.resolution, 1e-6))),
        )

        occupied_grid_cells = []
        for grid_y in range(self.height):
            for grid_x in range(self.width):
                if not self.is_free(
                    grid_x,
                    grid_y,
                    occupancy_threshold,
                    treat_unknown_as_occupied,
                ):
                    occupied_grid_cells.append((grid_x, grid_y))

        if inflation_radius_cells == 0:
            for occupied_x, occupied_y in occupied_grid_cells:
                inflated_obstacle_grid[occupied_y, occupied_x] = 1
            return inflated_obstacle_grid

        for occupied_x, occupied_y in occupied_grid_cells:
            min_grid_x = max(0, occupied_x - inflation_radius_cells)
            max_grid_x = min(self.width - 1, occupied_x + inflation_radius_cells)
            min_grid_y = max(0, occupied_y - inflation_radius_cells)
            max_grid_y = min(self.height - 1, occupied_y + inflation_radius_cells)

            for grid_x in range(min_grid_x, max_grid_x + 1):
                for grid_y in range(min_grid_y, max_grid_y + 1):
                    if math.hypot(grid_x - occupied_x, grid_y - occupied_y) <= inflation_radius_cells:
                        inflated_obstacle_grid[grid_y, grid_x] = 1

        return inflated_obstacle_grid


def extract_yaw_from_pose(pose: Pose) -> float:
    quaternion = pose.orientation
    sin_yaw_cos_pitch = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)


def normalize_angle(angle_radians: float) -> float:
    while angle_radians > math.pi:
        angle_radians -= 2.0 * math.pi
    while angle_radians < -math.pi:
        angle_radians += 2.0 * math.pi
    return angle_radians


def compute_polyline_length(world_points: List[WorldPoint]) -> float:
    if len(world_points) < 2:
        return 0.0

    total_length = 0.0
    for point_index in range(1, len(world_points)):
        total_length += math.dist(world_points[point_index - 1], world_points[point_index])

    return total_length