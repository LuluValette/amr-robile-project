import math
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy.ndimage import binary_dilation


@dataclass
class OccupancyGridMap:
    map_width: int = 0
    map_height: int = 0
    map_resolution: float = 0.05
    origin_world_x: float = 0.0
    origin_world_y: float = 0.0
    occupancy_data: Optional[np.ndarray] = None  # shape=(height, width), values in [-1, 100]

    def is_valid(self) -> bool:
        return self.occupancy_data is not None and self.map_width > 0 and self.map_height > 0

    def world_to_cell(self, world_x: float, world_y: float) -> Tuple[int, int]:
        cell_x = int((world_x - self.origin_world_x) / self.map_resolution)
        cell_y = int((world_y - self.origin_world_y) / self.map_resolution)
        return cell_x, cell_y

    def cell_to_world(self, cell_x: int, cell_y: int) -> Tuple[float, float]:
        world_x = self.origin_world_x + (cell_x + 0.5) * self.map_resolution
        world_y = self.origin_world_y + (cell_y + 0.5) * self.map_resolution
        return world_x, world_y

    def contains_cell(self, cell_x: int, cell_y: int) -> bool:
        return 0 <= cell_x < self.map_width and 0 <= cell_y < self.map_height

    def get_occupancy(self, cell_x: int, cell_y: int) -> int:
        if not self.contains_cell(cell_x, cell_y):
            return 100
        return int(self.occupancy_data[cell_y, cell_x])

    def is_free_cell(self, cell_x: int, cell_y: int, occupied_threshold: int = 50) -> bool:
        occupancy_value = self.get_occupancy(cell_x, cell_y)
        return occupancy_value >= 0 and occupancy_value < occupied_threshold

    def is_unknown_cell(self, cell_x: int, cell_y: int) -> bool:
        return self.get_occupancy(cell_x, cell_y) < 0


def normalize_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def euclidean_distance_cells(cell_a: Tuple[int, int], cell_b: Tuple[int, int]) -> float:
    return math.hypot(cell_a[0] - cell_b[0], cell_a[1] - cell_b[1])


def inflate_obstacles(
    occupancy_grid: OccupancyGridMap,
    inflation_radius_m: float = 0.20,
    occupied_threshold: int = 50
) -> np.ndarray:
    inflated_grid = np.array(occupancy_grid.occupancy_data, copy=True)
    inflation_radius_cells = max(1, int(math.ceil(inflation_radius_m / occupancy_grid.map_resolution)))

    # Build a circular structuring element
    diameter = 2 * inflation_radius_cells + 1
    structuring_element = np.zeros((diameter, diameter), dtype=bool)
    for dy in range(-inflation_radius_cells, inflation_radius_cells + 1):
        for dx in range(-inflation_radius_cells, inflation_radius_cells + 1):
            if dx * dx + dy * dy <= inflation_radius_cells * inflation_radius_cells:
                structuring_element[dy + inflation_radius_cells, dx + inflation_radius_cells] = True

    # Dilate the obstacle mask in one vectorized operation
    obstacle_mask = inflated_grid >= occupied_threshold
    dilated_mask = binary_dilation(obstacle_mask, structure=structuring_element)
    inflated_grid[dilated_mask] = 100

    return inflated_grid


def compute_astar_path(
    occupancy_grid: OccupancyGridMap,
    start_cell: Tuple[int, int],
    goal_cell: Tuple[int, int],
    occupied_threshold: int = 50,
    allow_diagonal_motion: bool = True,
    inflation_radius_m: float = 0.20
) -> List[Tuple[int, int]]:
    if not occupancy_grid.is_valid():
        return []

    inflated_grid = inflate_obstacles(occupancy_grid, inflation_radius_m, occupied_threshold)

    def is_traversable(cell: Tuple[int, int]) -> bool:
        cell_x, cell_y = cell
        if not occupancy_grid.contains_cell(cell_x, cell_y):
            return False
        occupancy_value = int(inflated_grid[cell_y, cell_x])
        return 0 <= occupancy_value < occupied_threshold

    if not is_traversable(start_cell) or not is_traversable(goal_cell):
        return []

    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal_motion:
        neighbor_offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    open_set = []
    heapq.heappush(open_set, (0.0, start_cell))

    came_from = {}
    cost_from_start = {start_cell: 0.0}

    while open_set:
        _, current_cell = heapq.heappop(open_set)

        if current_cell == goal_cell:
            path_cells = [current_cell]
            while current_cell in came_from:
                current_cell = came_from[current_cell]
                path_cells.append(current_cell)
            path_cells.reverse()
            return path_cells

        current_x, current_y = current_cell

        for offset_x, offset_y in neighbor_offsets:
            neighbor_cell = (current_x + offset_x, current_y + offset_y)

            if not is_traversable(neighbor_cell):
                continue

            step_cost = math.hypot(offset_x, offset_y)
            tentative_cost = cost_from_start[current_cell] + step_cost

            if tentative_cost < cost_from_start.get(neighbor_cell, float('inf')):
                came_from[neighbor_cell] = current_cell
                cost_from_start[neighbor_cell] = tentative_cost
                estimated_total_cost = tentative_cost + euclidean_distance_cells(neighbor_cell, goal_cell)
                heapq.heappush(open_set, (estimated_total_cost, neighbor_cell))

    return []


def simplify_path_to_waypoints(
    path_cells: List[Tuple[int, int]],
    turn_angle_threshold_deg: float = 25.0,
    minimum_spacing_cells: int = 8
) -> List[Tuple[int, int]]:
    if len(path_cells) <= 2:
        return path_cells[:]

    selected_waypoints = [path_cells[0]]
    turn_angle_threshold_rad = math.radians(turn_angle_threshold_deg)
    last_selected_index = 0

    for path_index in range(1, len(path_cells) - 1):
        prev_cell = path_cells[path_index - 1]
        current_cell = path_cells[path_index]
        next_cell = path_cells[path_index + 1]

        direction_before = (current_cell[0] - prev_cell[0], current_cell[1] - prev_cell[1])
        direction_after = (next_cell[0] - current_cell[0], next_cell[1] - current_cell[1])

        angle_before = math.atan2(direction_before[1], direction_before[0])
        angle_after = math.atan2(direction_after[1], direction_after[0])
        turn_angle = abs(normalize_angle(angle_after - angle_before))

        if turn_angle > turn_angle_threshold_rad or (path_index - last_selected_index) >= minimum_spacing_cells:
            selected_waypoints.append(current_cell)
            last_selected_index = path_index

    selected_waypoints.append(path_cells[-1])
    return selected_waypoints


def laser_scan_to_cartesian_points(
    scan_ranges: List[float],
    angle_min_rad: float,
    angle_increment_rad: float,
    min_valid_range: float,
    max_valid_range: float
) -> List[Tuple[float, float]]:
    points_xy = []

    for beam_index, measured_range in enumerate(scan_ranges):
        if math.isinf(measured_range) or math.isnan(measured_range):
            continue
        if measured_range < min_valid_range or measured_range > max_valid_range:
            continue

        beam_angle = angle_min_rad + beam_index * angle_increment_rad
        point_x = measured_range * math.cos(beam_angle)
        point_y = measured_range * math.sin(beam_angle)
        points_xy.append((point_x, point_y))

    return points_xy


def simulate_ray_cast(
    occupancy_grid: OccupancyGridMap,
    robot_x: float,
    robot_y: float,
    beam_angle_world: float,
    max_range: float,
    step_size: float = 0.05,
    occupied_threshold: int = 50
) -> float:
    """Vectorised DDA ray cast. Pre-computes all sample coordinates with numpy
    and looks up the occupancy grid in one batch, avoiding a Python loop per step."""
    cos_a = math.cos(beam_angle_world)
    sin_a = math.sin(beam_angle_world)
    n_steps = int(max_range / step_size) + 1

    distances = np.arange(n_steps, dtype=np.float32) * step_size
    sample_x = robot_x + distances * cos_a
    sample_y = robot_y + distances * sin_a

    cell_x = ((sample_x - occupancy_grid.origin_world_x) / occupancy_grid.map_resolution).astype(np.int32)
    cell_y = ((sample_y - occupancy_grid.origin_world_y) / occupancy_grid.map_resolution).astype(np.int32)

    in_bounds = (
        (cell_x >= 0) & (cell_x < occupancy_grid.map_width) &
        (cell_y >= 0) & (cell_y < occupancy_grid.map_height)
    )

    # First out-of-bounds step terminates the ray
    out_idx = np.argwhere(~in_bounds)
    first_oob = int(out_idx[0]) if len(out_idx) else n_steps

    if first_oob == 0:
        return 0.0

    # Only look up cells that are inside the grid
    cx = cell_x[:first_oob]
    cy = cell_y[:first_oob]
    occ = occupancy_grid.occupancy_data[cy, cx]

    hit = np.argwhere(occ >= occupied_threshold)
    if len(hit):
        return float(distances[int(hit[0])])

    # Ray exited the map without hitting anything
    return float(distances[first_oob - 1]) if first_oob < n_steps else max_range