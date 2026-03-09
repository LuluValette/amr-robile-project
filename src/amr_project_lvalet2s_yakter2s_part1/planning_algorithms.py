# planning_algorithms.py

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

GridCell = Tuple[int, int]
WorldPoint = Tuple[float, float]


def compute_astar_path(
    start_cell: GridCell,
    goal_cell: GridCell,
    inflated_obstacle_grid: np.ndarray,
    allow_diagonal_motion: bool = True,
) -> Optional[List[GridCell]]:
    grid_height, grid_width = inflated_obstacle_grid.shape

    def is_inside_grid(grid_cell: GridCell) -> bool:
        grid_x, grid_y = grid_cell
        return 0 <= grid_x < grid_width and 0 <= grid_y < grid_height

    def is_cell_free(grid_cell: GridCell) -> bool:
        grid_x, grid_y = grid_cell
        return inflated_obstacle_grid[grid_y, grid_x] == 0

    def estimate_cost_to_goal(cell_a: GridCell, cell_b: GridCell) -> float:
        return math.hypot(cell_a[0] - cell_b[0], cell_a[1] - cell_b[1])

    if not is_inside_grid(start_cell) or not is_inside_grid(goal_cell):
        return None

    if not is_cell_free(start_cell) or not is_cell_free(goal_cell):
        return None

    if allow_diagonal_motion:
        neighbor_offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    else:
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    open_set_heap = []
    heapq.heappush(open_set_heap, (0.0, start_cell))

    parent_by_cell: Dict[GridCell, GridCell] = {}
    cost_from_start: Dict[GridCell, float] = {start_cell: 0.0}
    estimated_total_cost: Dict[GridCell, float] = {
        start_cell: estimate_cost_to_goal(start_cell, goal_cell)
    }

    visited_cells = set()

    while open_set_heap:
        _, current_cell = heapq.heappop(open_set_heap)

        if current_cell in visited_cells:
            continue
        visited_cells.add(current_cell)

        if current_cell == goal_cell:
            reconstructed_path = [current_cell]
            while current_cell in parent_by_cell:
                current_cell = parent_by_cell[current_cell]
                reconstructed_path.append(current_cell)
            reconstructed_path.reverse()
            return reconstructed_path

        current_x, current_y = current_cell

        for offset_x, offset_y in neighbor_offsets:
            neighbor_x = current_x + offset_x
            neighbor_y = current_y + offset_y
            neighbor_cell = (neighbor_x, neighbor_y)

            if not is_inside_grid(neighbor_cell) or not is_cell_free(neighbor_cell):
                continue

            if offset_x != 0 and offset_y != 0:
                if not is_cell_free((current_x + offset_x, current_y)) or not is_cell_free((current_x, current_y + offset_y)):
                    continue
                movement_cost = math.sqrt(2.0)
            else:
                movement_cost = 1.0

            tentative_cost_from_start = cost_from_start[current_cell] + movement_cost

            if tentative_cost_from_start < cost_from_start.get(neighbor_cell, float('inf')):
                parent_by_cell[neighbor_cell] = current_cell
                cost_from_start[neighbor_cell] = tentative_cost_from_start

                neighbor_estimated_total_cost = (
                    tentative_cost_from_start
                    + estimate_cost_to_goal(neighbor_cell, goal_cell)
                )
                estimated_total_cost[neighbor_cell] = neighbor_estimated_total_cost

                heapq.heappush(
                    open_set_heap,
                    (neighbor_estimated_total_cost, neighbor_cell),
                )

    return None


def line_of_sight_is_clear(
    start_cell: GridCell,
    end_cell: GridCell,
    inflated_obstacle_grid: np.ndarray,
) -> bool:
    start_x, start_y = start_cell
    end_x, end_y = end_cell

    delta_x = abs(end_x - start_x)
    delta_y = abs(end_y - start_y)

    current_x = start_x
    current_y = start_y

    step_count = 1 + delta_x + delta_y
    x_increment = 1 if end_x > start_x else -1
    y_increment = 1 if end_y > start_y else -1
    error_value = delta_x - delta_y

    delta_x *= 2
    delta_y *= 2

    for _ in range(step_count):
        if inflated_obstacle_grid[current_y, current_x] != 0:
            return False

        if error_value > 0:
            current_x += x_increment
            error_value -= delta_y
        else:
            current_y += y_increment
            error_value += delta_x

    return True


def build_navigation_waypoints(
    grid_path: List[GridCell],
    inflated_obstacle_grid: np.ndarray,
    stride: int = 12,
    use_line_of_sight_simplification: bool = True,
) -> List[GridCell]:
    if not grid_path:
        return []

    if len(grid_path) <= 2:
        return grid_path

    sampled_waypoints = [grid_path[0]]
    for point_index in range(stride, len(grid_path) - 1, stride):
        sampled_waypoints.append(grid_path[point_index])

    if sampled_waypoints[-1] != grid_path[-1]:
        sampled_waypoints.append(grid_path[-1])

    turning_waypoints = [grid_path[0]]
    previous_direction = None

    for point_index in range(1, len(grid_path)):
        direction_x = grid_path[point_index][0] - grid_path[point_index - 1][0]
        direction_y = grid_path[point_index][1] - grid_path[point_index - 1][1]
        current_direction = (direction_x, direction_y)

        if previous_direction is not None and current_direction != previous_direction:
            turning_waypoints.append(grid_path[point_index - 1])

        previous_direction = current_direction

    turning_waypoints.append(grid_path[-1])

    merged_waypoints = [grid_path[0]]
    for waypoint_cell in sampled_waypoints[1:-1] + turning_waypoints[1:-1]:
        if waypoint_cell not in merged_waypoints:
            merged_waypoints.append(waypoint_cell)
    merged_waypoints.append(grid_path[-1])

    cell_index_lookup = {grid_cell: index for index, grid_cell in enumerate(grid_path)}
    ordered_waypoints = []
    for waypoint_cell in sorted(set(merged_waypoints), key=lambda cell: cell_index_lookup.get(cell, 1_000_000)):
        ordered_waypoints.append(waypoint_cell)

    if not use_line_of_sight_simplification:
        return ordered_waypoints

    simplified_waypoints = [ordered_waypoints[0]]
    anchor_waypoint = ordered_waypoints[0]

    for point_index in range(1, len(ordered_waypoints) - 1):
        next_waypoint = ordered_waypoints[point_index + 1]
        if not line_of_sight_is_clear(anchor_waypoint, next_waypoint, inflated_obstacle_grid):
            simplified_waypoints.append(ordered_waypoints[point_index])
            anchor_waypoint = ordered_waypoints[point_index]

    if simplified_waypoints[-1] != ordered_waypoints[-1]:
        simplified_waypoints.append(ordered_waypoints[-1])

    return simplified_waypoints