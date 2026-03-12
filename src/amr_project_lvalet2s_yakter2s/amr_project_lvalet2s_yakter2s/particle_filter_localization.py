import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped, Quaternion, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import tf2_ros

from .common import OccupancyGridMap, simulate_ray_cast, normalize_angle


@dataclass
class ParticleState:
    x: float
    y: float
    yaw: float
    weight: float


class ParticleFilterLocalizationNode(Node):
    def __init__(self) -> None:
        super().__init__('particle_filter_localization')

        self.declare_parameter('num_particles', 300)
        self.declare_parameter('random_particle_ratio', 0.05)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('scan_downsample_count', 24)
        self.declare_parameter('measurement_sigma', 0.25)
        self.declare_parameter('motion_noise_linear', 0.02)
        self.declare_parameter('motion_noise_angular', 0.04)
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('particles_topic', '/pf_particles')
        self.declare_parameter('estimated_pose_topic', '/pf_pose')
        self.declare_parameter('kidnap_weight_threshold', 0.002)
        self.declare_parameter('kidnap_recovery_ratio', 0.5)

        self.map_data = OccupancyGridMap()
        self.latest_scan: Optional[LaserScan] = None
        self.previous_odom_pose: Optional[Tuple[float, float, float]] = None
        self.current_odom_pose: Optional[Tuple[float, float, float]] = None
        self.particles: List[ParticleState] = []
        self.is_initialized = False

        self.create_subscription(OccupancyGrid, self.get_parameter('map_topic').value, self.handle_map, 10)
        self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self.handle_scan, 20)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.handle_odometry, 50)

        self.particles_publisher = self.create_publisher(
            PoseArray,
            self.get_parameter('particles_topic').value,
            10
        )
        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            self.get_parameter('estimated_pose_topic').value,
            10
        )
        self.pose_stamped_publisher = self.create_publisher(PoseStamped, '/pf_pose_stamped', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.create_timer(0.5, self.publish_filter_state)

        self.get_logger().info('ParticleFilterLocalizationNode started.')

    def handle_map(self, msg: OccupancyGrid) -> None:
        self.map_data.map_width = msg.info.width
        self.map_data.map_height = msg.info.height
        self.map_data.map_resolution = msg.info.resolution
        self.map_data.origin_world_x = msg.info.origin.position.x
        self.map_data.origin_world_y = msg.info.origin.position.y
        self.map_data.occupancy_data = np.array(msg.data, dtype=np.int16).reshape(
            (self.map_data.map_height, self.map_data.map_width)
        )

        if not self.is_initialized:
            self.initialize_particles_uniformly()

    def handle_scan(self, msg: LaserScan) -> None:
        self.latest_scan = msg

        if self.is_initialized and self.current_odom_pose is not None:
            self.update_particle_weights_from_scan()
            self.resample_particles()

    def handle_odometry(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        )

        odom_pose = (position.x, position.y, yaw)
        self.current_odom_pose = odom_pose

        if not self.is_initialized:
            return

        if self.previous_odom_pose is None:
            self.previous_odom_pose = odom_pose
            return

        delta_x = odom_pose[0] - self.previous_odom_pose[0]
        delta_y = odom_pose[1] - self.previous_odom_pose[1]
        translation_distance = math.hypot(delta_x, delta_y)
        rotation_delta = normalize_angle(odom_pose[2] - self.previous_odom_pose[2])

        self.apply_motion_update(translation_distance, rotation_delta)
        self.previous_odom_pose = odom_pose

    def initialize_particles_uniformly(self) -> None:
        if not self.map_data.is_valid():
            return

        occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        free_cells = np.argwhere(
            (self.map_data.occupancy_data >= 0) &
            (self.map_data.occupancy_data < occupied_threshold)
        )

        if len(free_cells) == 0:
            self.get_logger().warn('No free cells available for particle initialization.')
            return

        num_particles = int(self.get_parameter('num_particles').value)
        self.particles = []

        for _ in range(num_particles):
            cell_y, cell_x = free_cells[random.randrange(len(free_cells))]
            world_x, world_y = self.map_data.cell_to_world(int(cell_x), int(cell_y))
            yaw = random.uniform(-math.pi, math.pi)
            self.particles.append(ParticleState(world_x, world_y, yaw, 1.0 / num_particles))

        self.is_initialized = True
        self.get_logger().info(f'Initialized {num_particles} particles uniformly.')

    def apply_motion_update(self, translation_distance: float, rotation_delta: float) -> None:
        motion_noise_linear = float(self.get_parameter('motion_noise_linear').value)
        motion_noise_angular = float(self.get_parameter('motion_noise_angular').value)

        for particle in self.particles:
            noisy_translation = translation_distance + random.gauss(0.0, motion_noise_linear + 0.05 * abs(translation_distance))
            noisy_rotation = rotation_delta + random.gauss(0.0, motion_noise_angular + 0.05 * abs(rotation_delta))

            particle.x += noisy_translation * math.cos(particle.yaw)
            particle.y += noisy_translation * math.sin(particle.yaw)
            particle.yaw = normalize_angle(particle.yaw + noisy_rotation)

    def get_scan_sample_indices(self) -> List[int]:
        if self.latest_scan is None:
            return []

        desired_sample_count = int(self.get_parameter('scan_downsample_count').value)
        total_beams = len(self.latest_scan.ranges)

        if total_beams == 0:
            return []

        step = max(1, total_beams // desired_sample_count)
        return list(range(0, total_beams, step))[:desired_sample_count]

    def compute_particle_likelihood(self, particle: ParticleState) -> float:
        if self.latest_scan is None or not self.map_data.is_valid():
            return 1.0

        measurement_sigma = float(self.get_parameter('measurement_sigma').value)
        sampled_indices = self.get_scan_sample_indices()

        if not sampled_indices:
            return 1.0

        log_weight = 0.0
        valid_measurement_count = 0

        for beam_index in sampled_indices:
            measured_range = self.latest_scan.ranges[beam_index]

            if math.isinf(measured_range) or math.isnan(measured_range):
                continue
            if measured_range < self.latest_scan.range_min or measured_range > self.latest_scan.range_max:
                continue

            beam_angle_world = particle.yaw + self.latest_scan.angle_min + beam_index * self.latest_scan.angle_increment

            predicted_range = simulate_ray_cast(
                self.map_data,
                particle.x,
                particle.y,
                beam_angle_world,
                self.latest_scan.range_max,
                step_size=self.map_data.map_resolution
            )

            range_error = measured_range - predicted_range
            log_weight += -0.5 * (range_error * range_error) / max(1e-6, measurement_sigma * measurement_sigma)
            valid_measurement_count += 1

        if valid_measurement_count == 0:
            return 1e-9

        return math.exp(log_weight / valid_measurement_count)

    def update_particle_weights_from_scan(self) -> None:
        total_weight = 0.0
        occupied_threshold = int(self.get_parameter('occupied_threshold').value)

        for particle in self.particles:
            cell_x, cell_y = self.map_data.world_to_cell(particle.x, particle.y)

            if not self.map_data.contains_cell(cell_x, cell_y) or not self.map_data.is_free_cell(cell_x, cell_y, occupied_threshold):
                particle.weight = 1e-12
            else:
                particle.weight = max(1e-12, self.compute_particle_likelihood(particle))

            total_weight += particle.weight

        if total_weight <= 0.0:
            self.initialize_particles_uniformly()
            return

        for particle in self.particles:
            particle.weight /= total_weight

        # Kidnap detection: after normalisation the average weight is always 1/N,
        # so we use the maximum normalised weight instead. If even the best particle
        # is very close to uniform (max_weight ≈ 1/N), the filter has lost track.
        max_weight = max(p.weight for p in self.particles)
        kidnap_threshold = float(self.get_parameter('kidnap_weight_threshold').value)

        if max_weight < kidnap_threshold:
            self.get_logger().warn(
                f'Kidnap detected (max normalised weight={max_weight:.2e} < {kidnap_threshold:.2e}). '
                'Injecting recovery particles.'
            )
            self._inject_recovery_particles()

    def _inject_recovery_particles(self) -> None:
        """Replace a large fraction of the worst particles with random ones spread
        across the free space, while keeping the best ones to avoid discarding a
        correct hypothesis entirely."""
        occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        recovery_ratio = float(self.get_parameter('kidnap_recovery_ratio').value)

        free_cells = np.argwhere(
            (self.map_data.occupancy_data >= 0) &
            (self.map_data.occupancy_data < occupied_threshold)
        )
        if len(free_cells) == 0:
            return

        num_to_replace = int(len(self.particles) * recovery_ratio)

        # Sort ascending by weight so the weakest particles are replaced first
        self.particles.sort(key=lambda p: p.weight)

        for i in range(num_to_replace):
            cell_y, cell_x = free_cells[random.randrange(len(free_cells))]
            world_x, world_y = self.map_data.cell_to_world(int(cell_x), int(cell_y))
            self.particles[i] = ParticleState(
                world_x, world_y,
                random.uniform(-math.pi, math.pi),
                1.0 / len(self.particles)
            )

        # Re-normalise weights uniformly so no single particle dominates
        uniform_weight = 1.0 / len(self.particles)
        for particle in self.particles:
            particle.weight = uniform_weight

    def resample_particles(self) -> None:
        particle_count = len(self.particles)
        if particle_count == 0:
            return

        weights = [particle.weight for particle in self.particles]
        cumulative_weights = np.cumsum(weights)

        resampling_step = 1.0 / particle_count
        start_offset = random.uniform(0.0, resampling_step)
        current_index = 0

        random_particle_ratio = float(self.get_parameter('random_particle_ratio').value)
        occupied_threshold = int(self.get_parameter('occupied_threshold').value)

        free_cells = np.argwhere(
            (self.map_data.occupancy_data >= 0) &
            (self.map_data.occupancy_data < occupied_threshold)
        )

        new_particles: List[ParticleState] = []

        for sample_index in range(particle_count):
            if random.random() < random_particle_ratio and len(free_cells) > 0:
                cell_y, cell_x = free_cells[random.randrange(len(free_cells))]
                world_x, world_y = self.map_data.cell_to_world(int(cell_x), int(cell_y))
                new_particles.append(
                    ParticleState(world_x, world_y, random.uniform(-math.pi, math.pi), 1.0 / particle_count)
                )
                continue

            sample_value = start_offset + sample_index * resampling_step

            while current_index < particle_count - 1 and sample_value > cumulative_weights[current_index]:
                current_index += 1

            source_particle = self.particles[current_index]
            new_particles.append(
                ParticleState(source_particle.x, source_particle.y, source_particle.yaw, 1.0 / particle_count)
            )

        self.particles = new_particles

    def estimate_robot_pose(self) -> Optional[Tuple[float, float, float, np.ndarray]]:
        if not self.particles:
            return None

        normalized_weights = np.array([particle.weight for particle in self.particles], dtype=float)

        if normalized_weights.sum() <= 0:
            normalized_weights = np.ones(len(self.particles)) / len(self.particles)
        else:
            normalized_weights = normalized_weights / normalized_weights.sum()

        particle_x = np.array([particle.x for particle in self.particles])
        particle_y = np.array([particle.y for particle in self.particles])
        cos_yaw = np.array([math.cos(particle.yaw) for particle in self.particles])
        sin_yaw = np.array([math.sin(particle.yaw) for particle in self.particles])

        mean_x = float(np.sum(normalized_weights * particle_x))
        mean_y = float(np.sum(normalized_weights * particle_y))
        mean_yaw = math.atan2(
            float(np.sum(normalized_weights * sin_yaw)),
            float(np.sum(normalized_weights * cos_yaw))
        )

        deviations = np.vstack([
            particle_x - mean_x,
            particle_y - mean_y,
            np.array([normalize_angle(particle.yaw - mean_yaw) for particle in self.particles])
        ]).T

        covariance_matrix = np.zeros((3, 3), dtype=float)
        for weight, deviation in zip(normalized_weights, deviations):
            covariance_matrix += weight * np.outer(deviation, deviation)

        return mean_x, mean_y, mean_yaw, covariance_matrix

    def yaw_to_quaternion(self, yaw: float) -> Quaternion:
        quaternion = Quaternion()
        quaternion.z = math.sin(yaw / 2.0)
        quaternion.w = math.cos(yaw / 2.0)
        return quaternion

    def publish_filter_state(self) -> None:
        if not self.particles:
            return

        particle_array_msg = PoseArray()
        particle_array_msg.header.frame_id = 'map'
        particle_array_msg.header.stamp = self.get_clock().now().to_msg()

        for particle in self.particles:
            pose = Pose()
            pose.position.x = particle.x
            pose.position.y = particle.y
            pose.orientation = self.yaw_to_quaternion(particle.yaw)
            particle_array_msg.poses.append(pose)

        self.particles_publisher.publish(particle_array_msg)

        estimated_pose = self.estimate_robot_pose()
        if estimated_pose is None:
            return

        mean_x, mean_y, mean_yaw, covariance_matrix = estimated_pose

        pose_with_cov_msg = PoseWithCovarianceStamped()
        pose_with_cov_msg.header.frame_id = 'map'
        pose_with_cov_msg.header.stamp = self.get_clock().now().to_msg()
        pose_with_cov_msg.pose.pose.position.x = mean_x
        pose_with_cov_msg.pose.pose.position.y = mean_y
        pose_with_cov_msg.pose.pose.orientation = self.yaw_to_quaternion(mean_yaw)

        flat_covariance = [0.0] * 36
        flat_covariance[0] = float(covariance_matrix[0, 0])
        flat_covariance[1] = float(covariance_matrix[0, 1])
        flat_covariance[5] = float(covariance_matrix[0, 2])
        flat_covariance[6] = float(covariance_matrix[1, 0])
        flat_covariance[7] = float(covariance_matrix[1, 1])
        flat_covariance[11] = float(covariance_matrix[1, 2])
        flat_covariance[30] = float(covariance_matrix[2, 0])
        flat_covariance[31] = float(covariance_matrix[2, 1])
        flat_covariance[35] = float(covariance_matrix[2, 2])

        pose_with_cov_msg.pose.covariance = flat_covariance
        self.pose_publisher.publish(pose_with_cov_msg)

        pose_stamped_msg = PoseStamped()
        pose_stamped_msg.header = pose_with_cov_msg.header
        pose_stamped_msg.pose = pose_with_cov_msg.pose.pose
        self.pose_stamped_publisher.publish(pose_stamped_msg)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = mean_x
        t.transform.translation.y = mean_y
        t.transform.translation.z = 0.0
        t.transform.rotation = self.yaw_to_quaternion(mean_yaw)
        self.tf_broadcaster.sendTransform(t)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ParticleFilterLocalizationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()