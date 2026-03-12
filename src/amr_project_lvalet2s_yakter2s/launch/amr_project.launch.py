from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_name = 'amr_project_lvalet2s_yakter2s'
    pkg_share = get_package_share_directory(package_name)

    part1_config = os.path.join(pkg_share, 'config', 'part1_params.yaml')
    part2_config = os.path.join(pkg_share, 'config', 'part2_params.yaml')
    part3_config = os.path.join(pkg_share, 'config', 'part3_params.yaml')

    map_file = os.path.join(
        get_package_share_directory('robile_navigation'),
        'maps', 'closed_walls_map.yaml'
    )

    return LaunchDescription([
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_file, 'use_sim_time': False}]
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            output='screen',
            parameters=[{'autostart': True, 'node_names': ['map_server']}]
        ),
        Node(
            package=package_name,
            executable='particle_filter_localization',
            name='particle_filter_localization',
            parameters=[part2_config],
            output='screen',
        ),
        Node(
            package=package_name,
            executable='path_motion_planner',
            name='path_motion_planner',
            parameters=[part1_config],
            output='screen',
        ),
        Node(
            package=package_name,
            executable='frontier_explorer',
            name='frontier_explorer',
            parameters=[part3_config],
            output='screen',
        ),
    ])
