from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_name = 'amr_project_lvalet2s_yakter2s'
    config_file = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'part2_params.yaml'
    )

    return LaunchDescription([
        Node(
            package=package_name,
            executable='particle_filter_localization',
            name='particle_filter_localization',
            output='screen',
            parameters=[config_file]
        )
    ])