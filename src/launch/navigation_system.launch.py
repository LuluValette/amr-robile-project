from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    global_planner = Node(
        package='amr_project_lvalet2s_yakter2s_part1',
        executable='global_planner_node',
        name='global_planner_node',
        output='screen',
        parameters=['config/navigation_params.yaml'],
    )

    local_navigation = Node(
        package='amr_project_lvalet2s_yakter2s_part1',
        executable='local_navigation_node',
        name='local_navigation_node',
        output='screen',
        parameters=['config/navigation_params.yaml'],
    )

    return LaunchDescription([
        global_planner,
        local_navigation
    ])