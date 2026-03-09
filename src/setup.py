from setuptools import find_packages, setup
from glob import glob

package_name = 'amr_project_lvalet2s_yakter2s_part1'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'global_planner_node = amr_project_lvalet2s_yakter2s_part1.global_planner_node:main',
            'local_navigation_node = amr_project_lvalet2s_yakter2s_part1.local_navigation_node:main',
            'goal_sender_node = amr_project_lvalet2s_yakter2s_part1.goal_sender_node:main',
        ],
    },
)