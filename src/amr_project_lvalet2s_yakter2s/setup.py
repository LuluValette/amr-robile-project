from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'amr_project_lvalet2s_yakter2s'

setup(
    name=package_name,
    version='1.0.2',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py'),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_motion_planner = amr_project_lvalet2s_yakter2s.path_motion_planner:main',
            'particle_filter_localization = amr_project_lvalet2s_yakter2s.particle_filter_localization:main',
            'frontier_explorer = amr_project_lvalet2s_yakter2s.frontier_explorer:main',
            'goal_sender_node = amr_project_lvalet2s_yakter2s.goal_sender_node:main',
        ],
    },
)
