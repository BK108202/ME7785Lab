# Lab 4 03/06/2025 Testing only go to goal
from setuptools import find_packages, setup

package_name = 'teamfoobar_navigate_to_goal'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name +'/launch', ['launch/launch_navigate_to_goal.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ccanezo',
    maintainer_email='ccanezo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        # 'go_to_goal = teamfoobar_navigate_to_goal.go_to_goal:main',
        'get_object_range = teamfoobar_navigate_to_goal.get_object_range:main',
        'avoid_obstacle = teamfoobar_navigate_to_goal.avoid_obstacle:main',
        # 'state_manager = teamfoobar_navigate_to_goal.state_manager:main',
        ],
    },
)
