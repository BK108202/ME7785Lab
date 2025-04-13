from setuptools import find_packages, setup

package_name = 'teamfoobar_goal'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name +'/launch', ['launch/launch_goal.launch.py']),
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
        'obstacle_detector = teamfoobar_goal.obstacle_detector:main',
        'waypoint_navigator = teamfoobar_goal.waypoint_navigator:main',
        ],
    },
)
