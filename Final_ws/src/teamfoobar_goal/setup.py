from setuptools import find_packages, setup

package_name = 'teamfoobar_goal'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    include_package_data=True,  # Ensure package data specified in package_data is included.
    package_data={
        # This will include knn_model.xml from the teamfoobar_goal package directory.
        package_name: ['knn_model.xml'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/launch_goal.launch.py']),
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
            'sign_recognition = teamfoobar_goal.sign_recognition:main',
        ],
    },
)
