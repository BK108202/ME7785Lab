#!/usr/bin/env python3
import os
from glob import glob
from setuptools import setup

package_name = 'teamfoobar_goal'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        # Install package.xml.
        ('share/' + package_name, ['package.xml']),
        # Include the launch files.
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Include the resource files (if any).
        (os.path.join('share', package_name, 'resource'), glob('resource/*')),
        # Add the srv files.
        (os.path.join('share', package_name, 'srv'), glob('srv/*.srv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='ccanezo',
    author_email='ccanezo@todo.todo',
    maintainer='ccanezo',
    maintainer_email='ccanezo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'obstacle_detector = teamfoobar_goal.obstacle_detector:main',
            'sign_recognition = teamfoobar_goal.sign_recognition:main',
            'sign_classifier_server = teamfoobar_goal.sign_classifier_server:main',
            'waypoint_navigator = teamfoobar_goal.waypoint_navigator:main',
        ],
    },
)
