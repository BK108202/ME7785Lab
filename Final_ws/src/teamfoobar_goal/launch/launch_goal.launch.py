# Lab 4 03/06/2025 Testing Only Go to Goal

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='teamfoobar_goal',
            executable='obstacle_detector',
            name='obstacle_detector'
        ),
        Node(
            package='teamfoobar_goal',
            executable='sign_recognition',
            name='sign_recognition'
        ),
        Node(
            package='teamfoobar_goal',
            executable='waypoint_navigator',
            name='waypoint_navigator'
        )
    ])
