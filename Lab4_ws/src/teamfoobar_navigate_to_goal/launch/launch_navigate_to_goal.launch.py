# Lab 4 03/06/2025 Testing Only Go to Goal

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='teamfoobar_navigate_to_goal',
            executable='avoid_obstacle',
            name='avoid_obstacle'
        ),
        Node(
            package='teamfoobar_navigate_to_goal',
            executable='get_object_range',
            name='get_object_range'
        ),
        Node(
            package='teamfoobar_navigate_to_goal',
            executable='state_manager',
            name='state_manager'
        ),
        Node(
            package='teamfoobar_navigate_to_goal',
            executable='go_to_goal',
            name='go_to_goal',
        )
    ])
