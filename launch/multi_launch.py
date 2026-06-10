#!/usr/bin/env python3
"""
multi_mission_launch.py
-----------------------
Launch para misión con múltiples cargas ArUco.

Uso:
  ros2 launch pick_drop_nav multi_mission_launch.py \
    target_ids:="0,1" \
    pickup_x:=2.0 pickup_y:=0.0 \
    dropoff_x:=2.0 dropoff_y:=2.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    bug_alg    = LaunchConfiguration('bug_algorithm').perform(context)
    target_ids = LaunchConfiguration('target_ids').perform(context)

    return [
        # # 1. Detector ArUco multi-carga
        # Node(
        #     package='pick_drop_nav',
        #     executable='multi_aruco_detector_node',
        #     name='aruco_detector_node',
        #     parameters=[{
        #         'calib_file':    LaunchConfiguration('calib_file').perform(context),
        #         'target_ids':    target_ids,
        #         'use_flip':      LaunchConfiguration('use_flip').perform(context).lower() == 'true',
        #         'marker_length': 0.10,
        #         'frame_skip':    2,
        #         'publish_image': False,
        #     }]
        # ),

        # 2. EKF Localización (sin cambios)
        Node(
            package='pick_drop_nav',
            executable='localisation',
            name='localisation',
            parameters=[{
                'use_clock_topic': False,
                'K_R':    0.1015834066,
                'K_L':    0.1110887664,
                'r_dd':   0.1,
                'r_aa':   0.1,
                'aruco_map': '{"10": [-1.6, 0.0], "11": [1.6, 0.0], "13": [0.0, 1.3], "14": [0.0, -1.3]}',  # <-- editar IDs y coords aquí
            }]
        ),

        # 3. Bug2
        Node(
            package='pick_drop_nav',
            executable=bug_alg,
            name='bug_algorithm',
            parameters=[{'standalone': False}]
        ),

        # 4. Center & Approach multi-carga
        Node(
            package='pick_drop_nav',
            executable='multi_center_and_approach',
            name='center_and_approach',
            parameters=[{
                'standalone':    False,
                'stop_dist':     0.15,
                'brake_margin':  0.10,
                'w_min':         0.10,
                'yaw_thr':       0.20,
                'k_offset':     -0.3,
            }]
        ),

        # 5. Coordinador multi-carga
        Node(
            package='pick_drop_nav',
            executable='multi_mission_coordinator',
            name='mission_coordinator',
            parameters=[{
                'pickup_x':        float(LaunchConfiguration('pickup_x').perform(context)),
                'pickup_y':        float(LaunchConfiguration('pickup_y').perform(context)),
                'dropoff_x':       float(LaunchConfiguration('dropoff_x').perform(context)),
                'dropoff_y':       float(LaunchConfiguration('dropoff_y').perform(context)),
                'target_ids':      target_ids,
                'nav_tolerance':   0.15,
                'ca_trigger_dist': 1.0,
                'servo_wait':      2.0,
            }]
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('target_ids',     default_value='12'),
        DeclareLaunchArgument('pickup_x',       default_value='-1.45'),
        DeclareLaunchArgument('pickup_y',       default_value='0.0'),
        DeclareLaunchArgument('dropoff_x',      default_value='1.45'),
        DeclareLaunchArgument('dropoff_y',      default_value='0.0'),
        DeclareLaunchArgument('bug_algorithm',  default_value='bug2'),
        DeclareLaunchArgument('calib_file',     default_value='~/calib.pckl'),
        DeclareLaunchArgument('use_flip',       default_value='true'),
        OpaqueFunction(function=launch_setup),
    ])