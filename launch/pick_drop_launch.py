#!/usr/bin/env python3
"""
mission_launch.py
-----------------
Launch file para la misión completa del Puzzlebot.

Argumentos:
  pickup_x, pickup_y    : coordenadas del punto de carga   (default: 2.0, 0.0)
  dropoff_x, dropoff_y  : coordenadas del punto de descarga (default: 2.0, 2.0)
  bug_algorithm         : 'bug0' o 'bug2'                  (default: 'bug2')
  calib_file            : ruta al archivo calib.pckl        (default: ~/calib.pckl)
  target_id             : ID del ArUco de carga             (default: 0)
  use_flip              : flip horizontal de cámara         (default: true)
  aruco_map             : mapa de ArUcos para EKF (JSON)

Uso:
  # Con Bug2 (default):
  ros2 launch pick_drop_nav mission_launch.py \
    pickup_x:=2.0 pickup_y:=0.0 dropoff_x:=2.0 dropoff_y:=2.0

  # Con Bug0:
  ros2 launch pick_drop_nav mission_launch.py \
    bug_algorithm:=bug0 pickup_x:=2.0 pickup_y:=0.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    bug_alg = LaunchConfiguration('bug_algorithm').perform(context)

    # Nodo de navegación — bug0 o bug2 según parámetro
    bug_node = Node(
        package='pick_drop_nav',
        executable=bug_alg,
        name='bug_algorithm',
        parameters=[{'standalone': False}]
    )

    nodes = [

        # 1. Detector ArUco
        # Node(
        #     package='aruco_detector',
        #     executable='aruco_detector_node',
        #     name='aruco_detector_node',
        #     parameters=[{
        #         'calib_file':    LaunchConfiguration('calib_file').perform(context),
        #         'target_id':     int(LaunchConfiguration('target_id').perform(context)),
        #         'use_flip':      LaunchConfiguration('use_flip').perform(context).lower() == 'true',
        #         'marker_length': 0.10,
        #         'frame_skip':    2,
        #         'publish_image': False,
        #     }]
        # ),

        # 2. EKF Localización
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

        # 3. Navegación (bug0 o bug2)
        bug_node,

        # 4. Center & Approach
        Node(
            package='pick_drop_nav',
            executable='center_and_approach',
            name='center_and_approach',
            parameters=[{
                'standalone':    False,
                'stop_dist':     0.15,
                'brake_margin':  0.10,
                'w_min':         0.10,
                'yaw_thr':       0.20,
            }]
        ),

        # Node(
        #     package='pick_drop_nav',
        #     executable='center2',
        #     name='center2',
        #     parameters=[{
        #         'standalone':    False,
        #         'stop_dist':     0.15,
        #         'brake_margin':  0.10,
        #         'w_min':         0.10,
        #         'yaw_thr':       0.20,
        #         'perp_dist':     0.50,   # distancia del punto perp al ArUco
        #         'perp_tol':      0.12,   # tolerancia llegada al punto perp
        #         'alpha_thr':     0.08,   # ~4.5° para orienting
        #         'k_angular':     1.2,    # ganancia orientación
        #     }]
        # ),

        # 5. Coordinador de misión
        Node(
            package='pick_drop_nav',
            executable='mission_coordinator',
            name='mission_coordinator',
            parameters=[{
                'pickup_x':         float(LaunchConfiguration('pickup_x').perform(context)),
                'pickup_y':         float(LaunchConfiguration('pickup_y').perform(context)),
                'dropoff_x':        float(LaunchConfiguration('dropoff_x').perform(context)),
                'dropoff_y':        float(LaunchConfiguration('dropoff_y').perform(context)),
                'nav_tolerance':    0.15,
                'ca_trigger_dist':  1.0,
                'servo_wait':       2.0,
            }]
        ),
    ]

    return nodes


def generate_launch_description():

    args = [
        DeclareLaunchArgument('pickup_x',       default_value='-1.45'),
        DeclareLaunchArgument('pickup_y',       default_value='0.0'),
        DeclareLaunchArgument('dropoff_x',      default_value='1.45'),
        DeclareLaunchArgument('dropoff_y',      default_value='0.0'),
        DeclareLaunchArgument('bug_algorithm',  default_value='bug2',
                              description='bug0 o bug2'),
        DeclareLaunchArgument('calib_file',     default_value='~/calib.pckl'),
        DeclareLaunchArgument('target_id',      default_value='12'),
        DeclareLaunchArgument('use_flip',       default_value='true'),

    ]

    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])
