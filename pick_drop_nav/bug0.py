#!/usr/bin/env python3
"""
bug0.py — Bug0 con integración al coordinador de misión.

Bug0: sigue la pared hasta que el camino al objetivo esté libre.

Integración con coordinador:
  - Suscribe /mission_state: solo actúa en estados de navegación
    ('navigate_to_pickup' | 'navigate_to_dropoff' | 'navigate_to_origin')
  - Parámetro 'standalone': si True ignora mission_state y siempre navega
"""

import rclpy, math
from rclpy.node import Node
from nav_msgs.msg import Odometry
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import numpy as np


class BugAlgorithm(Node):

    SECTOR_LIMITS = [-158.0, -112.0, -68.0, 68.0, 112.0, 158.0]
    MIN_VALID     = 0.12

    NAV_STATES = {
        'navigate_to_pickup',
        'navigate_to_dropoff',
        'navigate_to_origin',
    }

    def __init__(self):
        super().__init__('bug_algorithm')
        self.get_logger().info('Bug0 Algorithm node has been initialized.')

        self.declare_parameter('standalone', False)
        self.standalone  = self.get_parameter('standalone').value
        self._nav_active = self.standalone

        self.create_timer(0.1, self.state_machine)
        self.create_subscription(Odometry,  'odom',           self.odom_callback,   10)
        self.create_subscription(Pose,      'target',         self.target_callback, 10)
        self.create_subscription(LaserScan, 'scan',           self._scan_cb,        10)
        self.create_subscription(String,    '/mission_state', self._mission_cb,     10)

        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self._wall_side     = None
        self.state          = 'stop_robot'
        self.k_linear       = 0.2
        self.k_angular      = 1.0
        self.turn_until     = 7/8 * math.pi
        self.current_pose   = []
        self.target_pose    = []
        self.got_new_target = False
        self.twist_msg      = Twist()
        self._data          = [float('inf')] * 6

    # ── Callbacks ─────────────────────────────────────────────────────────
    def _mission_cb(self, msg: String):
        was_active       = self._nav_active
        self._nav_active = self.standalone or (msg.data in self.NAV_STATES)
        if not self._nav_active and was_active:
            # Acaba de desactivarse → frenar el robot una vez
            self.stop_robot()
        if self._nav_active and not was_active and self.target_pose:
            self.got_new_target = True

    def _scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=msg.range_max,
                               posinf=msg.range_max, neginf=msg.range_max)
        ranges = np.where(ranges < self.MIN_VALID, msg.range_max, ranges)

        angles_deg = (np.degrees(msg.angle_min)
                      + np.arange(len(ranges)) * np.degrees(msg.angle_increment))
        s = self.SECTOR_LIMITS
        mask_fl = (angles_deg >  s[0]) & (angles_deg <= s[1])
        mask_l  = (angles_deg >  s[1]) & (angles_deg <= s[2])
        mask_b  = (angles_deg >  s[2]) & (angles_deg <= s[3])
        mask_r  = (angles_deg >  s[3]) & (angles_deg <= s[4])
        mask_fr = (angles_deg >  s[4]) & (angles_deg <= s[5])
        mask_f  = (angles_deg >  s[5]) | (angles_deg <= s[0])

        def safe_min(mask):
            vals = ranges[mask]
            return float(np.min(vals)) if vals.size > 0 else float('inf')

        self._data = [safe_min(mask_f),  safe_min(mask_fl), safe_min(mask_l),
                      safe_min(mask_b),  safe_min(mask_r),  safe_min(mask_fr)]

    def odom_callback(self, msg):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = self.get_yaw_from_quaternion(msg.pose.pose.orientation)
        self.current_pose = [x, y, yaw]

    def target_callback(self, msg):
        new_target = [msg.x, msg.y, msg.theta]
        if len(self.target_pose) == 0 or new_target != self.target_pose:
            self.target_pose    = new_target
            self.got_new_target = True
            self.get_logger().info(f'New target received: {self.target_pose}')

    # ── Máquina de estados ────────────────────────────────────────────────
    def state_machine(self):
        if len(self.current_pose) == 0:
            return

        if not self._nav_active:
            # No publicar nada — dejar que el nodo activo controle cmd_vel
            return

        # Transiciones
        if self.state == 'go_to_goal' and self.atTarget():
            self.state = 'stop_robot'
            self.got_new_target = False

        if self.state == 'stop_robot' and self.gotNewTarget():
            self.state = 'go_to_goal'

        if self.state == 'go_to_goal' and self.is_obstacle_ahead_to_close():
            self._wall_side = self._choose_wall_side()
            self.state = 'follow_wall'

        if self.state == 'follow_wall' and self.isWallCleared():
            self.state = 'go_to_goal'

        if self.state == 'follow_wall' and self.atTarget():
            self.state = 'stop_robot'
            self.got_new_target = False

        # Acciones
        if   self.state == 'stop_robot':  self.stop_robot()
        elif self.state == 'go_to_goal':  self.go_to_goal()
        elif self.state == 'follow_wall': self.follow_wall()

    # ── Comportamientos ───────────────────────────────────────────────────
    def atTarget(self):
        if not self.current_pose or not self.target_pose:
            return False
        dx = self.target_pose[0] - self.current_pose[0]
        dy = self.target_pose[1] - self.current_pose[1]
        return math.sqrt(dx**2 + dy**2) < 0.1

    def gotNewTarget(self):
        if self.got_new_target:
            self.got_new_target = False
            return True
        return False

    def is_obstacle_ahead_to_close(self):
        return self._data[0] < 0.4

    def isWallCleared(self):
        if not self.current_pose or not self.target_pose:
            return False
        Ex  = self.target_pose[0] - self.current_pose[0]
        Ey  = self.target_pose[1] - self.current_pose[1]
        Eq  = math.atan2(Ey, Ex) - self.current_pose[2]
        Eq  = math.atan2(math.sin(Eq), math.cos(Eq))
        return self._data[0] > 0.8 and abs(Eq) < 0.2

    def _choose_wall_side(self):
        if not self.current_pose or not self.target_pose:
            return 'left'
        angle_to_goal = math.atan2(
            self.target_pose[1] - self.current_pose[1],
            self.target_pose[0] - self.current_pose[0])
        angle_error = angle_to_goal - self.current_pose[2]
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        return 'right' if angle_error < 0 else 'left'

    def follow_wall(self):
        front, front_left, _, _, _, front_right = self._data
        dist_to_wall = 0.3
        tol          = 0.05

        if self._wall_side == 'right':
            if front < 0.4:                          v, w = 0.0,  0.5
            elif front_right < dist_to_wall - tol:   v, w = 0.1,  0.33
            elif front_right > dist_to_wall + tol:   v, w = 0.1, -0.33
            else:                                    v, w = 0.2,  0.0
        else:
            if front < 0.4:                         v, w = 0.0, -0.5
            elif front_left < dist_to_wall - tol:   v, w = 0.1, -0.33
            elif front_left > dist_to_wall + tol:   v, w = 0.1,  0.33
            else:                                   v, w = 0.2,  0.0
        self.move_robot(v, w)

    def go_to_goal(self):
        if not self.current_pose or not self.target_pose:
            return
        err_x = self.target_pose[0] - self.current_pose[0]
        err_y = self.target_pose[1] - self.current_pose[1]
        desired_heading  = math.atan2(err_y, err_x)
        err_theta        = desired_heading - self.current_pose[2]
        err_theta        = math.atan2(math.sin(err_theta), math.cos(err_theta))
        angular_velocity = self.k_angular * err_theta
        if abs(err_theta) > self.turn_until:
            self.move_robot(0.0, angular_velocity)
        else:
            self.move_robot(0.2, angular_velocity)

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def move_robot(self, v, w):
        self.twist_msg.linear.x  = v
        self.twist_msg.angular.z = w
        self.cmd_vel_publisher.publish(self.twist_msg)

    def stop_robot(self):
        self.move_robot(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = BugAlgorithm()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()