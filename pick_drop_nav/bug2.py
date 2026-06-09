#!/usr/bin/env python3
"""
bug2.py — Bug2 con integración al coordinador de misión.

Cambios respecto al standalone:
  - Suscribe /mission_state: solo actúa en estados de navegación
    ('navigate_to_pickup' | 'navigate_to_dropoff' | 'navigate_to_origin')
  - En cualquier otro estado se detiene y no publica cmd_vel
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
        super().__init__('bug2_algorithm')
        self.get_logger().info('Bug2 Algorithm node has been initialized.')

        self.declare_parameter('standalone', False)
        self.standalone = self.get_parameter('standalone').value

        self.create_timer(0.1, self.state_machine)
        self.create_subscription(Odometry, 'odom',    self.odom_callback,    10)
        self.create_subscription(Pose,     'target',  self.target_callback,  10)
        self.create_subscription(LaserScan,'scan',    self._scan_cb,         10)
        self.create_subscription(String,   '/mission_state', self._mission_cb, 10)

        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self._wall_side = None
        self._nav_active = self.standalone   # si standalone, siempre activo

        self.state = 'stop_robot'
        self.k_linear   = 0.2
        self.k_angular  = 0.5
        self.turn_until = 1/8 * math.pi
        self.current_pose   = []
        self.target_pose    = []
        self.got_new_target = False
        self.twist_msg = Twist()
        self._data = [float('inf')] * 6

        self._ml_a = 0.0
        self._ml_b = 0.0
        self._ml_c = 0.0
        self._dist_at_wall_entry = float('inf')
        self._mline_epsilon   = 0.15
        self._progress_margin = 0.10

    # ── Callbacks ─────────────────────────────────────────────────────────
    def _mission_cb(self, msg: String):
        was_active       = self._nav_active
        self._nav_active = self.standalone or (msg.data in self.NAV_STATES)

        if not self._nav_active and was_active:
            # Acaba de desactivarse → frenar el robot una vez
            self.stop_robot()
        # Reset estado interno cuando se reactiva la navegación
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

    # ── Bug2 ──────────────────────────────────────────────────────────────
    def _compute_mline(self):
        x0, y0 = self.current_pose[0], self.current_pose[1]
        xT, yT = self.target_pose[0],  self.target_pose[1]
        self._ml_a =  yT - y0
        self._ml_b = -(xT - x0)
        self._ml_c =  xT * y0 - yT * x0
        self.get_logger().info(
            f'M-line: {self._ml_a:.3f}x + {self._ml_b:.3f}y + {self._ml_c:.3f} = 0')

    def _dist_to_mline(self):
        denom = math.sqrt(self._ml_a**2 + self._ml_b**2)
        if denom < 1e-9:
            return 0.0
        x, y = self.current_pose[0], self.current_pose[1]
        return abs(self._ml_a*x + self._ml_b*y + self._ml_c) / denom

    def _dist_to_goal(self):
        if not self.current_pose or not self.target_pose:
            return float('inf')
        dx = self.target_pose[0] - self.current_pose[0]
        dy = self.target_pose[1] - self.current_pose[1]
        return math.sqrt(dx**2 + dy**2)

    def mlineAgainWithProgress(self):
        on_mline = self._dist_to_mline() < self._mline_epsilon
        progress = self._dist_to_goal() < (self._dist_at_wall_entry - self._progress_margin)
        front_ok = self._data[0] > 0.5
        return on_mline and progress and front_ok

    def _choose_wall_side(self):
        angle_to_goal = math.atan2(
            self.target_pose[1] - self.current_pose[1],
            self.target_pose[0] - self.current_pose[0])
        angle_error = angle_to_goal - self.current_pose[2]
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        return 'right' if angle_error < 0 else 'left'

    # ── Máquina de estados ────────────────────────────────────────────────
    def state_machine(self):
        if len(self.current_pose) == 0:
            return

        # Si el coordinador no activa navegación → detenerse
        if not self._nav_active:
            # No publicar nada — dejar que el nodo activo controle cmd_vel
            return

        # Transiciones
        if self.state == 'go_to_goal' and self.atTarget():
            self.state = 'stop_robot'
            self.get_logger().info('STATE → stop_robot (at target)')

        if self.state == 'stop_robot' and self.gotNewTarget():
            self._compute_mline()
            self.state = 'go_to_goal'
            self.get_logger().info('STATE → go_to_goal')

        if self.state == 'go_to_goal' and self.is_obstacle_ahead_to_close():
            self._wall_side               = self._choose_wall_side()
            self._dist_at_wall_entry      = self._dist_to_goal()
            self.state = 'follow_wall'
            self.get_logger().info(
                f'STATE → follow_wall (side={self._wall_side})')

        if self.state == 'follow_wall' and self.mlineAgainWithProgress():
            self.state = 'go_to_goal'
            self.get_logger().info('STATE → go_to_goal (m-line progress)')

        if self.state == 'follow_wall' and self.atTarget():
            self.state = 'stop_robot'
            self.get_logger().info('STATE → stop_robot (at target from wall)')

        # Acciones
        if self.state == 'stop_robot':   self.stop_robot()
        elif self.state == 'go_to_goal': self.go_to_goal()
        elif self.state == 'follow_wall':self.follow_wall()

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

    def follow_wall(self):
        front, front_left, _, _, _, front_right = self._data
        dist_to_wall = 0.3
        tol          = 0.05

        if self._wall_side == 'right':
            if front < 0.4:                          v, w = 0.0,  0.5
            elif front_right < dist_to_wall - tol:   v, w = 0.1,  0.33
            elif front_right > dist_to_wall + tol:   v, w = 0.1, -0.33
            else:                                    v, w = 0.1,  0.0
        else:
            if front < 0.4:                         v, w = 0.0, -0.5
            elif front_left < dist_to_wall - tol:   v, w = 0.1, -0.33
            elif front_left > dist_to_wall + tol:   v, w = 0.1,  0.33
            else:                                   v, w = 0.1,  0.0
        self.move_robot(v, w)

    def go_to_goal(self):
        err_x = self.target_pose[0] - self.current_pose[0]
        err_y = self.target_pose[1] - self.current_pose[1]
        desired_heading  = math.atan2(err_y, err_x)
        err_theta        = desired_heading - self.current_pose[2]
        err_theta        = math.atan2(math.sin(err_theta), math.cos(err_theta))
        angular_velocity = self.k_angular * err_theta
        if abs(err_theta) > self.turn_until:
            self.move_robot(0.0, angular_velocity)
        else:
            self.move_robot(0.1, angular_velocity)

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