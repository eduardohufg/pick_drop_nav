#!/usr/bin/env python3
"""
localisation.py  —  EKF con odometría + corrección ArUco
=========================================================
Paso de predicción : odometría de encoders  (L_k dinámica, modelo exacto)
Paso de corrección : detecciones ArUco      (Float32MultiArray)

Mejoras respecto a versión anterior:
  - Modelo cinemático exacto: v/w*(sin(θ+w·dt)-sin(θ)) en lugar de aproximación
  - Parsing robusto de Float32MultiArray [id, d_k, alpha_k, ...]
  - Mismo filtro de aruco_map (solo IDs conocidos)

Mapa de ArUcos — parámetro 'aruco_map' en formato JSON:
  '{"1": [1.6, 0.0], "2": [0.0, 1.6]}'
"""

import rclpy, math, json
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32, Float32MultiArray
from rosgraph_msgs.msg import Clock as ClockMsg
from rclpy import qos
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
import numpy as np


class Localisation(Node):
    def __init__(self):
        super().__init__('localisation')

        qos_vel = qos.qos_profile_sensor_data

        # ── Parámetros ───────────────────────────────────────────────────
        self.declare_parameter('rate',             20.0)
        self.declare_parameter('child_frame_id',   'base_footprint')
        self.declare_parameter('x0',               0.0)
        self.declare_parameter('y0',               0.0)
        self.declare_parameter('theta0',           0.0)
        self.declare_parameter('cov_x',            0.0)
        self.declare_parameter('cov_y',            0.0)
        self.declare_parameter('cov_yaw',          0.0)
        self.declare_parameter('K_R',              0.01262)
        self.declare_parameter('K_L',              0.01332)
        self.declare_parameter('use_clock_topic',  False)

        # Ruido de observación de la cámara
        self.declare_parameter('r_dd',             0.01)    # σ²_d
        self.declare_parameter('r_aa',             0.01)    # σ²_α

        # Mapa de ArUcos: JSON {"id": [x_mi, y_mi], ...}
        self.declare_parameter('aruco_map', '{"0": [1.0, 0.0]}')

        # ── Leer parámetros ──────────────────────────────────────────────
        self.rate            = self.get_parameter('rate').value
        child_frame          = self.get_parameter('child_frame_id').value
        cov_x                = self.get_parameter('cov_x').value
        cov_y                = self.get_parameter('cov_y').value
        cov_yaw              = self.get_parameter('cov_yaw').value
        self.K_R             = self.get_parameter('K_R').value
        self.K_L             = self.get_parameter('K_L').value
        self.use_clock_topic = self.get_parameter('use_clock_topic').value

        r_dd = self.get_parameter('r_dd').value
        r_aa = self.get_parameter('r_aa').value
        self.Rk = np.array([[r_dd, 0.0 ],
                             [0.0,  r_aa]], dtype=float)

        raw_map = json.loads(self.get_parameter('aruco_map').value)
        self.aruco_map = {int(k): np.array(v) for k, v in raw_map.items()}

        if not self.aruco_map:
            self.get_logger().warn(
                'aruco_map vacío — EKF solo usará odometría')
        else:
            self.get_logger().info(
                f'{len(self.aruco_map)} ArUco(s) de localización: '
                f'{list(self.aruco_map.keys())} | '
                f'IDs desconocidos ignorados automáticamente.')

        # ── Estado inicial ───────────────────────────────────────────────
        self.x     = self.get_parameter('x0').value
        self.y     = self.get_parameter('y0').value
        self.theta = self.get_parameter('theta0').value

        self.Ek = np.zeros((3, 3), dtype=float)
        self.Ek[0, 0] = cov_x
        self.Ek[1, 1] = cov_y
        self.Ek[2, 2] = cov_yaw

        self.Lk = np.zeros((2, 2), dtype=float)

        # ── ROS ──────────────────────────────────────────────────────────
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)

        self.create_subscription(Float32, 'VelocityEncR',
                                 self._enc_r_cb, qos_vel)
        self.create_subscription(Float32, 'VelocityEncL',
                                 self._enc_l_cb, qos_vel)
        self.create_subscription(ClockMsg, '/clock',
                                 self._clock_cb, 10)
        self.create_subscription(Float32MultiArray, '/aruco_detections',
                                 self._aruco_cb, 10)

        self._aruco_buf = []

        # ── TF ───────────────────────────────────────────────────────────
        self.tf_msg = TransformStamped()
        self.tf_msg.header.frame_id = 'world'
        self.tf_msg.child_frame_id  = child_frame
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Odometría ────────────────────────────────────────────────────
        self.wr = 0.0
        self.wl = 0.0
        self.r  = 0.05
        self.L  = 0.18

        # ── Tiempo ───────────────────────────────────────────────────────
        self.clock_ns = 0
        self.t0_ns    = 0
        self.t0       = self.get_clock().now()

        # ── Odom msg ─────────────────────────────────────────────────────
        self.odom = Odometry()
        self.odom.header.frame_id = 'world'
        self.odom.child_frame_id  = child_frame

        self.create_timer(1.0 / self.rate, self.timer_callback)

        src = '/clock topic' if self.use_clock_topic else 'wall clock'
        self.get_logger().info(
            f'Localisation+EKF | time={src} | '
            f'K_R={self.K_R:.5f} K_L={self.K_L:.5f} | '
            f'ArUcos={list(self.aruco_map.keys())}')

    # ── Callbacks ─────────────────────────────────────────────────────────
    def _enc_r_cb(self, msg): self.wr = msg.data
    def _enc_l_cb(self, msg): self.wl = msg.data

    def _clock_cb(self, msg):
        self.clock_ns = msg.clock.sec * 1_000_000_000 + msg.clock.nanosec

    def _aruco_cb(self, msg: Float32MultiArray):
        """
        Formato: [id0, d_k0, alpha_k0,  id1, d_k1, alpha_k1, ...]
        Solo acumula los IDs que están en aruco_map.
        """
        data = msg.data
        detections = []
        for i in range(0, len(data) - 2, 3):
            marker_id = int(data[i])
            d_k       = float(data[i + 1])
            alpha_k   = float(data[i + 2])
            if marker_id in self.aruco_map:
                detections.append((marker_id, d_k, alpha_k))
        self._aruco_buf = detections

    # ── Timer principal ────────────────────────────────────────────────────
    def timer_callback(self):

        # ── Tiempo ──────────────────────────────────────────────────────
        if self.use_clock_topic:
            if self.clock_ns == 0:
                return
            now_ns     = self.clock_ns
            dt         = (now_ns - self.t0_ns) / 1e9
            self.t0_ns = now_ns
            now_stamp  = Time(nanoseconds=now_ns).to_msg()
        else:
            now       = self.get_clock().now()
            dt        = (now - self.t0).nanoseconds / 1e9
            self.t0   = now
            now_stamp = now.to_msg()

        if dt <= 0.0 or dt > 1.0:
            return

        # ════════════════════════════════════════════════════════════════
        # PASO 1 — PREDICCIÓN
        # ════════════════════════════════════════════════════════════════
        v = (self.wr + self.wl) * self.r / 2.0
        w = (self.wr - self.wl) * self.r / self.L

        # L_k dinámica
        self.Lk = np.array([
            [self.K_R * abs(self.wr), 0.0                    ],
            [0.0,                     self.K_L * abs(self.wl)],
        ], dtype=float)

        # Jacobiano H_k respecto al estado
        Hk = np.array([
            [1.0, 0.0, -v * math.sin(self.theta) * dt],
            [0.0, 1.0,  v * math.cos(self.theta) * dt],
            [0.0, 0.0,  1.0],
        ], dtype=float)

        # Jacobiano F_k respecto a la entrada
        Fk = 0.5 * self.r * dt * np.array([
            [math.cos(self.theta),  math.cos(self.theta)],
            [math.sin(self.theta),  math.sin(self.theta)],
            [2.0 / self.L,         -2.0 / self.L        ],
        ], dtype=float)

        # Covarianza predicha
        self.Ek = Hk @ self.Ek @ Hk.T + Fk @ self.Lk @ Fk.T

        # ── Modelo cinemático EXACTO ─────────────────────────────────────
        if abs(w) > 1e-6:
            self.x += (v / w) * (math.sin(self.theta + w * dt) - math.sin(self.theta))
            self.y += (v / w) * (-math.cos(self.theta + w * dt) + math.cos(self.theta))
        else:
            self.x += v * math.cos(self.theta) * dt
            self.y += v * math.sin(self.theta) * dt
        self.theta += w * dt
        self.theta  = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # ════════════════════════════════════════════════════════════════
        # PASO 2 — CORRECCIÓN ArUco
        # ════════════════════════════════════════════════════════════════
        for (marker_id, d_k, alpha_k) in self._aruco_buf:
            mi     = self.aruco_map[marker_id]
            x_mi, y_mi = mi[0], mi[1]

            dx = x_mi - self.x
            dy = y_mi - self.y
            d2 = dx**2 + dy**2
            d  = math.sqrt(d2)

            if d < 1e-6:
                continue

            # Predicción de la medición
            d_pred     = d
            alpha_pred = math.atan2(dy, dx) - self.theta
            alpha_pred = math.atan2(math.sin(alpha_pred), math.cos(alpha_pred))

            # Jacobiano G_k (2×3)
            Gk = np.array([
                [-dx / d,   -dy / d,   0.0],
                [ dy / d2,  -dx / d2, -1.0],
            ], dtype=float)

            # Ganancia de Kalman
            S  = Gk @ self.Ek @ Gk.T + self.Rk
            try:
                Kk = self.Ek @ Gk.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                self.get_logger().error(
                    f'Matriz singular en corrección ArUco {marker_id}, saltando.')
                continue

            # Innovación
            innov_d     = d_k - d_pred
            innov_alpha = alpha_k - alpha_pred
            innov_alpha = math.atan2(math.sin(innov_alpha), math.cos(innov_alpha))

            innov = np.array([[innov_d], [innov_alpha]], dtype=float)

            # Actualización estado
            delta      = Kk @ innov
            self.x    += float(delta[0])
            self.y    += float(delta[1])
            self.theta += float(delta[2])
            self.theta  = math.atan2(math.sin(self.theta), math.cos(self.theta))

            # Actualización covarianza
            self.Ek = (np.eye(3) - Kk @ Gk) @ self.Ek

            self.get_logger().debug(
                f'ArUco {marker_id} | d={d_k:.3f} α={math.degrees(alpha_k):.1f}° | '
                f'Δx={float(delta[0]):+.4f} Δy={float(delta[1]):+.4f} '
                f'Δθ={math.degrees(float(delta[2])):+.2f}°')

        self._aruco_buf = []

        # ════════════════════════════════════════════════════════════════
        # PUBLICAR odom + TF
        # ════════════════════════════════════════════════════════════════
        q = quaternion_from_euler(0, 0, self.theta)

        self.tf_msg.header.stamp             = now_stamp
        self.tf_msg.transform.translation.x  = self.x
        self.tf_msg.transform.translation.y  = self.y
        self.tf_msg.transform.translation.z  = 0.0
        self.tf_msg.transform.rotation.x     = q[0]
        self.tf_msg.transform.rotation.y     = q[1]
        self.tf_msg.transform.rotation.z     = q[2]
        self.tf_msg.transform.rotation.w     = q[3]

        self.odom.header.stamp               = now_stamp
        self.odom.pose.pose.position.x       = self.x
        self.odom.pose.pose.position.y       = self.y
        self.odom.pose.pose.position.z       = 0.0
        self.odom.pose.pose.orientation.x    = q[0]
        self.odom.pose.pose.orientation.y    = q[1]
        self.odom.pose.pose.orientation.z    = q[2]
        self.odom.pose.pose.orientation.w    = q[3]

        self.odom.pose.covariance[0]  = self.Ek[0, 0]
        self.odom.pose.covariance[1]  = self.Ek[0, 1]
        self.odom.pose.covariance[5]  = self.Ek[0, 2]
        self.odom.pose.covariance[6]  = self.Ek[1, 0]
        self.odom.pose.covariance[7]  = self.Ek[1, 1]
        self.odom.pose.covariance[11] = self.Ek[1, 2]
        self.odom.pose.covariance[30] = self.Ek[2, 0]
        self.odom.pose.covariance[31] = self.Ek[2, 1]
        self.odom.pose.covariance[35] = self.Ek[2, 2]

        v_pub = (self.wr + self.wl) * self.r / 2.0
        w_pub = (self.wr - self.wl) * self.r / self.L
        self.odom.twist.twist.linear.x  = v_pub
        self.odom.twist.twist.angular.z = w_pub

        self.odom_pub.publish(self.odom)
        self.tf_broadcaster.sendTransform(self.tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Localisation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()