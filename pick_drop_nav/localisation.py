#!/usr/bin/env python3
"""
localisation.py  —  EKF con odometría + corrección ArUco
=========================================================
Paso de predicción : odometría de encoders  (L_k dinámica)
Paso de corrección : detecciones ArUco      (R_k con bias)

Mapa de ArUcos — parámetro 'aruco_map' en formato JSON:
  '{"0": [1.0, 0.0], "1": [2.0, 1.5], "2": [0.0, 2.0]}'
  clave = ID del marcador (string), valor = [x_mi, y_mi] en metros

Corrección de bias de la cámara (caracterizada experimentalmente):
  d_corr     = d_k / (1 + BIAS_D)   →  1.2246
  alpha_corr = alpha_k / (1 + BIAS_A) →  0.6105
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

        # Ruido de observación de la cámara (R_k fija — de la caracterización)
        self.declare_parameter('r_dd',             0.000016)   # σ²_d
        self.declare_parameter('r_aa',             0.000001)   # σ²_α
        self.declare_parameter('r_da',             0.0)        # covarianza cruzada

        # Mapa de ArUcos: JSON  {"id": [x_mi, y_mi], ...}
        self.declare_parameter('aruco_map',
                               '{"0": [1.0, 0.0]}')

        # ── Leer parámetros ──────────────────────────────────────────────
        self.rate         = self.get_parameter('rate').value
        child_frame       = self.get_parameter('child_frame_id').value
        cov_x             = self.get_parameter('cov_x').value
        cov_y             = self.get_parameter('cov_y').value
        cov_yaw           = self.get_parameter('cov_yaw').value
        self.K_R          = self.get_parameter('K_R').value
        self.K_L          = self.get_parameter('K_L').value
        self.use_clock_topic = self.get_parameter('use_clock_topic').value

        r_dd = self.get_parameter('r_dd').value
        r_aa = self.get_parameter('r_aa').value
        r_da = self.get_parameter('r_da').value
        self.Rk = np.matrix([[r_dd, r_da],
                             [r_da, r_aa]])

        # Mapa ArUcos: {int_id: np.array([x_mi, y_mi])}
        raw_map = json.loads(self.get_parameter('aruco_map').value)
        self.aruco_map = {int(k): np.array(v) for k, v in raw_map.items()}
        self.get_logger().info(f'Mapa ArUcos cargado: {self.aruco_map}')

        # Advertencia si el aruco_map tiene múltiples marcadores — OK
        # Advertencia si aruco_map está vacío
        if not self.aruco_map:
            self.get_logger().warn(
                'aruco_map está vacío — el EKF solo usará odometría sin corrección ArUco')
        else:
            self.get_logger().info(
                f'{len(self.aruco_map)} ArUco(s) de localización registrados: '
                f'{list(self.aruco_map.keys())} | '
                f'Cualquier otro ID detectado será ignorado automáticamente.')

        # ── Bias de cámara (caracterizado experimentalmente) ─────────────
        self.BIAS_D = 0.2246    # d_corr     = d_k / 1.2246
        self.BIAS_A = -0.3895   # alpha_corr = alpha_k / 0.6105

        # ── Estado inicial ───────────────────────────────────────────────
        self.x     = self.get_parameter('x0').value
        self.y     = self.get_parameter('y0').value
        self.theta = self.get_parameter('theta0').value

        self.Ek = np.matrix([[cov_x, 0.0,   0.0    ],
                             [0.0,   cov_y, 0.0    ],
                             [0.0,   0.0,   cov_yaw]])
        self.Lk = np.zeros((2, 2))

        # ── ROS — publishers / subscribers ──────────────────────────────
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.create_subscription(Float32, 'VelocityEncR',
                                 self._enc_r_cb, qos_vel)
        self.create_subscription(Float32, 'VelocityEncL',
                                 self._enc_l_cb, qos_vel)
        self.create_subscription(ClockMsg, '/clock',
                                 self._clock_cb, 10)
        self.create_subscription(Float32MultiArray, '/aruco_detections',
                                 self._aruco_cb, 10)

        # Buffer de detecciones ArUco — se procesa en timer_callback
        self._aruco_buf = []   # lista de (id, d_k, alpha_k)

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

        # ── Odometry msg ─────────────────────────────────────────────────
        self.odom = Odometry()
        self.odom.header.frame_id = 'world'
        self.odom.child_frame_id  = child_frame

        self.create_timer(1.0 / self.rate, self.timer_callback)

        src = '/clock topic' if self.use_clock_topic else 'wall clock'
        self.get_logger().info(
            f'Localisation+EKF iniciado | time={src} | '
            f'K_R={self.K_R:.5f} K_L={self.K_L:.5f} | '
            f'ArUcos={list(self.aruco_map.keys())}')

    # ── Callbacks de sensores ────────────────────────────────────────────
    def _enc_r_cb(self, msg):  self.wr = msg.data
    def _enc_l_cb(self, msg):  self.wl = msg.data
    def _clock_cb(self, msg):
        self.clock_ns = msg.clock.sec * 1_000_000_000 + msg.clock.nanosec

    def _aruco_cb(self, msg: Float32MultiArray):
        """Acumula detecciones ArUco en el buffer para procesarlas en el timer."""
        data = msg.data
        detections = []
        for i in range(0, len(data) - 2, 3):
            marker_id = int(data[i])
            d_k       = float(data[i + 1])
            alpha_k   = float(data[i + 2])
            if marker_id in self.aruco_map:
                detections.append((marker_id, d_k, alpha_k))
        self._aruco_buf = detections   # reemplaza — solo usamos el frame más reciente

    # ── Timer principal ──────────────────────────────────────────────────
    def timer_callback(self):

        # ── Tiempo ───────────────────────────────────────────────────────
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
        # PASO 1 — PREDICCIÓN  (odometría + L_k dinámica)
        # ════════════════════════════════════════════════════════════════
        self.Lk = np.matrix([
            [self.K_R * abs(self.wr), 0.0                    ],
            [0.0,                     self.K_L * abs(self.wl)],
        ])

        v = (self.wr + self.wl) * self.r / 2.0
        w = (self.wr - self.wl) * self.r / self.L

        # Jacobiano del modelo de movimiento respecto al estado
        Hk = np.matrix([
            [1, 0, -dt * v * math.sin(self.theta)],
            [0, 1,  dt * v * math.cos(self.theta)],
            [0, 0,  1],
        ])

        # Jacobiano del modelo de movimiento respecto a la entrada (ruido)
        Fk = 0.5 * self.r * dt * np.matrix([
            [math.cos(self.theta),  math.cos(self.theta)],
            [math.sin(self.theta),  math.sin(self.theta)],
            [2.0 / self.L,         -2.0 / self.L        ],
        ])

        # Covarianza predicha
        self.Ek = Hk * self.Ek * Hk.T + Fk * self.Lk * Fk.T

        # Estado predicho  μ_k⁻
        self.x     += v * math.cos(self.theta) * dt
        self.y     += v * math.sin(self.theta) * dt
        self.theta += w * dt

        # ════════════════════════════════════════════════════════════════
        # PASO 2 — CORRECCIÓN  (ArUco — una iteración por marcador)
        # ════════════════════════════════════════════════════════════════
        for (marker_id, d_k_raw, alpha_k_raw) in self._aruco_buf:

            # -- Corrección de bias de la cámara -------------------------
            d_k     = d_k_raw     / (1.0 + self.BIAS_D)   # / 1.2246
            alpha_k = alpha_k_raw / (1.0 + self.BIAS_A)   # / 0.6105

            # -- Posición conocida del ArUco en el mundo -----------------
            mi = self.aruco_map[marker_id]
            x_mi, y_mi = mi[0], mi[1]

            # -- Diferencias de posición (usando estado predicho μ_k⁻) ---
            dx = x_mi - self.x
            dy = y_mi - self.y
            d2 = dx**2 + dy**2
            d  = math.sqrt(d2)

            if d < 1e-6:   # evitar división por cero
                continue

            # -- Modelo de observación esperado  y_k = g(m_i, μ_k⁻) -----
            d_pred     =  d
            alpha_pred =  math.atan2(dy, dx) - self.theta

            # Normalizar ángulo a [-π, π]
            alpha_pred = math.atan2(math.sin(alpha_pred),
                                    math.cos(alpha_pred))

            # -- Jacobiano G_k  (2×3) — de la diapositiva ----------------
            #
            #  G_k = [ -dx/d        -dy/d         0  ]
            #        [  dy/d²       -dx/d²        -1  ]
            #
            Gk = np.matrix([
                [-dx / d,   -dy / d,   0.0],
                [ dy / d2,  -dx / d2, -1.0],
            ])

            # -- Ganancia de Kalman K_k -----------------------------------
            S  = Gk * self.Ek * Gk.T + self.Rk          # innovación covarianza
            Kk = self.Ek * Gk.T * np.linalg.inv(S)      # 3×2

            # -- Innovación  z_ik - g(m_i, μ_k⁻) ------------------------
            innov_d     = d_k - d_pred
            innov_alpha = alpha_k - alpha_pred
            innov_alpha = math.atan2(math.sin(innov_alpha),
                                     math.cos(innov_alpha))   # normalizar

            innov = np.matrix([[innov_d],
                               [innov_alpha]])

            # -- Actualización del estado --------------------------------
            delta  = Kk * innov
            self.x     += float(delta[0])
            self.y     += float(delta[1])
            self.theta += float(delta[2])
            self.theta  = math.atan2(math.sin(self.theta),
                                     math.cos(self.theta))

            # -- Actualización de la covarianza --------------------------
            I3         = np.eye(3)
            self.Ek    = (I3 - Kk * Gk) * self.Ek

            self.get_logger().debug(
                f'ArUco {marker_id} | d={d_k:.3f} α={math.degrees(alpha_k):.1f}° | '
                f'Δx={float(delta[0]):+.4f} Δy={float(delta[1]):+.4f} '
                f'Δθ={math.degrees(float(delta[2])):+.2f}°')

        # Limpiar buffer tras procesar
        self._aruco_buf = []

        # ════════════════════════════════════════════════════════════════
        # PUBLICAR  odom + TF
        # ════════════════════════════════════════════════════════════════
        q = quaternion_from_euler(0, 0, self.theta)

        self.tf_msg.header.stamp            = now_stamp
        self.tf_msg.transform.translation.x = self.x
        self.tf_msg.transform.translation.y = self.y
        self.tf_msg.transform.translation.z = 0.0
        self.tf_msg.transform.rotation.x    = q[0]
        self.tf_msg.transform.rotation.y    = q[1]
        self.tf_msg.transform.rotation.z    = q[2]
        self.tf_msg.transform.rotation.w    = q[3]

        self.odom.header.stamp              = now_stamp
        self.odom.pose.pose.position.x      = self.x
        self.odom.pose.pose.position.y      = self.y
        self.odom.pose.pose.position.z      = 0.0
        self.odom.pose.pose.orientation.x   = q[0]
        self.odom.pose.pose.orientation.y   = q[1]
        self.odom.pose.pose.orientation.z   = q[2]
        self.odom.pose.pose.orientation.w   = q[3]

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
