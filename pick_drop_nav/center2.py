#!/usr/bin/env python3
"""
center_and_approach.py
----------------------
Flujo mejorado con punto perpendicular en frame mundo:

  idle           — esperando ArUco
  go_to_perp     — calcula punto perpendicular en coords mundo y navega con bug
  orienting      — sobre la línea perpendicular, gira hasta alpha_k ≈ 0
  approaching    — avanza recto hasta stop_dist
  aligning       — corrección final de yaw en sitio
  picking        — bajar servo → avanzar → subir servo → retroceder
  depositing     — bajar servo → retroceder → subir servo
  done           — publica ca_status='done'

El punto perpendicular se calcula UNA VEZ al detectar el ArUco y se fija
en coordenadas del mundo usando /odom. El nodo bug navega a ese punto.
Una vez llegado, gira para orientarse y hace el approach final.

Integración:
  Suscribe  /mission_state   String         — actúa en 'pick' o 'deposit'
  Suscribe  /aruco_target    Float32MultiArray — [offset_x, d_k, yaw, alpha_k]
  Suscribe  /odom            Odometry
  Publica   /ca_status       String         — 'idle'|'running'|'done'
  Publica   /target          Pose           — punto perpendicular para bug
  Publica   /cmd_vel         Twist
  Publica   /ServoAngle      Float32
"""

import rclpy, math
from rclpy.node import Node
from nav_msgs.msg import Odometry
from turtlesim.msg import Pose
from std_msgs.msg import Float32MultiArray, Float32, String
from geometry_msgs.msg import Twist


class CenterAndApproach(Node):

    # ── Pasos picking ──────────────────────────────────────────────────────
    PICK_LOWER   = 'lower_servo'
    PICK_ADVANCE = 'advance'
    PICK_RAISE   = 'raise_servo'
    PICK_REVERSE = 'reverse'
    PICK_DONE    = 'pick_done'

    # ── Pasos depositing ───────────────────────────────────────────────────
    DEP_LOWER   = 'dep_lower_servo'
    DEP_REVERSE = 'dep_reverse'
    DEP_RAISE   = 'dep_raise_servo'
    DEP_DONE    = 'dep_done'

    SERVO_DOWN = 55.0
    SERVO_UP   = -80.0
    PICK_DIST  = 0.20
    PICK_SPEED = 0.06
    SERVO_WAIT = 1.0

    def __init__(self):
        super().__init__('center_and_approach')

        self.declare_parameter('stop_dist',    0.15)   # distancia final al ArUco
        self.declare_parameter('perp_dist',    0.50)   # distancia del punto perp al ArUco
        self.declare_parameter('perp_tol',     0.12)   # tolerancia llegada punto perp
        self.declare_parameter('center_thr',   0.05)   # umbral offset_x
        self.declare_parameter('yaw_thr',      0.20)   # umbral yaw [rad]
        self.declare_parameter('alpha_thr',    0.08)   # umbral alpha_k para orienting [rad] ~4.5°
        self.declare_parameter('k_offset',    -1.0)
        self.declare_parameter('k_yaw',        0.8)
        self.declare_parameter('k_angular',    1.2)    # ganancia orientación
        self.declare_parameter('w_min',        0.10)
        self.declare_parameter('v_approach',   0.08)
        self.declare_parameter('brake_margin', 0.10)
        self.declare_parameter('timeout',      3.0)
        self.declare_parameter('standalone',   False)

        self.stop_dist    = self.get_parameter('stop_dist').value
        self.perp_dist    = self.get_parameter('perp_dist').value
        self.perp_tol     = self.get_parameter('perp_tol').value
        self.center_thr   = self.get_parameter('center_thr').value
        self.yaw_thr      = self.get_parameter('yaw_thr').value
        self.alpha_thr    = self.get_parameter('alpha_thr').value
        self.k_offset     = self.get_parameter('k_offset').value
        self.k_yaw        = self.get_parameter('k_yaw').value
        self.k_angular    = self.get_parameter('k_angular').value
        self.w_min        = self.get_parameter('w_min').value
        self.v_approach   = self.get_parameter('v_approach').value
        self.brake_margin = self.get_parameter('brake_margin').value
        self.timeout      = self.get_parameter('timeout').value
        self.standalone   = self.get_parameter('standalone').value

        # ── Subscripciones ─────────────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray, '/aruco_target', self._aruco_cb, 10)
        self.create_subscription(
            String, '/mission_state', self._mission_cb, 10)
        self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)

        # ── Publishers ─────────────────────────────────────────────────────
        self.pub_cmd  = self.create_publisher(Twist,   '/cmd_vel',    10)
        self.pub_tgt  = self.create_publisher(Pose,    '/target',     10)
        self.pub_srv  = self.create_publisher(Float32, '/ServoAngle', 10)
        self.pub_stat = self.create_publisher(String,  '/ca_status',  10)

        self.create_timer(0.1, self._loop)

        # ── Estado ArUco ────────────────────────────────────────────────────
        self._offset_x = 0.0
        self._dist     = float('inf')
        self._yaw      = 0.0
        self._alpha_k  = 0.0
        self._stamp    = None
        self._yaw_buf  = []
        self._YAW_N    = 5

        # ── Estado odometría ────────────────────────────────────────────────
        self._rx    = 0.0
        self._ry    = 0.0
        self._rtheta= 0.0

        # ── Punto perpendicular fijado en frame mundo ────────────────────────
        self._perp_wx = None   # coordenada x mundo
        self._perp_wy = None   # coordenada y mundo

        # ── Estado máquina ──────────────────────────────────────────────────
        self._phase          = 'idle'
        self._mission_active = self.standalone
        self._deposit_active = False
        self._pick_offset_x  = 0.0
        self._pick_step      = None
        self._pick_step_start = None

        self.get_logger().info(
            f'Center & Approach | standalone={self.standalone} | '
            f'stop={self.stop_dist} m | perp={self.perp_dist} m')

    # ── Callbacks ──────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        self._rx     = msg.pose.pose.position.x
        self._ry     = msg.pose.pose.position.y
        q            = msg.pose.pose.orientation
        self._rtheta = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z))

    def _mission_cb(self, msg: String):
        was_pick    = self._mission_active
        was_deposit = self._deposit_active

        self._mission_active = (msg.data == 'pick')
        self._deposit_active = (msg.data == 'deposit')

        # Freno solo si secuencia NO está en curso
        if not self._mission_active and was_pick and self._phase not in ('picking',):
            self._stop()
        if not self._deposit_active and was_deposit and self._phase not in ('depositing',):
            self._stop()

        # Activar pick — reset a idle
        if self._mission_active and not was_pick:
            self._phase    = 'idle'
            self._perp_wx  = None
            self._perp_wy  = None
            self._yaw_buf.clear()
            self.get_logger().info('Misión PICK activada → idle')

        # Activar deposit — inicia secuencia inmediatamente
        if self._deposit_active and not was_deposit:
            self._phase           = 'depositing'
            self._pick_step       = self.DEP_LOWER
            self._pick_step_start = self._now()
            self.get_logger().info('Misión DEPOSIT activada → depositing')

    def _aruco_cb(self, msg: Float32MultiArray):
        if len(msg.data) < 4:
            return
        self._offset_x = float(msg.data[0])
        self._dist     = float(msg.data[1])
        raw_yaw        = float(msg.data[2])
        self._alpha_k  = float(msg.data[3])
        self._stamp    = self.get_clock().now()

        self._yaw_buf.append(raw_yaw)
        if len(self._yaw_buf) > self._YAW_N:
            self._yaw_buf.pop(0)
        self._yaw = sum(self._yaw_buf) / len(self._yaw_buf)

        # Al detectar por primera vez en idle → calcular punto perp y fijar
        if self._phase == 'idle' and self._mission_active:
            self._compute_and_fix_perp()
            self._phase = 'go_to_perp'
            self.get_logger().info(
                f'ArUco detectado → go_to_perp | '
                f'perp_world=({self._perp_wx:.3f}, {self._perp_wy:.3f})')

    def _aruco_visible(self):
        if self._stamp is None:
            return False
        return (self.get_clock().now() - self._stamp).nanoseconds / 1e9 < self.timeout

    # ── Geometría ───────────────────────────────────────────────────────────
    def _compute_and_fix_perp(self):
        """
        Calcula el punto perpendicular en coordenadas del mundo y lo fija.
        Solo se llama una vez al inicio de go_to_perp.

        ArUco en mundo:
          ax = rx + d_k * cos(rθ + alpha_k)
          ay = ry + d_k * sin(rθ + alpha_k)

        Normal del ArUco (dirección hacia el robot):
          normal = rθ + alpha_k + π + yaw_aruco

        Punto perp a perp_dist del ArUco sobre esa normal:
          px = ax + perp_dist * cos(normal)
          py = ay + perp_dist * sin(normal)
        """
        ax = self._rx + self._dist * math.cos(self._rtheta + self._alpha_k)
        ay = self._ry + self._dist * math.sin(self._rtheta + self._alpha_k)

        normal = self._rtheta + self._alpha_k + math.pi + self._yaw
        self._perp_wx = ax + self.perp_dist * math.cos(normal)
        self._perp_wy = ay + self.perp_dist * math.sin(normal)

    # ── Helpers ────────────────────────────────────────────────────────────
    def _apply_w_min(self, w):
        if 0 < abs(w) < self.w_min:
            w = math.copysign(self.w_min, w)
        return w

    def _dist_to_perp(self):
        if self._perp_wx is None:
            return float('inf')
        dx = self._perp_wx - self._rx
        dy = self._perp_wy - self._ry
        return math.sqrt(dx*dx + dy*dy)

    def _publish(self, v, w):
        msg = Twist()
        msg.linear.x  = float(v)
        msg.angular.z = float(w)
        self.pub_cmd.publish(msg)

    def _stop(self):
        self._publish(0.0, 0.0)

    def _send_target(self, x, y):
        msg   = Pose()
        msg.x = float(x)
        msg.y = float(y)
        self.pub_tgt.publish(msg)

    def _set_servo(self, angle):
        msg      = Float32()
        msg.data = float(angle)
        self.pub_srv.publish(msg)
        self.get_logger().info(f'Servo → {angle}°')

    def _pub_status(self, status: str):
        msg      = String()
        msg.data = status
        self.pub_stat.publish(msg)

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    # ── Loop principal ──────────────────────────────────────────────────────
    def _loop(self):
        # Fases autónomas — se ejecutan hasta el final sin importar mission_state
        if self._phase == 'picking':
            self._pub_status('running')
            self._do_picking()
            return

        if self._phase == 'depositing':
            self._pub_status('running')
            self._do_depositing()
            return

        if self._phase == 'done':
            self._pub_status('done')
            return

        # Fases controladas por mission_state
        active = self._mission_active or self._deposit_active or self.standalone
        if not active:
            self._pub_status('idle')
            return

        if not self._aruco_visible() and self._phase not in ('idle', 'go_to_perp'):
            self.get_logger().warn('ArUco perdido → idle')
            self._phase   = 'idle'
            self._perp_wx = None
            self._perp_wy = None
            self._stop()

        self._pub_status('running' if self._phase != 'idle' else 'idle')

        if   self._phase == 'idle':        pass
        elif self._phase == 'go_to_perp':  self._do_go_to_perp()
        elif self._phase == 'orienting':   self._do_orienting()
        elif self._phase == 'approaching': self._do_approaching()
        elif self._phase == 'aligning':    self._do_aligning()

    # ── Fase 1: ir al punto perpendicular ───────────────────────────────────
    def _do_go_to_perp(self):
        """
        Publica el punto perpendicular como /target para que bug lo navegue.
        Cuando el robot llega dentro de perp_tol, pasa a orienting.
        """
        if self._perp_wx is None:
            return

        d = self._dist_to_perp()

        # Publicar target para bug continuamente
        self._send_target(self._perp_wx, self._perp_wy)

        if d < self.perp_tol:
            self._stop()
            # Dejar de enviar target al bug — poner target en posición actual
            self._send_target(self._rx, self._ry)
            self._phase = 'orienting'
            self.get_logger().info(
                f'Punto perp alcanzado (d={d:.3f} m) → orienting')
        else:
            self.get_logger().info(
                f'[go_to_perp] d={d:.3f} m → '
                f'perp=({self._perp_wx:.3f},{self._perp_wy:.3f})')

    # ── Fase 2: orientarse de frente al ArUco ───────────────────────────────
    def _do_orienting(self):
        """
        Gira en sitio usando alpha_k hasta quedar de frente al ArUco.
        alpha_k ≈ 0 → robot mirando directo al ArUco.
        """
        if not self._aruco_visible():
            self.get_logger().warn('ArUco no visible en orienting — esperando')
            self._stop()
            return

        alpha_err = math.atan2(math.sin(self._alpha_k),
                               math.cos(self._alpha_k))

        if abs(alpha_err) < self.alpha_thr:
            self._stop()
            self._phase = 'approaching'
            self.get_logger().info(
                f'Orientado → approaching | '
                f'alpha={math.degrees(self._alpha_k):+.1f}°')
            return

        w = self.k_angular * alpha_err
        w = self._apply_w_min(w)
        self._publish(0.0, w)
        self.get_logger().info(
            f'[orienting] alpha={math.degrees(self._alpha_k):+.1f}° w={w:+.3f}')

    # ── Fase 3: avanzar recto al ArUco ──────────────────────────────────────
    def _do_approaching(self):
        if self._dist <= self.stop_dist + self.brake_margin:
            self._stop()
            self._phase = 'aligning'
            self.get_logger().info(
                f'Distancia OK ({self._dist:.3f} m) → aligning')
            return

        # Corrección suave de alpha durante approach
        alpha_err = math.atan2(math.sin(self._alpha_k),
                               math.cos(self._alpha_k))
        w = self.k_angular * alpha_err * 0.4
        w = self._apply_w_min(w) if abs(alpha_err) > self.alpha_thr else 0.0

        remaining = self._dist - self.stop_dist
        v = max(0.06, min(self.v_approach, 0.5 * remaining))
        self._publish(v, w)
        self.get_logger().info(
            f'[approaching] d={self._dist:.3f} m | '
            f'alpha={math.degrees(self._alpha_k):+.1f}° v={v:.3f} w={w:+.3f}')

    # ── Fase 4: alineación final ─────────────────────────────────────────────
    def _do_aligning(self):
        """Corrección final de offset y yaw con el robot quieto."""
        offset_ok = abs(self._offset_x) < self.center_thr
        yaw_ok    = abs(self._yaw)      < self.yaw_thr

        if offset_ok and yaw_ok:
            self._stop()
            self._pick_offset_x   = self._offset_x
            self._phase           = 'picking'
            self._pick_step       = self.PICK_LOWER
            self._pick_step_start = self._now()
            self.get_logger().info(
                f'Alineado → picking | offset_frozen={self._pick_offset_x:+.3f}')
            return

        # Corrección simultánea con peso adaptativo
        yaw_weight = max(0.0, 1.0 - abs(self._offset_x) / 0.15)
        w = (-self.k_offset * self._offset_x
             + self.k_yaw * 0.3 * yaw_weight * self._yaw)
        w = max(-0.20, min(0.20, w))
        w = self._apply_w_min(w)
        self._publish(0.0, w)
        self.get_logger().info(
            f'[aligning] offset={self._offset_x:+.3f} '
            f'yaw={math.degrees(self._yaw):+.1f}° w={w:+.3f}')

    # ── Fase 5: picking ───────────────────────────────────────────────────────
    def _do_picking(self):
        now     = self._now()
        elapsed = now - self._pick_step_start

        if self._pick_step == self.PICK_LOWER:
            self._stop()
            self._set_servo(self.SERVO_DOWN)
            if elapsed >= self.SERVO_WAIT:
                self._pick_step       = self.PICK_ADVANCE
                self._pick_step_start = now
                self.get_logger().info('Servo abajo → avanzando')

        elif self._pick_step == self.PICK_ADVANCE:
            dist_done = elapsed * self.PICK_SPEED
            if dist_done >= self.PICK_DIST:
                self._stop()
                self._pick_step       = self.PICK_RAISE
                self._pick_step_start = now
                self.get_logger().info('Avance completo → subiendo servo')
            else:
                w = -self.k_offset * self._pick_offset_x * 0.3
                self._publish(self.PICK_SPEED, w)

        elif self._pick_step == self.PICK_RAISE:
            self._stop()
            self._set_servo(self.SERVO_UP)
            if elapsed >= self.SERVO_WAIT:
                self._pick_step       = self.PICK_REVERSE
                self._pick_step_start = now
                self.get_logger().info('Servo arriba → retrocediendo')

        elif self._pick_step == self.PICK_REVERSE:
            dist_done = elapsed * self.PICK_SPEED
            if dist_done >= self.PICK_DIST:
                self._stop()
                self._phase     = 'done'
                self._pick_step = self.PICK_DONE
                self.get_logger().info('CAJA RECOGIDA ✓')
            else:
                self._publish(-self.PICK_SPEED, 0.0)

    # ── Fase 6: depositing ────────────────────────────────────────────────────
    def _do_depositing(self):
        now     = self._now()
        elapsed = now - self._pick_step_start

        if self._pick_step == self.DEP_LOWER:
            self._stop()
            self._set_servo(self.SERVO_DOWN)
            if elapsed >= self.SERVO_WAIT:
                self._pick_step       = self.DEP_REVERSE
                self._pick_step_start = now
                self.get_logger().info('Depósito: servo abajo → retrocediendo')

        elif self._pick_step == self.DEP_REVERSE:
            dist_done = elapsed * self.PICK_SPEED
            if dist_done >= self.PICK_DIST:
                self._stop()
                self._pick_step       = self.DEP_RAISE
                self._pick_step_start = now
                self.get_logger().info('Depósito: retroceso completo → subiendo servo')
            else:
                self._publish(-self.PICK_SPEED, 0.0)

        elif self._pick_step == self.DEP_RAISE:
            self._stop()
            self._set_servo(self.SERVO_UP)
            if elapsed >= self.SERVO_WAIT:
                self._pick_step = self.DEP_DONE
                self._phase     = 'done'
                self.get_logger().info('CARGA DEPOSITADA ✓')


def main(args=None):
    rclpy.init(args=args)
    node = CenterAndApproach()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()