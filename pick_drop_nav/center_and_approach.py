#!/usr/bin/env python3
"""
center_and_approach.py
----------------------
Flujo interno:
  idle        — esperando ArUco
  centering   — gira hasta offset_x ≈ 0
  approaching — avanza hasta stop_dist
  aligning    — corrección final de yaw
  picking     — secuencia: bajar servo → avanzar → subir servo → retroceder
  depositing  — secuencia: bajar servo → retroceder → subir servo
  done        — publica ca_status='done' y espera reset

Integración con coordinador:
  Suscribe  /mission_state  String  — actúa en 'pick' o 'deposit'
  Publica   /ca_status      String  — 'idle' | 'running' | 'done'

Regla de cmd_vel:
  - Solo publica cuando está activamente ejecutando una fase
  - picking y depositing se ejecutan hasta el final aunque mission_state cambie
  - idle y done son silenciosos
"""

import rclpy, math
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32, String
from geometry_msgs.msg import Twist


class CenterAndApproach(Node):

    PICK_LOWER   = 'lower_servo'
    PICK_ADVANCE = 'advance'
    PICK_RAISE   = 'raise_servo'
    PICK_REVERSE = 'reverse'
    PICK_DONE    = 'pick_done'

    DEP_LOWER    = 'dep_lower_servo'
    DEP_REVERSE  = 'dep_reverse'
    DEP_RAISE    = 'dep_raise_servo'
    DEP_DONE     = 'dep_done'

    SERVO_DOWN = 55.0
    SERVO_UP   = -80.0
    PICK_DIST  = 0.25
    PICK_SPEED = 0.06
    SERVO_WAIT = 1.0

    def __init__(self):
        super().__init__('center_and_approach')

        self.declare_parameter('stop_dist',    0.15)
        self.declare_parameter('center_thr',   0.05)
        self.declare_parameter('yaw_thr',      0.3)
        self.declare_parameter('k_offset',    -0.3)
        self.declare_parameter('k_yaw',        0.8)
        self.declare_parameter('w_min',        0.10)
        self.declare_parameter('v_approach',   0.08)
        self.declare_parameter('brake_margin', 0.10)
        self.declare_parameter('timeout',      3.0)
        self.declare_parameter('standalone',   False)

        self.stop_dist    = self.get_parameter('stop_dist').value
        self.center_thr   = self.get_parameter('center_thr').value
        self.yaw_thr      = self.get_parameter('yaw_thr').value
        self.k_offset     = self.get_parameter('k_offset').value
        self.k_yaw        = self.get_parameter('k_yaw').value
        self.w_min        = self.get_parameter('w_min').value
        self.v_approach   = self.get_parameter('v_approach').value
        self.brake_margin = self.get_parameter('brake_margin').value
        self.timeout      = self.get_parameter('timeout').value
        self.standalone   = self.get_parameter('standalone').value

        self.create_subscription(
            Float32MultiArray, '/aruco_target', self._aruco_cb, 10)
        self.create_subscription(
            String, '/mission_state', self._mission_cb, 10)

        self.pub      = self.create_publisher(Twist,   '/cmd_vel',    10)
        self.pub_srv  = self.create_publisher(Float32, '/ServoAngle', 10)
        self.pub_stat = self.create_publisher(String,  '/ca_status',  10)

        self.create_timer(0.1, self._loop)

        # Estado ArUco
        self._offset_x = 0.0
        self._dist     = float('inf')
        self._yaw      = 0.0
        self._alpha_k  = 0.0
        self._stamp    = None
        self._yaw_buf  = []
        self._YAW_N    = 5

        # Estado máquina
        self._phase          = 'idle'
        self._mission_active = self.standalone
        self._deposit_active = False
        self._pick_offset_x  = 0.0
        self._pick_step      = None
        self._pick_step_start = None

        self.get_logger().info(
            f'Center & Approach | standalone={self.standalone} | '
            f'stop={self.stop_dist} m | w_min={self.w_min}')

    # ── Callbacks ─────────────────────────────────────────────────────────
    def _mission_cb(self, msg: String):
        was_pick    = self._mission_active
        was_deposit = self._deposit_active

        self._mission_active = (msg.data == 'pick')
        self._deposit_active = (msg.data == 'deposit')

        # Freno solo si la secuencia NO está en curso
        if not self._mission_active and was_pick and self._phase not in ('picking',):
            self._stop()

        if not self._deposit_active and was_deposit and self._phase not in ('depositing',):
            self._stop()

        # Activar pick
        if self._mission_active and not was_pick:
            self._phase = 'idle'
            self._yaw_buf.clear()
            self.get_logger().info('Misión PICK activada → idle')

        # Activar deposit — inicia secuencia inmediatamente
        if self._deposit_active and not was_deposit:
            self._phase           = 'depositing'
            self._pick_step       = self.DEP_LOWER
            self._pick_step_start = self._now()
            self.get_logger().info('Misión DEPOSIT activada → secuencia depósito')

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

        if self._phase == 'idle' and self._mission_active:
            self._phase = 'centering'
            self.get_logger().info(
                f'ArUco detectado → centering | '
                f'd={self._dist:.3f} m | offset={self._offset_x:+.3f}')

    def _aruco_visible(self):
        if self._stamp is None:
            return False
        return (self.get_clock().now() - self._stamp).nanoseconds / 1e9 < self.timeout

    # ── Helpers ───────────────────────────────────────────────────────────
    def _apply_w_min(self, w):
        if 0 < abs(w) < self.w_min:
            w = math.copysign(self.w_min, w)
        return w

    def _publish(self, v, w):
        msg = Twist()
        msg.linear.x  = float(v)
        msg.angular.z = float(w)
        self.pub.publish(msg)

    def _stop(self):
        self._publish(0.0, 0.0)

    def _set_servo(self, angle):
        msg = Float32()
        msg.data = float(angle)
        self.pub_srv.publish(msg)
        self.get_logger().info(f'Servo → {angle}°')

    def _pub_status(self, status: str):
        msg = String()
        msg.data = status
        self.pub_stat.publish(msg)

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    # ── Loop principal ────────────────────────────────────────────────────
    def _loop(self):

        # ── Fases autónomas — se ejecutan hasta el final sin importar
        #    mission_state. El coordinador espera ca_status='done'. ────────
        if self._phase == 'picking':
            self._pub_status('running')
            self._do_picking()
            return

        if self._phase == 'depositing':
            self._pub_status('running')
            self._do_depositing()
            return

        if self._phase == 'done':
            self._pub_status('done')   # silencioso en cmd_vel
            return

        # ── Fases controladas por mission_state ───────────────────────────
        active = self._mission_active or self._deposit_active or self.standalone
        if not active:
            self._pub_status('idle')
            return   # silencio total en cmd_vel

        if not self._aruco_visible() and self._phase not in ('idle',):
            self.get_logger().warn('ArUco perdido → idle')
            self._phase = 'idle'
            self._stop()

        self._pub_status('running' if self._phase != 'idle' else 'idle')

        if   self._phase == 'idle':        pass   # silencio
        elif self._phase == 'centering':   self._do_centering()
        elif self._phase == 'approaching': self._do_approaching()
        elif self._phase == 'aligning':    self._do_aligning()

    # ── Fase: centrar ─────────────────────────────────────────────────────
    def _do_centering(self):
        if abs(self._offset_x) < self.center_thr:
            self._phase = 'approaching'
            self.get_logger().info(
                f'Centrado → approaching | offset={self._offset_x:+.3f}')
            return
        w = -self.k_offset * self._offset_x
        w = self._apply_w_min(w)
        self._publish(0.0, w)
        self.get_logger().info(
            f'[centering] offset={self._offset_x:+.3f} w={w:+.3f}')

    # ── Fase: approach ────────────────────────────────────────────────────
    def _do_approaching(self):
        if self._dist <= self.stop_dist + self.brake_margin:
            self._stop()
            self._phase = 'aligning'
            self.get_logger().info(
                f'Distancia OK ({self._dist:.3f} m) → aligning')
            return
        w = -self.k_offset * self._offset_x * 0.8
        w = self._apply_w_min(w) if abs(self._offset_x) > self.center_thr else w
        remaining = self._dist - self.stop_dist
        v = max(0.06, min(self.v_approach, 0.5 * remaining))
        self._publish(v, w)
        self.get_logger().info(
            f'[approaching] d={self._dist:.3f} m | offset={self._offset_x:+.3f} '
            f'v={v:.3f} w={w:+.3f}')

    # ── Fase: alineación final ────────────────────────────────────────────
    def _do_aligning(self):
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

        yaw_weight = max(0.0, 1.0 - abs(self._offset_x) / 0.15)
        w = (-self.k_offset * self._offset_x
             + self.k_yaw * 0.3 * yaw_weight * self._yaw)
        w = max(-0.20, min(0.20, w))
        w = self._apply_w_min(w)
        self._publish(0.0, w)
        self.get_logger().info(
            f'[aligning] offset={self._offset_x:+.3f} '
            f'yaw={math.degrees(self._yaw):+.1f}° w={w:+.3f}')

    # ── Fase: picking ─────────────────────────────────────────────────────
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
                w = -self.k_offset * self._pick_offset_x * 5.0
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

    # ── Fase: depositing ──────────────────────────────────────────────────
    def _do_depositing(self):
        """
        1. Baja servo  (55°)  → suelta la carga
        2. Retrocede 0.20 m   → se aleja dejando la carga
        3. Sube servo  (-80°) → recoge el tenedor vacío
        """
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