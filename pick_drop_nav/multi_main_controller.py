#!/usr/bin/env python3
"""
multi_mission_coordinator.py
-----------------------------
Coordinador de misión para múltiples cargas ArUco.

Flujo por carga:
  navigate_to_pickup → pick → navigate_to_dropoff → deposit

Repite para cada ID de carga en orden ascendente, luego regresa al origen.

Parámetros nuevos respecto al coordinador original:
  target_ids   : lista de IDs de carga, e.g. "0,1,2"
  pickup_x/y   : punto de búsqueda de cargas (robot navega aquí para buscar)
"""

import rclpy, math, json
from rclpy.node import Node
from nav_msgs.msg import Odometry
from turtlesim.msg import Pose
from std_msgs.msg import String, Float32, Float32MultiArray


class MultiMissionCoordinator(Node):

    IDLE                = 'idle'
    NAVIGATE_TO_PICKUP  = 'navigate_to_pickup'
    PICK                = 'pick'
    NAVIGATE_TO_DROPOFF = 'navigate_to_dropoff'
    DEPOSIT             = 'deposit'
    NAVIGATE_TO_ORIGIN  = 'navigate_to_origin'
    DONE                = 'done'

    def __init__(self):
        super().__init__('mission_coordinator')

        self.declare_parameter('pickup_x',           2.0)
        self.declare_parameter('pickup_y',           0.0)
        self.declare_parameter('dropoff_x',          2.0)
        self.declare_parameter('dropoff_y',          2.0)
        self.declare_parameter('nav_tolerance',      0.25)
        self.declare_parameter('ca_trigger_dist',    1.0)
        self.declare_parameter('servo_wait',         2.0)
        self.declare_parameter('target_ids',         '0,1')   # IDs de carga separados por coma
        self.declare_parameter('aruco_timeout',      2.0)     # segundos para considerar ArUco visible

        self.pickup_x    = self.get_parameter('pickup_x').value
        self.pickup_y    = self.get_parameter('pickup_y').value
        self.dropoff_x   = self.get_parameter('dropoff_x').value
        self.dropoff_y   = self.get_parameter('dropoff_y').value
        self.nav_tol     = self.get_parameter('nav_tolerance').value
        self.ca_trig     = self.get_parameter('ca_trigger_dist').value
        self.srv_wait    = self.get_parameter('servo_wait').value
        self.aruco_tout  = self.get_parameter('aruco_timeout').value

        # Construir lista de cargas pendientes en orden ascendente
        raw_ids = self.get_parameter('target_ids').value
        self._pending_ids = sorted([int(x.strip()) for x in raw_ids.split(',')])
        self._current_id  = None   # ID de la carga que se está recogiendo ahora

        self.get_logger().info(
            f'Multi-Mission | cargas pendientes: {self._pending_ids} | '
            f'pickup=({self.pickup_x},{self.pickup_y}) | '
            f'dropoff=({self.dropoff_x},{self.dropoff_y})')

        # Publishers
        self.pub_target  = self.create_publisher(Pose,   '/target',         10)
        self.pub_state   = self.create_publisher(String, '/mission_state',  10)
        self.pub_ca_id   = self.create_publisher(String, '/ca_target_id',   10)  # ID activo → C&A

        # Subscribers
        self.create_subscription(Odometry,         '/odom',             self._odom_cb,   10)
        self.create_subscription(String,           '/ca_status',        self._ca_cb,     10)
        self.create_subscription(Float32MultiArray,'/aruco_targets',    self._aruco_cb,  10)

        # Estado interno
        self._state      = self.IDLE
        self._pose       = None
        self._ca_status  = 'idle'
        self._aruco_data = {}   # {id: {'dist': float, 'stamp': float}}

        self.create_timer(0.2, self._tick)

    # ── Callbacks ─────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        q   = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self._pose = [x, y, yaw]
        if self._state == self.IDLE:
            self.get_logger().info('Odometría recibida → iniciando misión')
            self._start_next_pickup()

    def _ca_cb(self, msg: String):
        self._ca_status = msg.data

    def _aruco_cb(self, msg: Float32MultiArray):
        """
        Formato: [id, offset_x, d_k, yaw, alpha_k,  id, offset_x, ...]
        Guarda distancia y timestamp por ID.
        """
        data = msg.data
        now  = self._now()
        for i in range(0, len(data) - 4, 5):
            aruco_id = int(data[i])
            d_k      = float(data[i + 2])
            self._aruco_data[aruco_id] = {'dist': d_k, 'stamp': now}

    # ── Lógica de selección ───────────────────────────────────────────────
    def _start_next_pickup(self):
        """Toma el siguiente ID pendiente y arranca la navegación."""
        if not self._pending_ids:
            self.get_logger().info('Todas las cargas recogidas → navigate_to_origin')
            self._transition(self.NAVIGATE_TO_ORIGIN)
            return

        self._current_id = self._pending_ids[0]   # menor ID primero
        self.get_logger().info(
            f'→ Recogiendo carga ID={self._current_id} | '
            f'restantes: {self._pending_ids}')
        self._pub_ca_id(self._current_id)
        self._transition(self.NAVIGATE_TO_PICKUP)

    def _aruco_visible(self, aruco_id: int) -> bool:
        if aruco_id not in self._aruco_data:
            return False
        return self._now() - self._aruco_data[aruco_id]['stamp'] < self.aruco_tout

    def _aruco_dist(self, aruco_id: int) -> float:
        if aruco_id not in self._aruco_data:
            return float('inf')
        return self._aruco_data[aruco_id]['dist']

    # ── Tick principal ─────────────────────────────────────────────────────
    def _tick(self):
        if self._pose is None:
            return

        self._publish_state()

        # ── Navegar al pickup ──────────────────────────────────────────────
        if self._state == self.NAVIGATE_TO_PICKUP:
            self._send_target(self.pickup_x, self.pickup_y)
            near    = self._dist_to(self.pickup_x, self.pickup_y) < self.ca_trig
            visible = self._aruco_visible(self._current_id)

            if near and visible:
                self.get_logger().info(
                    f'ArUco ID={self._current_id} visible a '
                    f'{self._aruco_dist(self._current_id):.2f} m → PICK')
                self._transition(self.PICK)
            elif near and not visible:
                self.get_logger().warn(
                    f'Cerca del pickup pero ID={self._current_id} no visible — esperando...',
                    throttle_duration_sec=2.0)

        # ── Pick ───────────────────────────────────────────────────────────
        elif self._state == self.PICK:
            if self._ca_status == 'go_to_perp':
                self._publish_state_override('go_to_perp')
                return
            elif self._ca_status == 'done':
                # Carga recogida — quitarla de pendientes
                if self._current_id in self._pending_ids:
                    self._pending_ids.remove(self._current_id)
                self.get_logger().info(
                    f'Carga ID={self._current_id} recogida ✓ | '
                    f'restantes: {self._pending_ids}')
                self._transition(self.NAVIGATE_TO_DROPOFF)
                return

        # ── Navegar al dropoff ─────────────────────────────────────────────
        elif self._state == self.NAVIGATE_TO_DROPOFF:
            self._send_target(self.dropoff_x, self.dropoff_y)
            if self._dist_to(self.dropoff_x, self.dropoff_y) < self.nav_tol:
                self._send_target(self._pose[0], self._pose[1])
                self._transition(self.DEPOSIT)

        # ── Depositar ──────────────────────────────────────────────────────
        elif self._state == self.DEPOSIT:
            if self._ca_status == 'done':
                # Después de depositar, ir por la siguiente carga o al origen
                self._ca_status = 'idle'   # reset para la próxima carga
                self._start_next_pickup()
                return

        # ── Regresar al origen ─────────────────────────────────────────────
        elif self._state == self.NAVIGATE_TO_ORIGIN:
            self._send_target(0.0, 0.0)
            if self._dist_to(0.0, 0.0) < self.nav_tol:
                self._transition(self.DONE)

        elif self._state == self.DONE:
            self.get_logger().info('MISIÓN COMPLETA ✓', throttle_duration_sec=10.0)

        self._publish_state()

    # ── Helpers ───────────────────────────────────────────────────────────
    def _transition(self, new_state: str):
        self.get_logger().info(f'MISIÓN: {self._state} → {new_state}')
        self._state = new_state
        self._ca_status = 'idle'   # reset ca_status en cada transición

    def _dist_to(self, x, y):
        dx = x - self._pose[0]
        dy = y - self._pose[1]
        return math.sqrt(dx*dx + dy*dy)

    def _send_target(self, x, y):
        msg   = Pose()
        msg.x = float(x)
        msg.y = float(y)
        self.pub_target.publish(msg)

    def _pub_ca_id(self, aruco_id: int):
        msg      = String()
        msg.data = str(aruco_id)
        self.pub_ca_id.publish(msg)

    def _publish_state(self):
        msg      = String()
        msg.data = self._state
        self.pub_state.publish(msg)

    def _publish_state_override(self, state: str):
        msg      = String()
        msg.data = state
        self.pub_state.publish(msg)

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9


def main(args=None):
    rclpy.init(args=args)
    node = MultiMissionCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()