import socket
import struct
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class Observation:
    timestamp: float
    image_data_main: bytes
    image_data_right: bytes
    joint_states: Iterable[float]
    task_name: str


class SocketPolicyClient:
    """Simple TCP client that mirrors the Unity SocketPolicyClient protocol.

    Protocol (big-endian):
    - timestamp: float32
    - main image length: uint32
    - main image bytes
    - right image length: uint32
    - right image bytes
    - joint count: uint32
    - joint states: float32 * count
    - task name length: uint32
    - task name bytes (UTF-8)

    Response:
    - steps: uint32
    - dims: uint32
    - action values: float32 * (steps * dims)
    """

    def __init__(self, host: str, port: int, timeout_s: Optional[float] = 2.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if timeout_s is not None:
            self._sock.settimeout(timeout_s)
        self._sock.connect((host, port))

    def close(self) -> None:
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()

    def __enter__(self) -> "SocketPolicyClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_action(
        self,
        image_data_main: bytes,
        image_data_right: bytes,
        joint_states: Iterable[float],
        timestamp: float,
        task_name: str,
    ) -> List[List[float]]:
        message = self._serialize_observation(
            timestamp, image_data_main, image_data_right, joint_states, task_name
        )
        self._send_message(message)
        response = self._receive_message()
        return self._parse_action_chunks(response)

    def get_action_chunks(
        self,
        image_data_main: bytes,
        image_data_right: bytes,
        joint_states: Iterable[float],
        timestamp: float,
        task_name: str,
        chunk_size: int,
    ) -> List[List[float]]:
        message = self._serialize_observation(
            timestamp, image_data_main, image_data_right, joint_states, task_name
        )
        self._send_message(message)
        response = self._receive_message()
        chunks = self._parse_action_chunks(response)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        return chunks[:chunk_size]

    def iter_action_chunks(
        self,
        image_data_main: bytes,
        image_data_right: bytes,
        joint_states: Iterable[float],
        timestamp: float,
        task_name: str,
        chunk_size: int,
    ):
        for chunk in self.get_action_chunks(
            image_data_main, image_data_right, joint_states, timestamp, task_name, chunk_size
        ):
            yield chunk

    def _serialize_observation(
        self,
        timestamp: float,
        image_data_main: bytes,
        image_data_right: bytes,
        joint_states: Iterable[float],
        task_name: str,
    ) -> bytes:
        joint_list = list(joint_states)
        task_bytes = task_name.encode("utf-8")
        parts = [
            struct.pack(">f", float(timestamp)),
            struct.pack(">I", len(image_data_main)),
            image_data_main,
            struct.pack(">I", len(image_data_right)),
            image_data_right,
            struct.pack(">I", len(joint_list)),
            b"".join(struct.pack(">f", float(v)) for v in joint_list),
            struct.pack(">I", len(task_bytes)),
            task_bytes,
        ]
        return b"".join(parts)

    def _parse_action_chunks(self, data: bytes) -> List[List[float]]:
        if len(data) < 8:
            return []
        (n_steps, n_dim) = struct.unpack(">II", data[:8])
        total_floats = n_steps * n_dim
        expected_size = 8 + (total_floats * 4)
        if len(data) < expected_size:
            return []
        floats = struct.unpack(f">{total_floats}f", data[8:expected_size])
        actions: List[List[float]] = []
        for i in range(n_steps):
            start = i * n_dim
            end = start + n_dim
            actions.append(list(floats[start:end]))
        return actions

    def _send_message(self, payload: bytes) -> None:
        header = struct.pack(">I", len(payload))
        self._sock.sendall(header + payload)

    def _receive_message(self) -> bytes:
        header = self._recv_exact(4)
        (length,) = struct.unpack(">I", header)
        return self._recv_exact(length)

    def _recv_exact(self, count: int) -> bytes:
        chunks = []
        received = 0
        while received < count:
            chunk = self._sock.recv(count - received)
            if not chunk:
                raise ConnectionError("Server disconnected")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)


def dummy_action_server(host: str = "127.0.0.1", port: int = 5555) -> None:
    """Minimal server for local testing. Returns 4 zeros as action."""

    def recv_exact(conn: socket.socket, count: int) -> bytes:
        chunks = []
        received = 0
        while received < count:
            chunk = conn.recv(count - received)
            if not chunk:
                raise ConnectionError("Client disconnected")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        conn, _ = srv.accept()
        with conn:
            while True:
                header = recv_exact(conn, 4)
                (length,) = struct.unpack(">I", header)
                _payload = recv_exact(conn, length)
                # Always respond with 1 step of 4 zeros
                response = struct.pack(">II", 1, 4) + struct.pack(">ffff", 0.0, 0.0, 0.0, 0.0)
                conn.sendall(struct.pack(">I", len(response)) + response)


def run_robot_loop():
    import time

    try:
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.so_follower.so_follower import SO101Follower
    except Exception as exc:  # pragma: no cover
        raise ImportError("LeRobot SO-101 classes not available") from exc

    robot_config = SO101FollowerConfig(port="/dev/arm_right")
    robot = SO101Follower(robot_config)
    robot.connect()

    motor_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    host = "100.85.166.124"
    port = 9000
    dt = 1.0 / 30.0

    def get_my_camera_image_pair():
        return b"", b""

    with SocketPolicyClient(host, port) as client:
        while True:
            loop_start = time.perf_counter()
            image_main, image_right = get_my_camera_image_pair()
            robot_obs = robot.get_observation()
            joint_states = [robot_obs[f"{name}.pos"] for name in motor_names]
            action_chunk = client.get_action(
                image_main, image_right, joint_states, timestamp=time.time(), task_name="task"
            )
            if not action_chunk:
                continue
            steps_to_execute = min(10, len(action_chunk))
            for i in range(steps_to_execute):
                step_start = time.perf_counter()
                action_values = action_chunk[i]
                action_dict = {f"{name}.pos": val for name, val in zip(motor_names, action_values)}
                robot.send_action(action_dict)
                step_duration = time.perf_counter() - step_start
                time.sleep(max(0.0, dt - step_duration))
            loop_duration = time.perf_counter() - loop_start
            if loop_duration < dt:
                time.sleep(dt - loop_duration)


if __name__ == "__main__":
    # Example usage (client)
    # with SocketPolicyClient("127.0.0.1", 5555) as client:
    #     actions = client.get_action(b"", b"", [0.0, 0.0, 0.0], 0.0, "task")
    #     print(actions)

    # Example usage (server)
    # dummy_action_server()

    # Example usage (robot control loop)
    run_robot_loop()
