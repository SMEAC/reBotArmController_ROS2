from __future__ import annotations

import math
import time

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class JointPublisherTest(Node):
    def __init__(self) -> None:
        super().__init__("jointPublisherTest")

        self.declare_parameter("joint_states_topic", "/rebotarm/joint_states")
        self.declare_parameter("command_mode", "pos_vel")
        self.declare_parameter("command_topic", "")
        self.declare_parameter("min_position", 0.05)
        self.declare_parameter("max_position", 0.2)
        self.declare_parameter("period", 10.0)
        self.declare_parameter("sample_rate", 20.0)
        self.declare_parameter("cycles", 0)
        self.declare_parameter("joint_names", ["joint2", "joint3"])
        self.declare_parameter("publish_gripper", False)
        self.declare_parameter("gripper_position", 0.0)

        joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.command_mode = str(
            self.get_parameter("command_mode").value
        ).strip().lower()
        command_topic = str(self.get_parameter("command_topic").value)
        self.min_position = float(self.get_parameter("min_position").value)
        self.max_position = float(self.get_parameter("max_position").value)
        self.period = float(self.get_parameter("period").value)
        self.sample_rate = float(self.get_parameter("sample_rate").value)
        self.cycles = int(self.get_parameter("cycles").value)
        self.joint_names_to_move = list(self.get_parameter("joint_names").value)
        self.publish_gripper = bool(self.get_parameter("publish_gripper").value)
        self.gripper_position = float(self.get_parameter("gripper_position").value)

        if self.max_position < self.min_position:
            raise ValueError("max_position must be >= min_position")
        if self.period <= 0.0 or self.sample_rate <= 0.0:
            raise ValueError("period and sample_rate must be greater than zero")
        if self.cycles < 0:
            raise ValueError("cycles must be non-negative (0 means repeat until stopped)")
        if not self.joint_names_to_move:
            raise ValueError("joint_names must contain at least one joint")
        if self.command_mode not in ("pos_vel", "mit"):
            raise ValueError("command_mode must be 'pos_vel' or 'mit'")
        if not command_topic:
            command_topic = f"/rebotarm/joints/cmd/{self.command_mode}"

        self.command_topic = command_topic
        self.latest_joint_state: JointState | None = None
        self.create_subscription(
            JointState,
            joint_states_topic,
            self._joint_state_callback,
            qos_profile_sensor_data,
        )
        self.publisher = self.create_publisher(JointTrajectory, command_topic, 10)
        self.finished = False
        self.publish_timer = None

    def _joint_state_callback(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def run(self) -> bool:
        if not self._wait_for_joint_state():
            self.get_logger().error("Timed out waiting for joint states")
            return False

        self._prepare_joint_layout()
        self.start_time = time.monotonic()
        self.publish_timer = self.create_timer(
            1.0 / self.sample_rate,
            self._publish_command,
        )
        self.get_logger().info(
            f"Streaming full joint setpoints on {self.command_topic} "
            f"using {self.command_mode} mode"
        )
        while rclpy.ok() and not self.finished:
            rclpy.spin_once(self, timeout_sec=0.1)
        return True

    def _wait_for_joint_state(self, timeout_sec: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and self.latest_joint_state is None:
            if time.monotonic() >= deadline:
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.latest_joint_state is not None

    def _prepare_joint_layout(self) -> None:
        assert self.latest_joint_state is not None
        arm_joints = [
            (name, position)
            for name, position in zip(
                self.latest_joint_state.name,
                self.latest_joint_state.position,
            )
            if name not in (
                "gripper_joint1",
                "gripper_joint2",
                "finger_left",
                "finger_right",
            )
        ]
        self.joint_names = [name for name, _ in arm_joints]
        self.current_positions = [float(position) for _, position in arm_joints]
        missing_joints = [
            name for name in self.joint_names_to_move if name not in self.joint_names
        ]
        if missing_joints:
            raise ValueError(f"joint_states does not contain {missing_joints}")
        self.joint_indices = [
            self.joint_names.index(name) for name in self.joint_names_to_move
        ]

    def _publish_command(self) -> None:
        elapsed = time.monotonic() - self.start_time
        if self.cycles and elapsed >= self.period * self.cycles:
            elapsed = self.period * self.cycles
            self.finished = True

        midpoint = (self.min_position + self.max_position) / 2.0
        amplitude = (self.max_position - self.min_position) / 2.0
        target_position = midpoint - amplitude * math.cos(
            2.0 * math.pi * elapsed / self.period
        )
        positions = self.current_positions.copy()
        for joint_index in self.joint_indices:
            positions[joint_index] = target_position
        joint_names = list(self.joint_names)
        if self.publish_gripper:
            joint_names.append("gripper")
            positions.append(self.gripper_position)

        point = JointTrajectoryPoint()
        point.positions = positions
        horizon_ns = round(2.0 * 1e9 / self.sample_rate)
        horizon_sec, horizon_nanosec = divmod(horizon_ns, 1_000_000_000)
        point.time_from_start = Duration(
            sec=int(horizon_sec),
            nanosec=int(horizon_nanosec),
        )
        if self.command_mode == "mit":
            point.velocities = [0.0] * len(positions)
            point.effort = [0.0] * len(positions)

        trajectory = JointTrajectory()
        trajectory.joint_names = joint_names
        trajectory.points = [point]
        self.publisher.publish(trajectory)

        if self.finished:
            self.publish_timer.cancel()
            self.get_logger().info(
                f"Completed sinusoidal stream cycle(s): {self.cycles}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointPublisherTest()
    try:
        if not node.run():
            raise SystemExit(1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()