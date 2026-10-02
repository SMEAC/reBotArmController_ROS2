from __future__ import annotations

import math

from rclpy.qos import QoSProfile, ReliabilityPolicy
from trajectory_msgs.msg import JointTrajectory


class MotorPassthrough:
    def __init__(self, node, hardware, namespace: str, arbitration: str) -> None:
        self._node = node
        self._hardware = hardware
        self._arbitration = arbitration
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._subscriptions = []

        self._subscribe(
            JointTrajectory,
            f"/{namespace}/joints/cmd/pos_vel",
            self._make_combined_callback("pos_vel"),
            qos,
        )
        self._subscribe(
            JointTrajectory,
            f"/{namespace}/joints/cmd/mit",
            self._make_combined_callback("mit"),
            qos,
        )

    def _subscribe(self, msg_type, topic: str, callback, qos: QoSProfile) -> None:
        self._subscriptions.append(
            self._node.create_subscription(
                msg_type,
                topic,
                callback,
                qos,
                callback_group=self._node.reentrant_group,
            )
        )

    def _make_combined_callback(self, mode: str) -> object:
        topic = f"/joints/cmd/{mode}"

        def _callback(msg) -> None:
            if not self._can_send_lowlevel(
                topic,
                allow_preempt=True,
            ):
                return

            try:
                positions, velocities, torques = self._trajectory_setpoints(msg)
                self._hardware.set_lowlevel_joint_targets(
                    mode,
                    positions,
                    velocities=velocities,
                    torques=torques,
                )
            except Exception as exc:
                self._node.get_logger().warn(
                    f"combined {mode} command rejected: {exc}"
                )
            finally:
                self._node.publish_arm_status()

        return _callback

    def _trajectory_setpoints(
        self,
        msg: JointTrajectory,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        if len(msg.points) != 1:
            raise ValueError("streaming command must contain exactly one trajectory point")
        names = list(msg.joint_names)
        if not names:
            raise ValueError("joint_names must not be empty")
        if len(names) != len(set(names)):
            raise ValueError("joint_names must not contain duplicates")

        point = msg.points[0]
        if len(point.positions) != len(names):
            raise ValueError("point.positions length must match joint_names")
        positions = {name: float(value) for name, value in zip(names, point.positions)}
        if any(not math.isfinite(value) for value in positions.values()):
            raise ValueError("position values must be finite")

        arm_names = set(self._hardware.joint_names)
        missing_arm = arm_names - set(names)
        if missing_arm:
            raise ValueError(f"missing arm joints: {sorted(missing_arm)}")
        gripper_name = self._hardware.gripper_command_name
        allowed_names = arm_names | ({gripper_name} if gripper_name else set())
        unknown_names = set(names) - allowed_names
        if unknown_names:
            raise ValueError(f"unknown command joints: {sorted(unknown_names)}")

        velocities = self._optional_point_values(point.velocities, names, "velocities")
        torques = self._optional_point_values(point.effort, names, "effort")
        return positions, velocities, torques

    @staticmethod
    def _optional_point_values(
        values,
        names: list[str],
        label: str,
    ) -> dict[str, float]:
        if not values:
            return {}
        if len(values) != len(names):
            raise ValueError(
                f"point.{label} length must match joint_names when provided"
            )
        result = {name: float(value) for name, value in zip(names, values)}
        if any(not math.isfinite(value) for value in result.values()):
            raise ValueError(f"{label} values must be finite")
        return result

    def _can_send_lowlevel(self, label: str, *, allow_preempt: bool) -> bool:
        state = self._hardware.state_machine
        if state in ("GRAVITY_COMP", "SAFE_HOMING"):
            self._node.get_logger().warn(f"rejecting {label} in state {state}")
            return False
        if state == "TRAJ_RUNNING":
            if self._arbitration == "reject" or not allow_preempt:
                self._node.get_logger().warn(
                    f"rejecting {label} while trajectory is running"
                )
                return False
            self._node.get_logger().warn(
                f"preempting trajectory for {label}"
            )
            self._hardware.stop_motion()
        return True
