import unittest

from rebotarmcontroller.motor_passthrough import MotorPassthrough
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class _FakeHardware:
    joint_names = ["joint1", "joint2"]
    gripper_command_name = "gripper"


class CombinedTrajectoryCommandTest(unittest.TestCase):
    @staticmethod
    def _parser():
        parser = object.__new__(MotorPassthrough)
        parser._hardware = _FakeHardware()
        return parser

    @staticmethod
    def _message(names, positions, velocities=None, effort=None, point_count=1):
        msg = JointTrajectory()
        msg.joint_names = names
        for _ in range(point_count):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = velocities or []
            point.effort = effort or []
            msg.points.append(point)
        return msg

    def test_parses_named_positions_and_optional_mit_values(self):
        msg = self._message(
            ["joint2", "joint1", "gripper"],
            [0.2, -0.1, 0.5],
            velocities=[0.0, 0.1, -0.2],
            effort=[0.3, 0.0, 0.1],
        )

        result = self._parser()._trajectory_setpoints(msg)

        self.assertEqual(
            result,
            (
                {"joint2": 0.2, "joint1": -0.1, "gripper": 0.5},
                {"joint2": 0.0, "joint1": 0.1, "gripper": -0.2},
                {"joint2": 0.3, "joint1": 0.0, "gripper": 0.1},
            ),
        )

    def test_rejects_missing_arm_joints(self):
        msg = self._message(["joint1"], [0.0])

        with self.assertRaisesRegex(ValueError, "missing arm joints"):
            self._parser()._trajectory_setpoints(msg)

    def test_rejects_multiple_trajectory_points(self):
        msg = self._message(["joint1", "joint2"], [0.0, 0.0], point_count=2)

        with self.assertRaisesRegex(ValueError, "exactly one trajectory point"):
            self._parser()._trajectory_setpoints(msg)

    def test_rejects_invalid_optional_field_lengths(self):
        msg = self._message(
            ["joint1", "joint2"],
            [0.0, 0.0],
            velocities=[0.0],
        )

        with self.assertRaisesRegex(ValueError, "velocities length"):
            self._parser()._trajectory_setpoints(msg)


if __name__ == "__main__":
    unittest.main()