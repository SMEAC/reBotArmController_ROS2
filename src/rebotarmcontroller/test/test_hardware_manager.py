import math
import threading
import unittest

import numpy as np

from rebotarmcontroller.hardware_manager import HardwareManager
from rebotarmcontroller.hardware_config import _add_runtime_config


class GripperPositionLimitTest(unittest.TestCase):
    @staticmethod
    def _hardware(open_position, close_position):
        hardware = object.__new__(HardwareManager)
        hardware.gripper_open_position = float(open_position)
        hardware.gripper_close_position = float(close_position)
        hardware._configure_gripper_position_limits()
        return hardware

    def test_dm_limits_use_configured_range(self):
        hardware = self._hardware(open_position=-5.0, close_position=0.0)

        for position in (-5.0, -1.0, 0.0):
            with self.subTest(position=position):
                self.assertEqual(hardware.validate_gripper_position(position), position)

        for position in (-5.1, 2.5, math.nan, math.inf):
            with self.subTest(position=position):
                with self.assertRaises(ValueError):
                    hardware.validate_gripper_position(position)

    def test_rs_limits_use_configured_range(self):
        hardware = self._hardware(open_position=5.0, close_position=0.0)

        for position in (0.0, 1.0, 5.0):
            with self.subTest(position=position):
                self.assertEqual(hardware.validate_gripper_position(position), position)

        for position in (5.1, -1.0, math.nan, math.inf):
            with self.subTest(position=position):
                with self.assertRaises(ValueError):
                    hardware.validate_gripper_position(position)

    def test_reversed_open_and_close_ordering_is_supported(self):
        hardware = self._hardware(open_position=0.0, close_position=3.0)

        self.assertEqual(hardware.validate_gripper_position(1.5), 1.5)

    def test_invalid_limit_configuration_is_rejected(self):
        cases = (
            (-5.0, -5.0),
            (math.nan, 0.0),
            (5.0, math.inf),
        )
        for open_position, close_position in cases:
            with self.subTest(
                open_position=open_position,
                close_position=close_position,
            ):
                with self.assertRaises(ValueError):
                    self._hardware(open_position, close_position)


class _RecordingArmGroup:
    def __init__(self):
        self.pos_vel_commands = []

    def send_pos_vel(self, positions, vlim):
        self.pos_vel_commands.append((positions.copy(), vlim.copy()))


class LowLevelStreamerTest(unittest.TestCase):
    @staticmethod
    def _hardware():
        hardware = object.__new__(HardwareManager)
        hardware._cmd_lock = threading.RLock()
        hardware._control_output_enabled = True
        hardware._lowlevel_mode = "pos_vel"
        hardware._lowlevel_target_pos = np.array([1.0, -1.0])
        hardware._lowlevel_command_pos = np.array([0.0, 1.0])
        hardware._lowlevel_max_velocity = np.array([0.5, 2.0])
        hardware._lowlevel_vlim = np.array([3.0, 4.0])
        hardware._arm_group = _RecordingArmGroup()
        return hardware

    def test_loop_sends_latest_target_with_velocity_limited_interpolation(self):
        hardware = self._hardware()

        hardware._lowlevel_loop_cb(None, 0.1)

        positions, vlim = hardware._arm_group.pos_vel_commands[-1]
        np.testing.assert_allclose(positions, [0.05, 0.8])
        np.testing.assert_allclose(vlim, [3.0, 4.0])

        hardware._lowlevel_target_pos[:] = [0.06, 0.7]
        hardware._lowlevel_loop_cb(None, 0.1)

        positions, _ = hardware._arm_group.pos_vel_commands[-1]
        np.testing.assert_allclose(positions, [0.06, 0.7])
        self.assertEqual(len(hardware._arm_group.pos_vel_commands), 2)

    def test_hardware_config_keeps_per_joint_velocity_limits(self):
        velocities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        joint_names = [f"joint{index}" for index in range(1, 7)]
        config = {
            "groups": {"arm": {"joints": joint_names}},
            "control": {
                "arm_control_mode": "posvel",
                "mit_kp": [1.0] * 6,
                "mit_kd": [1.0] * 6,
                "lowlevel_max_velocity": velocities,
            },
            "gravity_compensation": {"kp": 0.0, "kd": 0.0},
            "joints": [],
        }

        _add_runtime_config(config)

        self.assertEqual(
            config["_runtime"]["control"]["lowlevel_max_velocity"],
            velocities,
        )


class _FakeRobot:
    def __init__(self, has_gripper):
        self.has_gripper = has_gripper


class _FakeJointGroup:
    def __init__(self, joint_names):
        self.joint_names = joint_names


class CombinedLowLevelTargetTest(unittest.TestCase):
    @staticmethod
    def _hardware(with_gripper=False):
        hardware = object.__new__(HardwareManager)
        hardware._cmd_lock = threading.RLock()
        hardware._robot = _FakeRobot(with_gripper)
        hardware._arm_group = _FakeJointGroup(["joint1", "joint2"])
        hardware._gripper_group = (
            _FakeJointGroup(["gripper"]) if with_gripper else None
        )
        hardware._gripper_name = "gripper" if with_gripper else ""
        hardware.gripper_open_position = 1.0
        hardware.gripper_close_position = 0.0
        hardware._configure_gripper_position_limits()
        hardware._lowlevel_target_pos = np.zeros(2)
        hardware._lowlevel_target_vel = np.zeros(2)
        hardware._lowlevel_target_tau = np.zeros(2)
        hardware._lowlevel_vlim = np.ones(2)
        hardware._lowlevel_gripper_mode = None
        hardware._lowlevel_gripper_target_pos = np.zeros(1)
        hardware._lowlevel_gripper_target_vel = np.zeros(1)
        hardware._lowlevel_gripper_target_tau = np.zeros(1)
        hardware._gripper_target_position = None
        hardware._endpos_ctrl = type("EndPose", (), {"_gripper_target": 0.0})()
        hardware._state_machine = "IDLE"
        hardware._begin_lowlevel_streaming = lambda mode: None
        hardware._begin_gripper_lowlevel = lambda mode: None
        return hardware

    def test_named_arm_targets_are_mapped_to_configured_joint_order(self):
        hardware = self._hardware()

        hardware.set_lowlevel_joint_targets(
            "pos_vel",
            {"joint2": -0.25, "joint1": 0.5},
        )

        np.testing.assert_allclose(hardware._lowlevel_target_pos, [0.5, -0.25])
        np.testing.assert_allclose(hardware._lowlevel_target_vel, [0.0, 0.0])
        self.assertEqual(hardware.state_machine, "LOWLEVEL_STREAMING")

    def test_combined_target_updates_optional_gripper_and_mit_fields(self):
        hardware = self._hardware(with_gripper=True)

        hardware.set_lowlevel_joint_targets(
            "mit",
            {"joint1": 0.1, "joint2": -0.1, "gripper": 0.8},
            velocities={"joint1": 0.2, "gripper": -0.3},
            torques={"joint2": 0.4, "gripper": 0.5},
        )

        np.testing.assert_allclose(hardware._lowlevel_target_pos, [0.1, -0.1])
        np.testing.assert_allclose(hardware._lowlevel_target_vel, [0.2, 0.0])
        np.testing.assert_allclose(hardware._lowlevel_target_tau, [0.0, 0.4])
        self.assertEqual(hardware._lowlevel_gripper_mode, "mit")
        self.assertAlmostEqual(hardware._lowlevel_gripper_target_pos[0], 0.8)
        self.assertAlmostEqual(hardware._lowlevel_gripper_target_vel[0], -0.3)
        self.assertAlmostEqual(hardware._lowlevel_gripper_target_tau[0], 0.5)

    def test_missing_arm_joint_is_rejected(self):
        hardware = self._hardware()

        with self.assertRaisesRegex(ValueError, "missing arm joints"):
            hardware.set_lowlevel_joint_targets("pos_vel", {"joint1": 0.1})

    def test_gripper_command_outside_configured_limits_is_rejected(self):
        hardware = self._hardware(with_gripper=True)

        with self.assertRaisesRegex(ValueError, "outside safe range"):
            hardware.set_lowlevel_joint_targets(
                "pos_vel",
                {"joint1": 0.0, "joint2": 0.0, "gripper": 1.1},
            )


if __name__ == "__main__":
    unittest.main()
