import math
import sys
from pathlib import Path

# joyconrobotics is a local package without setup.py; add its parent to sys.path.
_JOYCONROBOTICS_DIR = Path("D:/perpet/git/LLM/XLeRobot/software")
if str(_JOYCONROBOTICS_DIR) not in sys.path:
    sys.path.insert(0, str(_JOYCONROBOTICS_DIR))

from joyconrobotics import JoyconRobotics  # type: ignore[import-untyped]

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_MOTOR_LIMITS

from ..device_base import Device


class FixedAxesJoyconRobotics(JoyconRobotics):
    """JoyconRobotics with fixed-axis stick mapping for SO101 EE teleoperation.

    Vertical stick: controls X (forward/back) and Z simultaneously.
    Horizontal stick: controls Y (left/right shoulder pan).
    R button: Z up.  Stick button: Z down.  Home: reset.  ZR: toggle gripper.
    """

    def common_update(self):
        speed_scale = 0.0008

        # Vertical stick → X axis (forward/back) + coupled Z
        joycon_stick_v = (
            self.joycon.get_stick_right_vertical()
            if self.joycon.is_right()
            else self.joycon.get_stick_left_vertical()
        )
        joycon_stick_v_0 = 1800
        joycon_stick_v_threshold = 300
        joycon_stick_v_range = 1000
        if abs(joycon_stick_v - joycon_stick_v_0) > joycon_stick_v_threshold:
            delta_v = speed_scale * (joycon_stick_v - joycon_stick_v_0) / joycon_stick_v_range
            self.position[0] += delta_v * self.dof_speed[0] * self.direction_reverse[0] * self.direction_vector[0]
            self.position[2] += delta_v * self.dof_speed[1] * self.direction_reverse[1] * self.direction_vector[2]

        # Horizontal stick → Y axis (left/right)
        joycon_stick_h = (
            self.joycon.get_stick_right_horizontal()
            if self.joycon.is_right()
            else self.joycon.get_stick_left_horizontal()
        )
        joycon_stick_h_0 = 2000
        joycon_stick_h_threshold = 300
        joycon_stick_h_range = 1000
        if abs(joycon_stick_h - joycon_stick_h_0) > joycon_stick_h_threshold:
            self.position[1] += (
                speed_scale
                * (joycon_stick_h - joycon_stick_h_0)
                / joycon_stick_h_range
                * self.dof_speed[1]
                * self.direction_reverse[1]
            )

        # R / stick buttons: Z up / Z down
        joycon_button_up = self.joycon.get_button_r() if self.joycon.is_right() else self.joycon.get_button_l()
        if joycon_button_up == 1:
            self.position[2] += speed_scale * self.dof_speed[2] * self.direction_reverse[2]

        joycon_button_down = (
            self.joycon.get_button_r_stick() if self.joycon.is_right() else self.joycon.get_button_l_stick()
        )
        if joycon_button_down == 1:
            self.position[2] -= speed_scale * self.dof_speed[2] * self.direction_reverse[2]

        # X / B buttons: fine X control
        joycon_button_xup = self.joycon.get_button_x() if self.joycon.is_right() else self.joycon.get_button_up()
        joycon_button_xback = self.joycon.get_button_b() if self.joycon.is_right() else self.joycon.get_button_down()
        if joycon_button_xup == 1:
            self.position[0] += 0.001 * self.dof_speed[0]
        elif joycon_button_xback == 1:
            self.position[0] -= 0.001 * self.dof_speed[0]

        # Home button → reset EE position
        joycon_button_home = (
            self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        )
        if joycon_button_home == 1:
            self.position = self.offset_position_m.copy()

        # Event-driven buttons (gripper, episode control)
        for event_type, status in self.button.events():
            if (self.joycon.is_right() and event_type == "plus" and status == 1) or (
                self.joycon.is_left() and event_type == "minus" and status == 1
            ):
                self.reset_button = 1
                self.reset_joycon()
            elif self.joycon.is_right() and event_type == "a":
                self.next_episode_button = status
            elif self.joycon.is_right() and event_type == "y":
                self.restart_episode_button = status
            elif (
                (self.joycon.is_right() and event_type == "zr") or (self.joycon.is_left() and event_type == "zl")
            ) and not self.change_down_to_gripper:
                self.gripper_toggle_button = status
            elif (
                (self.joycon.is_right() and event_type == "stick_r_btn")
                or (self.joycon.is_left() and event_type == "stick_l_btn")
            ) and self.change_down_to_gripper:
                self.gripper_toggle_button = status
            else:
                self.reset_button = 0

        if self.gripper_toggle_button == 1:
            self.gripper_state = self.gripper_close if self.gripper_state == self.gripper_open else self.gripper_open
            self.gripper_toggle_button = 0

        if self.joycon.is_right():
            if self.next_episode_button == 1:
                self.button_control = 1
            elif self.restart_episode_button == 1:
                self.button_control = -1
            elif self.reset_button == 1:
                self.button_control = 8
            else:
                self.button_control = 0

        return self.position, self.gripper_state, self.button_control


def _ik_motor_degrees(x, y, l1=0.1159, l2=0.1350):
    """Compute shoulder_lift and elbow_flex in motor-space degrees via 2-link planar IK.

    Args:
        x: EE forward distance (m).
        y: EE height (m).
        l1: Upper arm length (m).
        l2: Lower arm length (m).

    Returns:
        (shoulder_lift_motor_deg, elbow_flex_motor_deg) compatible with
        SO101_FOLLOWER_MOTOR_LIMITS range [-100, 100].
    """
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2
    r_min = abs(l1 - l2)

    if r > r_max:
        scale = r_max / r
        x, y, r = x * scale, y * scale, r_max
    elif 0 < r < r_min:
        scale = r_min / r
        x, y, r = x * scale, y * scale, r_min

    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    theta2 = math.pi - math.acos(max(-1.0, min(1.0, cos_theta2)))

    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    joint2 = max(-0.1, min(3.45, theta1 + theta1_offset))
    joint3 = max(-0.2, min(math.pi, theta2 + theta2_offset))

    # URDF radians → motor-space degrees
    shoulder_lift_deg = 90.0 - math.degrees(joint2)
    elbow_flex_deg = math.degrees(joint3) - 90.0

    return shoulder_lift_deg, elbow_flex_deg


class SO101JoyConEE(Device):
    """Joy-Con (R) controller for SO101 using EE-space IK with absolute joint position control.

    Tracks the Joy-Con 6D pose via JoyconRobotics and maps it to SO101 joint angles
    through a 2-link planar IK, outputting motor-space degree targets for the simulation.

    Key bindings:
    ====================== ==========================================
    Description              Key
    ====================== ==========================================
    Forward / Backward       Stick Y (coupled X+Z motion)
    Left / Right (pan)       Stick X → shoulder_pan
    Up (Z only)              R button
    Down (Z only)            Stick button
    Wrist Pitch              Joy-Con pitch tilt
    Wrist Roll               Joy-Con roll tilt
    Reset EE position        Home button
    Gripper Toggle           ZR button
    ====================== ==========================================
    """

    def __init__(self, env, sensitivity: float = 1.0, debug: bool = False):  # noqa: ARG002
        super().__init__(env, "joycon_ee")
        self._debug = debug
        self._motor_limits = SO101_FOLLOWER_MOTOR_LIMITS

        # Rest-pose EE offsets matching the reference calibration
        self._x0 = 0.1629
        self._z0 = 0.1131

        self._joycon = FixedAxesJoyconRobotics("right", dof_speed=[2, 2, 2, 1, 1, 1])

        # Cached control state (updated once per advance() call)
        self._last_pose = [self._x0, 0.0, self._z0, 0.0, 0.0, 0.0]
        self._last_gripper = 0

    def __del__(self):
        super().__del__()

    def _add_device_control_description(self):
        self._display_controls_table.add_row(["Stick Y", "forward / backward (EE x+z coupled)"])
        self._display_controls_table.add_row(["Stick X", "shoulder pan (left / right)"])
        self._display_controls_table.add_row(["R button", "EE up (Z)"])
        self._display_controls_table.add_row(["Stick button", "EE down (Z)"])
        self._display_controls_table.add_row(["Joy-Con pitch", "wrist pitch"])
        self._display_controls_table.add_row(["Joy-Con roll", "wrist roll"])
        self._display_controls_table.add_row(["Home", "reset EE position"])
        self._display_controls_table.add_row(["ZR", "toggle gripper"])

    def get_device_state(self) -> dict:
        """Compute motor-space degree targets from the current Joy-Con state.

        Returns:
            Dict mapping joint name → motor-space degrees, compatible with
            SO101_FOLLOWER_MOTOR_LIMITS and convert_action_from_so101_leader().
        """
        x, y, z, roll_, pitch_, _ = self._last_pose
        gripper = self._last_gripper

        ee_x = self._x0 + x
        ee_z = self._z0 + z

        shoulder_lift_deg, elbow_flex_deg = _ik_motor_degrees(ee_x, ee_z)
        pitch_deg = -pitch_ * 300.0 + 20.0
        roll_deg = roll_ * 50.0

        joint_state = {
            "shoulder_pan": y * 300.0,
            "shoulder_lift": shoulder_lift_deg,
            "elbow_flex": elbow_flex_deg,
            "wrist_flex": -shoulder_lift_deg - elbow_flex_deg + pitch_deg,
            "wrist_roll": roll_deg,
            "gripper": 60.0 if gripper == 1 else 20.0,
        }

        if self._debug:
            print(
                f"[JoyConEE] x={x:.3f} y={y:.3f} z={z:.3f} | "
                f"lift={shoulder_lift_deg:.1f} flex={elbow_flex_deg:.1f} "
                f"wrist={joint_state['wrist_flex']:.1f} roll={roll_deg:.1f} "
                f"gripper={joint_state['gripper']}"
            )

        return joint_state

    def input2action(self) -> dict:
        ac_dict = super().input2action()
        ac_dict["motor_limits"] = self._motor_limits
        return ac_dict

    def reset(self):
        pass

    def advance(self):
        # Read JoyCon state once per frame and cache it
        pose, gripper, button_control = self._joycon.get_control()
        self._last_pose = pose
        self._last_gripper = gripper

        # button_control: 8=Plus(start/reset), 1=A(success), -1=Y(fail), 0=none
        if button_control == 8:
            if not self._started:
                self._started = True
                self._reset_state = False
            else:
                self._started = False
                self._reset_state = True
                if "R" in self._additional_callbacks:
                    self._additional_callbacks["R"]()
        elif button_control == 1:
            self._started = False
            self._reset_state = True
            if "N" in self._additional_callbacks:
                self._additional_callbacks["N"]()
        elif button_control == -1:
            self._started = False
            self._reset_state = True
            if "R" in self._additional_callbacks:
                self._additional_callbacks["R"]()

        return super().advance()
