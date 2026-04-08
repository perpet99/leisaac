import numpy as np

from ..device_base import Device
from .gamepad_utils import GamepadController


class SO101JoyCon(Device):
    """Joy-Con (R) controller for sending SE(3) commands as delta poses for so101 single arm.

    Assumes Joy-Con (R) connected via Bluetooth on Windows 11 (direct HID, no XInput wrapper).
    If using BetterJoy or similar XInput wrapper, use SO101Gamepad with --teleop_device gamepad instead.

    Key bindings:
    ====================== ==========================================
    Description              Key
    ====================== ==========================================
    Forward / Backward       Stick Y (up / down)
    Left / Right (shoulder)  Stick X (left / right)
    Up / Down                ZR (up) / R (down)  [toggle each press]
    Rotate (Yaw) Left/Right  SL / SR
    Rotate (Pitch) Up/Down   Y / A
    Gripper Open / Close     B / X
    ====================== ==========================================

    Tip: Run with --debug_joycon to print raw axis/button values for remapping.
    """

    def __init__(self, env, sensitivity: float = 1.0, debug: bool = False):
        super().__init__(env, "joycon")

        self.pos_sensitivity = 0.01 * sensitivity
        self.joint_sensitivity = 0.15 * sensitivity
        self.rot_sensitivity = 0.15 * sensitivity
        self.debug = debug

        self._gamepad = GamepadController()
        self._gamepad.start()

        name = self._gamepad.name
        if "joy-con" not in name and "joycon" not in name and "joy con" not in name:
            raise ValueError(
                f"Joy-Con (R) not detected. Found: '{self._gamepad.joystick.get_name()}'\n"
                "Make sure Joy-Con (R) is connected via Bluetooth.\n"
                "If using BetterJoy/XInput, use --teleop_device gamepad instead."
            )

        self._create_key_mapping()
        self._action_update_list = [self._update_arm_action]

        # command buffers (dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper)
        self._delta_action = np.zeros(8)

        self.asset_name = "robot"
        self.robot_asset = self.env.scene[self.asset_name]
        self.target_frame = "gripper"
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self.target_frame_idx = body_idxs[0]

    def __del__(self):
        super().__del__()
        self._gamepad.stop()

    def _add_device_control_description(self):
        self._display_controls_table.add_row(["Stick Y (up)", "forward"])
        self._display_controls_table.add_row(["Stick Y (down)", "backward"])
        self._display_controls_table.add_row(["Stick X (left)", "shoulder left"])
        self._display_controls_table.add_row(["Stick X (right)", "shoulder right"])
        self._display_controls_table.add_row(["ZR", "up"])
        self._display_controls_table.add_row(["R", "down"])
        self._display_controls_table.add_row(["SL", "rotate left (yaw)"])
        self._display_controls_table.add_row(["SR", "rotate right (yaw)"])
        self._display_controls_table.add_row(["Y", "rotate up (pitch)"])
        self._display_controls_table.add_row(["A", "rotate down (pitch)"])
        self._display_controls_table.add_row(["B", "gripper open"])
        self._display_controls_table.add_row(["X", "gripper close"])

    def get_device_state(self):
        return self._convert_delta_from_frame(self._delta_action)

    def reset(self):
        self._delta_action[:] = 0.0

    def advance(self):
        self._delta_action[:] = 0.0
        self._update_action()
        return super().advance()

    def _create_key_mapping(self):
        self._ACTION_DELTA_MAPPING = {
            "forward": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "backward": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "left": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * self.joint_sensitivity,
            "right": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) * self.joint_sensitivity,
            "up": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "down": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "rotate_up": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]) * self.rot_sensitivity,
            "rotate_down": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) * self.rot_sensitivity,
            "rotate_left": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "rotate_right": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "gripper_open": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) * self.joint_sensitivity,
            "gripper_close": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) * self.joint_sensitivity,
        }
        # (action_name, controller_name, reverse[optional])
        self._INPUT_KEY_MAPPING_LIST = [
            ("backward", "STICK_X", True),   # stick up = forward (Y axis inverted on JoyCon)
            ("forward", "STICK_X"),
            ("left", "STICK_Y", True),
            ("right", "STICK_Y"),
            ("up", "ZR"),
            ("down", "R"),
            ("rotate_left", "SL"),
            ("rotate_right", "SR"),
            ("rotate_up", "Y"),
            ("rotate_down", "A"),
            ("gripper_open", "B"),
            ("gripper_close", "X"),
        ]

    def _update_action(self):
        self._gamepad.update()
        for update_func in self._action_update_list:
            update_func()

    def _update_arm_action(self):
        controller_state = self._gamepad.get_state()

        if self.debug:
            # Active buttons
            active_buttons = [
                name
                for name, idx in self._gamepad.mappings["buttons"].items()
                if idx < len(controller_state.buttons) and controller_state.buttons[idx]
            ]
            # Active axes (outside deadzone)
            active_axes = {
                name: round(controller_state.axes[idx], 3)
                for name, idx in self._gamepad.mappings["axes"].items()
                if idx < len(controller_state.axes) and controller_state.axes[idx] != 0.0
            }
            print(
                f"[JoyCon] buttons={active_buttons or '-'}  axes={active_axes or '-'}  "
                f"raw_buttons={list(controller_state.buttons)}  raw_axes={[round(a,3) for a in controller_state.axes]}"
            )

        for input_key_mapping in self._INPUT_KEY_MAPPING_LIST:
            action_name, controller_name = input_key_mapping[0], input_key_mapping[1]
            reverse = input_key_mapping[2] if len(input_key_mapping) > 2 else False
            is_activate, is_positive = self._gamepad.lookup_controller_state(controller_state, controller_name, reverse)
            if is_activate and is_positive:
                self._delta_action += self._ACTION_DELTA_MAPPING[action_name]
