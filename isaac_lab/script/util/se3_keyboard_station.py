import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

import carb
import omni

from omni.isaac.lab.devices.device_base import DeviceBase


class Se3KeyboardAGV(DeviceBase):
    def __init__(self, pos_sensitivity: float = 0.01, rot_sensitivity: float = 0.01):
        # store inputs
        self.num_actions = 5 + 6 + 4
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(
                event, *args
            ),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._delta_pos = np.zeros(self.num_actions)  # (x, y, z)
        self._pressed_keys = set()  # Track pressed keys
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t-----------------------AGV-----------------------\n"
        msg += "\tmove: WASD\n"
        msg += "\tpin: Z/X\n"
        msg += "\tz-axis: Q/E\n"
        msg += "\t---------------------IONIQ_5---------------------\n"
        msg += "\tmove: IJKL\n"
        msg += "\t-----------------------LIFT----------------------\n"
        msg += "\tz-axis: O/P\n"
        return msg

    def reset(self):
        # default flags
        self._delta_pos = np.zeros(self.num_actions)  # (x, y, z)

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, bool]:
        """Provides the result from keyboard event state.

        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        for key in self._pressed_keys:
            self._delta_pos += self._INPUT_KEY_MAPPING.get(
                key, np.zeros(self.num_actions)
            )

        # if not any(key in self._pressed_keys for key in ["W", "A", "S", "D", "I", "J", "K", "L"]):
        #     self._delta_pos[0:6] *= 0.99
        #     self._delta_pos[0:6] = [
        #         0 if value <= 1e-10 else value for value in self._delta_pos[0:6]
        #     ]

        return self._delta_pos

    def _on_keyboard_event(self, event, *args, **kwargs):
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # if event.input.name == "L":
            #     self.reset()
            if event.input.name in ["O", "P", "Q", "E"]:
                self._pressed_keys.add(event.input.name)
            else:
                self._delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["W", "A", "S", "D", "I", "J", "K", "L"]:
                self._delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in self._pressed_keys:
                self._pressed_keys.remove(event.input.name)
            # if event.input.name in self._pressed_keys:
            #     self._pressed_keys.remove(event.input.name)
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # x-axis (forward)
            "S": np.asarray([2, 2, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0])
            * self.pos_sensitivity,
            "W": np.asarray([-2, -2, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0])
            * self.pos_sensitivity,
            # y-axis (left-right)
            "A": np.asarray([1, -1, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0])
            * self.pos_sensitivity,
            "D": np.asarray([-1, 1, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0])
            * self.pos_sensitivity,
            # z-axis (up-down)
            "Q": np.asarray([0, 0, 0.1, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0])
            * self.pos_sensitivity,
            "E": np.asarray([0, 0, -0.1, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0])
            * self.pos_sensitivity,
            # z-axis (up-down)
            "Z": np.asarray(
                [0, 0, 0, -0.003, -0.003] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0]
            ),
            "X": np.asarray(
                [0, 0, 0, 0.003, 0.003] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0]
            ),
            # z-axis (up-down)
            "O": np.asarray([0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0.1, 0.1, 0.1, 0.1])
            * self.pos_sensitivity,
            "P": np.asarray(
                [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [-0.1, -0.1, -0.1, -0.1]
            )
            * self.pos_sensitivity,
            # ioniq
            "I": np.asarray(
                [0, 0, 0, 0, 0]
                + [1, -1, 1, -1, 0, 0]  # (rrw, rlw, frw, flw, frn, fln)
                + [0, 0, 0, 0]
            )
            * self.pos_sensitivity,
            "J": np.asarray([0, 0, 0, 0, 0] + [0, 0, 0, 0, 20, 20] + [0, 0, 0, 0])
            * self.pos_sensitivity,
            "K": np.asarray([0, 0, 0, 0, 0] + [-1, 1, -1, 1, 0, 0] + [0, 0, 0, 0])
            * self.pos_sensitivity,
            "L": np.asarray([0, 0, 0, 0, 0] + [0, 0, 0, 0, -20, -20] + [0, 0, 0, 0])
            * self.pos_sensitivity,
        }
