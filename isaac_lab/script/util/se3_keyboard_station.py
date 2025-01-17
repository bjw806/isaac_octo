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
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._delta_pos = np.zeros(2)  # (x, y, z)
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
        msg += "\t----------------------------------------------\n"
        msg += "\tMove arm along x-axis: W/S\n"
        msg += "\tMove arm along y-axis: A/D\n"
        msg += "\tMove arm along z-axis: Q/E\n"
        return msg

    def reset(self):
        # default flags
        self._delta_pos = np.zeros(2)  # (x, y, z)

    def add_callback(self, key: str, func: Callable):

        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, bool]:
        """Provides the result from keyboard event state.

        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        for key in self._pressed_keys:
            self._delta_pos += self._INPUT_KEY_MAPPING.get(key, np.zeros(2))
        return self._delta_pos

    def _on_keyboard_event(self, event, *args, **kwargs):
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            self._pressed_keys.add(event.input.name)
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._pressed_keys:
                self._pressed_keys.remove(event.input.name)
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # x-axis (forward)
            "S": np.asarray([1, -1]) * self.pos_sensitivity,
            "W": np.asarray([-1, 1]) * self.pos_sensitivity,
            # y-axis (left-right)
            "A": np.asarray([.5, .5]) * self.pos_sensitivity,
            "D": np.asarray([-.5, -.5]) * self.pos_sensitivity,
            # z-axis (up-down)
            # "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            # "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
        }
