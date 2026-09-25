# VIRTUAL CURSOR - CURSOR / MOUSE CONTROL MODULE

import math
import pyautogui

from config import (
    FRAME_REDUCTION,
    SMOOTHING_FACTOR,
    CURSOR_DEADBAND,
    MIN_SMOOTHING_FACTOR,
    MAX_SMOOTHING_FACTOR,
    FAST_MOVEMENT_THRESHOLD
)


class CursorController:
    """
    Handles camera-to-screen mapping, cursor smoothing,
    and PyAutoGUI mouse operations.
    """

    def __init__(self):

        self.screen_width, self.screen_height = (
            pyautogui.size()
        )

        pyautogui.FAILSAFE = False

        self.smoothed_x = None
        self.smoothed_y = None

        self.previous_raw_x = None
        self.previous_raw_y = None

    def map_to_screen(
        self,
        x,
        y,
        frame_width,
        frame_height
    ):
        """
        Map camera coordinates to screen coordinates.
        """

        raw_screen_x = int(
            (x - FRAME_REDUCTION)
            / (frame_width - 2 * FRAME_REDUCTION)
            * self.screen_width
        )

        raw_screen_y = int(
            (y - FRAME_REDUCTION)
            / (frame_height - 2 * FRAME_REDUCTION)
            * self.screen_height
        )

        raw_screen_x = max(
            0,
            min(self.screen_width - 1, raw_screen_x)
        )

        raw_screen_y = max(
            0,
            min(self.screen_height - 1, raw_screen_y)
        )

        return raw_screen_x, raw_screen_y

    def initialize_position(self, raw_x, raw_y):

        if self.smoothed_x is None:

            self.smoothed_x = raw_x
            self.smoothed_y = raw_y

            self.previous_raw_x = raw_x
            self.previous_raw_y = raw_y

    def get_adaptive_factor(
        self,
        raw_x,
        raw_y
    ):
        """
        Adjust smoothing according to movement speed.

        Slow movement -> stronger smoothing.
        Fast movement -> weaker smoothing.
        """

        if self.previous_raw_x is None:

            return SMOOTHING_FACTOR

        movement = math.hypot(
            raw_x - self.previous_raw_x,
            raw_y - self.previous_raw_y
        )

        self.previous_raw_x = raw_x
        self.previous_raw_y = raw_y

        if movement >= FAST_MOVEMENT_THRESHOLD:

            return MIN_SMOOTHING_FACTOR

        # Interpolate between minimum and maximum smoothing
        ratio = movement / FAST_MOVEMENT_THRESHOLD

        factor = (
            MAX_SMOOTHING_FACTOR
            - ratio
            * (
                MAX_SMOOTHING_FACTOR
                - MIN_SMOOTHING_FACTOR
            )
        )

        return factor

    def move_smoothly(self, raw_x, raw_y):
        """
        Apply adaptive EMA smoothing and cursor deadband.
        """

        self.initialize_position(
            raw_x,
            raw_y
        )

        # Ignore very small movement
        movement = math.hypot(
            raw_x - self.smoothed_x,
            raw_y - self.smoothed_y
        )

        if movement < CURSOR_DEADBAND:

            return

        adaptive_factor = self.get_adaptive_factor(
            raw_x,
            raw_y
        )

        self.smoothed_x = (
            adaptive_factor * raw_x
            + (1 - adaptive_factor)
            * self.smoothed_x
        )

        self.smoothed_y = (
            adaptive_factor * raw_y
            + (1 - adaptive_factor)
            * self.smoothed_y
        )

        pyautogui.moveTo(
            int(self.smoothed_x),
            int(self.smoothed_y)
        )

    # --------------------------------------------------------
    # Mouse actions
    # --------------------------------------------------------

    def left_click(self):
        pyautogui.click()

    def double_click(self):
        pyautogui.doubleClick()

    def right_click(self):
        pyautogui.rightClick()

    def mouse_down(self):
        pyautogui.mouseDown()

    def mouse_up(self):
        pyautogui.mouseUp()

    def scroll(self, amount):
        pyautogui.scroll(int(amount))

    def reset_smoothing(self):

        self.smoothed_x = None
        self.smoothed_y = None

        self.previous_raw_x = None
        self.previous_raw_y = None