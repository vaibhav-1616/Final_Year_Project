# VIRTUAL CURSOR - GESTURE RECOGNITION

import math
import time

from config import (
    CLICK_DISTANCE_THRESHOLD,
    RIGHT_CLICK_DISTANCE_THRESHOLD,
    SCROLL_DISTANCE_THRESHOLD,

    CLICK_HOLD_TIME,
    DRAG_HOLD_TIME,
    DOUBLE_CLICK_WINDOW,

    EFFECTIVE_CLICK_HOLD_TIME,
    EFFECTIVE_RIGHT_CLICK_HOLD_TIME,
    EFFECTIVE_DRAG_HOLD_TIME,

    GESTURE_STABILITY_TIME,

    SCROLL_DEADBAND,
    SCROLL_MULTIPLIER,

    FIST_HOLD_TIME,
    FIST_REARM_TIME,

    STATE_IDLE,
    STATE_LEFT_HOLD,
    STATE_DRAGGING,
    STATE_RIGHT_HOLD,
    STATE_SCROLLING,
    STATE_PAUSED
)


class GestureEngine:
    """
    Recognizes gestures and manages the interaction state machine.
    """

    def __init__(self):

        self.state = STATE_IDLE
        self.paused = False

        # Left pinch
        self.pinch_start_time = None
        self.clicked_this_pinch = False

        # Right pinch
        self.right_pinch_start_time = None
        self.right_clicked_this_pinch = False

        # Double click
        self.last_click_time = None

        # Scroll
        self.scroll_prev_y = None

        # Temporal gesture stabilization
        self.candidate_gesture = None
        self.candidate_start_time = None

        # Fist pause/resume
        self.fist_start_time = None
        self.last_pause_toggle_time = 0

    @staticmethod
    def distance(point_a, point_b):
        """Euclidean distance between two 2D points."""

        return math.hypot(
            point_a[0] - point_b[0],
            point_a[1] - point_b[1]
        )

    def is_fist(self, landmarks):
        """
        Detect a closed fist using the relative position of
        fingertips and their corresponding joints.

        MediaPipe landmark indices:
        Index:  8 tip, 6 PIP
        Middle: 12 tip, 10 PIP
        Ring:   16 tip, 14 PIP
        Pinky:  20 tip, 18 PIP
        """

        index_closed = landmarks[8].y > landmarks[6].y
        middle_closed = landmarks[12].y > landmarks[10].y
        ring_closed = landmarks[16].y > landmarks[14].y
        pinky_closed = landmarks[20].y > landmarks[18].y

        closed_count = sum([
            index_closed,
            middle_closed,
            ring_closed,
            pinky_closed
        ])

        return closed_count >= 3

    def update_fist_state(self, hand_data):
        """
        Check whether a fist has been held long enough
        to toggle pause/resume.

        Returns:
            True if a pause/resume toggle occurred.
        """

        landmarks = hand_data["all_landmarks"]

        fist_detected = self.is_fist(landmarks)

        if fist_detected:

            if self.fist_start_time is None:
                self.fist_start_time = time.time()

            elapsed = (
                time.time() - self.fist_start_time
            )

            enough_time_since_toggle = (
                time.time() - self.last_pause_toggle_time
                >= FIST_REARM_TIME
            )

            if (
                elapsed >= FIST_HOLD_TIME
                and enough_time_since_toggle
            ):

                self.paused = not self.paused
                self.last_pause_toggle_time = time.time()

                if self.paused:
                    self.state = STATE_PAUSED

                else:
                    self.state = STATE_IDLE

                # Reset all interaction states
                self.reset_gesture_timers()

                return True

        else:
            self.fist_start_time = None

        return False

    def stabilize(self, gesture_name):
        """
        Require a gesture to remain present for a short period
        before considering it stable.
        """

        now = time.time()

        if self.candidate_gesture != gesture_name:

            self.candidate_gesture = gesture_name
            self.candidate_start_time = now

            return False

        if (
            self.candidate_start_time is not None
            and now - self.candidate_start_time
            >= GESTURE_STABILITY_TIME
        ):

            return True

        return False

    def process(self, hand_data):
        """
        Process the detected hand and return an action.
        """

        # ----------------------------------------------------
        # Pause / Resume gesture
        # ----------------------------------------------------

        if self.update_fist_state(hand_data):

            return {
                "action": "none",
                "state": self.state,
                "move_cursor": False,
                "paused": self.paused
            }

        # If paused, ignore all normal gestures
        if self.paused:

            self.state = STATE_PAUSED

            return {
                "action": "none",
                "state": STATE_PAUSED,
                "move_cursor": False,
                "paused": True
            }

        index = hand_data["index"]
        thumb = hand_data["thumb"]
        middle = hand_data["middle"]

        # ----------------------------------------------------
        # Gesture distances
        # ----------------------------------------------------

        click_distance = self.distance(
            thumb,
            index
        )

        right_click_distance = self.distance(
            thumb,
            middle
        )

        scroll_distance = self.distance(
            index,
            middle
        )

        # ----------------------------------------------------
        # LEFT CLICK / DOUBLE CLICK / DRAG
        # ----------------------------------------------------

        if (
            click_distance < CLICK_DISTANCE_THRESHOLD
            and click_distance <= right_click_distance
        ):

            if not self.stabilize("LEFT_PINCH"):

                self.state = STATE_LEFT_HOLD

                return {
                    "action": "none",
                    "state": self.state,
                    "move_cursor": False,
                    "paused": False
                }

            if self.pinch_start_time is None:

                now = time.time()

                # Double-click detection
                if (
                    self.last_click_time is not None
                    and now - self.last_click_time
                    < DOUBLE_CLICK_WINDOW
                ):

                    self.last_click_time = None
                    self.clicked_this_pinch = True

                    self.pinch_start_time = now

                    return {
                        "action": "double_click",
                        "state": STATE_LEFT_HOLD,
                        "move_cursor": False,
                        "paused": False
                    }

                self.clicked_this_pinch = False
                self.pinch_start_time = now

            elapsed = (
                time.time() - self.pinch_start_time
            )

            # ------------------------------------------------
            # Drag
            # ------------------------------------------------

            if elapsed >= EFFECTIVE_DRAG_HOLD_TIME:

                if self.state != STATE_DRAGGING:

                    self.state = STATE_DRAGGING

                    return {
                        "action": "mouse_down",
                        "state": STATE_DRAGGING,
                        "move_cursor": True,
                        "paused": False
                    }

                self.state = STATE_DRAGGING

                return {
                    "action": "move",
                    "state": STATE_DRAGGING,
                    "move_cursor": True,
                    "paused": False
                }

            # ------------------------------------------------
            # Single click
            # ------------------------------------------------

            elif (
                elapsed >= EFFECTIVE_CLICK_HOLD_TIME
                and not self.clicked_this_pinch
            ):

                self.clicked_this_pinch = True
                self.last_click_time = time.time()
                self.state = STATE_LEFT_HOLD

                return {
                    "action": "click",
                    "state": STATE_LEFT_HOLD,
                    "move_cursor": False,
                    "paused": False
                }

            else:

                self.state = STATE_LEFT_HOLD

                return {
                    "action": "none",
                    "state": STATE_LEFT_HOLD,
                    "move_cursor": False,
                    "paused": False
                }

        # ----------------------------------------------------
        # RIGHT CLICK
        # ----------------------------------------------------

        elif (
            right_click_distance
            < RIGHT_CLICK_DISTANCE_THRESHOLD
        ):

            if not self.stabilize("RIGHT_PINCH"):

                self.state = STATE_RIGHT_HOLD

                return {
                    "action": "none",
                    "state": STATE_RIGHT_HOLD,
                    "move_cursor": False,
                    "paused": False
                }

            if self.right_pinch_start_time is None:

                self.right_pinch_start_time = time.time()
                self.right_clicked_this_pinch = False

            elapsed = (
                time.time()
                - self.right_pinch_start_time
            )

            if (
                elapsed >= EFFECTIVE_RIGHT_CLICK_HOLD_TIME
                and not self.right_clicked_this_pinch
            ):

                self.right_clicked_this_pinch = True
                self.state = STATE_RIGHT_HOLD

                return {
                    "action": "right_click",
                    "state": STATE_RIGHT_HOLD,
                    "move_cursor": False,
                    "paused": False
                }

            self.state = STATE_RIGHT_HOLD

            return {
                "action": "none",
                "state": STATE_RIGHT_HOLD,
                "move_cursor": False,
                "paused": False
            }

        # ----------------------------------------------------
        # SCROLL
        # ----------------------------------------------------

        elif (
            scroll_distance
            < SCROLL_DISTANCE_THRESHOLD
        ):

            if not self.stabilize("SCROLL"):

                self.state = STATE_SCROLLING

                return {
                    "action": "none",
                    "state": STATE_SCROLLING,
                    "move_cursor": False,
                    "paused": False
                }

            self.state = STATE_SCROLLING

            index_y = index[1]

            if self.scroll_prev_y is None:

                self.scroll_prev_y = index_y

                return {
                    "action": "none",
                    "state": STATE_SCROLLING,
                    "move_cursor": False,
                    "paused": False
                }

            dy = self.scroll_prev_y - index_y

            self.scroll_prev_y = index_y

            if abs(dy) > SCROLL_DEADBAND:

                return {
                    "action": "scroll",
                    "amount": dy * SCROLL_MULTIPLIER,
                    "state": STATE_SCROLLING,
                    "move_cursor": False,
                    "paused": False
                }

            return {
                "action": "none",
                "state": STATE_SCROLLING,
                "move_cursor": False,
                "paused": False
            }

        # ----------------------------------------------------
        # NORMAL CURSOR MOVEMENT
        # ----------------------------------------------------

        else:

            was_dragging = (
                self.state == STATE_DRAGGING
            )

            self.reset_gesture_timers()

            self.state = STATE_IDLE

            if was_dragging:

                return {
                    "action": "mouse_up",
                    "state": STATE_IDLE,
                    "move_cursor": True,
                    "paused": False
                }

            return {
                "action": "move",
                "state": STATE_IDLE,
                "move_cursor": True,
                "paused": False
            }

    def handle_no_hand(self):
        """
        Safety behavior when the hand disappears.
        """

        was_dragging = (
            self.state == STATE_DRAGGING
        )

        self.reset_gesture_timers()

        if was_dragging:

            self.state = STATE_IDLE

            return {
                "action": "mouse_up",
                "state": STATE_IDLE,
                "move_cursor": False,
                "paused": self.paused
            }

        if self.paused:

            self.state = STATE_PAUSED

        else:

            self.state = STATE_IDLE

        return {
            "action": "none",
            "state": self.state,
            "move_cursor": False,
            "paused": self.paused
        }

    def reset_gesture_timers(self):

        self.pinch_start_time = None
        self.clicked_this_pinch = False

        self.right_pinch_start_time = None
        self.right_clicked_this_pinch = False

        self.scroll_prev_y = None

        self.candidate_gesture = None
        self.candidate_start_time = None