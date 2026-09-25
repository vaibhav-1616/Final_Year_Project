# VIRTUAL CURSOR - VISUALIZATION MODULE

import cv2

from config import (
    FRAME_REDUCTION,
    WINDOW_NAME,
    EXIT_KEY,

    STATE_IDLE,
    STATE_LEFT_HOLD,
    STATE_DRAGGING,
    STATE_RIGHT_HOLD,
    STATE_SCROLLING,
    STATE_PAUSED
)


class Visualizer:
    """
    Handles the OpenCV visual interface.
    """

    STATE_COLORS = {

        STATE_IDLE:
            (200, 200, 200),

        STATE_LEFT_HOLD:
            (0, 140, 255),

        STATE_DRAGGING:
            (0, 0, 255),

        STATE_RIGHT_HOLD:
            (255, 0, 255),

        STATE_SCROLLING:
            (0, 255, 255),

        STATE_PAUSED:
            (0, 165, 255)
    }

    def __init__(self):

        self.mp_draw = None
        self.mp_hands = None

    def initialize_mediapipe_draw(
        self,
        mp_draw,
        mp_hands
    ):

        self.mp_draw = mp_draw
        self.mp_hands = mp_hands

    def draw(
        self,
        frame,
        hand_data,
        state,
        fps,
        paused=False
    ):

        height, width, _ = frame.shape

        # ----------------------------------------------------
        # Active cursor region
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (
                FRAME_REDUCTION,
                FRAME_REDUCTION
            ),
            (
                width - FRAME_REDUCTION,
                height - FRAME_REDUCTION
            ),
            (255, 0, 255),
            2
        )

        # ----------------------------------------------------
        # Hand landmarks
        # ----------------------------------------------------

        if hand_data is not None:

            landmarks = hand_data["landmarks"]

            if (
                self.mp_draw is not None
                and self.mp_hands is not None
            ):

                self.mp_draw.draw_landmarks(
                    frame,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

            index_x, index_y = hand_data["index"]
            thumb_x, thumb_y = hand_data["thumb"]
            middle_x, middle_y = hand_data["middle"]

            # Index
            cv2.circle(
                frame,
                (index_x, index_y),
                8,
                (0, 255, 0),
                cv2.FILLED
            )

            # Thumb
            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                8,
                (255, 0, 0),
                cv2.FILLED
            )

            # Middle
            cv2.circle(
                frame,
                (middle_x, middle_y),
                8,
                (0, 255, 255),
                cv2.FILLED
            )

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        state_color = self.STATE_COLORS.get(
            state,
            (255, 255, 255)
        )

        cv2.putText(
            frame,
            f"STATE: {state}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            state_color,
            2
        )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (width - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Pause indicator
        # ----------------------------------------------------

        if paused:

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (width, height),
                (0, 0, 0),
                -1
            )

            frame = cv2.addWeighted(
                overlay,
                0.45,
                frame,
                0.55,
                0
            )

            cv2.putText(
                frame,
                "SYSTEM PAUSED",
                (
                    width // 2 - 150,
                    height // 2
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 165, 255),
                3
            )

            cv2.putText(
                frame,
                "Hold FIST again to resume",
                (
                    width // 2 - 180,
                    height // 2 + 40
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return frame

    def show(self, frame):

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

    def should_exit(self):

        if cv2.waitKey(1) & 0xFF == ord(EXIT_KEY):

            return True

        try:

            if (
                cv2.getWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_VISIBLE
                ) < 1
            ):

                return True

        except cv2.error:

            return True

        return False

    def close(self):

        cv2.destroyAllWindows()