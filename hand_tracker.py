# VIRTUAL CURSOR - HAND TRACKING MODULE

import cv2
import mediapipe as mp

from config import (
    CAM_WIDTH,
    CAM_HEIGHT,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)


class HandTracker:
    """
    Handles webcam input and MediaPipe hand landmark detection.
    """

    def __init__(self):

        self.cap = cv2.VideoCapture(0)

        self.cap.set(3, CAM_WIDTH)
        self.cap.set(4, CAM_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError("Webcam unable to open!")

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

    def get_frame(self):
        """
        Capture a frame and detect hand landmarks.

        Returns:
            success
            frame
            hand_data
        """

        success, frame = self.cap.read()

        if not success:
            return False, None, None

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # OpenCV BGR -> MediaPipe RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb_frame)

        hand_data = None

        if results.multi_hand_landmarks:

            # Project uses one hand
            hand_landmarks = results.multi_hand_landmarks[0]

            landmarks = hand_landmarks.landmark

            height, width, _ = frame.shape

            # Index fingertip
            index_x = int(landmarks[8].x * width)
            index_y = int(landmarks[8].y * height)

            # Thumb fingertip
            thumb_x = int(landmarks[4].x * width)
            thumb_y = int(landmarks[4].y * height)

            # Middle fingertip
            middle_x = int(landmarks[12].x * width)
            middle_y = int(landmarks[12].y * height)

            hand_data = {
                "landmarks": hand_landmarks,

                "all_landmarks": landmarks,

                "index": (
                    index_x,
                    index_y
                ),

                "thumb": (
                    thumb_x,
                    thumb_y
                ),

                "middle": (
                    middle_x,
                    middle_y
                )
            }

        return True, frame, hand_data

    def release(self):

        self.cap.release()
        self.hands.close()