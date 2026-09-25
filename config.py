# VIRTUAL CURSOR - CONFIGURATION


# ----- Camera -----
CAM_WIDTH = 640
CAM_HEIGHT = 480
FRAME_REDUCTION = 100


# ----- Cursor Smoothing -----
SMOOTHING_FACTOR = 0.6

# Small movement below this value is ignored to reduce jitter
CURSOR_DEADBAND = 3

# Adaptive smoothing limits
MIN_SMOOTHING_FACTOR = 0.35
MAX_SMOOTHING_FACTOR = 0.75

# Movement considered "fast"
FAST_MOVEMENT_THRESHOLD = 35


# ----- Gesture Distance Thresholds -----
CLICK_DISTANCE_THRESHOLD = 40
RIGHT_CLICK_DISTANCE_THRESHOLD = 40
SCROLL_DISTANCE_THRESHOLD = 40


# ----- Gesture Timing -----
CLICK_HOLD_TIME = 0.5
DRAG_HOLD_TIME = 1.0
DOUBLE_CLICK_WINDOW = 0.4

# Small temporal confirmation period to reduce accidental activation
GESTURE_STABILITY_TIME = 0.08


# ----- Scrolling -----
SCROLL_DEADBAND = 4
SCROLL_MULTIPLIER = 3


# ----- Pause / Resume -----
FIST_HOLD_TIME = 1.0

# Prevent repeated pause/resume toggles while fist remains closed
FIST_REARM_TIME = 0.8


# ----- MediaPipe -----
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7


# ----- States -----
STATE_IDLE = "IDLE"
STATE_LEFT_HOLD = "LEFT_HOLD"
STATE_DRAGGING = "DRAGGING"
STATE_RIGHT_HOLD = "RIGHT_HOLD"
STATE_SCROLLING = "SCROLLING"
STATE_PAUSED = "PAUSED"


# ----- Application -----
WINDOW_NAME = "Virtual Cursor"
EXIT_KEY = "x"


# ----- Safety Mode -----
# When enabled, gesture confirmation is slightly more conservative.
SAFE_MODE = True

if SAFE_MODE:
    EFFECTIVE_CLICK_HOLD_TIME = 0.6
    EFFECTIVE_RIGHT_CLICK_HOLD_TIME = 0.6
    EFFECTIVE_DRAG_HOLD_TIME = 1.1
else:
    EFFECTIVE_CLICK_HOLD_TIME = CLICK_HOLD_TIME
    EFFECTIVE_RIGHT_CLICK_HOLD_TIME = CLICK_HOLD_TIME
    EFFECTIVE_DRAG_HOLD_TIME = DRAG_HOLD_TIME