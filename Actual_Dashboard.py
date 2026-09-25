# ============================================================
# AI-DRIVEN TOUCHLESS VIRTUAL CURSOR DASHBOARD
# ============================================================
# Run:
# streamlit run Final_implementation\Actual_Dashboard.py

import time
import threading

import cv2
import streamlit as st

from hand_tracker import HandTracker
from gestures import GestureEngine
from cursor_controller import CursorController

from config import (
    CAM_WIDTH,
    CAM_HEIGHT,
    FRAME_REDUCTION,
    SMOOTHING_FACTOR,
    CURSOR_DEADBAND,
    CLICK_DISTANCE_THRESHOLD,
    RIGHT_CLICK_DISTANCE_THRESHOLD,
    SCROLL_DISTANCE_THRESHOLD,
    CLICK_HOLD_TIME,
    DRAG_HOLD_TIME,
    DOUBLE_CLICK_WINDOW,
    SCROLL_DEADBAND,
    SCROLL_MULTIPLIER,
    FIST_HOLD_TIME,
    SAFE_MODE,

    STATE_IDLE,
    STATE_LEFT_HOLD,
    STATE_DRAGGING,
    STATE_RIGHT_HOLD,
    STATE_SCROLLING,
    STATE_PAUSED
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Virtual Cursor",
    page_icon="🖐️",
    layout="wide"
)


# ============================================================
# SHARED DATA
# ============================================================

class DashboardData:

    def __init__(self):
        self.lock = threading.Lock()

        self.frame = None

        self.running = False
        self.hand_detected = False

        self.state = STATE_IDLE
        self.paused = False

        self.left_clicks = 0
        self.double_clicks = 0
        self.right_clicks = 0
        self.drag_actions = 0
        self.scroll_events = 0
        self.pause_toggles = 0


# ============================================================
# SYSTEM
# ============================================================

class DashboardSystem:

    def __init__(self):

        self.data = DashboardData()

        self.tracker = None
        self.gesture_engine = None
        self.cursor = None

        self.thread = None
        self.stop_event = threading.Event()

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def start(self):

        if self.thread is not None and self.thread.is_alive():
            return

        self.stop_event.clear()

        with self.data.lock:
            self.data.running = True

        self.thread = threading.Thread(
            target=self.run,
            daemon=True
        )

        self.thread.start()

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    def stop(self):

        self.stop_event.set()

        with self.data.lock:
            self.data.running = False

    # --------------------------------------------------------
    # MAIN CAMERA LOOP
    # --------------------------------------------------------

    def run(self):

        try:

            # Create actual project components
            self.tracker = HandTracker()
            self.gesture_engine = GestureEngine()
            self.cursor = CursorController()

            while not self.stop_event.is_set():

                # ------------------------------------------------
                # GET CAMERA FRAME
                # ------------------------------------------------

                success, frame, hand_data = self.tracker.get_frame()

                if not success or frame is None:
                    continue

                height, width, _ = frame.shape

                # ------------------------------------------------
                # HAND DETECTED
                # ------------------------------------------------

                if hand_data is not None:

                    with self.data.lock:
                        self.data.hand_detected = True

                    index_x, index_y = hand_data["index"]

                    # Convert camera position to screen position
                    raw_x, raw_y = self.cursor.map_to_screen(
                        index_x,
                        index_y,
                        width,
                        height
                    )

                    self.cursor.initialize_position(
                        raw_x,
                        raw_y
                    )

                    # ------------------------------------------------
                    # GESTURE ENGINE
                    # ------------------------------------------------

                    old_paused = self.gesture_engine.paused

                    action_data = self.gesture_engine.process(
                        hand_data
                    )

                    new_paused = self.gesture_engine.paused

                    # Pause / resume counter
                    if old_paused != new_paused:

                        with self.data.lock:
                            self.data.pause_toggles += 1

                    # ------------------------------------------------
                    # EXECUTE ACTION
                    # ------------------------------------------------

                    action = action_data.get("action")

                    if action == "click":

                        self.cursor.left_click()

                        with self.data.lock:
                            self.data.left_clicks += 1

                    elif action == "double_click":

                        self.cursor.double_click()

                        with self.data.lock:
                            self.data.double_clicks += 1

                    elif action == "right_click":

                        self.cursor.right_click()

                        with self.data.lock:
                            self.data.right_clicks += 1

                    elif action == "mouse_down":

                        self.cursor.mouse_down()

                        with self.data.lock:
                            self.data.drag_actions += 1

                    elif action == "mouse_up":

                        self.cursor.mouse_up()

                    elif action == "scroll":

                        self.cursor.scroll(
                            action_data.get("amount", 0)
                        )

                        with self.data.lock:
                            self.data.scroll_events += 1

                    # ------------------------------------------------
                    # CURSOR MOVEMENT
                    # ------------------------------------------------

                    if action_data.get("move_cursor"):

                        self.cursor.move_smoothly(
                            raw_x,
                            raw_y
                        )

                # ------------------------------------------------
                # NO HAND
                # ------------------------------------------------

                else:

                    with self.data.lock:
                        self.data.hand_detected = False

                    action_data = (
                        self.gesture_engine.handle_no_hand()
                    )

                    if action_data.get("action") == "mouse_up":
                        self.cursor.mouse_up()

                # ------------------------------------------------
                # PUBLISH VISUALIZED FRAME
                # ------------------------------------------------

                # HandTracker's returned frame is used directly.
                # This preserves its actual landmark visualization.

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                with self.data.lock:

                    self.data.frame = rgb_frame

                    self.data.state = (
                        self.gesture_engine.state
                    )

                    self.data.paused = (
                        self.gesture_engine.paused
                    )

        except Exception as e:

            print(f"Dashboard error: {e}")

        finally:

            # Safety: release mouse if dragging
            if (
                self.gesture_engine is not None
                and self.gesture_engine.state == STATE_DRAGGING
            ):
                self.cursor.mouse_up()

            # Release camera
            if self.tracker is not None:
                self.tracker.release()

            with self.data.lock:
                self.data.running = False


# ============================================================
# SESSION STATE
# ============================================================

if "system" not in st.session_state:
    st.session_state.system = DashboardSystem()

system = st.session_state.system
data = system.data


# ============================================================
# HEADER
# ============================================================

st.title("🖐️ AI-Driven Touchless Virtual Cursor")

st.caption(
    "Assistive Human-Computer Interaction System"
)

st.divider()


# ============================================================
# CONTROL BAR
# ============================================================

c1, c2, c3 = st.columns([1, 1, 2])

with c1:

    if st.button(
        "▶ Start System",
        use_container_width=True
    ):
        system.start()

with c2:

    if st.button(
        "⏹ Stop System",
        use_container_width=True
    ):
        system.stop()

with c3:

    with data.lock:
        running = data.running

    if running:
        st.success("🟢 SYSTEM RUNNING")
    else:
        st.info("⚪ SYSTEM STOPPED")


# ============================================================
# LIVE CAMERA + STATE
# ============================================================

st.subheader("Live Hand Tracking")

@st.fragment(run_every=0.20)
def live_monitor():

    camera_col, status_col = st.columns(
        [2.5, 1]
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    with camera_col:

        with data.lock:
            frame = data.frame

        if frame is not None:

            st.image(
                frame,
                channels="RGB",
                use_container_width=True
            )

        else:

            st.info(
                "Start the system to activate "
                "the camera and hand tracker."
            )

    # --------------------------------------------------------
    # STATE PANEL
    # --------------------------------------------------------

    with status_col:

        st.markdown("### System State")

        with data.lock:
            state = data.state
            paused = data.paused
            hand = data.hand_detected

        if paused:

            st.warning("⏸ PAUSED")

        elif state == STATE_DRAGGING:

            st.error("🖱️ DRAGGING")

        elif state == STATE_LEFT_HOLD:

            st.warning("🤏 LEFT HOLD")

        elif state == STATE_RIGHT_HOLD:

            st.warning("🤏 RIGHT HOLD")

        elif state == STATE_SCROLLING:

            st.info("↕️ SCROLLING")

        else:

            st.success("🟢 IDLE")

        st.divider()

        if hand:

            st.success("🖐 HAND DETECTED")

        else:

            st.warning("NO HAND")

        st.divider()

        st.markdown("### Interaction")

        m1, m2 = st.columns(2)

        with data.lock:

            left = data.left_clicks
            double = data.double_clicks
            right = data.right_clicks
            drag = data.drag_actions
            scroll = data.scroll_events
            pause = data.pause_toggles

        m1.metric("Left Click", left)
        m2.metric("Double Click", double)

        m1.metric("Right Click", right)
        m2.metric("Drag", drag)

        m1.metric("Scroll", scroll)
        m2.metric("Pause / Resume", pause)


live_monitor()


# ============================================================
# GESTURE MAPPING
# ============================================================

st.subheader("Gesture Mapping")

st.markdown(
    """
| Gesture | Action |
|---|---|
| ☝️ Index finger | Cursor movement |
| 🤏 Thumb + Index | Left click |
| 🤏 Rapid second pinch | Double click |
| 🤏 Held Thumb + Index | Drag |
| 🤏 Thumb + Middle | Right click |
| ✌️ Index + Middle | Scroll |
| ✊ Fist held | Pause / Resume |
"""
)


# ============================================================
# CONFIGURATION
# ============================================================

with st.expander("⚙️ System Parameters"):

    c1, c2 = st.columns(2)

    with c1:

        st.write(f"**Camera:** {CAM_WIDTH} × {CAM_HEIGHT}")

        st.write(
            f"**Active Region:** "
            f"{FRAME_REDUCTION}px"
        )

        st.write(
            f"**Smoothing:** "
            f"{SMOOTHING_FACTOR}"
        )

        st.write(
            f"**Cursor Deadband:** "
            f"{CURSOR_DEADBAND}px"
        )

        st.write(
            f"**Click Threshold:** "
            f"{CLICK_DISTANCE_THRESHOLD}px"
        )

        st.write(
            f"**Right Click Threshold:** "
            f"{RIGHT_CLICK_DISTANCE_THRESHOLD}px"
        )

    with c2:

        st.write(
            f"**Scroll Threshold:** "
            f"{SCROLL_DISTANCE_THRESHOLD}px"
        )

        st.write(
            f"**Click Hold:** "
            f"{CLICK_HOLD_TIME}s"
        )

        st.write(
            f"**Drag Hold:** "
            f"{DRAG_HOLD_TIME}s"
        )

        st.write(
            f"**Double Click Window:** "
            f"{DOUBLE_CLICK_WINDOW}s"
        )

        st.write(
            f"**Scroll Deadband:** "
            f"{SCROLL_DEADBAND}px"
        )

        st.write(
            f"**Fist Hold:** "
            f"{FIST_HOLD_TIME}s"
        )

        st.write(
            f"**Safe Mode:** "
            f"{'Enabled' if SAFE_MODE else 'Disabled'}"
        )


# ============================================================
# ARCHITECTURE
# ============================================================

with st.expander("🏗️ System Architecture"):

    st.code(
        """
Webcam
   │
   ▼
HandTracker
(MediaPipe)
   │
   │  Hand landmarks
   ▼
GestureEngine
(FSM + Gesture Logic)
   │
   ▼
CursorController
(EMA + PyAutoGUI)
   │
   ▼
Computer UI


Dashboard
   │
   ├── Live Hand Visualization
   ├── FSM State
   ├── Interaction Status
   ├── Gesture Mapping
   └── System Parameters
        """,
        language="text"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Driven Touchless HCI System | "
    "MediaPipe + OpenCV + PyAutoGUI + Streamlit"
)